"""Corpus B generation driver: single-SCC constant forced response.

    python corpus_b_generator.py --corpus_a_dir corpusA --out_dir corpusB --workers 8
    python corpus_b_generator.py --corpus_a_dir corpusA --out_dir corpusB --preset full --device cuda

Forcing model (spec §7.2): for each forced gene b_k, the production term is
multiplied by the constant forcing level u_k in (min(1,F_k), max(1,F_k)):
    dot(x_{b_k}) = P_{b_k} * R_{b_k}(x) * u_k - K_{b_k} * x_{b_k}

Implementation via parameter absorption: u_k is absorbed into P as
    P_forced[b_k] = P[b_k] * u_k
The solver sees an autonomous ODE with modified production rates. Zero changes
to solver.py. The forced xmax is automatically correct because prepare_params
recomputes it from the modified P values.

Performance design (over the first draft):
  - All draws for all graphs of the same n are batched into a single vectorized
    solve_draws call (matching the Corpus A generator pattern).
  - Sweeps tile all SWEEP_POINTS forcing levels into one batch solve.
  - Corpus A index (graph -> sample indices) is precomputed at worker init.
  - Each training draw gets its own fresh kinetic parameters; the sweep uses a
    single Phase-1 base draw per graph only to locate the bifurcation boundary.
  - Per-graph contributions are accumulated at the graph level, not the draw
    level, to reduce ShardBuilder allocation overhead.
  - Multi-port sweep correctly holds all other ports at midpoint while sweeping
    each port independently (bug fix from first draft).

Sections
  1. Presets and configuration
  2. Corpus A index construction
  3. Forcing specification sampling
  4. Forcing absorption into parameters
  5. Bifurcation sweep over the forcing axis (batched)
  6. Stratified forcing-level sampling
  7. Batched draw construction and solving
  8. Shard assembly
  9. Per-shard driver and multiprocess entry point

Patches applied vs original
  [P1] sample_u_vals: n0 decrement for neutral draw now uses max(0, n0-1) to
       prevent n0 going negative when n_draws is very small (e.g. 1).
  [P2] sample_u_vals: strata counts n0/n1/n2 each floored at 0, not 1,
       so the [:n_draws] clip at the end is always safe even at tiny budgets.
  [P3] _slice_results: added assertion that the total draws in results_list
       match `total`, so a future reordering of accum_meta/accum_par is caught
       immediately rather than silently corrupting fixed-point data.
  [P4] run_shard_b: added assertion that cursor reaches total_draws after
       scattering results back, guarding against accum_meta/accum_par length
       mismatch.
  [P5] sweep_one_port: dom_hopf all-NaN guard — if every sweep point is
       non-hyperbolic we skip the Hopf-delta check rather than crashing on
       np.nanmin of an all-NaN slice.
  [P6] build_corpus_a_index: cast ca_s_graph to plain ndarray once before the
       loop (mmap arrays iterate very slowly element-by-element on some numpy
       versions; the cast makes the loop ~10x faster at worker init).
  [P7] _init_worker_b: s_fp_ptr loaded from the cache key that actually exists
       (s_fp_ptr.npy, not fp_ptr.npy) — the original had an implicit assumption
       about cache layout that fails if the cache was built with an older
       train.py.
"""
import argparse
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, fields
from collections import defaultdict

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

import topology
from data import N_MAX
from generate import sample_params
from solver import (SolverConfig, corner_count, solve_draws,
                    summarize_solution, verify_batch)

# ---------------------------------------------------------------------------
# 1. Presets and configuration
# ---------------------------------------------------------------------------
PRESETS = {
    "quick": dict(draws_per_graph=[12] * 9,
                  seeds=[48] * 9),
    "valid": dict(draws_per_graph=[15, 15, 20, 18, 16, 14, 12, 10, 8],
                  seeds=[96, 128, 160, 192, 224, 256, 288, 320, 352]),
    "full":  dict(draws_per_graph=[30, 28, 25, 22, 20, 16, 14, 12, 10],
                  seeds=[96, 128, 160, 192, 224, 320, 448, 512, 576]),
    # exhaust: 2x draws vs "full" for n=2..6 to cover the full breadth of
    # conditioned dynamical behaviors; n=7-10 entries are never accessed
    # because exhaust Corpus A contains no n>6 graphs.
    "exhaust": dict(draws_per_graph=[60, 56, 50, 44, 40, 16, 14, 12, 10],
                    seeds=[96, 128, 160, 192, 224, 320, 448, 512, 576]),
}

STRATUM_FRACS = (0.40, 0.35, 0.25)
PORT_PROBS    = np.array([0.60, 0.30, 0.10])

SWEEP_POINTS      = 16
BISECT_STEPS      = 6
PROBE_SEEDS       = 16
PROBE_CORNERS     = 8
HOPF_DELTA_THRESH = 0.30
BIF_WINDOW        = 0.30
TC1_DRAW_MULTIPLIER = 2.0
INCLUDE_NEUTRAL   = True

BIF_NONE        = np.uint8(0)
BIF_SADDLE_NODE = np.uint8(1)
BIF_HOPF        = np.uint8(2)


@dataclass
class CorpusBConfig:
    corpus_a_dir:     str   = "corpusA"
    out_dir:          str   = "corpusB"
    preset:           str   = "full"
    seed:             int   = 0
    workers:          int   = 8
    threads:          int   = 1
    device:           str   = "cpu"
    graphs_per_shard: int   = 64
    elements_per_batch: int = 0
    rounds:           int   = 5
    newton_steps:     int   = 60
    eps_box:          float = 0.05
    verify:           bool  = True
    max_ports:        int   = 3
    oversample_tc1:   bool  = True


def as_bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def parse_config(argv=None):
    p = argparse.ArgumentParser()
    for f in fields(CorpusBConfig):
        kind = as_bool if isinstance(f.default, bool) else type(f.default)
        p.add_argument(f"--{f.name}", type=kind, default=f.default)
    cfg = CorpusBConfig(**vars(p.parse_args(argv)))
    if cfg.preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    return cfg


def elements_budget(cfg):
    if cfg.elements_per_batch > 0:
        return cfg.elements_per_batch
    return 12_000_000 if cfg.device.startswith("cuda") else 600_000


def batch_size(n, scfg, elements):
    k = max(scfg.seeds[n - 2] // 2, 8) + corner_count(n, scfg.max_corners)
    return int(np.clip(elements // (n * n * k), 8, 8192))


def probe_scfg(scfg):
    from generate import probe_config
    return probe_config(scfg)


# ---------------------------------------------------------------------------
# 2. Corpus A index construction
# ---------------------------------------------------------------------------
def build_corpus_a_index(ca_s_graph, ca_n_fp, ca_s_cons):
    """Build {graph_id: (sample_indices, mean_consistent_n_fp)} dict.

    [P6] Cast mmap arrays to plain ndarrays before iterating — element-wise
    access on memory-mapped arrays is ~10x slower on some numpy versions.
    """
    # [P6] cast once here instead of inside the loop
    ca_s_graph = np.asarray(ca_s_graph)
    ca_n_fp    = np.asarray(ca_n_fp)
    ca_s_cons  = np.asarray(ca_s_cons)

    idx = defaultdict(list)
    for i, g in enumerate(ca_s_graph):
        idx[int(g)].append(i)
    result = {}
    for g, idxs in idx.items():
        idxs = np.array(idxs, dtype=np.int64)
        cons_mask = ca_s_cons[idxs].astype(bool)
        if cons_mask.any():
            nfp = int(np.median(ca_n_fp[idxs[cons_mask]]))
        else:
            nfp = int(np.median(ca_n_fp[idxs]))
        result[g] = (idxs, nfp)
    return result


def get_auto_nfp(g, ca_index):
    if g in ca_index:
        return ca_index[g][1]
    return 1


def get_warm_fps(g, ca_index, ca_fp_state, ca_s_fp_ptr, ca_s_n_fp, n):
    if g not in ca_index:
        return None
    idxs, _ = ca_index[g]
    s_idx = int(idxs[len(idxs) // 2])
    k = int(ca_s_n_fp[s_idx])
    if k == 0:
        return None
    f0 = int(ca_s_fp_ptr[s_idx])
    fps = np.asarray(ca_fp_state[f0:f0 + k, :n], dtype=np.float64)
    return fps


# ---------------------------------------------------------------------------
# 3. Forcing specification sampling
# ---------------------------------------------------------------------------
def sample_port_spec(adj, n, n_ports, rng):
    edge_bool = (adj != 0).astype(np.float32)
    two_cycle = (edge_bool @ edge_bool.T).diagonal()
    self_loop = (adj.diagonal() != 0).astype(np.float32) * 2.0
    out_deg   = edge_bool.sum(axis=1)
    in_deg    = edge_bool.sum(axis=0)
    weights   = in_deg + out_deg + two_cycle + self_loop + 1.0
    weights  /= weights.sum()

    genes = rng.choice(n, size=min(n_ports, n), replace=False, p=weights)
    ports = []
    for gene in genes:
        is_act = rng.random() < 0.60
        F = float(np.exp(rng.uniform(math.log(10.0), math.log(100.0)) if is_act
                         else rng.uniform(math.log(0.01), math.log(0.3))))
        ports.append({
            "gene":   int(gene),
            "F":      F,
            "u_min":  float(min(1.0, F)),
            "u_max":  float(max(1.0, F)),
            "T":      float(np.exp(rng.uniform(math.log(0.5), math.log(5.0)))),
            "n_hill": int(rng.choice(5, p=[0.00, 0.10, 0.30, 0.35, 0.25])) + 1,
        })
    return ports


# ---------------------------------------------------------------------------
# 4. Forcing absorption into parameters
# ---------------------------------------------------------------------------
def apply_forcing(par, port_genes, u_vals):
    """Absorb forcing into P: P_forced[b_k] = P[b_k] * u_k."""
    P_forced = par["P"].copy()
    for k, gene in enumerate(port_genes):
        P_forced[:, gene] *= u_vals[:, k]
    return {**par, "P": P_forced}


# ---------------------------------------------------------------------------
# 5. Bifurcation sweep (batched)
# ---------------------------------------------------------------------------
def sweep_one_port(par_base_1, port_gene, u_min, u_max, n,
                   p_scfg, gen, device, warm_fps=None):
    u_grid = np.exp(np.linspace(
        math.log(max(u_min, 1e-6)), math.log(u_max), SWEEP_POINTS))

    rep = {k: np.repeat(v, SWEEP_POINTS, axis=0) for k, v in par_base_1.items()}
    u_batch = u_grid[:, None]
    par_sweep = apply_forcing(rep, [port_gene], u_batch)

    warm_t = None
    if warm_fps is not None and warm_fps.shape[0] > 0:
        warm_t = [warm_fps] * SWEEP_POINTS

    res = solve_draws(par_sweep, p_scfg, n, gen, device, warm=warm_t)
    fp_counts = res["counts"].astype(np.int64)
    certified = res["cons"].astype(bool)

    # res["hopf"] is FLAT over all FPs of all sweep-point draws — size = sum(fp_counts),
    # NOT SWEEP_POINTS. Reshape by draw using fp_counts to get per-draw dominant ratio.
    hopf_flat = res["hopf"]   # [sum(fp_counts)]
    dom_hopf  = np.full(SWEEP_POINTS, np.nan, dtype=np.float64)
    fp_cursor = 0
    for gi in range(SWEEP_POINTS):
        k = int(fp_counts[gi])
        if k > 0:
            hr = hopf_flat[fp_cursor : fp_cursor + k].astype(np.float64)
            # Replace 0.0 (sentinel for "no complex eigenvalues") with nan
            hr = np.where(hr == 0.0, np.nan, hr)
            if not np.all(np.isnan(hr)):
                dom_hopf[gi] = np.nanmin(hr)
            # else: stays nan — no complex eigenvalues at any FP of this draw
        fp_cursor += k

    transitions = []
    for gi in range(SWEEP_POINTS - 1):
        if not (certified[gi] and certified[gi + 1]):
            continue
        if fp_counts[gi] != fp_counts[gi + 1]:
            lo_u, hi_u = u_grid[gi], u_grid[gi + 1]
            c_lo = fp_counts[gi]
            for _ in range(BISECT_STEPS):
                mid_u = math.sqrt(lo_u * hi_u)
                par_m = apply_forcing(par_base_1, [port_gene],
                                      np.array([[mid_u]]))
                r_m = solve_draws(par_m, p_scfg, n, gen, device)
                if r_m["cons"][0] and r_m["counts"][0] == c_lo:
                    lo_u = mid_u
                else:
                    hi_u = mid_u
            transitions.append({
                "u_bif":    math.sqrt(lo_u * hi_u),
                "bif_type": BIF_SADDLE_NODE,
                "count_lo": int(c_lo),
                "count_hi": int(fp_counts[gi + 1]),
            })
        else:
            # [P5] Only check Hopf delta when both points have a finite hopf ratio
            hr_lo, hr_hi = dom_hopf[gi], dom_hopf[gi + 1]
            if (not np.isnan(hr_lo)) and (not np.isnan(hr_hi)):
                if abs(hr_hi - hr_lo) > HOPF_DELTA_THRESH:
                    transitions.append({
                        "u_bif":    math.sqrt(u_grid[gi] * u_grid[gi + 1]),
                        "bif_type": BIF_HOPF,
                        "count_lo": int(fp_counts[gi]),
                        "count_hi": int(fp_counts[gi + 1]),
                    })

    return transitions, u_grid, fp_counts, certified


def sweep_ports(ports, par_base_1, n, p_scfg, gen, device, warm_fps=None):
    n_ports = len(ports)
    port_transitions = []

    for k, port in enumerate(ports):
        if n_ports == 1:
            base_for_sweep = par_base_1
        else:
            other_genes = [ports[j]["gene"] for j in range(n_ports) if j != k]
            other_mids  = np.array([[
                math.sqrt(ports[j]["u_min"] * ports[j]["u_max"])
                for j in range(n_ports) if j != k
            ]])
            base_for_sweep = apply_forcing(par_base_1, other_genes, other_mids)

        transitions, u_grid, fp_counts, cert = sweep_one_port(
            base_for_sweep, port["gene"],
            port["u_min"], port["u_max"],
            n, p_scfg, gen, device, warm_fps)
        port_transitions.append((transitions, u_grid, fp_counts, cert))

    return port_transitions


# ---------------------------------------------------------------------------
# 6. Stratified forcing-level sampling
# ---------------------------------------------------------------------------
def sample_u_vals(ports, port_transitions, n_draws, auto_nfp, rng):
    """Sample forcing levels stratified into three strata.

    [P1] Neutral-draw decrement uses max(0, n0-1) so n0 never goes negative.
    [P2] All stratum counts are floored at 0 (not 1) so the [:n_draws] clip
         at the end is safe even when n_draws is very small.
    """
    n_ports = len(ports)
    # [P2] floor at 0, not 1 — the [:n_draws] clip below handles under-fill
    n0 = max(0, round(STRATUM_FRACS[0] * n_draws))
    n1 = max(0, round(STRATUM_FRACS[1] * n_draws))
    n2 = max(0, n_draws - n0 - n1)

    u_list    = []
    strata    = []
    bif_types = []
    bif_us    = []

    if INCLUDE_NEUTRAL:
        u_list.append(np.ones((1, n_ports)))
        strata.append(0); bif_types.append(BIF_NONE); bif_us.append(np.nan)
        # [P1] guard: n0 must not go below 0
        n0 = max(0, n0 - 1)

    # --- Stratum 0: broad log-uniform ---
    if n0 > 0:
        u0 = np.column_stack([
            np.exp(rng.uniform(
                math.log(max(p["u_min"], 1e-6)),
                math.log(p["u_max"]), n0))
            for p in ports
        ]).reshape(n0, n_ports)
        for i in range(n0):
            u_list.append(u0[i:i+1])
            strata.append(0); bif_types.append(BIF_NONE); bif_us.append(np.nan)

    # --- Stratum 1: near-bifurcation ---
    all_transitions = []
    for k, (transitions, _, _, _) in enumerate(port_transitions):
        for tr in transitions:
            all_transitions.append((k, tr))

    if all_transitions and n1 > 0:
        draws_per_tr = max(1, n1 // len(all_transitions))
        for k, tr in all_transitions:
            u_bif = tr["u_bif"]
            lo = max(math.log(max(ports[k]["u_min"], 1e-6)),
                     math.log(u_bif) - BIF_WINDOW)
            hi = min(math.log(ports[k]["u_max"]),
                     math.log(u_bif) + BIF_WINDOW)
            if lo >= hi:
                lo = math.log(max(ports[k]["u_min"], 1e-6))
                hi = math.log(ports[k]["u_max"])
            u_near = np.column_stack([
                np.exp(rng.uniform(
                    math.log(max(ports[j]["u_min"], 1e-6)),
                    math.log(ports[j]["u_max"]), draws_per_tr))
                if j != k else
                np.exp(rng.uniform(lo, hi, draws_per_tr))
                for j in range(n_ports)
            ]).reshape(draws_per_tr, n_ports)
            for i in range(draws_per_tr):
                u_list.append(u_near[i:i+1])
                strata.append(1)
                bif_types.append(tr["bif_type"])
                bif_us.append(float(u_bif))
        remaining_1 = n1 - len(all_transitions) * draws_per_tr
        if remaining_1 > 0:
            k0, tr0 = all_transitions[0]
            u_bif0 = tr0["u_bif"]
            lo0 = max(math.log(max(ports[k0]["u_min"], 1e-6)),
                      math.log(u_bif0) - BIF_WINDOW)
            hi0 = min(math.log(ports[k0]["u_max"]),
                      math.log(u_bif0) + BIF_WINDOW)
            u_fill = np.column_stack([
                np.exp(rng.uniform(
                    math.log(max(ports[j]["u_min"], 1e-6)),
                    math.log(ports[j]["u_max"]), remaining_1))
                if j != k0 else
                np.exp(rng.uniform(lo0, hi0, remaining_1))
                for j in range(n_ports)
            ]).reshape(remaining_1, n_ports)
            for i in range(remaining_1):
                u_list.append(u_fill[i:i+1])
                strata.append(1)
                bif_types.append(tr0["bif_type"])
                bif_us.append(float(u_bif0))
    elif n1 > 0:
        # No transitions found: fall back to broad draws for stratum 1
        u_fill = np.column_stack([
            np.exp(rng.uniform(
                math.log(max(p["u_min"], 1e-6)),
                math.log(p["u_max"]), n1))
            for p in ports
        ]).reshape(n1, n_ports)
        for i in range(n1):
            u_list.append(u_fill[i:i+1])
            strata.append(1); bif_types.append(BIF_NONE); bif_us.append(np.nan)

    # --- Stratum 2: opposite regime ---
    if n2 > 0:
        target_count = 1 if auto_nfp >= 3 else 3
        opp_u_candidates = []
        for k, (_, u_grid, fp_counts, cert) in enumerate(port_transitions):
            opp_mask = cert & (fp_counts == target_count)
            if opp_mask.any():
                opp_u_candidates.append((k, u_grid[opp_mask]))
        if opp_u_candidates:
            u_opp = np.column_stack([
                np.exp(rng.uniform(
                    math.log(max(ports[j]["u_min"], 1e-6)),
                    math.log(ports[j]["u_max"]), n2))
                for j in range(n_ports)
            ]).reshape(n2, n_ports)
            k_opp, opp_vals = opp_u_candidates[0]
            chosen_opp = np.exp(rng.choice(np.log(opp_vals), n2))
            jitter = rng.normal(0.0, 0.08, n2)
            u_opp[:, k_opp] = np.clip(
                np.exp(np.log(chosen_opp) + jitter),
                ports[k_opp]["u_min"], ports[k_opp]["u_max"])
            for i in range(n2):
                u_list.append(u_opp[i:i+1])
                strata.append(2); bif_types.append(BIF_NONE); bif_us.append(np.nan)
        else:
            u_fill = np.column_stack([
                np.exp(rng.uniform(
                    math.log(max(p["u_min"], 1e-6)),
                    math.log(p["u_max"]), n2))
                for p in ports
            ]).reshape(n2, n_ports)
            for i in range(n2):
                u_list.append(u_fill[i:i+1])
                strata.append(2); bif_types.append(BIF_NONE); bif_us.append(np.nan)

    if not u_list:
        # Absolute fallback: n_draws broad draws (should never happen in practice)
        u_list = [np.column_stack([
            np.exp(rng.uniform(
                math.log(max(p["u_min"], 1e-6)),
                math.log(p["u_max"]), n_draws))
            for p in ports
        ]).reshape(n_draws, n_ports)[i:i+1] for i in range(n_draws)]
        strata    = [0] * n_draws
        bif_types = [BIF_NONE] * n_draws
        bif_us    = [np.nan] * n_draws

    u_arr     = np.vstack(u_list)[:n_draws]
    strata    = np.array(strata[:n_draws],    dtype=np.int64)
    bif_types = np.array(bif_types[:n_draws], dtype=np.uint8)
    bif_us    = np.array(bif_us[:n_draws],    dtype=np.float32)

    for k, port in enumerate(ports):
        u_arr[:, k] = np.clip(u_arr[:, k], port["u_min"], port["u_max"])

    return u_arr, strata, bif_types, bif_us


# ---------------------------------------------------------------------------
# 7. Batched draw construction and solving
# ---------------------------------------------------------------------------
def build_draws_for_graph(g, adj, n, ports, u_arr, scfg, rng, seed_base):
    n_draws    = u_arr.shape[0]
    port_genes = [p["gene"] for p in ports]
    par_batch  = sample_params(1, np.repeat(adj[None], n_draws, axis=0), rng)
    par_forced = apply_forcing(par_batch, port_genes, u_arr)
    return par_forced


# ---------------------------------------------------------------------------
# 8. Shard assembly
# ---------------------------------------------------------------------------
class CorpusBShardBuilder:
    """Accumulates Corpus B data for one shard."""

    def __init__(self, topo, graph_ids):
        self.topo  = topo
        self.ids   = np.asarray(graph_ids, dtype=np.int64)
        self.local = {int(g): i for i, g in enumerate(self.ids)}
        self.parts = {}

    def _put(self, key, val):
        self.parts.setdefault(key, []).append(np.asarray(val))

    def add_graph(self, n, g, ports, u_arr, strata, bif_types, bif_us,
                  par_forced, res, auto_nfp):
        n_draws  = u_arr.shape[0]
        n_ports  = len(ports)
        g_local  = self.local[int(g)]

        port_genes = np.array([p["gene"]   for p in ports], dtype=np.uint8)
        port_F     = np.array([p["F"]      for p in ports], dtype=np.float32)
        port_T     = np.array([p["T"]      for p in ports], dtype=np.float32)
        port_nh    = np.array([p["n_hill"] for p in ports], dtype=np.uint8)

        def pad2d(a):
            """Pad a [B, n] array to [B, N_MAX] with ones (neutral kinetic value)."""
            out = np.ones((a.shape[0], N_MAX), dtype=np.float32)
            out[:, :n] = a
            return out

        self._put("s_graph",        np.full(n_draws, g_local, dtype=np.int64))
        self._put("s_P",            pad2d(par_forced["P"][:, :n].astype(np.float32)))
        self._put("s_K",            pad2d(par_forced["K"][:, :n].astype(np.float32)))
        self._put("s_xmax",         pad2d(res["xmax"].astype(np.float32)))
        self._put("s_n_fp",         res["counts"].astype(np.int32))
        self._put("s_cons",         res["cons"].astype(bool))
        self._put("s_bsum",         res["bsum"].astype(np.float32))
        self._put("s_phase",        strata.astype(np.uint8))
        self._put("cb_n_ports",     np.full(n_draws, n_ports, dtype=np.uint8))
        self._put("cb_stratum",     strata.astype(np.uint8))
        self._put("cb_bif_type",    bif_types)
        self._put("cb_bif_u",       bif_us)
        self._put("cb_bif_crossed", (res["counts"] != auto_nfp).astype(bool))
        self._put("cb_auto_nfp",    np.full(n_draws, auto_nfp, dtype=np.uint8))
        self._put("cb_port_ptr_len",np.full(n_draws, n_ports, dtype=np.int64))
        self._put("cb_port_gene",   np.tile(port_genes, n_draws))
        self._put("cb_port_u",      u_arr.astype(np.float32).ravel())
        self._put("cb_port_F",      np.tile(port_F, n_draws))
        self._put("cb_port_T",      np.tile(port_T, n_draws))
        self._put("cb_port_nh",     np.tile(port_nh, n_draws))
        mask = par_forced["adj"][0] != 0
        n_e  = int(mask.sum())
        self._put("ep_len",  np.full(n_draws, n_e, dtype=np.int64))
        self._put("ep_T",    par_forced["T"][:, mask].astype(np.float32).ravel())
        self._put("ep_nh",   par_forced["nh"][:, mask].astype(np.uint8).ravel())
        self._put("ep_F",    par_forced["F"][:, mask].astype(np.float32).ravel())
        M = res["state"].shape[0]

        def pad(a, rows):
            out = np.full((rows, N_MAX), np.nan, np.float32)
            out[:, :n] = a
            return out

        self._put("fp_state",    pad(res["state"],    M))
        self._put("fp_charpoly", pad(res["charpoly"], M))
        self._put("fp_stab",     res["stab"].astype(np.uint8))
        self._put("fp_bidx",     res["bidx"].astype(np.int8))
        self._put("fp_hopf",     res["hopf"].astype(np.float32))
        self._put("fp_zeroeig",  res["zeroeig"].astype(np.float32))
        self._put("fp_res",      res["resid"].astype(np.float32))

    def finalize(self):
        topo, ids = self.topo, self.ids
        cat  = {k: np.concatenate(v, axis=0) for k, v in self.parts.items()}
        ptr  = topo["t_edge_ptr"]
        n_e  = (ptr[ids + 1] - ptr[ids]).astype(np.int64)

        def starts(lengths):
            return np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

        def take(key):
            return np.concatenate([topo[key][ptr[g]:ptr[g+1]] for g in ids])

        port_lens = cat.pop("cb_port_ptr_len").astype(np.int64)
        ep_lens   = cat.pop("ep_len").astype(np.int64)
        n_fp_arr  = cat["s_n_fp"].astype(np.int64)

        out = {
            "g_n":        topo["t_n"][ids],
            "g_n_edges":  n_e.astype(np.uint8),
            "g_tc":       topo["t_tc"][ids],
            "g_pos_fb":   topo["t_pos"][ids],
            "g_neg_fb":   topo["t_neg"][ids],
            "g_dt":       topo["t_dt"][ids],
            "g_split":    topo["t_split"][ids],
            "g_edge_ptr": starts(n_e),
            "e_src":      take("t_src"),
            "e_tgt":      take("t_tgt"),
            "e_sgn":      take("t_sgn"),
            "s_ep_ptr":   starts(ep_lens),
            "s_fp_ptr":   starts(n_fp_arr),
            "s_cb_ptr":   starts(port_lens),
        }
        out.update(cat)
        out["_sizes"] = np.array([
            ids.size,
            out["e_src"].shape[0],
            cat["s_graph"].shape[0],
            out["ep_T"].shape[0] if "ep_T" in out else 0,
            out["fp_state"].shape[0],
            out["cb_port_gene"].shape[0],   # cb group total
        ], dtype=np.int64)
        return out


# ---------------------------------------------------------------------------
# 9. Per-shard driver and multiprocess entry point
# ---------------------------------------------------------------------------
def run_shard_b(shard_id, graph_ids, topo, ca_index,
                ca_fp_state, ca_s_fp_ptr, ca_s_n_fp, cfg):
    preset   = PRESETS[cfg.preset]
    rng      = np.random.default_rng([cfg.seed, shard_id, 99])
    device   = torch.device(cfg.device)
    gen      = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2**62)))
    scfg     = SolverConfig(
        seeds=tuple(preset["seeds"]),
        rounds=cfg.rounds,
        newton_steps=cfg.newton_steps,
        eps_box=cfg.eps_box)
    p_scfg   = probe_scfg(scfg)
    elements = elements_budget(cfg)
    builder  = CorpusBShardBuilder(topo, graph_ids)
    ns       = topo["t_n"][graph_ids].astype(np.int64)
    checked  = False

    for n in range(topology.N_MIN, N_MAX + 1):
        gn    = graph_ids[ns == n]
        if gn.size == 0:
            continue
        adj_g = topology.dense_adjacency(topo, gn, n)
        bsz   = batch_size(n, scfg, elements)
        draws_budget = preset["draws_per_graph"][n - 2]

        accum_par  = []
        accum_meta = []

        for gi, g in enumerate(gn):
            adj = adj_g[gi]
            tc  = int(topo["t_tc"][g])

            if cfg.oversample_tc1 and tc == 1:
                n_draws = min(int(draws_budget * TC1_DRAW_MULTIPLIER),
                              draws_budget * 3)
            else:
                n_draws = draws_budget

            n_ports = int(rng.choice(
                [1, 2, 3][:cfg.max_ports],
                p=PORT_PROBS[:cfg.max_ports] / PORT_PROBS[:cfg.max_ports].sum()))
            ports = sample_port_spec(adj, n, n_ports, rng)
            if not ports:
                continue

            auto_nfp   = get_auto_nfp(int(g), ca_index)
            par_base_1 = sample_params(1, adj[None], rng)
            warm_fps   = get_warm_fps(int(g), ca_index,
                                      ca_fp_state, ca_s_fp_ptr, ca_s_n_fp, n)
            port_trans = sweep_ports(ports, par_base_1, n,
                                     p_scfg, gen, device, warm_fps)
            u_arr, strata, bif_types, bif_us = sample_u_vals(
                ports, port_trans, n_draws, auto_nfp, rng)
            par_forced = build_draws_for_graph(
                g, adj, n, ports, u_arr, scfg, rng, shard_id * 100000 + gi)

            accum_par.append(par_forced)
            accum_meta.append((g, ports, u_arr, strata, bif_types, bif_us,
                                auto_nfp, u_arr.shape[0]))

        if not accum_par:
            continue

        par_all = {k: np.concatenate([p[k] for p in accum_par], axis=0)
                   for k in accum_par[0]}
        total_draws = par_all["P"].shape[0]

        results_all = []
        for s in range(0, total_draws, bsz):
            par_sl = {k: v[s:s + bsz] for k, v in par_all.items()}
            res_sl = solve_draws(par_sl, scfg, n, gen, device)
            if cfg.verify and not checked:
                verify_batch(par_sl, res_sl, scfg, device)
                st = summarize_solution(res_sl)
                print(f"  shard {shard_id} B verified n={n}: "
                      f"mean_fp {st['mean_fp']:.2f}  "
                      f"certified {100*st['certified']:.1f}%  "
                      f"multistable {100*st['multistable']:.1f}%", flush=True)
                checked = True
            results_all.append(res_sl)

        # [P4] Integrity check: meta list must account for exactly total_draws
        meta_total = sum(nd for (*_, nd) in accum_meta)
        assert meta_total == total_draws, (
            f"shard {shard_id} n={n}: accum_meta draws ({meta_total}) "
            f"!= par_all draws ({total_draws}). "
            "Check that accum_meta and accum_par are appended in lockstep."
        )

        cursor = 0
        for (g, ports, u_arr, strata, bif_types, bif_us, auto_nfp, nd), par_g \
                in zip(accum_meta, accum_par):
            res_g = _slice_results(results_all, cursor, nd, total_draws)
            builder.add_graph(n, g, ports, u_arr, strata, bif_types, bif_us,
                               par_g, res_g, auto_nfp)
            cursor += nd

        # [P4] Confirm full consumption
        assert cursor == total_draws, (
            f"shard {shard_id} n={n}: cursor {cursor} != total_draws {total_draws} "
            "after scattering results. Logic error in the scatter loop."
        )

    return builder.finalize()


def _slice_results(results_list, start, length, total):
    """Extract rows [start:start+length] from a list of result batches.

    [P3] Assert that the concatenated results cover exactly `total` draws so
    a future reordering of accum_meta/accum_par is caught immediately.
    """
    draw_counts = np.concatenate([r["counts"] for r in results_list])
    draw_cons   = np.concatenate([r["cons"]   for r in results_list])
    draw_bsum   = np.concatenate([r["bsum"]   for r in results_list])
    draw_xmax   = np.concatenate([r["xmax"]   for r in results_list], axis=0)

    # [P3] Structural assertion — all results_list entries must cover `total` draws
    assert draw_counts.shape[0] == total, (
        f"_slice_results: concatenated draw count ({draw_counts.shape[0]}) "
        f"!= expected total ({total}). results_list may be out of order."
    )

    target_counts = draw_counts[start:start + length]
    out = {
        "counts": target_counts,
        "cons":   draw_cons[start:start + length],
        "bsum":   draw_bsum[start:start + length],
        "xmax":   draw_xmax[start:start + length],
    }

    fp_starts = int(np.cumsum(draw_counts)[start - 1]) if start > 0 else 0
    fp_end    = fp_starts + int(target_counts.sum())

    for key in ("state", "charpoly", "stab", "bidx", "hopf", "zeroeig", "resid"):
        arr = np.concatenate([r[key] for r in results_list], axis=0)
        out[key] = arr[fp_starts:fp_end]

    return out


# ---- Worker process globals and pool plumbing ----

_TOPO_B  = None
_CFG_B   = None
_CA_IDX  = None
_CA_FPS  = None
_CA_FPTR = None
_CA_NFP  = None


def _init_worker_b(topo_path, cfg_dict, threads, cache_dir):
    global _TOPO_B, _CFG_B, _CA_IDX, _CA_FPS, _CA_FPTR, _CA_NFP
    torch.set_num_threads(threads)
    _TOPO_B = topology.load_topologies(topo_path)
    _CFG_B  = CorpusBConfig(**cfg_dict)

    ca_s_graph = np.load(os.path.join(cache_dir, "s_graph.npy"), mmap_mode="r")
    ca_n_fp    = np.load(os.path.join(cache_dir, "s_n_fp.npy"),  mmap_mode="r")
    ca_s_cons  = np.load(os.path.join(cache_dir, "s_cons.npy"),  mmap_mode="r")

    _CA_IDX  = build_corpus_a_index(
        np.asarray(ca_s_graph), np.asarray(ca_n_fp), np.asarray(ca_s_cons))

    _CA_FPS  = np.load(os.path.join(cache_dir, "fp_state.npy"),  mmap_mode="r")
    # [P7] Use the correct cache key (s_fp_ptr, not fp_ptr)
    _CA_FPTR = np.load(os.path.join(cache_dir, "s_fp_ptr.npy"),  mmap_mode="r")
    _CA_NFP  = ca_n_fp


def _work_b(job):
    shard_id, ids, path = job
    t0 = time.time()
    arrays = run_shard_b(shard_id, ids, _TOPO_B, _CA_IDX,
                         _CA_FPS, _CA_FPTR, _CA_NFP, _CFG_B)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    sizes = arrays["_sizes"]
    cons  = float(arrays["s_cons"].mean())  if arrays["s_cons"].size  else 0.0
    bif   = float(arrays["cb_bif_crossed"].mean()) if "cb_bif_crossed" in arrays else 0.0
    return shard_id, int(sizes[2]), int(sizes[4]), cons, bif, time.time() - t0


def generate_b(cfg):
    shard_dir = os.path.join(cfg.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    topo_path = os.path.join(cfg.corpus_a_dir, "topologies.npz")
    cache_dir = os.path.join(cfg.corpus_a_dir, "cache")
    if not os.path.exists(topo_path):
        raise FileNotFoundError(f"topologies.npz not found in {cfg.corpus_a_dir}")
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(
            f"Corpus A cache not found at {cache_dir}. "
            f"Run: python train.py --mode cache --corpus_dir {cfg.corpus_a_dir}")

    # [P7] Verify the specific cache arrays needed at worker-init time exist
    required_cache_files = ["s_graph.npy", "s_n_fp.npy", "s_cons.npy",
                            "fp_state.npy", "s_fp_ptr.npy"]
    missing = [f for f in required_cache_files
               if not os.path.exists(os.path.join(cache_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing cache arrays: {missing}. "
            f"Re-run: python train.py --mode cache --corpus_dir {cfg.corpus_a_dir}")

    topo     = topology.load_topologies(topo_path)
    n_graphs = int(topo["t_n"].shape[0])
    rng      = np.random.default_rng(cfg.seed)
    order    = rng.permutation(n_graphs)
    chunks   = [np.sort(order[i:i + cfg.graphs_per_shard])
                for i in range(0, n_graphs, cfg.graphs_per_shard)]
    jobs     = [(i, ids, os.path.join(shard_dir, f"shard_b_{i:04d}.npz"))
                for i, ids in enumerate(chunks)
                if not os.path.exists(
                    os.path.join(shard_dir, f"shard_b_{i:04d}.npz"))]

    preset = PRESETS[cfg.preset]
    avg_d  = np.mean(preset["draws_per_graph"])
    print(f"Corpus B: {n_graphs} graphs, {len(chunks)} shards, "
          f"{len(jobs)} to generate ({len(chunks)-len(jobs)} cached), "
          f"device {cfg.device}", flush=True)
    print(f"Avg {avg_d:.1f} draws/graph -> ~{int(n_graphs*avg_d):,} samples", flush=True)

    t_start, done, n_samp = time.time(), 0, 0

    def report(r):
        nonlocal done, n_samp
        done += 1; n_samp += r[1]
        rate = (time.time() - t_start) / done
        print(f"shard {r[0]:4d}: {r[1]:7d} samples  {r[2]:8d} FPs  "
              f"cert {100*r[3]:5.1f}%  bif_crossed {100*r[4]:5.1f}%  "
              f"{r[5]:5.1f}s | {done}/{len(jobs)}  "
              f"eta {rate*(len(jobs)-done)/60:5.1f} min", flush=True)

    if cfg.workers <= 1:
        _init_worker_b(topo_path, asdict(cfg), cfg.threads, cache_dir)
        for job in jobs:
            report(_work_b(job))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(cfg.workers,
                      initializer=_init_worker_b,
                      initargs=(topo_path, asdict(cfg), cfg.threads, cache_dir)) as pool:
            for r in pool.imap_unordered(_work_b, jobs):
                report(r)

    print(f"Corpus B: {n_samp:,} samples in {(time.time()-t_start)/60:.1f} min",
          flush=True)


def main():
    generate_b(parse_config())


if __name__ == "__main__":
    main()

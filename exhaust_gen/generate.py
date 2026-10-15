"""Corpus A generation driver.

    python generate.py --preset quick --out_dir corpusQ --workers 4
    python generate.py --preset full  --out_dir corpusA --device cuda --workers 4
    python generate.py --mode audit   --preset full --out_dir corpusA --device cuda

Sections
  1. Presets and configuration
  2. Parameter samplers for the three draw phases
  3. Boundary-guided continuation (phase 2)
  4. Shard assembly
  5. Per-shard generation, the multiprocess driver, and the completeness audit

Draw phases (every sample is an independent parameter draw with its own complete label set)
  0 broad       independent log-uniform parameters, the naive prior
  1 switch      high cooperativity, strong fold-changes, and thresholds placed inside the
                expression range of the regulating gene, so the regulation is actually used
  2 boundary    one parameter of a phase-1 draw is scanned along a line, the change of the
                fixed-point count is bracketed by bisection on the SAME parameter vector, and
                samples are emitted on either side at distances from 1% to 50% of the scan
                half-width, so near-boundary labels stay learnable. Every emitted sample is
                then solved from scratch with the full seed budget, so its labels do not
                depend on how it was found.

Where the time goes
  A phase-2 sample costs far more than a phase-0 or phase-1 sample because locating a boundary
  takes about 8 + 6 probe solves per base draw. Three things keep that cheap: probes are
  warm-started from the base draw's own fixed points plus a few random seeds, so they need a
  fraction of the full seed budget; half of the base draws are chosen among the multistable
  phase-1 draws, which have a boundary nearby; and the probe grid is coarse.

Output layout: <out_dir>/topologies.npz and <out_dir>/shards/shard_XXXX.npz, which
data.build_cache merges into the memory-mapped training cache.
"""
import argparse
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, fields, replace

# One thread per worker process. Without this every worker's BLAS (used by numpy's
# eigenvalue routine) starts one thread per core and the workers fight over the machine.
# Set before numpy and torch are imported; spawned workers inherit it.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch

import topology
from data import N_MAX, load_sample, open_cache
from solver import (SolverConfig, corner_count, solve_draws, summarize_solution,
                    verify_batch)

# ---------------------------------------------------------------------------
# 1. Presets and configuration
# ---------------------------------------------------------------------------
PRESETS = {
    # topos: quota for n = 4 .. 10 (n = 2, 3 are enumerated); draws, seeds: n = 2 .. 10
    "quick": dict(topos=[12, 12, 8, 8, 8, 8, 8], draws=[16] * 9, seeds=[48] * 9),
    "valid": dict(topos=[150, 150, 100, 80, 80, 60, 60],
                  draws=[20, 6, 32, 28, 24, 20, 16, 16, 12],
                  seeds=[96, 128, 160, 192, 224, 256, 288, 320, 352]),
    "full": dict(topos=[6000, 10000, 7000, 5000, 4000, 3000, 2400],
                 draws=[120, 100, 80, 30, 25, 20, 16, 12, 10],
                 seeds=[96, 128, 160, 192, 224, 320, 448, 512, 576]),
    # exhaust: same topology breadth as "full" (n=2,3 exhaustive; n=4-10 sampled),
    #          but 2x the parameter draws per topology (~40 for large n) to improve
    #          dynamical coverage — the 4-5 dynamical regimes each topology admits
    #          are captured by parameter breadth, not structural enumeration.
    "exhaust": dict(
        topos=[6000, 10000, 7000, 5000, 4000, 3000, 2400],
        draws=[240, 200, 160, 60, 50, 40, 32, 24, 20],
        seeds=[96, 128, 160, 192, 224, 320, 448, 512, 576],
    ),
}
PHASE_FRACS = (0.30, 0.50, 0.20)
HILL_WEIGHTS = {0: [0.10, 0.20, 0.30, 0.25, 0.15],      # probability of n_Hill = 1 .. 5
                1: [0.00, 0.10, 0.30, 0.35, 0.25]}
GRID_POINTS = 8                                           # scan points per boundary search
BISECT_STEPS = 4                                          # bracket of 2 / 7 / 2**4 of the scan half-width
MIN_BOUNDARY_DISTANCE = 0.01                              # emitted samples lie this far (fraction of the scan
MAX_BOUNDARY_DISTANCE = 0.5                               # half-width) to this far from the located boundary
PROBE_SEEDS = 24                                          # random seeds per probe solve (12 in its one round)
PROBE_CORNERS = 16
KIND_LIMITS = (math.log(5.0), math.log(6.0), math.log(8.0))   # scan half-width: edge F, edge T, gene P


@dataclass
class GenConfig:
    preset: str = "quick"
    out_dir: str = "corpusA"
    mode: str = "generate"            # generate | audit
    seed: int = 0
    workers: int = 8
    threads: int = 1                  # torch threads per worker
    device: str = "cpu"               # cpu | cuda
    graphs_per_shard: int = 64
    elements_per_batch: int = 0       # draws x seeds x n^2 solved together; 0 = automatic by device
    rounds: int = 5
    newton_steps: int = 60
    eps_box: float = 0.05
    verify: bool = True               # independent checks on the first batch of every shard
    audit_samples: int = 2000         # samples re-solved by --mode audit


def as_bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    for f in fields(GenConfig):
        kind = as_bool if isinstance(f.default, bool) else type(f.default)
        parser.add_argument(f"--{f.name}", type=kind, default=f.default)
    cfg = GenConfig(**vars(parser.parse_args(argv)))
    if cfg.preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    if cfg.mode not in ("generate", "audit"):
        raise ValueError("mode must be generate or audit")
    return cfg


def phase_counts(total):
    n0 = int(round(PHASE_FRACS[0] * total))
    n1 = int(round(PHASE_FRACS[1] * total))
    return n0, n1, max(total - n0 - n1, 0)


def elements_budget(cfg):
    """Elements of one [B, K, N, N] tensor per solve. The memory of a solve is about fifteen
    such tensors in float64: 600k elements is cache-sized for a CPU core, 12M is about 1.5 GB."""
    if cfg.elements_per_batch > 0:
        return cfg.elements_per_batch
    return 12_000_000 if cfg.device.startswith("cuda") else 600_000


def batch_size(n, scfg, elements):
    """Draws per solve for gene count n: round-0 seeds are seeds / 2 random plus the corners."""
    k = max(scfg.seeds[n - 2] // 2, 8) + corner_count(n, scfg.max_corners)
    return int(np.clip(elements // (n * n * k), 8, 8192))


def probe_config(scfg):
    """Cheap solver settings for locating boundaries: one round, few random seeds, few corners."""
    return replace(scfg, seeds=(PROBE_SEEDS,) * len(scfg.seeds), rounds=1, confirm=False,
                   max_corners=PROBE_CORNERS)


# ---------------------------------------------------------------------------
# 2. Parameter samplers
# ---------------------------------------------------------------------------
def quantize(par):
    """Round parameters through float32. The cache stores float32, so rounding before the
    solve makes the stored parameters identical to the ones the fixed points belong to."""
    for k in ("P", "K", "T", "F"):
        par[k] = par[k].astype(np.float32).astype(np.float64)
    return par


def sample_params(phase, adj, rng):
    """Parameters for B draws of graphs with signed adjacency adj [B, N, N] (int8).

    P in [0.01, 100], K in [0.01, 10]. Activation fold-changes F > 1, repression F < 1.
    Phase 0 draws the threshold T log-uniformly in [0.1, 10], independent of the expression
    scale; for most draws the regulation then sits entirely above or below the expression
    range and the network is trivially monostable. Phases 1 and 2 place T within a factor 6
    of the geometric middle of the regulating gene's expression range,
    (P / K) * sqrt(product of the fold-changes into that gene)."""
    B, N, _ = adj.shape
    edge, act = adj != 0, adj > 0

    def logu(lo, hi, shape):
        return np.exp(rng.uniform(math.log(lo), math.log(hi), shape))

    P = logu(0.01, 100.0, (B, N))
    Kd = logu(0.01, 10.0, (B, N))
    if phase == 0:
        f_act, f_rep = (2.0, 100.0), (0.01, 0.5)
    else:
        f_act, f_rep = (10.0, 100.0), (0.01, 0.2)
    F = np.where(edge, np.where(act, logu(*f_act, (B, N, N)), logu(*f_rep, (B, N, N))), 1.0)
    nh = rng.choice(5, size=(B, N, N), p=HILL_WEIGHTS[min(phase, 1)]) + 1.0
    nh = np.where(edge, nh, 1.0)
    if phase == 0:
        T = logu(0.1, 10.0, (B, N, N))
    else:
        middle = P / Kd * np.sqrt(F.prod(axis=1))
        T = middle[:, :, None] * 10.0 ** rng.uniform(-0.8, 0.8, (B, N, N))
    T = np.where(edge, np.clip(T, 1e-3, 1e9), 1.0)
    return quantize({"P": P, "K": Kd, "adj": adj, "T": T, "nh": nh, "F": F})


# ---------------------------------------------------------------------------
# 3. Boundary-guided continuation
# ---------------------------------------------------------------------------
def apply_multiplier(par, kind, src, tgt, node, mult):
    """Copy of par with one parameter of each row scaled by mult.

    kind 0 scales the fold-change of edge (src, tgt) while keeping it on its side of 1, so an
    activation stays an activation; kind 1 scales the threshold of that edge; kind 2 scales
    the production rate of gene `node`."""
    out = {k: v.copy() for k, v in par.items()}
    r = np.arange(mult.shape[0])
    m0, m1, m2 = kind == 0, kind == 1, kind == 2
    if m0.any():
        F = out["F"][r[m0], src[m0], tgt[m0]]
        scaled = F * mult[m0]
        out["F"][r[m0], src[m0], tgt[m0]] = np.where(F > 1.0, np.maximum(scaled, 1.05), np.minimum(scaled, 0.95))
    if m1.any():
        out["T"][r[m1], src[m1], tgt[m1]] *= mult[m1]
    if m2.any():
        out["P"][r[m2], node[m2]] *= mult[m2]
    return quantize(out)


def locate_boundaries(base, warm, n, probe, gen, device, rng):
    """Samples straddling a change of the fixed-point count of each base draw.

    For every base draw one parameter is scanned over GRID_POINTS values; between two adjacent
    certified grid points whose fixed-point counts differ, the interval is bisected
    BISECT_STEPS times. Every probe solve is warm-started from the fixed points of the base
    draw (warm: list of arrays [c, N]), which follow the scan until a fold; random seeds
    catch the roots that are born at the fold. Two parameter sets per located boundary are
    returned, each on a random side of it at a distance between MIN_BOUNDARY_DISTANCE and
    MAX_BOUNDARY_DISTANCE of the scan half-width. Returns (params, base_index) or None if no
    base draw showed a change."""
    Bb = base["P"].shape[0]
    kind = rng.integers(0, 3, Bb)
    flat = (rng.random(base["adj"].shape) * (base["adj"] != 0)).reshape(Bb, -1).argmax(axis=1)
    src, tgt, node = flat // n, flat % n, rng.integers(0, n, Bb)
    limit = np.array(KIND_LIMITS)[kind]
    grid = np.linspace(-1.0, 1.0, GRID_POINTS)
    rep = np.repeat(np.arange(Bb), GRID_POINTS)
    scan = apply_multiplier({k: v[rep] for k, v in base.items()}, kind[rep], src[rep], tgt[rep],
                            node[rep], np.exp(limit[rep] * np.tile(grid, Bb)))
    res = solve_draws(scan, probe, n, gen, device, warm=[warm[i] for i in rep])
    count = res["counts"].reshape(Bb, GRID_POINTS)
    good = res["cons"].reshape(Bb, GRID_POINTS)
    change = good[:, :-1] & good[:, 1:] & (count[:, :-1] != count[:, 1:])
    sel = np.nonzero(change.any(axis=1))[0]
    if sel.size == 0:
        return None
    j = (rng.random(change.shape) * change).argmax(axis=1)[sel]
    lo, hi = grid[j], grid[j + 1]
    c_lo = count[sel, j]
    pb = {k: v[sel] for k, v in base.items()}
    kk, ss, tt, nn, ll = kind[sel], src[sel], tgt[sel], node[sel], limit[sel]
    warm_sel = [warm[i] for i in sel]
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        rm = solve_draws(apply_multiplier(pb, kk, ss, tt, nn, np.exp(ll * mid)), probe, n, gen, device,
                         warm=warm_sel)
        same = rm["cons"] & (rm["counts"] == c_lo)
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    # Two samples per boundary, each on a random side at a log-uniform distance from it. Samples
    # at the bracket ends themselves would come in pairs with nearly identical inputs and
    # different fixed-point counts, a label that no model can predict and that a large one
    # memorizes. Spread over distances, the label depends on the input in a learnable way.
    mid = 0.5 * (lo + hi)
    idx = np.concatenate([np.arange(sel.size), np.arange(sel.size)])
    dist = np.exp(rng.uniform(math.log(MIN_BOUNDARY_DISTANCE), math.log(MAX_BOUNDARY_DISTANCE), idx.size))
    t = np.clip(mid[idx] + rng.choice([-1.0, 1.0], size=idx.size) * dist, -1.5, 1.5)
    par = apply_multiplier({k: v[idx] for k, v in pb.items()}, kk[idx], ss[idx], tt[idx], nn[idx],
                           np.exp(ll[idx] * t))
    return par, np.concatenate([sel, sel])


# ---------------------------------------------------------------------------
# 4. Shard assembly
# ---------------------------------------------------------------------------
class ShardBuilder:
    """Accumulates solved batches and writes them in the layout data.build_cache merges.

    Samples of a shard are stored batch after batch; the fixed points and edge parameters of
    each sample are contiguous and reached through start pointers."""

    def __init__(self, topo, graph_ids):
        self.topo = topo
        self.ids = np.asarray(graph_ids, dtype=np.int64)
        self.local = {int(g): i for i, g in enumerate(self.ids)}
        self.parts = {}

    def _put(self, key, value):
        self.parts.setdefault(key, []).append(value)

    def add(self, n, g_b, phase, par, res):
        B = len(g_b)
        mask = par["adj"] != 0

        def pad(a, rows):
            out = np.full((rows, N_MAX), np.nan, dtype=np.float32)
            out[:, :n] = a
            return out

        self._put("s_graph", np.array([self.local[int(g)] for g in g_b], dtype=np.int64))
        self._put("s_P", pad(par["P"], B))
        self._put("s_K", pad(par["K"], B))
        self._put("s_xmax", pad(res["xmax"], B))
        self._put("s_n_fp", res["counts"].astype(np.int32))
        self._put("s_cons", res["cons"].astype(bool))
        self._put("s_bsum", res["bsum"].astype(np.float32))
        self._put("s_phase", np.full(B, phase, dtype=np.uint8))
        self._put("ep_len", mask.sum(axis=(1, 2)).astype(np.int64))
        self._put("ep_T", par["T"][mask].astype(np.float32))
        self._put("ep_nh", par["nh"][mask].astype(np.uint8))
        self._put("ep_F", par["F"][mask].astype(np.float32))
        M = res["state"].shape[0]
        self._put("fp_state", pad(res["state"], M))
        self._put("fp_charpoly", pad(res["charpoly"], M))
        self._put("fp_stab", res["stab"].astype(np.uint8))
        self._put("fp_bidx", res["bidx"].astype(np.int8))
        self._put("fp_hopf", res["hopf"].astype(np.float32))
        self._put("fp_zeroeig", res["zeroeig"].astype(np.float32))
        self._put("fp_res", res["resid"].astype(np.float32))

    def finalize(self):
        t, ids = self.topo, self.ids
        cat = {k: np.concatenate(v, axis=0) for k, v in self.parts.items()}
        ptr = t["t_edge_ptr"]
        n_e = (ptr[ids + 1] - ptr[ids]).astype(np.int64)
        starts = lambda lengths: np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        take = lambda key: np.concatenate([t[key][ptr[g]:ptr[g + 1]] for g in ids])
        out = {
            "g_n": t["t_n"][ids], "g_n_edges": n_e.astype(np.uint8), "g_tc": t["t_tc"][ids],
            "g_pos_fb": t["t_pos"][ids], "g_neg_fb": t["t_neg"][ids], "g_dt": t["t_dt"][ids],
            "g_split": t["t_split"][ids], "g_edge_ptr": starts(n_e),
            "e_src": take("t_src"), "e_tgt": take("t_tgt"), "e_sgn": take("t_sgn"),
            "s_ep_ptr": starts(cat.pop("ep_len")), "s_fp_ptr": starts(cat["s_n_fp"]),
        }
        out.update(cat)
        out["_sizes"] = np.array([ids.size, out["e_src"].shape[0], out["s_graph"].shape[0],
                                  out["ep_T"].shape[0], out["fp_state"].shape[0]], dtype=np.int64)
        return out


# ---------------------------------------------------------------------------
# 5. Shard generation, driver, audit
# ---------------------------------------------------------------------------
def split_roots(res):
    """Per-draw arrays of fixed points from the flat arrays of a solve."""
    return np.split(res["state"], np.cumsum(res["counts"])[:-1])


def run_boundary_phase(builder, n, per_graph, n_graphs, pool, scfg, probe, bsz, probe_bsz, gen, device, rng):
    """Phase 2: bases are phase-1 draws, half of them multistable when such draws exist.

    pool entries are (graphs, params, roots per draw, counts, certified flags). Emitted
    boundary samples are solved from scratch with the full solver settings."""
    g_pool = np.concatenate([e[0] for e in pool])
    par_pool = {k: np.concatenate([e[1][k] for e in pool]) for k in pool[0][1]}
    roots_pool = [r for e in pool for r in e[2]]
    multi = np.nonzero(np.concatenate([e[4] & (e[3] >= 3) for e in pool]))[0]
    target = n_graphs * per_graph
    cap = max(1, probe_bsz // GRID_POINTS)           # each base becomes GRID_POINTS probe draws
    per_base, n_bases = 0.8, 0                       # emitted samples per base draw, updated as bases are used
    got_par, got_g, n_got = [], [], 0
    for _ in range(30):
        if n_got >= target:
            break
        # Only as many bases as the samples still missing require. Sizing by batch capacity
        # instead computes up to a thousand bases when a shard needs a handful of samples.
        bases = int(np.clip(math.ceil(1.25 * (target - n_got) / per_base), 4, cap))
        n_bases += bases
        n_multi = bases // 2 if multi.size else 0
        pick = np.concatenate([rng.choice(multi, n_multi) if n_multi else np.zeros(0, dtype=np.int64),
                               rng.integers(0, g_pool.shape[0], bases - n_multi)])
        out = locate_boundaries({k: v[pick] for k, v in par_pool.items()}, [roots_pool[i] for i in pick],
                                n, probe, gen, device, rng)
        if out is not None:
            par_e, src_idx = out
            got_par.append(par_e)
            got_g.append(g_pool[pick][src_idx])
            n_got += src_idx.shape[0]
        per_base = max(n_got / n_bases, 0.05)
    if n_got == 0:
        return
    par_e = {k: np.concatenate([p[k] for p in got_par])[:target] for k in got_par[0]}
    g_e = np.concatenate(got_g)[:target]
    for s in range(0, g_e.shape[0], bsz):
        par = {k: v[s:s + bsz] for k, v in par_e.items()}
        builder.add(n, g_e[s:s + bsz], 2, par, solve_draws(par, scfg, n, gen, device))


def run_shard(shard_id, graph_ids, topo, cfg):
    preset = PRESETS[cfg.preset]
    rng = np.random.default_rng([cfg.seed, shard_id])
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2 ** 62)))
    scfg = SolverConfig(seeds=tuple(preset["seeds"]), rounds=cfg.rounds,
                        newton_steps=cfg.newton_steps, eps_box=cfg.eps_box)
    probe = probe_config(scfg)
    elements = elements_budget(cfg)
    builder = ShardBuilder(topo, graph_ids)
    ns = topo["t_n"][graph_ids].astype(np.int64)
    checked = False
    clock = np.zeros((N_MAX + 1, 3))        # seconds by (n, phase)
    for n in range(topology.N_MIN, N_MAX + 1):
        gn = graph_ids[ns == n]
        if gn.size == 0:
            continue
        adj_g = topology.dense_adjacency(topo, gn, n)
        bsz = batch_size(n, scfg, elements)
        counts = phase_counts(preset["draws"][n - 2])
        pool = []
        for phase in (0, 1):
            if counts[phase] == 0:
                continue
            g_rep = np.repeat(gn, counts[phase])
            a_rep = np.repeat(adj_g, counts[phase], axis=0)
            for s in range(0, g_rep.shape[0], bsz):
                t0 = time.time()
                par = sample_params(phase, a_rep[s:s + bsz], rng)
                res = solve_draws(par, scfg, n, gen, device)
                if cfg.verify and not checked:
                    verify_batch(par, res, scfg, device)
                    st = summarize_solution(res)
                    print(f"  shard {shard_id} verified on n={n}: mean_fp {st['mean_fp']:.2f} max_fp {st['max_fp']} "
                          f"certified {100 * st['certified']:.1f}% multistable {100 * st['multistable']:.1f}% "
                          f"seeds converged {100 * st['seed_ok']:.1f}%", flush=True)
                    checked = True
                builder.add(n, g_rep[s:s + bsz], phase, par, res)
                if phase == 1:
                    pool.append((g_rep[s:s + bsz], par, split_roots(res), res["counts"], res["cons"]))
                clock[n, phase] += time.time() - t0
        if counts[2] > 0 and pool:
            t0 = time.time()
            run_boundary_phase(builder, n, counts[2], gn.size, pool, scfg, probe, bsz,
                               batch_size(n, probe, elements), gen, device, rng)
            clock[n, 2] += time.time() - t0
    return builder.finalize(), clock


_TOPO = None
_CFG = None


def _init_worker(topo_path, cfg_dict, threads):
    global _TOPO, _CFG
    torch.set_num_threads(threads)
    _TOPO = topology.load_topologies(topo_path)
    _CFG = GenConfig(**cfg_dict)


def _work(job):
    shard_id, ids, path = job
    t0 = time.time()
    arrays, clock = run_shard(shard_id, ids, _TOPO, _CFG)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    sizes = arrays["_sizes"]
    cons = float(arrays["s_cons"].mean())
    return shard_id, int(sizes[2]), int(sizes[4]), cons, time.time() - t0, clock


def print_clock(clock_total):
    print("core-seconds by n:  broad    switch  boundary   (all workers summed)")
    for n in range(topology.N_MIN, N_MAX + 1):
        if clock_total[n].sum() > 0:
            print(f"  n={n:2d}: {clock_total[n, 0]:9.0f} {clock_total[n, 1]:9.0f} {clock_total[n, 2]:9.0f}")
    print(f"  total: {clock_total.sum():9.0f} core-seconds", flush=True)


def generate(cfg):
    preset = PRESETS[cfg.preset]
    shard_dir = os.path.join(cfg.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    topo_path = os.path.join(cfg.out_dir, "topologies.npz")
    if not os.path.exists(topo_path):
        exhaust_n = preset.get("exhaust_n", 3)
        topology.build_and_save(preset["topos"], cfg.seed, topo_path, exhaust_n=exhaust_n)
    topo = topology.load_topologies(topo_path)
    n_graphs = int(topo["t_n"].shape[0])
    order = np.random.default_rng(cfg.seed).permutation(n_graphs)
    chunks = [np.sort(order[i:i + cfg.graphs_per_shard]) for i in range(0, n_graphs, cfg.graphs_per_shard)]
    jobs = [(i, ids, os.path.join(shard_dir, f"shard_{i:04d}.npz")) for i, ids in enumerate(chunks)
            if not os.path.exists(os.path.join(shard_dir, f"shard_{i:04d}.npz"))]
    print(f"{n_graphs} graphs, {len(chunks)} shards, {len(jobs)} to generate "
          f"({len(chunks) - len(jobs)} already present), device {cfg.device}", flush=True)

    t_start, done, n_samples = time.time(), 0, 0
    clock_total = np.zeros((N_MAX + 1, 3))

    def report(r):
        nonlocal done, n_samples
        done += 1
        n_samples += r[1]
        clock_total[:] += r[5]
        rate = (time.time() - t_start) / done
        print(f"shard {r[0]:4d}: {r[1]:7d} samples, {r[2]:8d} fixed points, certified {100 * r[3]:5.1f}%, "
              f"{r[4]:6.1f}s | {done}/{len(jobs)} done, eta {rate * (len(jobs) - done) / 60:6.1f} min", flush=True)
        if done % 20 == 0:
            print_clock(clock_total)

    if cfg.workers <= 1:
        _init_worker(topo_path, asdict(cfg), cfg.threads)
        for job in jobs:
            report(_work(job))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(cfg.workers, initializer=_init_worker,
                      initargs=(topo_path, asdict(cfg), cfg.threads)) as pool:
            for r in pool.imap_unordered(_work, jobs):
                report(r)
    print(f"generated {n_samples} samples in {(time.time() - t_start) / 60:.1f} min", flush=True)
    print_clock(clock_total)


def audit(cfg):
    """Estimate how many fixed points the production seed budget misses.

    A random subset of cached samples is solved again with four times the production seeds
    and up to three rounds. Reported per n: the share of samples in which the larger budget
    found additional fixed points, the mean number found, the share in which it found fewer
    (clustering differences), and the share certified. The index sum cannot see a missed
    saddle-node pair, so this is the measurement of label completeness. It needs the cache
    built from <out_dir> (train.py --mode cache)."""
    preset = PRESETS[cfg.preset]
    device = torch.device(cfg.device)
    a = open_cache(os.path.join(cfg.out_dir, "cache"))
    n_fp = np.asarray(a["s_n_fp"])
    g_n = np.asarray(a["g_n"])[np.asarray(a["s_graph"])]
    rng = np.random.default_rng(cfg.seed + 1)
    cand = np.nonzero(n_fp >= 1)[0]
    pick = rng.choice(cand, size=min(cfg.audit_samples, cand.size), replace=False)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 2)
    heavy = SolverConfig(seeds=tuple(4 * s for s in preset["seeds"]), rounds=3,
                         newton_steps=cfg.newton_steps, eps_box=cfg.eps_box)
    elements = elements_budget(cfg)
    print(f"audit: {pick.size} samples re-solved with 4x seeds")
    print("  n | samples | gained fixed points % | mean gained | fewer % | certified now %")
    gained_total, count_total = 0, 0
    for n in range(topology.N_MIN, N_MAX + 1):
        sel = pick[g_n[pick] == n]
        if sel.size == 0:
            continue
        d = [load_sample(a, int(s), int(n_fp[s])) for s in sel]
        P = np.stack([x["P"][:n] for x in d]).astype(np.float64)
        Kd = np.stack([x["K"][:n] for x in d]).astype(np.float64)
        adj = np.stack([x["adj"][:n, :n] for x in d]).astype(np.int8)
        T, nh, F = (np.stack([x[k][:n, :n] for x in d]).astype(np.float64) for k in ("T", "nh", "F"))
        bsz = batch_size(n, heavy, elements)
        counts, cons = [], []
        for s in range(0, sel.size, bsz):
            sl = slice(s, s + bsz)
            res = solve_draws({"P": P[sl], "K": Kd[sl], "adj": adj[sl], "T": T[sl], "nh": nh[sl], "F": F[sl]},
                              heavy, n, gen, device)
            counts.append(res["counts"])
            cons.append(res["cons"])
        diff = np.concatenate(counts) - n_fp[sel].astype(np.int64)
        gained_total += int((diff > 0).sum())
        count_total += int(sel.size)
        print(f" {n:2d} | {sel.size:7d} | {100 * (diff > 0).mean():21.2f} | {diff[diff > 0].mean() if (diff > 0).any() else 0.0:11.2f}"
              f" | {100 * (diff < 0).mean():7.2f} | {100 * np.concatenate(cons).mean():15.1f}", flush=True)
    print(f"overall: {100 * gained_total / max(count_total, 1):.2f}% of samples gain fixed points under the larger budget")


def main():
    cfg = parse_config()
    if cfg.mode == "audit":
        audit(cfg)
    else:
        generate(cfg)


if __name__ == "__main__":
    main()
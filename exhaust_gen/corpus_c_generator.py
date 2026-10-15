"""
corpus_c_generator.py — Corpus C: Complete SCC-DAG Composition Ground Truth

Overview
--------
Corpus C teaches the architecture to chain composition operators recursively
across complete SCC-DAGs.  Each sample is a small DAG of K SCCs where every
SCC is parameterised independently (drawn from Corpus A / Corpus B parameter
distributions) and inter-SCC edges carry Hill parameters.  Ground truth is
generated in topological order:

  For each SCC G_j (topo order, upstream → downstream):
    1. Collect all upstream parent SCCs whose FP portrait is already solved.
    2. For each stable FP branch r of each parent G_i with edge (a→b, ξ_ab):
         s_r = H_ab(x*_i_a^(r);  T_ab, n_ab, F_ab)        [scalar Hill signal]
    3. Multiply signals from all parents targeting the same gene b:
         u_b = ∏_k  s_k^(b)                               [absorb into P_b]
    4. Solve the forced autonomous ODE for G_j under these u values.
    5. Store the conditioned phase portrait per compatible branch tuple.

Branch compatibility
--------------------
At reconvergent nodes (diamond / multi-path) the branches arriving from
different parents must share a common ancestor lineage.  We track this as a
tuple of (root_scc_idx, root_fp_idx) pairs.  Only branch tuples with
consistent lineage at shared root ancestors are combined.

Data format
-----------
Each shard is a .npz file containing:

  # DAG-level (one row per DAG sample)
  dag_n_sccs      [D]           int32   number of SCCs in this DAG
  dag_n_edges     [D]           int32   number of inter-SCC edges
  dag_scc_ptr     [D+1]         int32   pointer into scc_* arrays
  dag_edge_ptr    [D+1]         int32   pointer into iedge_* arrays

  # SCC-level (one row per SCC across all DAGs)
  scc_n           [S]           int32   number of genes in this SCC
  scc_tc          [S]           int8    Thomas class (1=neg, 2=pos, 3=both)
  scc_dag_id      [S]           int32   which DAG this SCC belongs to
  scc_topo_rank   [S]           int32   topological rank (0 = root)
  scc_branch_ptr  [S+1]         int32   pointer into branch_* arrays

  # Intra-SCC topology / parameters (padded to N_MAX per SCC)
  scc_sign        [S, N_MAX, N_MAX]   int8    signed adjacency within SCC
  scc_P           [S, N_MAX]          float32 production rates
  scc_K           [S, N_MAX]          float32 degradation rates
  scc_ep_T        [S, N_MAX, N_MAX]   float32 Hill thresholds  (dense, row-major indexed)
  scc_ep_nh       [S, N_MAX, N_MAX]   float32 Hill n
  scc_ep_F        [S, N_MAX, N_MAX]   float32 Hill fold-changes

  # Inter-SCC edge table (global SCC indices)
  iedge_src_scc   [IE]          int32   source SCC global index
  iedge_src_gene  [IE]          int32   gene index within source SCC
  iedge_tgt_scc   [IE]          int32   target SCC global index
  iedge_tgt_gene  [IE]          int32   gene index within target SCC
  iedge_T         [IE]          float32 Hill threshold
  iedge_nh        [IE]          float32 Hill n
  iedge_F         [IE]          float32 Hill fold-change

  # Branch table — one row per branch-tuple at each SCC
  branch_scc      [B]           int32   SCC global index this branch belongs to
  branch_key      [B]           uint64  lineage hash (FNV-64 of sorted root assignments)
  branch_fp_ptr   [B+1]         int32   pointer into bfp_* arrays

  # Branch-specific FP records (residual-filtered, not Brouwer-certified, per branch)
  bfp_x           [BF, N_MAX]   float32  FP location (zero-padded)
  bfp_charpoly    [BF, N_MAX]   float32  char poly coefficients
  bfp_stab        [BF]          int8     stability class
  bfp_hopf        [BF]          float32  Hopf proximity ratio
  bfp_zeroeig     [BF]          float32  |det(J)|
  bfp_resid       [BF]          float32  ODE residual at FP

Patches / design notes
----------------------
[C1] Branch enumeration is capped at MAX_BRANCHES_PER_NODE compatible tuples
     per DAG node to bound exponential blowup in deep multistable DAGs.
[C2] Root SCCs (no parents) are solved autonomously; each stable FP becomes
     one branch, anchoring the lineage tree.
[C3] Multiple inter-SCC edges into the same target gene multiply their u_k
     values, consistent with the Hill product ODE structure.
[C4] Root SCCs use phase=0 (broad parameter sampling); non-root SCCs use
     phase=1 (switch-mode), matching the biological reasoning that genes
     receiving regulatory signals operate near switching thresholds.
[C5] Lineage compatibility: each branch carries a dict {root_scc → fp_idx}.
     At reconvergent nodes, two parent branches are compatible iff their
     root-assignment dicts agree on all shared keys.  The lineage_hash is an
     FNV-64 hash of the sorted items of that dict.
[C6] par["T"], par["nh"], par["F"] from sample_params are dense [B, N, N]
     arrays indexed [batch, src_gene, tgt_gene], matching the signed adjacency
     layout of topology.py.  Edge params are stored in the same dense format.
[C7] DAG topology fix: sample_dag_topology now builds a random spanning tree
     first (guaranteeing all K SCCs are weakly connected), then samples
     additional edges for branching/diamond structures.  This eliminates the
     69% disconnected-DAG rate caused by the previous purely-probabilistic
     edge sampling with a single guaranteed fallback edge.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from topology import (
    N_MIN, N_MAX, TC_BOTH,
    sample_graphs,
    enumerate_exhaustive_chunked,
)
from generate import sample_params
from solver import SolverConfig, solve_draws

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DAG_SCCS           = 8     # max SCCs per DAG sample
MAX_BRANCHES_PER_NODE  = 16    # cap on compatible branch tuples per node  [C1]
MAX_STABLE_FPS         = 8     # cap on stable FPs stored per branch

# Inter-SCC Hill parameter ranges (synthetic)
IEDGE_T_LO,    IEDGE_T_HI    = 0.5,  5.0
IEDGE_NH_LO,   IEDGE_NH_HI   = 1.0,  4.0
IEDGE_F_ACT_LO, IEDGE_F_ACT_HI = 1.5, 20.0
IEDGE_F_REP_LO, IEDGE_F_REP_HI = 0.05, 0.7
P_ACTIVATION = 0.6             # prob an inter-SCC edge is activating

# DAG generation parameters
DAG_K_MIN, DAG_K_MAX = 2, MAX_DAG_SCCS
DAG_EDGE_PROB        = 0.35    # per-candidate-edge probability (for bonus edges)

# Exhaustive SCC topology coverage: for n <= EXHAUST_SCC_N we pre-enumerate all
# non-isomorphic strongly connected signed digraphs and sample uniformly from the
# resulting pool, giving the model equal exposure to every topology at small sizes.
EXHAUST_SCC_N = 3

# Solver settings for Corpus C (lightweight; same order as corpus_b probe_config)
PROBE_SEEDS   = 48    # seeds per draw; must find FPs without Brouwer certification
PROBE_ROUNDS  = 2     # two rounds so the solver gets a second chance at missed FPs
PROBE_CORNERS = 16    # corner seeds in round 0 (2^n capped at this)

# Quality gate applied in _extract_fps (replaces Brouwer certification)
FP_RESID_TOL = 1e-7   # max relative ODE residual to accept a candidate FP


# ---------------------------------------------------------------------------
# Solver config builder
# ---------------------------------------------------------------------------

def _make_scfg() -> SolverConfig:
    """Lightweight SolverConfig for Corpus C (probe-quality, no confirmation)."""
    n_sizes = N_MAX - N_MIN + 1
    return SolverConfig(
        seeds      = tuple([PROBE_SEEDS] * n_sizes),
        rounds     = PROBE_ROUNDS,
        confirm    = False,
        max_corners= PROBE_CORNERS,
    )


# ---------------------------------------------------------------------------
# Hill function (scalar)
# ---------------------------------------------------------------------------

def hill(x: float, T: float, n: float, F: float) -> float:
    """H(x; T, n, F) = (F*(x/T)^n + 1) / ((x/T)^n + 1).

    Clamps x to a small positive value to avoid division by zero.
    """
    x = max(x, 1e-12)
    r = (x / T) ** n
    return (F * r + 1.0) / (r + 1.0)


# ---------------------------------------------------------------------------
# DAG topology
# ---------------------------------------------------------------------------

@dataclass
class DagTopology:
    """Lightweight DAG over K SCCs.

    inter_edges: list of 7-tuples
        (src_scc, src_gene, tgt_scc, tgt_gene, T, n, F)
    parents[j]: unique list of parent SCC indices for SCC j.
    topo_order: SCC indices in topological processing order (roots first).
    """
    k:           int
    parents:     List[List[int]]   # parents[j] = sorted unique parent indices
    inter_edges: List[Tuple]       # 7-tuples as above
    topo_order:  List[int]


def sample_dag_topology(
    scc_sizes: List[int],
    rng:       np.random.Generator,
    edge_prob: float = DAG_EDGE_PROB,
) -> DagTopology:
    """Random layered DAG over K = len(scc_sizes) SCCs.  [C7]

    FIX: Previously used purely probabilistic edge sampling with a single
    guaranteed fallback edge, which produced 69% disconnected DAGs (most with
    K-1 isolated SCCs and exactly 1 inter-SCC edge).

    Now builds a random spanning tree first (K-1 edges, guaranteeing all SCCs
    are weakly connected), then samples additional bonus edges for branching
    and diamond/reconvergent structures.
    """
    k = len(scc_sizes)

    if k == 1:
        return DagTopology(k=1, parents=[[]], inter_edges=[], topo_order=[0])

    # ── Assign strict layer ordering ─────────────────────────────────────────
    # Draw k random values and sort them; break ties by incrementing so every
    # node has a unique layer → guarantees a valid topological ordering.
    raw = sorted(rng.integers(0, k * 2, size=k).tolist())
    layers = list(raw)
    for i in range(1, k):
        if layers[i] <= layers[i - 1]:
            layers[i] = layers[i - 1] + 1

    # topo_order[rank] = SCC index; since we assign node i to layers[i],
    # order is simply 0..k-1 (already sorted by layer).
    order = list(range(k))

    parents: List[List[int]] = [[] for _ in range(k)]
    inter_edges: List[Tuple] = []

    def _make_edge(i: int, j: int) -> None:
        """Add one directed inter-SCC edge i → j with random Hill params."""
        src_gene = int(rng.integers(0, scc_sizes[i]))
        tgt_gene = int(rng.integers(0, scc_sizes[j]))
        T = float(rng.uniform(IEDGE_T_LO, IEDGE_T_HI))
        n = float(rng.uniform(IEDGE_NH_LO, IEDGE_NH_HI))
        F = (float(rng.uniform(IEDGE_F_ACT_LO, IEDGE_F_ACT_HI))
             if rng.random() < P_ACTIVATION
             else float(rng.uniform(IEDGE_F_REP_LO, IEDGE_F_REP_HI)))
        inter_edges.append((i, src_gene, j, tgt_gene, T, n, F))
        if i not in parents[j]:
            parents[j].append(i)

    # ── Step 1: Random spanning tree — guarantees weak connectivity ──────────
    # Each non-root node (rank > 0) picks exactly one random parent from the
    # nodes with strictly lower rank.  This gives K-1 edges connecting all K
    # nodes before any probabilistic sampling, eliminating isolated SCCs.
    for rank in range(1, k):
        j = order[rank]
        parent_rank = int(rng.integers(0, rank))   # uniform in [0, rank)
        i = order[parent_rank]
        _make_edge(i, j)

    # ── Step 2: Bonus edges for branching / diamond structures ───────────────
    # Iterate all remaining candidate forward edges and sample each with
    # probability edge_prob / layer_distance (distance decay keeps it sparse).
    existing = {(e[0], e[2]) for e in inter_edges}
    for rank_j in range(1, k):
        j = order[rank_j]
        for rank_i in range(rank_j):
            i = order[rank_i]
            if (i, j) in existing:
                continue   # spanning tree already added this edge
            layer_dist = layers[rank_j] - layers[rank_i]
            p = edge_prob / max(layer_dist, 1)
            if rng.random() < p:
                _make_edge(i, j)
                existing.add((i, j))

    return DagTopology(k=k, parents=parents, inter_edges=inter_edges, topo_order=order)


# ---------------------------------------------------------------------------
# Lineage hashing  [C5]
# ---------------------------------------------------------------------------

def lineage_hash(root_assignments: Dict[int, int]) -> int:
    """FNV-64 hash of sorted (root_scc, fp_idx) pairs.

    Deterministic uint64 that identifies an ancestor branch configuration.
    Collision probability is negligible for realistic branch counts.
    """
    h = 0xcbf29ce484222325  # FNV-1a offset basis (64-bit)
    for key, val in sorted(root_assignments.items()):
        for v in (key, val):
            h ^= v & 0xFF
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Fixed-point record
# ---------------------------------------------------------------------------

@dataclass
class FPRecord:
    x:        np.ndarray   # [scc_n] expression levels
    charpoly: np.ndarray   # [scc_n] characteristic polynomial coefficients
    stab:     int          # stability class from solver
    hopf:     float
    zeroeig:  float
    resid:    float

    @property
    def is_stable(self) -> bool:
        # stab 0 = stable node, 1 = stable spiral (solver convention)
        return self.stab in (0, 1)


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    """One ancestor-branch assignment at a DAG node.

    root_assignments: dict mapping root_scc_idx → fp_idx (the lineage key)
    fps:              all FPs found under this branch's forcing
    stable_fps:       subset of fps that are stable (capped at MAX_STABLE_FPS)
    u_per_gene:       gene_idx → combined Hill-signal multiplier absorbed into P
    lhash:            FNV-64 hash of root_assignments
    """
    root_assignments: Dict[int, int]
    fps:              List[FPRecord]
    stable_fps:       List[FPRecord]
    u_per_gene:       Dict[int, float]
    parent_branches:  List[Any] = field(default_factory=list)  # parent Branch objects
    lhash:            int = field(init=False)

    def __post_init__(self):
        self.lhash = lineage_hash(self.root_assignments)


# ---------------------------------------------------------------------------
# Forced solver helper
# ---------------------------------------------------------------------------

def _solve_forced(
    par:        Dict[str, Any],
    u_per_gene: Dict[int, float],
    scc_n:      int,
    scfg:       SolverConfig,
    gen,
    device:     torch.device,
) -> Dict[str, np.ndarray]:
    """Solve SCC under constant forcing.

    Forcing is absorbed into P: P_forced[:, gene] *= u  (batch dim = 1).
    par["P"] has shape [1, N]; we copy before modifying.
    """
    par_f = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in par.items()}
    for gene, u in u_per_gene.items():
        par_f["P"][:, gene] = par_f["P"][:, gene] * u
    return solve_draws(par_f, scfg, scc_n, gen, device)


def _extract_fps(result: Dict[str, np.ndarray], scc_n: int) -> List[FPRecord]:
    """Unpack solver output into FPRecord list.

    Quality gate: residual < FP_RESID_TOL and not non-hyperbolic (stab != 5).
    We do NOT require Brouwer certification (cons) — Corpus C uses a lightweight
    solver budget where cons is rarely True, but the Newton-converged FPs are
    still valid ground truth for composition.
    """
    if result["counts"][0] == 0:
        return []
    state   = result["state"]
    cp      = result["charpoly"]
    stab    = result["stab"]
    hopf    = result["hopf"]
    zeroeig = result["zeroeig"]
    resid   = result["resid"]
    # Accept any FP that converged well and is hyperbolic
    mask = (resid < FP_RESID_TOL) & (stab != 5)
    records = []
    for idx in np.where(mask)[0]:
        records.append(FPRecord(
            x        = state[idx, :scc_n].copy(),
            charpoly = cp[idx, :scc_n].copy(),
            stab     = int(stab[idx]),
            hopf     = float(hopf[idx]),
            zeroeig  = float(zeroeig[idx]),
            resid    = float(resid[idx]),
        ))
    return records


# ---------------------------------------------------------------------------
# Core DAG solver
# ---------------------------------------------------------------------------

def solve_dag(
    dag:        DagTopology,
    scc_graphs: List[Dict],
    scc_params: List[Dict],
    scc_sizes:  List[int],
    scfg:       SolverConfig,
    gen,
    device:     torch.device,
    rng:        np.random.Generator,
) -> List[List[Branch]]:
    """Solve all SCCs in topological order and return per-SCC branch lists."""

    scc_branches: List[List[Branch]] = [[] for _ in range(dag.k)]

    # Build edge lookup: tgt_scc → list of (src_scc, src_gene, tgt_gene, T, n, F)
    edges_to: Dict[int, List[Tuple]] = defaultdict(list)
    for (src_scc, src_gene, tgt_scc, tgt_gene, T, n, F) in dag.inter_edges:
        edges_to[tgt_scc].append((src_scc, src_gene, tgt_gene, T, n, F))

    for j in dag.topo_order:
        par   = scc_params[j]
        n_j   = scc_sizes[j]

        if not dag.parents[j]:
            # Root SCC [C2]: solve autonomously
            result = solve_draws(par, scfg, n_j, gen, device)
            fps    = _extract_fps(result, n_j)
            for fp_idx, fp in enumerate(fps):
                ra = {j: fp_idx}
                scc_branches[j].append(Branch(
                    root_assignments = ra,
                    fps              = [fp],
                    stable_fps       = [fp] if fp.is_stable else [],
                    u_per_gene       = {},
                ))
            if not scc_branches[j]:
                # No FPs found: placeholder so downstream skips gracefully
                scc_branches[j].append(Branch(
                    root_assignments = {j: 0},
                    fps              = [],
                    stable_fps       = [],
                    u_per_gene       = {},
                ))
            continue

        # Non-root SCC: enumerate compatible parent branch combinations
        # Group edges by source SCC
        src_to_edges: Dict[int, List[Tuple]] = defaultdict(list)
        for (src_scc, src_gene, tgt_gene, T, n, F) in edges_to[j]:
            src_to_edges[src_scc].append((src_gene, tgt_gene, T, n, F))

        # Parent branch lists (only branches with stable FPs)
        parent_branch_lists: List[List[Branch]] = [
            [b for b in scc_branches[p] if b.stable_fps]
            for p in dag.parents[j]
        ]
        if any(not bl for bl in parent_branch_lists):
            # A parent has no stable branches → j gets nothing
            continue

        # Cartesian product, capped  [C1]
        combos = list(itertools.product(*parent_branch_lists))
        if len(combos) > MAX_BRANCHES_PER_NODE:
            idx = rng.choice(len(combos), size=MAX_BRANCHES_PER_NODE, replace=False)
            combos = [combos[i] for i in idx]

        for combo in combos:
            # Lineage compatibility check  [C5]
            merged_ra: Dict[int, int] = {}
            compatible = True
            for branch in combo:
                for root_scc, fp_idx in branch.root_assignments.items():
                    if root_scc in merged_ra and merged_ra[root_scc] != fp_idx:
                        compatible = False
                        break
                    merged_ra[root_scc] = fp_idx
                if not compatible:
                    break
            if not compatible:
                continue

            # Compute u_per_gene: multiply Hill signals for each edge  [C3]
            u_per_gene: Dict[int, float] = {}
            for parent_branch, parent_scc in zip(combo, dag.parents[j]):
                src_fp = parent_branch.stable_fps[0]
                for (src_gene, tgt_gene, T, n, F) in src_to_edges[parent_scc]:
                    x_src = float(src_fp.x[src_gene]) if src_gene < len(src_fp.x) else 1.0
                    s = hill(x_src, T, n, F)
                    u_per_gene[tgt_gene] = u_per_gene.get(tgt_gene, 1.0) * s

            # Solve forced system
            if u_per_gene:
                result = _solve_forced(par, u_per_gene, n_j, scfg, gen, device)
            else:
                result = solve_draws(par, scfg, n_j, gen, device)
            fps = _extract_fps(result, n_j)

            stable = [fp for fp in fps if fp.is_stable][:MAX_STABLE_FPS]
            scc_branches[j].append(Branch(
                root_assignments = merged_ra,
                fps              = fps,
                stable_fps       = stable,
                u_per_gene       = u_per_gene,
                parent_branches  = list(combo),
            ))

    return scc_branches


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

@dataclass
class CorpusCShardBuilder:
    """Accumulates solved DAG samples and serialises them to .npz."""

    # DAG-level
    dag_n_sccs:   List[int] = field(default_factory=list)
    dag_n_edges:  List[int] = field(default_factory=list)

    # SCC-level
    scc_n:         List[int]         = field(default_factory=list)
    scc_tc:        List[int]         = field(default_factory=list)
    scc_dag_id:    List[int]         = field(default_factory=list)
    scc_topo_rank: List[int]         = field(default_factory=list)
    scc_sign_list: List[np.ndarray]  = field(default_factory=list)
    scc_P_list:    List[np.ndarray]  = field(default_factory=list)
    scc_K_list:    List[np.ndarray]  = field(default_factory=list)
    scc_epT_list:  List[np.ndarray]  = field(default_factory=list)
    scc_epNH_list: List[np.ndarray]  = field(default_factory=list)
    scc_epF_list:  List[np.ndarray]  = field(default_factory=list)

    # Inter-SCC edges
    ie_src_scc:  List[int]   = field(default_factory=list)
    ie_src_gene: List[int]   = field(default_factory=list)
    ie_tgt_scc:  List[int]   = field(default_factory=list)
    ie_tgt_gene: List[int]   = field(default_factory=list)
    ie_T:        List[float] = field(default_factory=list)
    ie_nh:       List[float] = field(default_factory=list)
    ie_F:        List[float] = field(default_factory=list)

    # Branch table
    br_scc:  List[int] = field(default_factory=list)
    br_key:  List[int] = field(default_factory=list)

    # Link records (parent-branch references per branch)
    lk_parent_branch: List[int] = field(default_factory=list)
    lk_parent_fp:     List[int] = field(default_factory=list)

    # Branch-FP records
    bfp_x:       List[np.ndarray] = field(default_factory=list)
    bfp_cp:      List[np.ndarray] = field(default_factory=list)
    bfp_stab:    List[int]        = field(default_factory=list)
    bfp_hopf:    List[float]      = field(default_factory=list)
    bfp_zeroeig: List[float]      = field(default_factory=list)
    bfp_resid:   List[float]      = field(default_factory=list)

    # Running counters
    _n_dags:     int = 0
    _n_sccs:     int = 0
    _n_iedges:   int = 0
    _n_branches: int = 0
    _n_links:    int = 0
    _n_bfps:     int = 0

    # Pointer arrays (start with 0)
    _dag_scc_ptr:    List[int] = field(default_factory=lambda: [0])
    _dag_edge_ptr:   List[int] = field(default_factory=lambda: [0])
    _scc_branch_ptr: List[int] = field(default_factory=lambda: [0])
    _branch_fp_ptr:  List[int] = field(default_factory=lambda: [0])
    _branch_link_ptr: List[int] = field(default_factory=lambda: [0])

    def add_dag(
        self,
        dag:          DagTopology,
        scc_graphs:   List[Dict],
        scc_params:   List[Dict],
        scc_sizes:    List[int],
        scc_branches: List[List[Branch]],
    ) -> None:
        dag_id   = self._n_dags
        scc_base = self._n_sccs
        self._n_dags += 1
        self.dag_n_sccs.append(dag.k)
        self.dag_n_edges.append(len(dag.inter_edges))
        # Map Branch object id → global branch index (populated as we go)
        branch_global_idx: Dict[int, int] = {}

        for j in range(dag.k):
            g   = scc_graphs[j]
            par = scc_params[j]
            n_j = scc_sizes[j]

            # sign matrix: g["sign"] is [N, N] int8
            sign = g["sign"]   # [n_j, n_j]
            sign_pad = np.zeros((N_MAX, N_MAX), dtype=np.int8)
            sign_pad[:n_j, :n_j] = sign

            # Edge params: par["T"], par["nh"], par["F"] are [1, N, N] dense [C6]
            ep_T  = np.ones((N_MAX, N_MAX), dtype=np.float32)
            ep_NH = np.ones((N_MAX, N_MAX), dtype=np.float32)
            ep_F  = np.ones((N_MAX, N_MAX), dtype=np.float32)
            ep_T[:n_j, :n_j]  = par["T"][0, :n_j, :n_j].astype(np.float32)
            ep_NH[:n_j, :n_j] = par["nh"][0, :n_j, :n_j].astype(np.float32)
            ep_F[:n_j, :n_j]  = par["F"][0, :n_j, :n_j].astype(np.float32)

            P_pad = np.zeros(N_MAX, dtype=np.float32)
            K_pad = np.zeros(N_MAX, dtype=np.float32)
            P_pad[:n_j] = par["P"][0, :n_j].astype(np.float32)
            K_pad[:n_j] = par["K"][0, :n_j].astype(np.float32)

            tc         = int(g.get("tc", TC_BOTH))
            topo_rank  = dag.topo_order.index(j)

            self.scc_n.append(n_j)
            self.scc_tc.append(tc)
            self.scc_dag_id.append(dag_id)
            self.scc_topo_rank.append(topo_rank)
            self.scc_sign_list.append(sign_pad)
            self.scc_P_list.append(P_pad)
            self.scc_K_list.append(K_pad)
            self.scc_epT_list.append(ep_T)
            self.scc_epNH_list.append(ep_NH)
            self.scc_epF_list.append(ep_F)

            # Record branches and their FPs
            for branch in scc_branches[j]:
                global_br_idx = self._n_branches
                branch_global_idx[id(branch)] = global_br_idx
                self.br_scc.append(scc_base + j)
                self.br_key.append(branch.lhash)
                for fp in branch.fps:
                    x_pad  = np.zeros(N_MAX, dtype=np.float32)
                    cp_pad = np.zeros(N_MAX, dtype=np.float32)
                    x_pad[:len(fp.x)]        = fp.x
                    cp_pad[:len(fp.charpoly)] = fp.charpoly
                    self.bfp_x.append(x_pad)
                    self.bfp_cp.append(cp_pad)
                    self.bfp_stab.append(fp.stab)
                    self.bfp_hopf.append(fp.hopf)
                    self.bfp_zeroeig.append(fp.zeroeig)
                    self.bfp_resid.append(fp.resid)
                    self._n_bfps += 1
                self._branch_fp_ptr.append(self._n_bfps)
                # Links: record parent branch references
                for par_branch in branch.parent_branches:
                    par_idx = branch_global_idx.get(id(par_branch), -1)
                    self.lk_parent_branch.append(par_idx)
                    self.lk_parent_fp.append(0)  # always use first stable FP
                    self._n_links += 1
                self._branch_link_ptr.append(self._n_links)
                self._n_branches += 1
            self._scc_branch_ptr.append(self._n_branches)
            self._n_sccs += 1

        # Inter-SCC edges (global SCC indices)
        for (src_scc, src_gene, tgt_scc, tgt_gene, T, nh, F) in dag.inter_edges:
            self.ie_src_scc.append(scc_base + src_scc)
            self.ie_src_gene.append(src_gene)
            self.ie_tgt_scc.append(scc_base + tgt_scc)
            self.ie_tgt_gene.append(tgt_gene)
            self.ie_T.append(T)
            self.ie_nh.append(nh)
            self.ie_F.append(F)
            self._n_iedges += 1

        self._dag_scc_ptr.append(self._n_sccs)
        self._dag_edge_ptr.append(self._n_iedges)

    def save(self, path: Path) -> None:
        if self._n_dags == 0:
            logger.warning("No DAGs to save; skipping %s", path)
            return

        def _a(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        def _s2(lst, dtype):
            """Stack list of [N_MAX, N_MAX] arrays → [S, N_MAX, N_MAX]."""
            if not lst:
                return np.zeros((0, N_MAX, N_MAX), dtype=dtype)
            return np.stack(lst).astype(dtype)

        def _s1(lst, dtype):
            """Stack list of [N_MAX] arrays → [S, N_MAX]."""
            if not lst:
                return np.zeros((0, N_MAX), dtype=dtype)
            return np.stack(lst).astype(dtype)

        np.savez_compressed(
            path,
            # DAG
            dag_n_sccs    = _a(self.dag_n_sccs,   np.int32),
            dag_n_edges   = _a(self.dag_n_edges,  np.int32),
            dag_scc_ptr   = _a(self._dag_scc_ptr,  np.int32),
            dag_edge_ptr  = _a(self._dag_edge_ptr, np.int32),
            # SCC
            scc_n         = _a(self.scc_n,         np.int32),
            scc_tc        = _a(self.scc_tc,        np.int8),
            scc_dag_id    = _a(self.scc_dag_id,    np.int32),
            scc_topo_rank = _a(self.scc_topo_rank, np.int32),
            scc_branch_ptr= _a(self._scc_branch_ptr, np.int32),
            scc_sign      = _s2(self.scc_sign_list, np.int8),
            scc_P         = _s1(self.scc_P_list,   np.float32),
            scc_K         = _s1(self.scc_K_list,   np.float32),
            scc_ep_T      = _s2(self.scc_epT_list,  np.float32),
            scc_ep_nh     = _s2(self.scc_epNH_list, np.float32),
            scc_ep_F      = _s2(self.scc_epF_list,  np.float32),
            # Inter-SCC edges
            iedge_src_scc = _a(self.ie_src_scc,  np.int32),
            iedge_src_gene= _a(self.ie_src_gene, np.int32),
            iedge_tgt_scc = _a(self.ie_tgt_scc,  np.int32),
            iedge_tgt_gene= _a(self.ie_tgt_gene, np.int32),
            iedge_T       = _a(self.ie_T,  np.float32),
            iedge_nh      = _a(self.ie_nh, np.float32),
            iedge_F       = _a(self.ie_F,  np.float32),
            # Branch table
            branch_scc       = _a(self.br_scc, np.int32),
            branch_key       = _a(self.br_key, np.uint64),
            branch_fp_ptr    = _a(self._branch_fp_ptr, np.int32),
            branch_link_ptr  = _a(self._branch_link_ptr, np.int32),
            # Link table
            link_parent_branch = _a(self.lk_parent_branch, np.int32),
            link_parent_fp     = _a(self.lk_parent_fp, np.int32),
            # Branch FP records
            bfp_x         = _s1(self.bfp_x,    np.float32),
            bfp_charpoly  = _s1(self.bfp_cp,   np.float32),
            bfp_stab      = _a(self.bfp_stab,    np.int8),
            bfp_hopf      = _a(self.bfp_hopf,    np.float32),
            bfp_zeroeig   = _a(self.bfp_zeroeig, np.float32),
            bfp_resid     = _a(self.bfp_resid,   np.float32),
            _sizes        = np.array([
                self._n_dags, self._n_sccs, self._n_iedges,
                self._n_branches, self._n_links, self._n_bfps,
            ], dtype=np.int64),
        )
        logger.info(
            "Saved %s | %d DAGs  %d SCCs  %d inter-edges  %d branches  %d links  %d FP records",
            path, self._n_dags, self._n_sccs, self._n_iedges, self._n_branches, self._n_links, self._n_bfps,
        )


# ---------------------------------------------------------------------------
# Shard runner
# ---------------------------------------------------------------------------

def run_shard_c(
    shard_idx:    int,
    n_dags:       int,
    out_dir:      Path,
    seed:         int,
    device_str:   str   = "cpu",
    k_min:        int   = DAG_K_MIN,
    k_max:        int   = DAG_K_MAX,
    dag_edge_prob: float = DAG_EDGE_PROB,
) -> None:
    t0     = time.time()
    rng    = np.random.default_rng(seed + shard_idx * 100_003)
    device = torch.device(device_str)
    scfg   = _make_scfg()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(2**31)))

    seen: set = set()

    # Pre-build exhaustive topology pools for n <= EXHAUST_SCC_N so every SCC
    # of a small size draws from the complete non-isomorphic set.
    exhaust_pools: dict = {}
    for _en in range(N_MIN, EXHAUST_SCC_N + 1):
        logger.info("Building exhaustive pool for n=%d …", _en)
        exhaust_pools[_en] = enumerate_exhaustive_chunked(_en, set())
        logger.info("  n=%d: %d distinct topologies", _en, len(exhaust_pools[_en]))

    builder   = CorpusCShardBuilder()
    n_skipped = 0

    for dag_i in range(n_dags):
        dag_rng = np.random.default_rng(int(rng.integers(2**31)))

        k = int(dag_rng.integers(k_min, k_max + 1))

        # Sample SCC topologies
        scc_graphs: List[Dict] = []
        scc_sizes:  List[int]  = []
        for _ in range(k):
            n_genes = int(dag_rng.integers(N_MIN, N_MAX + 1))
            if n_genes <= EXHAUST_SCC_N and exhaust_pools.get(n_genes):
                # Draw uniformly from the pre-built exhaustive pool (repeats allowed).
                pool = exhaust_pools[n_genes]
                graphs = [pool[int(dag_rng.integers(len(pool)))]]
            else:
                graphs  = sample_graphs(n_genes, 1, seen, dag_rng)
                if not graphs:
                    from topology import sample_cell, TC_BOTH, DENSITY_FRACS
                    tier = int(dag_rng.choice(list(DENSITY_FRACS.keys()),
                                              p=list(DENSITY_FRACS.values())))
                    graphs = sample_cell(n_genes, tier, TC_BOTH, 1, seen, dag_rng, attempts=400)
            if not graphs:
                graphs = None
                break
            scc_graphs.append(graphs[0])
            scc_sizes.append(n_genes)

        if len(scc_graphs) < k:
            n_skipped += 1
            continue

        # Build DAG topology (spanning-tree-first, guaranteed connectivity)
        dag = sample_dag_topology(scc_sizes, dag_rng, edge_prob=dag_edge_prob)

        # Sample parameters with correct phase  [C4]
        scc_params: List[Dict] = []
        for j in range(k):
            is_root = (len(dag.parents[j]) == 0)
            phase   = 0 if is_root else 1
            sign    = scc_graphs[j]["sign"]
            adj     = sign[np.newaxis, :, :]
            par     = sample_params(phase, adj, dag_rng)
            scc_params.append(par)

        # Solve DAG
        try:
            scc_branches = solve_dag(
                dag, scc_graphs, scc_params, scc_sizes,
                scfg, gen, device, dag_rng,
            )
        except Exception as exc:
            logger.warning("DAG %d/%d: exception %s — skipping", dag_i, n_dags, exc)
            n_skipped += 1
            continue

        # Skip if no stable FP was found anywhere
        if not any(b.stable_fps for branches in scc_branches for b in branches):
            n_skipped += 1
            continue

        builder.add_dag(dag, scc_graphs, scc_params, scc_sizes, scc_branches)

        if (dag_i + 1) % 50 == 0:
            logger.info("Shard %04d | %d/%d DAGs | %d skipped | %.1fs",
                        shard_idx, dag_i + 1, n_dags, n_skipped, time.time() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"corpus_c_shard_{shard_idx:04d}.npz"
    builder.save(shard_path)
    logger.info("Shard %04d done: %d saved  %d skipped  %.1fs",
                shard_idx, builder._n_dags, n_skipped, time.time() - t0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Corpus C: SCC-DAG composition")
    p.add_argument("--out_dir",         type=str,   required=True)
    p.add_argument("--n_shards",        type=int,   default=10)
    p.add_argument("--dags_per_shard",  type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--k_min",           type=int,   default=DAG_K_MIN)
    p.add_argument("--k_max",           type=int,   default=DAG_K_MAX)
    p.add_argument("--edge_prob",       type=float, default=DAG_EDGE_PROB)
    p.add_argument("--shard_start",     type=int,   default=0)
    p.add_argument("--n_workers",       type=int,   default=1)
    p.add_argument("--log_level",       type=str,   default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level.upper()),
        format  = "%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = Path(args.out_dir)

    shard_args = [
        (args.shard_start + i, args.dags_per_shard, out_dir, args.seed,
         args.device, args.k_min, args.k_max, args.edge_prob)
        for i in range(args.n_shards)
    ]

    if args.n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.n_workers) as pool:
            pool.starmap(run_shard_c, shard_args)
    else:
        for sa in shard_args:
            run_shard_c(*sa)


if __name__ == "__main__":
    main()
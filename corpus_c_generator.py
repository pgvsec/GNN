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
A branch is the complete conditioned portrait of one SCC under one choice of stable
fixed point in each parent SCC.  Every branch carries a lineage, the dict
{ancestor_scc: chosen_fp_idx} over *all* ancestors, and links to the parent branches (and
the parent fixed point inside each) that produced its forcing.  At reconvergent nodes
(diamond / multi-path) the options offered by different parents are combined only when
their lineages agree on every SCC they share, so a child never mixes two different states
of the same ancestor.

Data format
-----------
Each shard is a .npz file containing:

  # DAG-level (one row per DAG sample)
  dag_n_sccs      [D]           int32   number of SCCs in this DAG
  dag_n_edges     [D]           int32   number of inter-SCC edges
  dag_scc_ptr     [D+1]         int32   pointer into scc_* arrays
  dag_edge_ptr    [D+1]         int32   pointer into iedge_* arrays
  dag_family      [D]           int8    sampler's requested family, index into FAMILIES
  dag_shape       [D]           int8    realised shape: 0 chain, 1 tree, 2 merge, 3 reconvergent
  dag_depth       [D]           int8    longest root-to-leaf path in edges
  dag_max_fanin   [D]           int8    largest number of parent SCCs of any node
  dag_n_reconv    [D]           int8    nodes reached by two paths from a common ancestor
  dag_n_roots     [D]           int8    SCCs with no parent
  dag_all_solved  [D]           bool    every SCC has at least one branch

  # SCC-level (one row per SCC across all DAGs)
  scc_n           [S]           int32   number of genes in this SCC
  scc_tc          [S]           int8    Thomas class (1=neg, 2=pos, 3=both)
  scc_dag_id      [S]           int32   which DAG this SCC belongs to
  scc_topo_rank   [S]           int32   topological rank (0 = root)
  scc_depth       [S]           int8    longest path from any root, in edges
  scc_n_parents   [S]           int8    number of parent SCCs
  scc_truncated   [S]           bool    branches of this SCC were subsampled (cap or scan)
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

  # Branch table — one row per conditioned portrait (SCC x choice of parent states)
  branch_scc      [B]           int32   SCC global index this branch belongs to
  branch_key      [B]           uint64  lineage hash (FNV-64 of the sorted lineage items)
  branch_cons     [B]           bool    Brouwer index sum == 1 (fixed-point set is complete)
  branch_bsum     [B]           int8    the Brouwer index sum itself
  branch_u        [B, N_MAX]    float32 forcing multiplier absorbed into P per gene (1 = none)
  branch_fp_ptr   [B+1]         int32   pointer into bfp_* arrays
  branch_link_ptr [B+1]         int32   pointer into link_* arrays (empty for roots)

  # Branch -> parent links, one row per parent SCC, in the order of the SCC's parents
  link_parent_branch [L]        int32   global branch index of the parent branch
  link_parent_fp     [L]        int32   fixed-point index inside that parent branch, i.e.
                                         row branch_fp_ptr[pb] + link_parent_fp of bfp_*.
                                         Always a stable fixed point.

  schema_version  []            int32   2 (v1 had no links, no cons/bsum, no shape labels)

  # Branch-specific FP records (residual-filtered, not Brouwer-certified, per branch)
  bfp_x           [BF, N_MAX]   float32  FP location (zero-padded)
  bfp_charpoly    [BF, N_MAX]   float32  char poly coefficients
  bfp_stab        [BF]          int8     stability class
  bfp_hopf        [BF]          float32  Hopf proximity ratio
  bfp_zeroeig     [BF]          float32  min|lambda| / spectral radius of J (0 = saddle-node)
  bfp_resid       [BF]          float32  ODE residual at FP

Patches / design notes
----------------------
[C1] Branch enumeration is capped at MAX_BRANCHES_PER_NODE compatible tuples
     per DAG node to bound exponential blowup in deep multistable DAGs.  When
     anything is dropped scc_truncated is set for that node.
[C2] Root SCCs (no parents) are solved autonomously and hold ONE branch with the
     full autonomous portrait (v1 made one branch per fixed point).
[C3] Multiple inter-SCC edges into the same target gene multiply their u_k
     values, consistent with the Hill product ODE structure.
[C4] Root SCCs use phase=0 (broad parameter sampling); non-root SCCs use
     phase=1 (switch-mode), matching the biological reasoning that genes
     receiving regulatory signals operate near switching thresholds.
[C5] Lineage compatibility: each branch carries a dict {ancestor_scc → fp_idx}
     over every ancestor.  Two parent options are compatible iff their dicts
     agree on all shared keys.  The lineage_hash is an FNV-64 hash of the sorted
     items of that dict.
[C6] par["T"], par["nh"], par["F"] from sample_params are dense [B, N, N]
     arrays indexed [batch, src_gene, tgt_gene], matching the signed adjacency
     layout of topology.py.  Edge params are stored in the same dense format.
[C7] DAG topology fix: sample_dag_topology now builds a random spanning tree
     first (guaranteeing all K SCCs are weakly connected), then samples
     additional edges for branching/diamond structures.  This eliminates the
     69% disconnected-DAG rate caused by the previous purely-probabilistic
     edge sampling with a single guaranteed fallback edge.
[C8] (v2) A parent offers one option per (branch, stable fixed point).  v1 used only
     stable_fps[0] of each parent branch, so every multistable intermediate SCC
     silently lost all but one of its states downstream.  Compatible combinations
     of options are what a child is solved under.
[C9] (v2) SCC graphs come from _sample_scc_graph with an explicit (Thomas class, density
     tier) draw.  v1 called sample_graphs(n, quota=1), which rounds every cell's quota
     to 0 and always fell back to TC_BOTH, so Corpus C had no positive-only or
     negative-only SCCs.
[C10] (v2) DAGs come from a shape family (chain, tree, reconvergent, fanin, multiroot,
     random) with a per-DAG depth bias.  v1's uniform-parent tree has expected depth
     ln k, so DAGs of depth 6+ essentially never occurred.  Realised depth, fan-in and
     shape are stored in dag_*.
[C11] (v2) A solve that returns no usable fixed point is retried once with a larger
     budget, since the invariant box guarantees one exists.  An SCC whose fixed points
     are all unstable is kept as is: it gets no stable options and its children are
     unsolved (dag_all_solved False), which is the oscillatory-regime pool for Version 2.
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
    N_MIN, N_MAX, TC_BOTH, TC_NEG,
    CLASS_FRACS, DENSITY_FRACS,
    sample_cell,
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
DAG_EDGE_PROB        = 0.35    # per-candidate-edge probability (random family only)

# DAG shape families. The family is the sampler's *request*; the realised shape, depth and
# fan-in are always recomputed from the final edge set (dag_labels) and stored separately.
FAMILIES  = ("chain", "tree", "reconvergent", "fanin", "multiroot", "random")
SHAPE_MIX = {"chain": 0.15, "tree": 0.20, "reconvergent": 0.25,
             "fanin": 0.15, "multiroot": 0.10, "random": 0.15}

# Realised shape codes (dag_shape)
SHAPE_CHAIN, SHAPE_TREE, SHAPE_MERGE, SHAPE_RECONV = 0, 1, 2, 3
SHAPE_NAMES = {SHAPE_CHAIN: "chain", SHAPE_TREE: "tree",
               SHAPE_MERGE: "merge", SHAPE_RECONV: "reconvergent"}

SCHEMA_VERSION  = 2     # bump whenever the shard layout or the meaning of a field changes
MAX_COMBO_SCAN  = 4096  # parent-option tuples scanned per node before random subsampling

# Solver settings for Corpus C (lightweight; same order as corpus_b probe_config)
PROBE_SEEDS   = 48    # seeds per draw; must find FPs without Brouwer certification
PROBE_ROUNDS  = 2     # two rounds so the solver gets a second chance at missed FPs
PROBE_CORNERS = 16    # corner seeds in round 0 (2^n capped at this)

# Quality gate applied in _extract_fps (replaces Brouwer certification)
FP_RESID_TOL = 1e-7   # max relative ODE residual to accept a candidate FP


# ---------------------------------------------------------------------------
# Solver config builder
# ---------------------------------------------------------------------------

def _make_scfg(seeds: int = PROBE_SEEDS, rounds: int = PROBE_ROUNDS,
               confirm: bool = False) -> SolverConfig:
    """SolverConfig for Corpus C. The default is a probe budget (no confirmation round);
    pass confirm=True and more rounds to certify, then read branch_cons for completeness."""
    n_sizes = N_MAX - N_MIN + 1
    return SolverConfig(
        seeds      = tuple([seeds] * n_sizes),
        rounds     = rounds,
        confirm    = confirm,
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


def _tree_edges(k: int, rng: np.random.Generator, q: float) -> set:
    """Single-root out-tree. Node j hangs under j-1 with probability q, else under a uniform
    earlier node. q = 1 is a chain, q = 0 a random recursive tree (depth about ln k)."""
    edges = set()
    for j in range(1, k):
        p = j - 1 if rng.random() < q else int(rng.integers(0, j))
        edges.add((p, j))
    return edges


def _random_edges(k: int, rng: np.random.Generator, edge_prob: float) -> set:
    """Spanning tree plus distance-decayed bonus edges (the [C7] sampler)."""
    raw = sorted(rng.integers(0, k * 2, size=k).tolist())
    layers = list(raw)
    for i in range(1, k):
        if layers[i] <= layers[i - 1]:
            layers[i] = layers[i - 1] + 1
    edges = set()
    for j in range(1, k):
        edges.add((int(rng.integers(0, j)), j))
    for j in range(1, k):
        for i in range(j):
            if (i, j) in edges:
                continue
            if rng.random() < edge_prob / max(layers[j] - layers[i], 1):
                edges.add((i, j))
    return edges


def _add_extra_edges(edges: set, k: int, rng: np.random.Generator, n_extra: int) -> set:
    """Add up to n_extra forward edges into nodes that already have a parent. In a
    single-root tree every such edge creates a reconvergent node."""
    cand = [(i, j) for j in range(2, k) for i in range(j) if (i, j) not in edges]
    for t in rng.permutation(len(cand))[:n_extra]:
        edges.add(cand[int(t)])
    return edges


def _family_edges(family: str, k: int, rng: np.random.Generator, edge_prob: float) -> set:
    if k == 2:
        return {(0, 1)}
    if family == "chain":
        return {(j - 1, j) for j in range(1, k)}
    if family == "random":
        return _random_edges(k, rng, edge_prob)
    q = float(rng.uniform(0.0, 1.0))                # per-DAG depth bias
    if family == "tree":
        return _tree_edges(k, rng, q)
    if family == "reconvergent":
        return _add_extra_edges(_tree_edges(k, rng, q), k, rng, int(rng.integers(1, 4)))
    if family == "fanin":
        if k < 4:
            return _family_edges("reconvergent", k, rng, edge_prob)
        edges = _tree_edges(k, rng, q)
        sink = int(rng.integers(3, k))              # needs >= 3 earlier nodes
        want = int(rng.integers(3, min(sink, 4) + 1))
        have = {i for (i, j) in edges if j == sink}
        pool = [i for i in range(sink) if i not in have]
        for t in rng.permutation(len(pool)):
            if len(have) >= want:
                break
            have.add(pool[int(t)])
            edges.add((pool[int(t)], sink))
        return edges
    if family == "multiroot":
        # Nodes 0 and 1 are roots of two separate trees, joined at one merge node.
        root_of = {0: 0, 1: 1}
        edges = set()
        for j in range(2, k):
            p = j - 1 if rng.random() < q else int(rng.integers(0, j))
            edges.add((p, j))
            root_of[j] = root_of[p]
        sink = int(rng.integers(2, k))
        cand = [p for p in range(sink) if root_of[p] != root_of[sink]]
        edges.add((cand[int(rng.integers(len(cand)))], sink))
        return edges
    raise ValueError(f"unknown DAG family {family!r}")


def sample_dag_topology(
    scc_sizes: List[int],
    rng:       np.random.Generator,
    family:    str   = "random",
    edge_prob: float = DAG_EDGE_PROB,
) -> DagTopology:
    """Weakly connected DAG over K = len(scc_sizes) SCCs, drawn from a shape family.

    Nodes are numbered in topological order (edges only go from lower to higher index).
    Every family keeps the DAG weakly connected, so no sample bundles independent
    sub-graphs (see validate_dag_connectivity.py). Use dag_labels() for the realised shape.
    """
    k = len(scc_sizes)
    if k == 1:
        return DagTopology(k=1, parents=[[]], inter_edges=[], topo_order=[0])

    edges = _family_edges(family, k, rng, edge_prob)

    parents: List[List[int]] = [[] for _ in range(k)]
    inter_edges: List[Tuple] = []
    for (i, j) in sorted(edges, key=lambda e: (e[1], e[0])):
        src_gene = int(rng.integers(0, scc_sizes[i]))
        tgt_gene = int(rng.integers(0, scc_sizes[j]))
        T = float(rng.uniform(IEDGE_T_LO, IEDGE_T_HI))
        n = float(rng.uniform(IEDGE_NH_LO, IEDGE_NH_HI))
        F = (float(rng.uniform(IEDGE_F_ACT_LO, IEDGE_F_ACT_HI))
             if rng.random() < P_ACTIVATION
             else float(rng.uniform(IEDGE_F_REP_LO, IEDGE_F_REP_HI)))
        inter_edges.append((i, src_gene, j, tgt_gene, T, n, F))
        parents[j].append(i)

    return DagTopology(k=k, parents=parents, inter_edges=inter_edges, topo_order=list(range(k)))


def dag_labels(dag: DagTopology) -> Dict[str, Any]:
    """Realised structure of a DAG, computed from its edges.

    depth        longest root-to-leaf path in edges (a 7-node chain has depth 6)
    max_fanin    largest number of distinct parent SCCs of any node
    n_reconv     nodes with two parents that share an ancestor (or one is the other's
                 ancestor), i.e. nodes reached by two distinct paths from a common source
    n_roots      SCCs with no parent
    shape        SHAPE_CHAIN / TREE / MERGE (several roots or fan-in, no shared ancestry) /
                 RECONV (at least one reconvergent node)
    scc_depth    per-SCC longest path from any root
    """
    k = dag.k
    anc = [set() for _ in range(k)]
    scc_depth = [0] * k
    for j in dag.topo_order:
        anc[j] = {j}
        for p in dag.parents[j]:
            anc[j] |= anc[p]
            scc_depth[j] = max(scc_depth[j], scc_depth[p] + 1)
    n_children = [0] * k
    for ps in dag.parents:
        for p in ps:
            n_children[p] += 1
    n_roots = sum(1 for ps in dag.parents if not ps)
    max_fanin = max((len(ps) for ps in dag.parents), default=0)
    n_reconv = sum(
        1 for ps in dag.parents
        if len(ps) >= 2 and any(anc[a] & anc[b] for a, b in itertools.combinations(ps, 2))
    )
    if n_reconv > 0:
        shape = SHAPE_RECONV
    elif n_roots > 1 or max_fanin > 1:
        shape = SHAPE_MERGE
    elif max(n_children, default=0) <= 1:
        shape = SHAPE_CHAIN
    else:
        shape = SHAPE_TREE
    return {"depth": max(scc_depth, default=0), "max_fanin": max_fanin, "n_reconv": n_reconv,
            "n_roots": n_roots, "shape": shape, "scc_depth": scc_depth}


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
    """The conditioned phase portrait of one SCC under one choice of upstream states.

    A branch is one forced solve. It holds the *complete* fixed-point portrait of the SCC
    (stable and unstable) under the forcing produced by one chosen stable fixed point in
    each parent SCC.

    lineage:    {ancestor_scc: chosen_fp_idx} for every ancestor, parents included (SCC
                indices are local to the DAG, 0..k-1). Two
                parent options are compatible iff their lineages agree on shared SCCs.
    fps:        every fixed point found under this forcing
    stable_idx: indices into fps of the stable ones (capped at MAX_STABLE_FPS); these are
                the states a child SCC can be forced by
    u_per_gene: gene_idx -> combined Hill signal multiplier absorbed into P
    links:      (parent_branch, fp_idx) per parent SCC, in dag.parents[j] order. Empty for roots.
    cons, bsum: Brouwer index sum (== 1 means the fixed-point set is complete) and its value
    lhash:      FNV-64 hash of lineage
    """
    lineage:    Dict[int, int]
    fps:        List[FPRecord]
    stable_idx: List[int]
    u_per_gene: Dict[int, float]
    links:      List[Tuple["Branch", int]]
    cons:       bool = False
    bsum:       int  = 0
    lhash:      int  = field(init=False)

    def __post_init__(self):
        self.lhash = lineage_hash(self.lineage)

    @property
    def stable_fps(self) -> List[FPRecord]:
        return [self.fps[i] for i in self.stable_idx]


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

def _solve_with_retry(par, u_per_gene, scc_n, scfg, gen, device):
    """Forced solve. If no usable fixed point comes out, retry once with 4x the seeds, two more
    rounds and confirmation [C11].

    The positively invariant box guarantees at least one fixed point, so an empty result is a
    solver miss of the probe budget, not a property of the SCC. (An SCC whose fixed points are
    all unstable is different: that is returned as is and ends up with no stable options.)
    """
    result = _solve_forced(par, u_per_gene, scc_n, scfg, gen, device)
    fps = _extract_fps(result, scc_n)
    if not fps:
        big = replace(scfg, seeds=tuple(4 * s for s in scfg.seeds),
                      rounds=scfg.rounds + 2, confirm=True)
        result = _solve_forced(par, u_per_gene, scc_n, big, gen, device)
        fps = _extract_fps(result, scc_n)
    return result, fps


def _merge_lineages(lineages: List[Dict[int, int]]) -> Optional[Dict[int, int]]:
    """Union of lineages, or None if two of them disagree on a shared SCC."""
    merged: Dict[int, int] = {}
    for lin in lineages:
        for scc, fp_idx in lin.items():
            if merged.setdefault(scc, fp_idx) != fp_idx:
                return None
    return merged


def _stable_options(branches: List[Branch], scc_idx: int) -> List[Tuple[Branch, int, Dict[int, int]]]:
    """Every (branch, stable fp) a child can be forced by, with the lineage that choice implies."""
    opts = []
    for b in branches:
        for fi in b.stable_idx:
            lin = dict(b.lineage)
            lin[scc_idx] = fi
            opts.append((b, fi, lin))
    return opts


def _compatible_combos(option_lists: List[List[Tuple]], rng: np.random.Generator, cap: int):
    """One option per parent, restricted to tuples with mutually consistent lineages.

    Small products are enumerated exhaustively; larger ones are sampled. Returns
    ([(combo, merged_lineage)], truncated) with at most `cap` entries; truncated is True
    whenever some compatible tuples were left out.
    """
    total = 1
    for ol in option_lists:
        total *= len(ol)
    combos: List[Tuple] = []
    truncated = False
    if total <= MAX_COMBO_SCAN:
        for combo in itertools.product(*option_lists):
            merged = _merge_lineages([o[2] for o in combo])
            if merged is not None:
                combos.append((combo, merged))
    else:
        truncated = True
        tried = set()
        for _ in range(MAX_COMBO_SCAN):
            idx = tuple(int(rng.integers(len(ol))) for ol in option_lists)
            if idx in tried:
                continue
            tried.add(idx)
            combo = tuple(ol[i] for ol, i in zip(option_lists, idx))
            merged = _merge_lineages([o[2] for o in combo])
            if merged is not None:
                combos.append((combo, merged))
            if len(combos) >= cap:
                break
    if len(combos) > cap:
        keep = sorted(int(i) for i in rng.choice(len(combos), size=cap, replace=False))
        combos = [combos[i] for i in keep]
        truncated = True
    return combos, truncated


def solve_dag(
    dag:        DagTopology,
    scc_graphs: List[Dict],
    scc_params: List[Dict],
    scc_sizes:  List[int],
    scfg:       SolverConfig,
    gen,
    device:     torch.device,
    rng:        np.random.Generator,
) -> Tuple[List[List[Branch]], List[bool]]:
    """Solve all SCCs in topological order.

    Returns (scc_branches, scc_truncated). scc_truncated[j] is True when branches of SCC j
    were dropped (the per-node cap, the combination scan, or more than MAX_STABLE_FPS stable
    fixed points), so a consumer knows that node's branch list is a subsample.
    """

    scc_branches: List[List[Branch]] = [[] for _ in range(dag.k)]
    scc_truncated: List[bool] = [False] * dag.k

    # Build edge lookup: tgt_scc → list of (src_scc, src_gene, tgt_gene, T, n, F)
    edges_to: Dict[int, List[Tuple]] = defaultdict(list)
    for (src_scc, src_gene, tgt_scc, tgt_gene, T, n, F) in dag.inter_edges:
        edges_to[tgt_scc].append((src_scc, src_gene, tgt_gene, T, n, F))

    for j in dag.topo_order:
        par   = scc_params[j]
        n_j   = scc_sizes[j]

        if not dag.parents[j]:
            # Root SCC [C2]: one branch holding the autonomous portrait
            result, fps = _solve_with_retry(par, {}, n_j, scfg, gen, device)
            stable = [i for i, fp in enumerate(fps) if fp.is_stable]
            if len(stable) > MAX_STABLE_FPS:
                stable = stable[:MAX_STABLE_FPS]
                scc_truncated[j] = True
            scc_branches[j].append(Branch(
                lineage={}, fps=fps, stable_idx=stable, u_per_gene={}, links=[],
                cons=bool(result["cons"][0]), bsum=int(result["bsum"][0]),
            ))
            continue

        # Non-root SCC: group edges by source SCC
        src_to_edges: Dict[int, List[Tuple]] = defaultdict(list)
        for (src_scc, src_gene, tgt_gene, T, n, F) in edges_to[j]:
            src_to_edges[src_scc].append((src_gene, tgt_gene, T, n, F))

        # Each parent offers one option per (branch, stable fixed point)  [C8]
        option_lists = [_stable_options(scc_branches[p], p) for p in dag.parents[j]]
        if any(not ol for ol in option_lists):
            # A parent has no stable state to force with → j gets nothing
            continue

        combos, trunc = _compatible_combos(option_lists, rng, MAX_BRANCHES_PER_NODE)  # [C1]
        scc_truncated[j] = trunc

        for combo, merged in combos:
            # Compute u_per_gene: multiply Hill signals for each edge  [C3]
            u_per_gene: Dict[int, float] = {}
            for (parent_branch, fp_idx, _lin), parent_scc in zip(combo, dag.parents[j]):
                src_fp = parent_branch.fps[fp_idx]
                for (src_gene, tgt_gene, T, n, F) in src_to_edges[parent_scc]:
                    x_src = float(src_fp.x[src_gene]) if src_gene < len(src_fp.x) else 1.0
                    s = hill(x_src, T, n, F)
                    u_per_gene[tgt_gene] = u_per_gene.get(tgt_gene, 1.0) * s

            result, fps = _solve_with_retry(par, u_per_gene, n_j, scfg, gen, device)
            stable = [i for i, fp in enumerate(fps) if fp.is_stable]
            if len(stable) > MAX_STABLE_FPS:
                stable = stable[:MAX_STABLE_FPS]
                scc_truncated[j] = True

            scc_branches[j].append(Branch(
                lineage    = merged,
                fps        = fps,
                stable_idx = stable,
                u_per_gene = u_per_gene,
                links      = [(pb, fi) for (pb, fi, _lin) in combo],
                cons       = bool(result["cons"][0]),
                bsum       = int(result["bsum"][0]),
            ))

    return scc_branches, scc_truncated


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------

@dataclass
class CorpusCShardBuilder:
    """Accumulates solved DAG samples and serialises them to .npz."""

    # DAG-level
    dag_n_sccs:     List[int]  = field(default_factory=list)
    dag_n_edges:    List[int]  = field(default_factory=list)
    dag_family:     List[int]  = field(default_factory=list)
    dag_shape:      List[int]  = field(default_factory=list)
    dag_depth:      List[int]  = field(default_factory=list)
    dag_max_fanin:  List[int]  = field(default_factory=list)
    dag_n_reconv:   List[int]  = field(default_factory=list)
    dag_n_roots:    List[int]  = field(default_factory=list)
    dag_all_solved: List[bool] = field(default_factory=list)

    # SCC-level
    scc_n:         List[int]         = field(default_factory=list)
    scc_tc:        List[int]         = field(default_factory=list)
    scc_dag_id:    List[int]         = field(default_factory=list)
    scc_topo_rank: List[int]         = field(default_factory=list)
    scc_depth:     List[int]         = field(default_factory=list)
    scc_n_parents: List[int]         = field(default_factory=list)
    scc_truncated: List[bool]        = field(default_factory=list)
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
    br_scc:  List[int]        = field(default_factory=list)
    br_key:  List[int]        = field(default_factory=list)
    br_cons: List[bool]       = field(default_factory=list)
    br_bsum: List[int]        = field(default_factory=list)
    br_u:    List[np.ndarray] = field(default_factory=list)

    # Branch -> parent links (one row per parent SCC of the branch's SCC)
    link_pb: List[int] = field(default_factory=list)   # global index of the parent branch
    link_fp: List[int] = field(default_factory=list)   # fp index inside that parent branch

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
    _n_bfps:     int = 0

    # Pointer arrays (start with 0)
    _dag_scc_ptr:    List[int] = field(default_factory=lambda: [0])
    _dag_edge_ptr:   List[int] = field(default_factory=lambda: [0])
    _scc_branch_ptr: List[int] = field(default_factory=lambda: [0])
    _branch_fp_ptr:  List[int] = field(default_factory=lambda: [0])
    _branch_link_ptr: List[int] = field(default_factory=lambda: [0])

    def add_dag(
        self,
        dag:           DagTopology,
        scc_graphs:    List[Dict],
        scc_params:    List[Dict],
        scc_sizes:     List[int],
        scc_branches:  List[List[Branch]],
        scc_truncated: List[bool],
        family:        str,
    ) -> None:
        lab      = dag_labels(dag)
        dag_id   = self._n_dags
        scc_base = self._n_sccs
        self._n_dags += 1
        self.dag_n_sccs.append(dag.k)
        self.dag_n_edges.append(len(dag.inter_edges))
        self.dag_family.append(FAMILIES.index(family))
        self.dag_shape.append(lab["shape"])
        self.dag_depth.append(lab["depth"])
        self.dag_max_fanin.append(lab["max_fanin"])
        self.dag_n_reconv.append(lab["n_reconv"])
        self.dag_n_roots.append(lab["n_roots"])
        self.dag_all_solved.append(all(len(b) > 0 for b in scc_branches))

        # Global branch index of every branch of this DAG, so links can point at parents
        gidx: Dict[int, int] = {}
        nxt = self._n_branches
        for j in range(dag.k):
            for b in scc_branches[j]:
                gidx[id(b)] = nxt
                nxt += 1

        for j in range(dag.k):
            g   = scc_graphs[j]
            par = scc_params[j]
            n_j = scc_sizes[j]

            # sign matrix: g["sign"] is [N, N] int8
            sign = g["sign"]   # [n_j, n_j]
            sign_pad = np.zeros((N_MAX, N_MAX), dtype=np.int8)
            sign_pad[:n_j, :n_j] = sign

            # Edge params: par["T"], par["nh"], par["F"] are [1, N, N] dense [C6]
            # Pad with neutral values (1.0) so absent-edge entries are never degenerate.
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
            self.scc_depth.append(lab["scc_depth"][j])
            self.scc_n_parents.append(len(dag.parents[j]))
            self.scc_truncated.append(bool(scc_truncated[j]))
            self.scc_sign_list.append(sign_pad)
            self.scc_P_list.append(P_pad)
            self.scc_K_list.append(K_pad)
            self.scc_epT_list.append(ep_T)
            self.scc_epNH_list.append(ep_NH)
            self.scc_epF_list.append(ep_F)

            # Record branches and their FPs
            for branch in scc_branches[j]:
                assert gidx[id(branch)] == self._n_branches
                self.br_scc.append(scc_base + j)
                self.br_key.append(branch.lhash)
                self.br_cons.append(branch.cons)
                self.br_bsum.append(branch.bsum)
                u = np.ones(N_MAX, dtype=np.float32)       # 1.0 = unforced gene
                for gene, val in branch.u_per_gene.items():
                    u[gene] = val
                self.br_u.append(u)
                for parent_branch, fp_idx in branch.links:
                    self.link_pb.append(gidx[id(parent_branch)])
                    self.link_fp.append(fp_idx)
                self._branch_link_ptr.append(len(self.link_pb))
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
            schema_version = np.array(SCHEMA_VERSION, dtype=np.int32),
            _sizes = np.array([
                self._n_dags,
                self._n_sccs,
                self._n_iedges,
                self._n_branches,
                len(self.link_pb),
                self._n_bfps,
            ], dtype=np.int64),
            # DAG
            dag_n_sccs    = _a(self.dag_n_sccs,   np.int32),
            dag_n_edges   = _a(self.dag_n_edges,  np.int32),
            dag_scc_ptr   = _a(self._dag_scc_ptr,  np.int32),
            dag_edge_ptr  = _a(self._dag_edge_ptr, np.int32),
            dag_family    = _a(self.dag_family,    np.int8),
            dag_shape     = _a(self.dag_shape,     np.int8),
            dag_depth     = _a(self.dag_depth,     np.int8),
            dag_max_fanin = _a(self.dag_max_fanin, np.int8),
            dag_n_reconv  = _a(self.dag_n_reconv,  np.int8),
            dag_n_roots   = _a(self.dag_n_roots,   np.int8),
            dag_all_solved= _a(self.dag_all_solved, np.bool_),
            # SCC
            scc_n         = _a(self.scc_n,         np.int32),
            scc_tc        = _a(self.scc_tc,        np.int8),
            scc_dag_id    = _a(self.scc_dag_id,    np.int32),
            scc_topo_rank = _a(self.scc_topo_rank, np.int32),
            scc_depth     = _a(self.scc_depth,     np.int8),
            scc_n_parents = _a(self.scc_n_parents, np.int8),
            scc_truncated = _a(self.scc_truncated, np.bool_),
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
            branch_scc    = _a(self.br_scc, np.int32),
            branch_key    = _a(self.br_key, np.uint64),
            branch_cons   = _a(self.br_cons, np.bool_),
            branch_bsum   = _a(self.br_bsum, np.int8),
            branch_u      = _s1(self.br_u, np.float32),
            branch_fp_ptr = _a(self._branch_fp_ptr, np.int32),
            branch_link_ptr = _a(self._branch_link_ptr, np.int32),
            link_parent_branch = _a(self.link_pb, np.int32),
            link_parent_fp     = _a(self.link_fp, np.int32),
            # Branch FP records
            bfp_x         = _s1(self.bfp_x,    np.float32),
            bfp_charpoly  = _s1(self.bfp_cp,   np.float32),
            bfp_stab      = _a(self.bfp_stab,    np.int8),
            bfp_hopf      = _a(self.bfp_hopf,    np.float32),
            bfp_zeroeig   = _a(self.bfp_zeroeig, np.float32),
            bfp_resid     = _a(self.bfp_resid,   np.float32),
        )
        logger.info(
            "Saved %s | %d DAGs  %d SCCs  %d inter-edges  %d branches  %d FP records",
            path, self._n_dags, self._n_sccs, self._n_iedges, self._n_branches, self._n_bfps,
        )


# ---------------------------------------------------------------------------
# Shard runner
# ---------------------------------------------------------------------------

def _sample_scc_graph(n: int, rng: np.random.Generator) -> Optional[Dict]:
    """One strongly connected signed graph on n genes, Thomas class and density tier drawn from
    the topology.py mixes [C9].

    topology.sample_graphs(n, quota=1) must not be used for this: it rounds every cell's
    quota (at most 0.6 * 0.4) down to zero and always falls back to TC_BOTH. If the drawn cell
    is empty at this n (for example the densest tier at n = 2) the neighbouring cells are tried,
    so the realised class is whatever is stored in g["tc"].

    Topologies are not de-duplicated here: the same graph may appear in many DAGs with
    different parameters, as in Corpus A.
    """
    tcs   = list(CLASS_FRACS)
    tiers = list(DENSITY_FRACS)
    tc    = int(rng.choice(tcs,   p=[CLASS_FRACS[t]   for t in tcs]))
    tier  = int(rng.choice(tiers, p=[DENSITY_FRACS[t] for t in tiers]))
    for t_c in [tc] + [c for c in tcs if c != tc]:
        for t_t in [tier] + [t for t in tiers if t != tier]:
            out = sample_cell(n, t_t, t_c, 1, set(), rng, attempts=300 if t_c == TC_NEG else 80)
            if out:
                return out[0]
    return None


def parse_shape_mix(text: str) -> Dict[str, float]:
    """'chain=0.2,tree=0.3' -> normalised {family: probability}; unnamed families get 0."""
    mix = {f: 0.0 for f in FAMILIES}
    for part in text.split(","):
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in mix:
            raise ValueError(f"unknown DAG family {name!r}; choose from {FAMILIES}")
        mix[name] = float(val)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("shape mix must have positive total weight")
    return {f: v / total for f, v in mix.items()}


def run_shard_c(
    shard_idx:     int,
    n_dags:        int,
    out_dir:       Path,
    seed:          int,
    device_str:    str   = "cpu",
    k_min:         int   = DAG_K_MIN,
    k_max:         int   = DAG_K_MAX,
    dag_edge_prob: float = DAG_EDGE_PROB,
    seeds:         int   = PROBE_SEEDS,
    rounds:        int   = PROBE_ROUNDS,
    confirm:       bool  = False,
    shape_mix:     Optional[Dict[str, float]] = None,
) -> None:
    t0     = time.time()
    rng    = np.random.default_rng(seed + shard_idx * 100_003)
    device = torch.device(device_str)
    scfg   = _make_scfg(seeds, rounds, confirm)

    mix      = shape_mix or SHAPE_MIX
    families = [f for f in FAMILIES if mix.get(f, 0.0) > 0.0]
    fam_p    = np.array([mix[f] for f in families], dtype=float)
    fam_p   /= fam_p.sum()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(2**31)))

    builder   = CorpusCShardBuilder()
    n_skipped = 0

    for dag_i in range(n_dags):
        dag_rng = np.random.default_rng(int(rng.integers(2**31)))

        k      = int(dag_rng.integers(k_min, k_max + 1))
        family = str(dag_rng.choice(families, p=fam_p))

        # Sample SCC topologies
        scc_graphs: List[Dict] = []
        scc_sizes = [int(dag_rng.integers(N_MIN, N_MAX + 1)) for _ in range(k)]
        for n_genes in scc_sizes:
            g = _sample_scc_graph(n_genes, dag_rng)
            if g is None:
                break
            scc_graphs.append(g)

        if len(scc_graphs) < k:
            n_skipped += 1
            continue

        # Build DAG topology (weakly connected, drawn from the requested shape family)
        dag = sample_dag_topology(scc_sizes, dag_rng, family=family, edge_prob=dag_edge_prob)

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
            scc_branches, scc_trunc = solve_dag(
                dag, scc_graphs, scc_params, scc_sizes,
                scfg, gen, device, dag_rng,
            )
        except Exception as exc:
            logger.warning("DAG %d/%d: exception %s — skipping", dag_i, n_dags, exc)
            n_skipped += 1
            continue

        # Skip if no stable FP was found anywhere
        if not any(b.stable_idx for branches in scc_branches for b in branches):
            n_skipped += 1
            continue

        builder.add_dag(dag, scc_graphs, scc_params, scc_sizes, scc_branches, scc_trunc, family)

        if (dag_i + 1) % 50 == 0:
            logger.info("Shard %04d | %d/%d DAGs | %d skipped | %.1fs",
                        shard_idx, dag_i + 1, n_dags, n_skipped, time.time() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"corpus_c_shard_{shard_idx:04d}.npz"
    builder.save(shard_path)
    shapes = np.bincount(np.array(builder.dag_shape, dtype=int), minlength=len(SHAPE_NAMES))
    logger.info("Shard %04d done: %d saved  %d skipped  %.1fs | shapes %s",
                shard_idx, builder._n_dags, n_skipped, time.time() - t0,
                {SHAPE_NAMES[i]: int(c) for i, c in enumerate(shapes)})


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
    p.add_argument("--k_max",           type=int,   default=DAG_K_MAX,
                   help="max SCCs per DAG; depth 6+ needs k >= 7")
    p.add_argument("--edge_prob",       type=float, default=DAG_EDGE_PROB,
                   help="bonus-edge probability of the 'random' family")
    p.add_argument("--shape_mix",       type=str,   default=None,
                   help="e.g. 'chain=0.2,tree=0.2,reconvergent=0.3,fanin=0.2,multiroot=0.1'; "
                        f"families: {', '.join(FAMILIES)}")
    p.add_argument("--seeds",           type=int,   default=PROBE_SEEDS,
                   help="Newton seeds per solve")
    p.add_argument("--rounds",          type=int,   default=PROBE_ROUNDS,
                   help="solver rounds per solve")
    p.add_argument("--confirm",         action="store_true",
                   help="require a confirmation round that finds nothing new (slower; "
                        "raises the Brouwer-certified fraction in branch_cons)")
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
    mix = parse_shape_mix(args.shape_mix) if args.shape_mix else None

    shard_args = [
        (args.shard_start + i, args.dags_per_shard, out_dir, args.seed,
         args.device, args.k_min, args.k_max, args.edge_prob,
         args.seeds, args.rounds, args.confirm, mix)
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

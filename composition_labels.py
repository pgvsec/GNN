"""Exact per-branch fixed-point labels from composition ground truth.

A composition case gives the full fixed-point set of a multi-SCC GRN. Because the
condensation DAG is acyclic, every global fixed point restricts to a fixed point of each
SCC under the constant forcing produced by the values of that SCC's upstream source genes.
So the per-SCC training labels can be read off the global set exactly:

  - Root SCC: its fixed points are the distinct restrictions to its genes.
  - Forced SCC: group the global fixed points by the values of the genes that send an
    inter-SCC edge into it (the forcing depends on nothing else). Each group is one forcing
    context (one branch); its fixed points are the distinct restrictions to the SCC's genes,
    and its forcing u is computed from the group's source-gene values with the same Hill
    product the assembler uses.

Two branches of the same SCC can have different counts (a child can be bistable under one
parent state and monostable under another), so every branch keeps its own label. This
replaces the modal-count label of the first finetune_count_heads.py, which trained every
branch toward the most common count.

Distinctness is decided in log space with a tolerance well above float32 noise and below
the solver's own merge tolerance scale, mirroring solver.cluster_roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

LOG_TOL = 1e-4      # two restrictions closer than this in every log coordinate are the same point


@dataclass
class BranchLabel:
    """One SCC under one forcing context, with its ground-truth fixed points."""
    scc_id: int
    is_root: bool
    u: np.ndarray            # [n_scc] forcing multipliers (all ones for a root)
    k: int                   # number of fixed points of the SCC in this context
    x: np.ndarray            # [k, n_scc] fixed points in SCC-local coordinates
    stab: np.ndarray         # [k] stability class of the SCC block at each point (-1 if unknown)
    src_values: np.ndarray   # [n_src] values of the upstream source genes (empty for a root)
    consistent: bool = True  # False when the GT is visibly incomplete for this branch: an even
                             # count (impossible for hyperbolic points), or upstream points with
                             # identical forcing that list different child points


def _cluster(rows: np.ndarray, tol: float = LOG_TOL) -> np.ndarray:
    """Label of each row [m, d] such that rows closer than tol in every log coordinate share a
    label (single linkage). Labels are 0..c-1 in order of first appearance."""
    m = rows.shape[0]
    if m == 0:
        return np.zeros(0, np.int64)
    if rows.shape[1] == 0:
        return np.zeros(m, np.int64)
    lg = np.log(np.clip(rows.astype(np.float64), 1e-300, None))
    close = np.abs(lg[:, None, :] - lg[None, :, :]).max(axis=-1) < tol
    lab = np.arange(m)
    for _ in range(m):
        new = np.where(close, lab[None, :], m).min(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
    _, first = np.unique(lab, return_index=True)
    remap = {lab[i]: j for j, i in enumerate(sorted(first))}
    return np.array([remap[v] for v in lab], np.int64)


def _source_genes(dag, scc_id: int) -> List[int]:
    """Global indices of the genes with an inter-SCC edge into scc_id, sorted."""
    src = set()
    for (pa, ch), edge_map in dag.inter_edges.items():
        if ch != scc_id:
            continue
        for edge_list in edge_map.values():
            for gi, _gj in edge_list:
                src.add(gi)
    return sorted(src)


def _forcing_from_sources(dag, scc_id: int, x_global: np.ndarray, T, nh, F) -> np.ndarray:
    """u [n_scc] from a global state, using the same Hill product as assemble._compute_forcing."""
    scc = dag.sccs[scc_id]
    u = np.ones(scc.n, dtype=np.float64)
    for (pa, ch), edge_map in dag.inter_edges.items():
        if ch != scc_id:
            continue
        for lj, edge_list in edge_map.items():
            for gi, gj in edge_list:
                x_i = max(float(x_global[gi]), 1e-300)
                r = (x_i / float(T[gi, gj])) ** float(nh[gi, gj])
                u[lj] *= (float(F[gi, gj]) * r + 1.0) / (r + 1.0)
    return u.astype(np.float32)


def block_stability(x_local, P_eff, K, adj, T, nh, F) -> int:
    """Stability class (model.py / solver convention) of one SCC block at x_local.

    0 stable node, 1 stable spiral, 2 unstable node, 3 unstable spiral, 4 saddle,
    5 non-hyperbolic. Uses the analytic Jacobian of the forced SCC."""
    n = x_local.shape[0]
    x = np.clip(x_local.astype(np.float64), 1e-300, None)
    J = np.zeros((n, n))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logr = nh * (np.log(x)[:, None] - np.log(T))
        r = np.exp(np.clip(logr, -700, 700))
        H = (F * r + 1.0) / (r + 1.0)
        dH = (F - 1.0) * nh * r / (x[:, None] * (1.0 + r) ** 2)
    edge = adj != 0
    Hm = np.where(edge, H, 1.0)
    prod = Hm.prod(axis=0)                                  # over sources, per target
    for i in range(n):           # target
        for j in range(n):       # source
            if edge[j, i]:
                J[i, j] += P_eff[i] * prod[i] / Hm[j, i] * dH[j, i]
        J[i, i] -= K[i]
    ev = np.linalg.eigvals(J)
    re, im = ev.real, ev.imag
    scale = max(np.abs(ev).max(), 1e-30)
    tol = 1e-9 * scale
    if (np.abs(re) <= tol).any():
        return 5
    cplx = (np.abs(im) > tol).any()
    n_pos = int((re > 0).sum())
    if n_pos == 0:
        return 1 if cplx else 0
    if n_pos == n:
        return 3 if cplx else 2
    return 4


def extract_branch_labels(dag, fp_state: np.ndarray, T, nh, F,
                          with_stability: bool = False) -> Dict[int, List[BranchLabel]]:
    """Per-SCC list of BranchLabel for one composition case.

    dag      : assemble.SCCDAG of the case
    fp_state : [M, N] ground-truth global fixed points
    T, nh, F : [N, N] global edge parameters (for the forcing of each branch)
    """
    out: Dict[int, List[BranchLabel]] = {}
    M = fp_state.shape[0]
    for scc in dag.sccs:
        genes = list(scc.genes)
        local = fp_state[:, genes] if M else np.zeros((0, scc.n), np.float32)
        is_root = len(scc.parents) == 0
        if is_root:
            groups = [np.arange(M)]
            src = []
        else:
            src = _source_genes(dag, scc.scc_id)
            glab = _cluster(fp_state[:, src]) if M else np.zeros(0, np.int64)
            groups = [np.nonzero(glab == g)[0] for g in range(int(glab.max()) + 1)] if M else []
        branches = []
        for idx in groups:
            if idx.size == 0:
                continue
            lab = _cluster(local[idx])
            consistent = True
            if not is_root:
                # Upstream points sharing this forcing must all see the same child set.
                up = [g for p in scc.parents for g in dag.sccs[p].genes]
                ulab = _cluster(fp_state[idx][:, up])
                sizes = {len(set(lab[ulab == c].tolist())) for c in range(int(ulab.max()) + 1)}
                consistent = sizes == {int(lab.max()) + 1}
            reps = np.array([idx[np.argmax(lab == c)] for c in range(int(lab.max()) + 1)])
            x = local[reps].astype(np.float32)
            if is_root:
                u = np.ones(scc.n, np.float32)
            else:
                u = _forcing_from_sources(dag, scc.scc_id, fp_state[idx[0]], T, nh, F)
            if with_stability:
                P_eff = scc.P.astype(np.float64) * u
                stab = np.array([block_stability(xi, P_eff, scc.K.astype(np.float64),
                                                 scc.adj_intra, scc.T.astype(np.float64),
                                                 scc.nh.astype(np.float64),
                                                 scc.F.astype(np.float64)) for xi in x], np.int64)
            else:
                stab = np.full(x.shape[0], -1, np.int64)
            branches.append(BranchLabel(
                scc_id=scc.scc_id, is_root=is_root, u=u, k=int(x.shape[0]), x=x, stab=stab,
                src_values=fp_state[idx[0], src].astype(np.float32) if src else np.zeros(0, np.float32),
                consistent=consistent and x.shape[0] % 2 == 1))
        out[scc.scc_id] = branches
    return out


def case_consistent(labels: Dict[int, List[BranchLabel]]) -> bool:
    """True when no branch of the case shows a visible GT gap."""
    return all(b.consistent for brs in labels.values() for b in brs)


def enumerate_count(dag, labels: Dict[int, List[BranchLabel]]) -> int:
    """Number of global fixed points implied by the branch labels.

    Walks the DAG in topological order: a root contributes each of its points; a forced SCC
    looks up the branch whose source values match the points chosen upstream and contributes
    each of that branch's points. Equals the ground-truth count M for consistent labels, so it
    is the self-check of the extraction (the sum formula N = sum_x |F_child(x)|, generalised
    to any DAG). A context with no matching branch contributes zero."""
    sccs = dag.sccs
    srcs = {s.scc_id: _source_genes(dag, s.scc_id) for s in sccs}
    n_genes = dag.n_genes

    def rec(i, x_global):
        if i == len(sccs):
            return 1
        scc = sccs[i]
        brs = labels.get(scc.scc_id, [])
        if not brs:
            return 0
        if len(scc.parents) == 0:
            br = brs[0]
        else:
            src = srcs[scc.scc_id]
            v = np.log(np.clip(x_global[src], 1e-300, None))
            d = [np.abs(np.log(np.clip(b.src_values.astype(np.float64), 1e-300, None)) - v).max()
                 for b in brs]
            j = int(np.argmin(d))
            if d[j] >= 10 * LOG_TOL:
                return 0
            br = brs[j]
        total = 0
        for xk in br.x:
            x_next = x_global.copy()
            x_next[list(scc.genes)] = xk
            total += rec(i + 1, x_next)
        return total

    return rec(0, np.ones(n_genes, np.float64))

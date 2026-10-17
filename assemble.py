"""Global fixed-point assembly for arbitrary large GRNs.

Pipeline:
  1. decompose_grn       — Tarjan SCC decomposition → condensation DAG
  2. run_inference        — walk the DAG, run Layer I on roots, Layer III on forced SCCs
  3. assemble_global_fps  — enumerate consistent branch paths → full N-dimensional FP vectors
  4. refine_fps           — Newton refinement of assembled FP predictions
  5. GRNPredictor         — convenience wrapper that does steps 1-4

Mathematical foundation
  The Jacobian of the full GRN at any global fixed point is block lower-triangular (because
  the condensation DAG is acyclic). Its eigenvalues are the union of the per-SCC block
  eigenvalues. Therefore:
    - Global stability = AND(local stability for each SCC on the path)
    - Global FP position = concatenation of SCC-level positions (padded to gene indices)
    - Global FP count = product of per-branch FP counts (when the DAG has one root)

Shapes
  All public functions take/return plain Python dicts and numpy arrays for portability.
  The internal inference helpers move to the torch device specified by the model.

Usage
  >>> predictor = GRNPredictor(l1_model, l3_model)
  >>> result = predictor.predict(adj, signs, P, K, T, nh, F)
  >>> print(result["global_fps"])       # [n_fps, N] ndarray
  >>> print(result["global_stab"])      # [n_fps] int array, stability class per FP
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
STAB_STABLE_NODE   = 0
STAB_STABLE_SPIRAL = 1
STAB_UNSTABLE_NODE = 2
STAB_UNSTABLE_SPIRAL = 3
STAB_SADDLE        = 4
STAB_NONHYP        = 5
STAB_UNKNOWN       = 6

STABLE_CLASSES = {STAB_STABLE_NODE, STAB_STABLE_SPIRAL}

# Tolerance for detecting "unforced" genes (u ≈ 1.0)
U_FORCED_TOL = 1e-5

# Max Newton steps in the post-assembly refinement
NEWTON_STEPS = 30

# ---------------------------------------------------------------------------
# 1. decompose_grn — Tarjan SCC decomposition + condensation DAG
# ---------------------------------------------------------------------------

@dataclass
class SccNode:
    """One strongly-connected component in the condensation DAG."""
    scc_id: int               # topological order index (0 = root, increasing downstream)
    genes: List[int]          # original gene indices in the full GRN (sorted)
    n: int                    # number of genes
    # Intra-SCC parameters (shape [n] or [n,n])
    adj_intra: np.ndarray     # signed adjacency among SCC genes [n, n]
    P: np.ndarray             # production rates [n]
    K: np.ndarray             # degradation rates [n]
    T: np.ndarray             # Hill thresholds [n, n]
    nh: np.ndarray            # Hill coefficients [n, n]
    F: np.ndarray             # fold-changes [n, n]
    # DAG linkage
    parents: List[int] = field(default_factory=list)   # scc_ids of parent SCCs
    children: List[int] = field(default_factory=list)  # scc_ids of child SCCs


@dataclass
class SCCDAG:
    """Condensation DAG of a GRN."""
    n_genes: int              # total genes in the original GRN
    sccs: List[SccNode]       # sorted in topological order (roots first)
    roots: List[int]          # scc_ids with no parents
    # Inter-SCC edges: for each (parent_scc, child_scc), a dict of
    #   { child_gene_local_idx : forcing_multiplier }
    # Forcing is precomputed from the parent SCC's predicted FP; stored
    # per (parent_scc_id, child_scc_id) after inference.
    inter_edges: Dict[Tuple[int,int], Dict[int, float]] = field(default_factory=dict)


def _tarjan_scc(n: int, adj_bool: np.ndarray) -> List[List[int]]:
    """Tarjan's SCC algorithm. Returns list of SCCs in reverse topological order
    (each SCC comes before the SCCs it feeds into).

    adj_bool: [n, n] bool, adj_bool[i,j] = True if edge i -> j.
    """
    index_counter = [0]
    stack = []
    lowlink = [0] * n
    index = [0] * n
    on_stack = [False] * n
    index_set = [False] * n
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        index_set[v] = True
        stack.append(v)
        on_stack[v] = True

        for w in range(n):
            if not adj_bool[v, w]:
                continue
            if not index_set[w]:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    # Use iterative DFS to avoid Python recursion limit on large GRNs
    for v in range(n):
        if not index_set[v]:
            # Iterative Tarjan
            call_stack = [(v, 0, iter(range(n)))]
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            index_set[v] = True
            stack.append(v)
            on_stack[v] = True

            while call_stack:
                node, _, neighbors = call_stack[-1]
                try:
                    w = next(neighbors)
                    if not adj_bool[node, w]:
                        continue
                    if not index_set[w]:
                        index[w] = index_counter[0]
                        lowlink[w] = index_counter[0]
                        index_counter[0] += 1
                        index_set[w] = True
                        stack.append(w)
                        on_stack[w] = True
                        call_stack.append((w, 0, iter(range(n))))
                    elif on_stack[w]:
                        lowlink[node] = min(lowlink[node], index[w])
                except StopIteration:
                    call_stack.pop()
                    if call_stack:
                        parent = call_stack[-1][0]
                        lowlink[parent] = min(lowlink[parent], lowlink[node])
                    if lowlink[node] == index[node]:
                        scc = []
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            scc.append(w)
                            if w == node:
                                break
                        sccs.append(scc)

    # reverse topological order → return as-is (Tarjan gives reverse topo)
    return sccs


def decompose_grn(
    adj: np.ndarray,      # [N, N] signed adjacency (int/float; nonzero = edge present)
    P: np.ndarray,        # [N] production rates
    K: np.ndarray,        # [N] degradation rates
    T: np.ndarray,        # [N, N] Hill thresholds
    nh: np.ndarray,       # [N, N] Hill coefficients
    F: np.ndarray,        # [N, N] fold-changes
) -> SCCDAG:
    """Decompose a GRN into its SCC condensation DAG.

    Parameters
    ----------
    adj : signed adjacency matrix [N, N]. adj[i,j] != 0 means edge i -> j.
    P, K : per-gene production and degradation rates [N].
    T, nh, F : per-edge Hill parameters [N, N], indexed [source, target].

    Returns
    -------
    SCCDAG with SCCs in topological order (roots first) and inter-SCC edge maps.
    """
    N = adj.shape[0]
    adj_bool = (adj != 0)

    # --- 1. Find SCCs (Tarjan gives reverse topological order) ---
    raw_sccs = _tarjan_scc(N, adj_bool)  # reverse topo order
    # Reverse to get topological order (roots first)
    raw_sccs = raw_sccs[::-1]
    n_sccs = len(raw_sccs)

    # Map gene -> scc_id
    gene_to_scc = np.zeros(N, dtype=int)
    for scc_id, genes in enumerate(raw_sccs):
        for g in genes:
            gene_to_scc[g] = scc_id

    # --- 2. Build condensation DAG ---
    # Condensation edge: scc_a -> scc_b if any gene in a feeds a gene in b and a != b
    cond_adj = np.zeros((n_sccs, n_sccs), dtype=bool)
    inter_edge_genes: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    # (src_global, tgt_global) for inter-SCC edges
    for i in range(N):
        for j in range(N):
            if adj_bool[i, j] and gene_to_scc[i] != gene_to_scc[j]:
                sa, sb = gene_to_scc[i], gene_to_scc[j]
                cond_adj[sa, sb] = True
                key = (sa, sb)
                if key not in inter_edge_genes:
                    inter_edge_genes[key] = []
                inter_edge_genes[key].append((i, j))

    # --- 3. Build SccNode objects ---
    scc_nodes: List[SccNode] = []
    for scc_id, genes in enumerate(raw_sccs):
        genes_sorted = sorted(genes)
        n_scc = len(genes_sorted)
        local_idx = {g: i for i, g in enumerate(genes_sorted)}

        # Extract intra-SCC submatrices
        adj_intra = np.zeros((n_scc, n_scc), dtype=adj.dtype)
        T_intra = np.ones((n_scc, n_scc), dtype=np.float32)
        nh_intra = np.ones((n_scc, n_scc), dtype=np.float32)
        F_intra = np.ones((n_scc, n_scc), dtype=np.float32)

        for gi in genes_sorted:
            li = local_idx[gi]
            for gj in genes_sorted:
                lj = local_idx[gj]
                adj_intra[li, lj] = adj[gi, gj]
                T_intra[li, lj] = T[gi, gj]
                nh_intra[li, lj] = nh[gi, gj]
                F_intra[li, lj] = F[gi, gj]

        P_scc = P[genes_sorted].astype(np.float32)
        K_scc = K[genes_sorted].astype(np.float32)

        parents = [s for s in range(n_sccs) if cond_adj[s, scc_id]]
        children = [s for s in range(n_sccs) if cond_adj[scc_id, s]]

        scc_nodes.append(SccNode(
            scc_id=scc_id,
            genes=genes_sorted,
            n=n_scc,
            adj_intra=adj_intra,
            P=P_scc,
            K=K_scc,
            T=T_intra,
            nh=nh_intra,
            F=F_intra,
            parents=parents,
            children=children,
        ))

    roots = [s.scc_id for s in scc_nodes if len(s.parents) == 0]

    # --- 4. Build inter-edge map (for forcing computation later) ---
    inter_edges: Dict[Tuple[int,int], Dict[int, List[Tuple[int, float]]]] = {}
    for (sa, sb), edge_pairs in inter_edge_genes.items():
        # local_idx_b = local index in SCC b of the target gene
        genes_b = scc_nodes[sb].genes
        local_b = {g: i for i, g in enumerate(genes_b)}
        if (sa, sb) not in inter_edges:
            inter_edges[(sa, sb)] = {}
        for (gi, gj) in edge_pairs:
            lj = local_b[gj]
            if lj not in inter_edges[(sa, sb)]:
                inter_edges[(sa, sb)][lj] = []
            # Store (gi_global, gj_global) and edge params for later Hill computation
            inter_edges[(sa, sb)][lj].append((gi, gj))

    dag = SCCDAG(
        n_genes=N,
        sccs=scc_nodes,
        roots=roots,
        inter_edges=inter_edges,
    )
    return dag


# ---------------------------------------------------------------------------
# 2. run_inference — walk the DAG, call models on each SCC
# ---------------------------------------------------------------------------

@dataclass
class BranchNode:
    """One fixed point of one SCC under one set of parent FP choices."""
    scc_id: int
    fp_idx: int              # which FP of this SCC (0-indexed)
    x_local: np.ndarray      # FP position in SCC-local gene coordinates [n_scc]
    stab: int                # stability class (STAB_* constants)
    exist_prob: float        # model existence confidence
    # Children: dict of {child_scc_id: [BranchNode, ...]}
    children: Dict[int, List["BranchNode"]] = field(default_factory=dict)


def _make_scc_batch(scc: SccNode, u: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    """Build a single-sample batch dict for one SCC under a forcing vector u.

    u : [n_scc] forcing multipliers (1.0 = unforced, from parent SCC).
    The batch is shaped [1, n_scc, ...] to match the model's expected [B, N, ...] format.
    Genes are already in SCC-local coordinates.
    """
    from data import N_MAX
    n = scc.n

    # Pad all arrays to N_MAX along the gene dimension
    def pad2d(arr, fill=0.0):
        """[n, n] -> [N_MAX, N_MAX]"""
        out = np.full((N_MAX, N_MAX), fill, dtype=np.float32)
        out[:n, :n] = arr.astype(np.float32)
        return out

    def pad1d(arr, fill=0.0):
        """[n] -> [N_MAX]"""
        out = np.full(N_MAX, fill, dtype=np.float32)
        out[:n] = arr.astype(np.float32)
        return out

    def t(arr):
        return torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    P_eff = scc.P * u[:n]  # Apply forcing by multiplying production rates
    xmax = np.maximum(P_eff / np.maximum(scc.K, 1e-6), 1e-6)
    F_full = scc.F.copy()
    gain = np.where(scc.adj_intra != 0, np.maximum(F_full, 1.0), 1.0)
    xmax_full = (1.05 * P_eff / np.maximum(scc.K, 1e-6) * gain.prod(axis=0))
    xmax_full = np.maximum(xmax_full, 1e-6)

    adj_signed = scc.adj_intra.astype(np.float32)

    batch = {
        "P":        t(pad1d(P_eff, fill=1.0)),    # data.py:228 np.ones(N_MAX)
        "K":        t(pad1d(scc.K, fill=1.0)),    # data.py:229 np.ones(N_MAX)
        "xmax":     t(pad1d(xmax_full, fill=1.0)),  # data.py:230 np.ones(N_MAX)
        "adj":      t(pad2d(adj_signed)),
        "T":        t(pad2d(scc.T, fill=1.0)),    # data.py:238 np.ones(N_MAX,N_MAX)
        "nh":       t(pad2d(scc.nh, fill=1.0)),   # data.py:239 np.ones(N_MAX,N_MAX)
        "F":        t(pad2d(F_full, fill=1.0)),   # data.py:240 np.ones(N_MAX,N_MAX)
        "node_mask": torch.zeros(1, N_MAX, dtype=torch.bool, device=device),
        "u":        t(pad1d(u[:n])),
        "n":        torch.tensor([n], dtype=torch.long, device=device),
    }
    batch["node_mask"][0, :n] = True
    return batch


def _compute_forcing(
    dag: SCCDAG,
    parent_scc_id: int,
    child_scc_id: int,
    parent_x_local: np.ndarray,   # FP of parent SCC in local coords
    global_adj: np.ndarray,
    T: np.ndarray,
    nh: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Compute the forcing vector u for child_scc given parent's fixed point.

    For each gene j in child_scc that receives an inter-SCC edge from parent_scc,
    compute u[j_local] = prod over incoming inter-SCC edges of H(x_i; F_ij, T_ij, n_ij).

    Returns u [n_child] with 1.0 for unforced genes and the Hill product for forced ones.
    """
    child_scc = dag.sccs[child_scc_id]
    parent_scc = dag.sccs[parent_scc_id]
    n_child = child_scc.n
    u = np.ones(n_child, dtype=np.float64)

    edge_key = (parent_scc_id, child_scc_id)
    if edge_key not in dag.inter_edges:
        return u.astype(np.float32)

    edge_map = dag.inter_edges[edge_key]  # local_child_gene -> [(gi_global, gj_global), ...]
    parent_genes = parent_scc.genes
    parent_local = {g: i for i, g in enumerate(parent_genes)}

    for lj, edge_list in edge_map.items():
        for (gi_global, gj_global) in edge_list:
            li = parent_local[gi_global]
            x_i = float(parent_x_local[li])
            T_ij = float(T[gi_global, gj_global])
            nh_ij = float(nh[gi_global, gj_global])
            F_ij = float(F[gi_global, gj_global])
            r = (max(x_i, 1e-300) / T_ij) ** nh_ij
            H = (F_ij * r + 1.0) / (r + 1.0)
            u[lj] *= H

    return u.astype(np.float32)


def _solver_fallback_scc(
    scc: SccNode,
    u: np.ndarray,
    device: torch.device,
) -> List[Tuple[np.ndarray, int, float]]:
    """Exact Newton solver for one SCC — used when the GNN outputs degenerate logits.

    This calls the same solver used for ground-truth comparison, but scoped to a single SCC.
    Returns list of (x_local [n], stability_class, exist_prob=1.0).
    """
    from solver import solve_draws, SolverConfig

    n = scc.n
    P_eff = (scc.P * u[:n]).astype(np.float64)
    par = {
        "P":   P_eff[np.newaxis].astype(np.float32),
        "K":   scc.K[np.newaxis].astype(np.float32),
        "adj": scc.adj_intra[np.newaxis].astype(np.float32),
        "T":   scc.T[np.newaxis].astype(np.float32),
        "nh":  scc.nh[np.newaxis].astype(np.float32),
        "F":   scc.F[np.newaxis].astype(np.float32),
    }
    seeds_for_n = (96, 128, 160, 192, 224, 256, 288, 320, 352)
    cfg = SolverConfig(seeds=seeds_for_n, rounds=3)
    gen = torch.Generator(device=device); gen.manual_seed(0)

    try:
        res = solve_draws(par, cfg, n, gen, device)
        n_fps = int(res["counts"][0])
        results = []
        for k in range(n_fps):
            x_k  = res["state"][k].astype(np.float32)
            stab = int(res["stab"][k])
            results.append((x_k, stab, 1.0))
        return results if results else []
    except Exception:
        return []


def _model_is_degenerate(out) -> bool:
    """Return True if the predicted FP-set slots have near-zero existence confidence.

    Checks the top-k exist_logits (the k slots decode_slots would return) rather
    than the global max across all K slots.  The global max check fails when a
    high-index slot that is never selected by decode_slots's count-then-rank
    strategy happens to have |logit| >= the threshold due to random init or
    structural bias, masking the fact that every *selected* slot is near 0.5.
    """
    logits = out["count_logit"].float()[0]          # [max_fp+1]
    el = out["exist_logit"].float()[0]               # [K_slots]
    # Predicted count k (odd-only argmax, identical to decode_slots)
    even = torch.arange(logits.shape[0], device=logits.device) % 2 == 0
    k = int(logits.masked_fill(even, float("-inf")).argmax().item())
    k = max(k, 1)
    # The top-k most-positive exist_logits are the slots decode_slots returns.
    # If even the most confident of those is near zero, the model has no signal.
    topk_logits = el.topk(k).values   # [k] most positive
    return bool(topk_logits.max().item() < 0.1)


def _infer_scc(
    scc: SccNode,
    u: np.ndarray,
    l1_model,
    l3_model,
    device: torch.device,
    refine_steps: int = 5,
) -> List[Tuple[np.ndarray, int, float]]:
    """Run the appropriate model on one SCC and return predicted FPs.

    For root SCCs (u all-ones) uses the Layer I model (SccFixedPointNet).
    For forced SCCs uses the Layer III model (CorpusCNet).
    If the model outputs degenerate (uniform) logits, falls back to the exact Newton solver.

    Returns list of (x_local [n], stability_class, exist_prob).
    """
    from model import decode_position, decode_slots, newton_refine

    is_root = np.all(np.abs(u - 1.0) < U_FORCED_TOL)
    batch = _make_scc_batch(scc, u, device)

    with torch.no_grad():
        if is_root:
            out = l1_model(batch)
        else:
            out, _out0 = l3_model(batch, freeze_backbone=True)

    # If the model output is degenerate (all logits ≈ 0), fall back to the exact solver.
    # This happens for very small SCCs (n=1,2) that are outside the training distribution.
    if _model_is_degenerate(out):
        if getattr(_infer_scc, "_verbose", False):
            print(f"    [SCC {scc.scc_id} n={scc.n}] degenerate model output → Newton fallback")
        return _solver_fallback_scc(scc, u, device)

    node_mask = batch["node_mask"]
    nm = node_mask.float()
    n = scc.n
    P   = batch["P"].float()
    Kd  = batch["K"].float()
    xmax = batch["xmax"].float()
    adj  = batch["adj"]
    T_t  = batch["T"].float()
    nh_t = batch["nh"].float()
    F_t  = batch["F"].float()

    exist = decode_slots(out)                              # [1, K] bool
    x_pred = decode_position(out["pos"].float(), xmax.unsqueeze(1))  # [1, K, N_MAX]
    # Newton refinement
    x_ref = newton_refine(x_pred, P, Kd, xmax, adj, T_t, nh_t, F_t, node_mask, refine_steps)

    stab = out["stab_logit"].float().argmax(-1)            # [1, K]
    exist_logit = out["exist_logit"].float()
    exist_prob = torch.sigmoid(exist_logit)                # [1, K]
    # Guard against NaN (can arise from very large logits or degenerate model outputs)
    exist_prob = torch.nan_to_num(exist_prob, nan=0.5, posinf=1.0, neginf=0.0)

    # Diagnostic: print count_logit distribution for this SCC
    if getattr(_infer_scc, "_verbose", False):
        cl = out["count_logit"].float()[0]           # [max_fp+1]
        even_mask = torch.arange(cl.shape[0]) % 2 == 0
        cl_odd = cl.clone(); cl_odd[even_mask] = float("-inf")
        k_pred = int(cl_odd.argmax().item())
        ep_top4 = exist_prob[0, :4].tolist()
        print(f"    [SCC {scc.scc_id} n={scc.n}] count_logit peak at k={k_pred}  "
              f"exist_prob[:4]={[f'{p:.3f}' for p in ep_top4]}")

    results = []
    K_slots = exist.shape[1]
    for k in range(K_slots):
        if not exist[0, k].item():
            continue
        x_k = x_ref[0, k, :n].cpu().numpy()  # trim to actual SCC size
        # Guard against NaN in predicted position (fall back to xmax midpoint)
        x_k = np.where(np.isfinite(x_k), x_k, xmax.squeeze(0)[:n].cpu().numpy() * 0.5)
        s_k = int(stab[0, k].item())
        ep_k = float(exist_prob[0, k].item())
        results.append((x_k, s_k, ep_k))

    return results


def run_inference(
    dag: SCCDAG,
    global_adj: np.ndarray,  # full [N, N] signed adjacency (for Hill parameter lookup)
    T: np.ndarray,           # [N, N] global Hill thresholds
    nh: np.ndarray,          # [N, N] global Hill coefficients
    F: np.ndarray,           # [N, N] global fold-changes
    l1_model,                # SccFixedPointNet (Layer I)
    l3_model,                # CorpusCNet (Layer III)
    device: torch.device,
    refine_steps: int = 5,
) -> Dict[int, List[BranchNode]]:
    """Walk the SCC DAG and run inference on each SCC.

    The DAG is processed in topological order (roots first). For each SCC, every
    combination of parent fixed-point choices is a distinct forcing context, and the
    model is run once per context.

    Returns a dict: scc_id -> list of BranchNode trees rooted at that SCC's FPs.
    The BranchNode.children dict is populated by the recursive descent.

    The returned dict for root SCCs contains the complete branch trees.
    """
    # scc_id -> list of (context_key, u_vector, BranchNode or None)
    # We store: scc_id -> list of BranchNode (each is one FP of that SCC under one forcing)
    # Key insight: for the assembly we need to track *which parent FP* caused which child FP.
    # We store the branch trees on the BranchNode.children dict.

    # scc_id -> list of BranchNode (all FPs of this SCC across all forcing contexts)
    all_branches: Dict[int, List[BranchNode]] = {s.scc_id: [] for s in dag.sccs}

    # Process in topological order
    for scc in dag.sccs:
        scc_id = scc.scc_id

        if len(scc.parents) == 0:
            # Root SCC: no forcing
            u = np.ones(scc.n, dtype=np.float32)
            fps = _infer_scc(scc, u, l1_model, l3_model, device, refine_steps)
            for fp_idx, (x_local, stab, exist_prob) in enumerate(fps):
                node = BranchNode(
                    scc_id=scc_id,
                    fp_idx=fp_idx,
                    x_local=x_local,
                    stab=stab,
                    exist_prob=exist_prob,
                )
                all_branches[scc_id].append(node)
        else:
            # Forced SCC: one call per parent FP combination
            # Collect parent BranchNodes
            parent_fps: List[List[BranchNode]] = [all_branches[p] for p in scc.parents]

            # Generate all combinations of parent FP choices
            import itertools
            parent_combos = list(itertools.product(*parent_fps))
            if len(parent_combos) == 0:
                continue  # no parent FPs → skip

            for combo in parent_combos:
                # Compute combined forcing from all parents
                u = np.ones(scc.n, dtype=np.float32)
                for parent_node in combo:
                    parent_scc_id = parent_node.scc_id
                    u_contribution = _compute_forcing(
                        dag, parent_scc_id, scc_id,
                        parent_node.x_local,
                        global_adj, T, nh, F,
                    )
                    # Multiply contributions (multiple parents multiply their Hills)
                    u *= u_contribution

                fps = _infer_scc(scc, u, l1_model, l3_model, device, refine_steps)
                for fp_idx, (x_local, stab, exist_prob) in enumerate(fps):
                    node = BranchNode(
                        scc_id=scc_id,
                        fp_idx=fp_idx,
                        x_local=x_local,
                        stab=stab,
                        exist_prob=exist_prob,
                    )
                    # Register this node as a child of all parent nodes in this combo
                    for parent_node in combo:
                        if scc_id not in parent_node.children:
                            parent_node.children[scc_id] = []
                        parent_node.children[scc_id].append(node)
                    all_branches[scc_id].append(node)

    return all_branches


# ---------------------------------------------------------------------------
# 3. assemble_global_fps — enumerate paths through the branch tree
# ---------------------------------------------------------------------------

def assemble_global_fps(
    dag: SCCDAG,
    all_branches: Dict[int, List[BranchNode]],
) -> Dict[str, np.ndarray]:
    """Enumerate all consistent paths through the branch tree and assemble global FP vectors.

    A "consistent path" is a complete assignment of one FP to every SCC such that each
    child SCC's FP was generated under the forcing produced by its chosen parent FPs.

    Returns
    -------
    dict with keys:
      "fps"           : [n_fps, N] float32 — global FP position vectors
      "stab"          : [n_fps] int — global stability class (AND of local stabs)
      "is_stable"     : [n_fps] bool — True if all SCCs on path are stable
      "scc_stabs"     : [n_fps, n_sccs] int — per-SCC stability for each global FP
      "exist_probs"   : [n_fps] float — product of per-SCC existence probabilities
      "n_fps"         : int — number of global FPs found
    """
    N = dag.n_genes
    n_sccs = len(dag.sccs)

    # We do a depth-first path enumeration from root(s).
    # Each path is represented as an ordered list of BranchNodes, one per SCC.
    # The global FP is assembled from the SCC-local positions.

    # Build a mapping: scc_id -> sorted gene list (for vector assembly)
    scc_genes = {s.scc_id: s.genes for s in dag.sccs}

    # For the path enumeration we treat the multi-root case by adding a virtual root
    # that has all actual roots as children.

    fps_list: List[np.ndarray] = []
    stab_list: List[int] = []
    scc_stab_list: List[np.ndarray] = []
    exist_prob_list: List[float] = []

    def _global_stab(stabs: List[int]) -> int:
        """Combine per-SCC stability labels into a global label.

        Rule (from block-triangular Jacobian argument):
          - If any SCC is non-hyperbolic (5) or unknown (6): return 5 (non-hyperbolic)
          - If any SCC is a saddle (4): return 4 (saddle)
          - If any SCC is unstable (2, 3): return 2 if all unstable are nodes, else 3
          - If all are stable: return 0 (node) if all are nodes, else 1 (spiral)
        """
        if any(s in (STAB_NONHYP, STAB_UNKNOWN) for s in stabs):
            return STAB_NONHYP
        if any(s == STAB_SADDLE for s in stabs):
            return STAB_SADDLE
        any_unstable = any(s in (STAB_UNSTABLE_NODE, STAB_UNSTABLE_SPIRAL) for s in stabs)
        any_stable   = any(s in (STAB_STABLE_NODE,   STAB_STABLE_SPIRAL)   for s in stabs)
        if any_unstable and any_stable:
            return STAB_SADDLE  # mixed → effectively a saddle
        if any_unstable:
            has_spiral = any(s == STAB_UNSTABLE_SPIRAL for s in stabs)
            return STAB_UNSTABLE_SPIRAL if has_spiral else STAB_UNSTABLE_NODE
        # All stable
        has_spiral = any(s == STAB_STABLE_SPIRAL for s in stabs)
        return STAB_STABLE_SPIRAL if has_spiral else STAB_STABLE_NODE

    def _enumerate_paths(scc_id_order: List[int], branch_nodes_by_scc: Dict[int, List[BranchNode]]):
        """Depth-first enumeration of complete paths. One SCC at a time in topo order."""
        # This is called with all SCCs in topological order.
        # For each root SCC, we enumerate all its FPs and recurse.

        def recurse(path: List[BranchNode], scc_idx: int):
            if scc_idx == len(scc_id_order):
                # Complete path — assemble global FP vector
                x_global = np.zeros(N, dtype=np.float32)
                stabs = []
                prob = 1.0
                for node in path:
                    g_list = scc_genes[node.scc_id]
                    x_global[g_list] = node.x_local[:len(g_list)]
                    stabs.append(node.stab)
                    ep = node.exist_prob if np.isfinite(node.exist_prob) else 0.5
                    prob *= ep
                fps_list.append(x_global)
                stab_list.append(_global_stab(stabs))
                scc_stab_list.append(np.array(stabs, dtype=np.int32))
                exist_prob_list.append(prob)
                return

            scc_id = scc_id_order[scc_idx]
            scc = dag.sccs[scc_id]

            if len(scc.parents) == 0:
                # Root: try all FPs of this SCC
                candidates = branch_nodes_by_scc.get(scc_id, [])
                for node in candidates:
                    recurse(path + [node], scc_idx + 1)
            else:
                # Forced: candidates are children of the parent node(s) already in path
                # Find parent nodes in path
                parent_nodes = [n for n in path if n.scc_id in scc.parents]
                if not parent_nodes:
                    # No parent in path yet (shouldn't happen in topo order) — skip
                    recurse(path, scc_idx + 1)
                    return
                # Children must appear in ALL parent nodes' children for this scc_id
                # We take the intersection of child node sets
                candidate_sets = [
                    set(id(c) for c in pn.children.get(scc_id, []))
                    for pn in parent_nodes
                ]
                # The candidates available to this scc are those that appeared under
                # the combination of all parents simultaneously (stored when we ran inference).
                # Since we stored children on each parent node, children that appeared under
                # a specific combo are in ALL parents' child lists.
                # Use id-based intersection:
                if len(candidate_sets) == 1:
                    valid_ids = candidate_sets[0]
                else:
                    valid_ids = candidate_sets[0]
                    for s in candidate_sets[1:]:
                        valid_ids = valid_ids & s

                # Build mapping from id to node for all candidate children
                all_children = {}
                for pn in parent_nodes:
                    for c in pn.children.get(scc_id, []):
                        all_children[id(c)] = c

                candidates = [all_children[i] for i in valid_ids]
                if not candidates:
                    # No valid children — this branch path is incomplete; skip
                    return
                for node in candidates:
                    recurse(path + [node], scc_idx + 1)

        recurse([], 0)

    # Get topological order of SCC ids (already sorted in dag.sccs)
    scc_id_order = [s.scc_id for s in dag.sccs]
    _enumerate_paths(scc_id_order, all_branches)

    if not fps_list:
        return {
            "fps": np.zeros((0, N), dtype=np.float32),
            "stab": np.zeros(0, dtype=np.int32),
            "is_stable": np.zeros(0, dtype=bool),
            "scc_stabs": np.zeros((0, n_sccs), dtype=np.int32),
            "exist_probs": np.zeros(0, dtype=np.float32),
            "n_fps": 0,
        }

    fps_arr    = np.stack(fps_list, axis=0)         # [n_fps, N]
    stab_arr   = np.array(stab_list, dtype=np.int32)
    scc_stab_arr = np.stack(scc_stab_list, axis=0) if scc_stab_list else np.zeros((len(fps_list), n_sccs), dtype=np.int32)
    prob_arr   = np.array(exist_prob_list, dtype=np.float32)
    is_stable  = np.array([s in STABLE_CLASSES for s in stab_list], dtype=bool)

    return {
        "fps":         fps_arr,
        "stab":        stab_arr,
        "is_stable":   is_stable,
        "scc_stabs":   scc_stab_arr,
        "exist_probs": prob_arr,
        "n_fps":       len(fps_list),
    }


# ---------------------------------------------------------------------------
# 4. Newton refinement on the full GRN
# ---------------------------------------------------------------------------

def refine_fps(
    fps: np.ndarray,          # [n_fps, N] predicted FP positions
    adj: np.ndarray,          # [N, N] signed adjacency
    P: np.ndarray,            # [N]
    K: np.ndarray,            # [N]
    T: np.ndarray,            # [N, N]
    nh: np.ndarray,           # [N, N]
    F: np.ndarray,            # [N, N]
    device: torch.device,
    tol: float = 1e-8,
    max_steps: int = NEWTON_STEPS,
    n_restarts: int = 3,
) -> Dict[str, np.ndarray]:
    """Newton refinement of assembled global FP predictions.

    Each predicted FP is used as a seed for Newton iteration on the full N-gene ODE.
    Additionally, `n_restarts` random perturbations of each seed are tried.

    Returns
    -------
    dict with keys:
      "fps_refined"    : [n_fps, N] after refinement (converged or best iterate)
      "residuals"      : [n_fps] max relative ODE residual after refinement
      "converged"      : [n_fps] bool — residual < tol after refinement
      "n_converged"    : int
    """
    from model import newton_refine as model_newton, hill_residual, decode_position

    n_fps, N_dim = fps.shape
    if n_fps == 0:
        return {
            "fps_refined": fps,
            "residuals": np.zeros(0, dtype=np.float32),
            "converged": np.zeros(0, dtype=bool),
            "n_converged": 0,
        }

    adj_t = torch.as_tensor(adj.astype(np.float32), device=device).unsqueeze(0)  # [1, N, N]
    P_t   = torch.as_tensor(P.astype(np.float32), device=device).unsqueeze(0)    # [1, N]
    K_t   = torch.as_tensor(K.astype(np.float32), device=device).unsqueeze(0)
    T_t   = torch.as_tensor(T.astype(np.float32), device=device).unsqueeze(0)
    nh_t  = torch.as_tensor(nh.astype(np.float32), device=device).unsqueeze(0)
    F_t   = torch.as_tensor(F.astype(np.float32), device=device).unsqueeze(0)

    # Invariant box — compute in float64 to avoid overflow on dense graphs,
    # then clamp to float32 range before converting.
    # gain[j] = prod_{i: adj[i,j]!=0} max(F[i,j], 1)  (product over incoming sources)
    gain = np.where(adj != 0, np.maximum(F.astype(np.float64), 1.0), 1.0).prod(axis=0)
    xmax_np = 1.05 * P.astype(np.float64) / np.maximum(K.astype(np.float64), 1e-6) * gain
    xmax_np = np.clip(xmax_np, 1e-6, 3.4e38).astype(np.float32)  # clamp to float32 max
    xmax_t = torch.as_tensor(xmax_np, device=device).unsqueeze(0)

    node_mask = torch.ones(1, N_dim, dtype=torch.bool, device=device)

    fps_refined = np.zeros_like(fps)
    residuals   = np.full(n_fps, np.inf, dtype=np.float32)

    with torch.no_grad():
        for i in range(n_fps):
            x0 = torch.as_tensor(fps[i:i+1], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 1, N]
            x0 = x0.clamp(1e-9 * xmax_t.unsqueeze(1), xmax_t.unsqueeze(1))

            # Try the predicted seed + n_restarts perturbations
            seeds = [x0]
            for _ in range(n_restarts):
                noise = torch.randn_like(x0) * 0.1
                xs = (x0 * torch.exp(noise)).clamp(
                    1e-9 * xmax_t.unsqueeze(1), xmax_t.unsqueeze(1))
                seeds.append(xs)

            best_x = x0.squeeze(0)  # [1, N]
            best_res = np.inf

            for seed in seeds:
                x_ref = model_newton(seed, P_t, K_t, xmax_t, adj_t, T_t, nh_t, F_t,
                                     node_mask, max_steps)
                res = hill_residual(x_ref, P_t, K_t, xmax_t, adj_t, T_t, nh_t, F_t)
                r = float(res.abs().max().item())
                if r < best_res:
                    best_res = r
                    best_x = x_ref.squeeze(0)

            fps_refined[i] = best_x[0].cpu().numpy()
            residuals[i] = best_res

    converged = residuals < tol
    return {
        "fps_refined":  fps_refined,
        "residuals":    residuals,
        "converged":    converged,
        "n_converged":  int(converged.sum()),
    }


# ---------------------------------------------------------------------------
# 5. GRNPredictor — convenience end-to-end wrapper
# ---------------------------------------------------------------------------

class GRNPredictor:
    """End-to-end fixed-point predictor for arbitrary large GRNs.

    Usage
    -----
    >>> pred = GRNPredictor(l1_model, l3_model, device=torch.device("cuda"))
    >>> result = pred.predict(adj, P, K, T, nh, F)
    >>> result["global_fps"]        # [n_fps, N] ndarray
    >>> result["global_stab"]       # [n_fps] stability classes
    >>> result["converged"]         # [n_fps] bool (Newton convergence)
    """

    def __init__(self, l1_model, l3_model, device: Optional[torch.device] = None,
                 refine_steps: int = 5, newton_steps: int = NEWTON_STEPS,
                 newton_restarts: int = 3, tol: float = 1e-8):
        self.l1_model = l1_model
        self.l3_model = l3_model
        self.device = device or torch.device("cpu")
        self.refine_steps = refine_steps
        self.newton_steps = newton_steps
        self.newton_restarts = newton_restarts
        self.tol = tol

        self.l1_model.eval()
        self.l3_model.eval()

    def predict(
        self,
        adj: np.ndarray,    # [N, N] signed adjacency
        P: np.ndarray,      # [N]
        K: np.ndarray,      # [N]
        T: np.ndarray,      # [N, N]
        nh: np.ndarray,     # [N, N]
        F: np.ndarray,      # [N, N]
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Predict global fixed points of a GRN.

        Returns a dict with:
          "dag"           : SCCDAG decomposition
          "all_branches"  : raw branch tree from inference
          "global_fps"    : [n_fps, N] assembled global FP positions (before refinement)
          "global_stab"   : [n_fps] global stability labels
          "global_is_stable": [n_fps] bool
          "scc_stabs"     : [n_fps, n_sccs] per-SCC stability
          "exist_probs"   : [n_fps] existence probability product
          "n_fps"         : int
          "fps_refined"   : [n_fps, N] after Newton refinement
          "residuals"     : [n_fps] Newton residuals
          "converged"     : [n_fps] bool
          "n_converged"   : int
        """
        N = adj.shape[0]
        if verbose:
            print(f"[GRNPredictor] Decomposing {N}-gene GRN...")

        # Step 1: decompose
        dag = decompose_grn(adj, P, K, T, nh, F)
        if verbose:
            print(f"  {len(dag.sccs)} SCCs, {len(dag.roots)} root(s), "
                  f"sizes: {[s.n for s in dag.sccs]}")

        # Step 2: run inference
        if verbose:
            print(f"[GRNPredictor] Running inference on each SCC...")
        _infer_scc._verbose = getattr(self, "verbose_scc", False)
        all_branches = run_inference(
            dag, adj, T, nh, F,
            self.l1_model, self.l3_model,
            self.device, self.refine_steps,
        )
        for scc_id, branches in all_branches.items():
            if verbose:
                print(f"  SCC {scc_id} (n={dag.sccs[scc_id].n}): {len(branches)} branch FPs")

        # Step 3: assemble global FPs
        if verbose:
            print(f"[GRNPredictor] Assembling global fixed points...")
        assembly = assemble_global_fps(dag, all_branches)
        if verbose:
            print(f"  {assembly['n_fps']} global FP(s) found")

        # Step 4: Newton refinement on full GRN
        if assembly["n_fps"] > 0:
            if verbose:
                print(f"[GRNPredictor] Running Newton refinement on full {N}-gene system...")
            refinement = refine_fps(
                assembly["fps"], adj, P, K, T, nh, F,
                self.device, self.tol, self.newton_steps, self.newton_restarts,
            )
            if verbose:
                nc = refinement["n_converged"]
                nf = assembly["n_fps"]
                print(f"  {nc}/{nf} FPs converged (residual < {self.tol:.0e})")
        else:
            refinement = {
                "fps_refined": np.zeros((0, N), dtype=np.float32),
                "residuals": np.zeros(0, dtype=np.float32),
                "converged": np.zeros(0, dtype=bool),
                "n_converged": 0,
            }

        return {
            "dag":              dag,
            "all_branches":     all_branches,
            "global_fps":       assembly["fps"],
            "global_stab":      assembly["stab"],
            "global_is_stable": assembly["is_stable"],
            "scc_stabs":        assembly["scc_stabs"],
            "exist_probs":      assembly["exist_probs"],
            "n_fps":            assembly["n_fps"],
            **refinement,
        }


# ---------------------------------------------------------------------------
# 6. Utilities
# ---------------------------------------------------------------------------

def load_l1_model(checkpoint_path: str, device: torch.device):
    """Load a Layer I (SccFixedPointNet) checkpoint."""
    from model import SccFixedPointNet
    from types import SimpleNamespace
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    weights = ck.get("ema", ck.get("model"))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def load_l3_model(checkpoint_path: str, l1_model, device: torch.device):
    """Load a Layer III (CorpusCNet) checkpoint, wrapping the given L1 backbone."""
    from model_c import CorpusCNet
    from types import SimpleNamespace
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = CorpusCNet(l1_model, cfg)
    weights = ck.get("ema", ck.get("model"))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def print_result_summary(result: Dict[str, Any]):
    """Print a human-readable summary of GRNPredictor output."""
    dag = result["dag"]
    N = dag.n_genes
    n_sccs = len(dag.sccs)
    print(f"\n{'='*60}")
    print(f"GRN: {N} genes, {n_sccs} SCCs")
    print(f"  SCC sizes: {[s.n for s in dag.sccs]}")
    print(f"  Root SCCs: {dag.roots}")
    print(f"\nFixed points found: {result['n_fps']}")
    if result["n_fps"] > 0:
        print(f"  Converged (Newton): {result['n_converged']}/{result['n_fps']}")
        print(f"  Stable FPs: {int(result['global_is_stable'].sum())}/{result['n_fps']}")
        stab_names = ["stable node", "stable spiral", "unstable node",
                      "unstable spiral", "saddle", "non-hyperbolic", "unknown"]
        for i in range(result["n_fps"]):
            s = result["global_stab"][i]
            conv = "✓" if result["converged"][i] else "✗"
            res  = result["residuals"][i]
            prob = result["exist_probs"][i]
            print(f"  FP {i}: {stab_names[s]}, "
                  f"Newton {conv} (res={res:.2e}), "
                  f"confidence={prob:.3f}")
    print(f"{'='*60}\n")

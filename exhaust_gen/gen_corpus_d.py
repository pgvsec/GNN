"""Generate Corpus D: composition training data for Layer III.

Each sample is a composed 3+3 = 6-gene graph (SCC_A → SCC_B with cross edges).
Both SCCs are drawn from Corpus A.  We solve the full 6-gene system for
ground-truth FPs, then emit TWO branch records per composition case:

  1. The SCC_A branch (root, u=ones, solved from the 6-gene GT restricted
     to genes 0..2 projected back via the intra-A equations — but we just
     take the GT FPs of SCC_A: those are the first 3 coordinates of any
     6-gene FP).
  2. The SCC_B branch (one non-root branch per stable SCC_A FP):
       u_b[gene_b] = product of Hill(x_a[cross_src], T, nh, F) for all
                     cross edges (cross_src -> gene_b)
       GT FPs = GT 6-gene FPs restricted to genes 3..5 that are compatible
                with the given SCC_A FP.

The output is a set of .npz shard files in the Corpus C shard format so that
build_cache_c() in data_c.py can merge them directly and train_layer3.py
can use cfg.cache_dir_c pointing to the merged cache.

Usage (from REPO_ROOT):
  PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp python exhaust_gen/gen_corpus_d.py \
      --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A/cache \
      --out_dir      /scratch/gilbreth/gupt1042/corpus_d/shards \
      --n_cases      50000 \
      --n_cross      2 \
      --shard_size   1000 \
      --seed         0
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Corpus C shard schema constants (must match data_c.py)
GROUP_ORDER_C = ("dag", "scc", "iedge", "branch", "link", "bfp")
N_MAX = 10       # must match data_c.N_MAX
SCHEMA_VERSION = 1

SCFG = SimpleNamespace(
    eps_box=0.05,
    newton_steps=200,
    tol=1e-10,
    accept_tol=1e-6,
    seeds=[0, 8, 16, 32, 64, 96],
    max_corners=256,
    rounds=8,
    confirm=True,
)


# ---------------------------------------------------------------------------
# Corpus A cache helpers (same as gen_composition_eval.py)
# ---------------------------------------------------------------------------

def load_corpus_cache(cache_dir):
    cache_dir = Path(cache_dir)
    def mm(name):
        return np.load(cache_dir / f"{name}.npy", mmap_mode="r")
    g_n        = mm("g_n")
    g_edge_ptr = mm("g_edge_ptr")
    e_src      = mm("e_src")
    e_tgt      = mm("e_tgt")
    e_sgn      = mm("e_sgn")
    s_graph    = mm("s_graph")
    s_P        = mm("s_P")
    s_K        = mm("s_K")
    s_ep_ptr   = mm("s_ep_ptr")
    s_fp_ptr   = mm("s_fp_ptr")
    s_n_fp     = mm("s_n_fp")
    ep_T       = mm("ep_T")
    ep_nh      = mm("ep_nh")
    ep_F       = mm("ep_F")
    fp_state   = mm("fp_state")
    fp_stab    = mm("fp_stab")
    n_g  = len(g_n)
    n_e  = len(e_src)
    n_s  = len(s_graph)
    n_ep = len(ep_T)
    n_fp = len(fp_stab)
    N_MAX_corp = s_P.shape[1] if s_P.ndim > 1 else 10
    return dict(n_g=n_g, n_e=n_e, n_s=n_s, n_ep=n_ep, n_fp=n_fp,
                N_MAX=N_MAX_corp,
                g_n=g_n, g_edge_ptr=g_edge_ptr,
                e_src=e_src, e_tgt=e_tgt, e_sgn=e_sgn,
                s_graph=s_graph, s_P=s_P, s_K=s_K,
                s_ep_ptr=s_ep_ptr, s_fp_ptr=s_fp_ptr, s_n_fp=s_n_fp,
                ep_T=ep_T, ep_nh=ep_nh, ep_F=ep_F,
                fp_state=fp_state, fp_stab=fp_stab)


def get_edge_ptr(C, gi):
    ep0 = int(C["g_edge_ptr"][gi])
    ep1 = int(C["g_edge_ptr"][gi + 1]) if gi + 1 < C["n_g"] else C["n_e"]
    return ep0, ep1


def get_sample_ptrs(C, si):
    ep0 = int(C["s_ep_ptr"][si])
    ep1 = int(C["s_ep_ptr"][si + 1]) if si + 1 < C["n_s"] else C["n_ep"]
    fp0 = int(C["s_fp_ptr"][si])
    fp1 = int(C["s_fp_ptr"][si + 1]) if si + 1 < C["n_s"] else C["n_fp"]
    return ep0, ep1, fp0, fp1


def build_3gene_params(C, si):
    gi = int(C["s_graph"][si])
    ep0, ep1 = get_edge_ptr(C, gi)
    e_src = C["e_src"][ep0:ep1].astype(int)
    e_tgt = C["e_tgt"][ep0:ep1].astype(int)
    e_sgn = C["e_sgn"][ep0:ep1].astype(float)

    adj = np.zeros((3, 3), dtype=np.float32)
    for s, t, sg in zip(e_src, e_tgt, e_sgn):
        adj[s, t] = sg

    ep_s0, ep_s1, _, _ = get_sample_ptrs(C, si)

    P  = C["s_P"][si, :3].astype(np.float64)
    K  = C["s_K"][si, :3].astype(np.float64)
    T_mat  = np.ones((3, 3), dtype=np.float64)
    nh_mat = np.ones((3, 3), dtype=np.float64)
    F_mat  = np.ones((3, 3), dtype=np.float64)
    ep_T  = C["ep_T"][ep_s0:ep_s1]
    ep_nh = C["ep_nh"][ep_s0:ep_s1]
    ep_F  = C["ep_F"][ep_s0:ep_s1]
    for i, (s, t) in enumerate(zip(e_src, e_tgt)):
        T_mat[s, t]  = float(ep_T[i])
        nh_mat[s, t] = float(ep_nh[i])
        F_mat[s, t]  = float(ep_F[i])

    par = dict(P=P[None], K=K[None], adj=adj[None],
               T=T_mat[None], nh=nh_mat[None], F=F_mat[None])
    return adj, par, e_src, e_tgt, e_sgn, T_mat, nh_mat, F_mat


def get_gt_fps(C, si):
    _, _, fp0, fp1 = get_sample_ptrs(C, si)
    states = C["fp_state"][fp0:fp1].astype(np.float32)
    stabs  = C["fp_stab"][fp0:fp1].astype(np.int64)
    return states, stabs


# ---------------------------------------------------------------------------
# Compose two 3-gene SCCs and solve the 6-gene system
# ---------------------------------------------------------------------------

def compose_6gene(adj_a, par_a, adj_b, par_b, cross_edges, rng):
    adj6 = np.zeros((6, 6), dtype=np.float32)
    adj6[:3, :3] = adj_a
    adj6[3:, 3:] = adj_b
    for (sa, tb, sg) in cross_edges:
        adj6[sa, 3 + tb] = sg

    P6  = np.concatenate([par_a["P"][0],  par_b["P"][0]])
    K6  = np.concatenate([par_a["K"][0],  par_b["K"][0]])
    T6  = np.ones((6, 6), dtype=np.float64)
    nh6 = np.ones((6, 6), dtype=np.float64)
    F6  = np.ones((6, 6), dtype=np.float64)

    T6[:3, :3]  = par_a["T"][0];  T6[3:, 3:]  = par_b["T"][0]
    nh6[:3, :3] = par_a["nh"][0]; nh6[3:, 3:] = par_b["nh"][0]
    F6[:3, :3]  = par_a["F"][0];  F6[3:, 3:]  = par_b["F"][0]

    for (sa, tb, sg) in cross_edges:
        tgt = 3 + tb
        P_src = P6[sa]; K_src = K6[sa]
        middle = P_src / K_src
        T6[sa, tgt]  = float(np.clip(middle * 10.0 ** rng.uniform(-0.8, 0.8), 1e-3, 1e9))
        nh6[sa, tgt] = float(rng.choice([1, 2, 3, 4, 5]))
        if sg > 0:
            F6[sa, tgt] = float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        else:
            F6[sa, tgt] = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))

    par6 = dict(P=P6[None], K=K6[None], adj=adj6[None],
                T=T6[None], nh=nh6[None], F=F6[None])
    return adj6, par6, T6, nh6, F6


def solve_6gene(par6, device):
    import torch
    from solver import solve_draws
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    res = solve_draws(par6, SCFG, n=6, gen=gen, device=device)
    counts = res["counts"]
    if counts[0] == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.uint8)
    states = res["state"].astype(np.float32)[:, :6]
    stabs  = res["stab"].astype(np.uint8)
    return states, stabs


def hill(x, T, nh, F):
    """Scalar Hill function H(x; T, nh, F)."""
    r = (x / T) ** nh
    return (F * r + 1.0) / (r + 1.0)


def compute_u_b(x_a, cross_edges, T6, nh6, F6):
    """Compute forcing vector u for SCC_B given SCC_A fixed point x_a [3].
    u[gene_b] = product of Hill(x_a[src], T6[src, 3+tgt], nh6, F6) over
    all cross edges pointing at gene_b.
    Returns u [3].
    """
    u = np.ones(3, dtype=np.float64)
    for (sa, tb, sg) in cross_edges:
        x_src = float(x_a[sa])
        T_val  = float(T6[sa, 3 + tb])
        nh_val = float(nh6[sa, 3 + tb])
        F_val  = float(F6[sa, 3 + tb])
        u[tb] *= hill(x_src, T_val, nh_val, F_val)
    return u


# ---------------------------------------------------------------------------
# Thomas class computation
# ---------------------------------------------------------------------------

def thomas_class(adj):
    """1=TC_NEG (only neg fb), 2=TC_POS (only pos fb), 3=TC_BOTH (both)."""
    n = adj.shape[0]
    has_pos = False
    has_neg = False
    # Check all directed cycles via DFS
    sign_stack = []
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(v, acc_sign):
        nonlocal has_pos, has_neg
        visited[v] = True
        rec_stack[v] = True
        for u in range(n):
            if adj[v, u] == 0:
                continue
            s = int(np.sign(adj[v, u]))
            new_sign = acc_sign * s
            if rec_stack[u]:
                # found a cycle
                if new_sign > 0:
                    has_pos = True
                else:
                    has_neg = True
            elif not visited[u]:
                dfs(u, new_sign)
        rec_stack[v] = False

    for start in range(n):
        if not visited[start]:
            dfs(start, 1)

    if has_pos and has_neg:
        return 3
    elif has_pos:
        return 2
    else:
        return 1


# ---------------------------------------------------------------------------
# Corpus C shard builder
# ---------------------------------------------------------------------------

def empty_shard():
    """Return a fresh shard dict with empty lists for each field."""
    return {
        # DAG level
        "dag_n_sccs":    [], "dag_n_edges":   [], "dag_family":    [],
        "dag_shape":     [], "dag_depth":      [], "dag_max_fanin": [],
        "dag_n_reconv":  [], "dag_n_roots":    [], "dag_all_solved":[],
        # SCC level
        "scc_n":         [], "scc_tc":         [], "scc_dag_id":    [],
        "scc_topo_rank": [], "scc_depth":       [], "scc_n_parents": [],
        "scc_truncated": [], "scc_sign":        [],
        "scc_P":         [], "scc_K":           [],
        "scc_ep_T":      [], "scc_ep_nh":       [], "scc_ep_F":      [],
        # Inter-SCC edges
        "iedge_src_scc": [], "iedge_src_gene":  [],
        "iedge_tgt_scc": [], "iedge_tgt_gene":  [],
        "iedge_T":       [], "iedge_nh":        [], "iedge_F":       [],
        # Branch level
        "branch_scc":    [], "branch_key":      [], "branch_cons":   [],
        "branch_bsum":   [], "branch_u":        [],
        # Links
        "link_parent_branch": [], "link_parent_fp": [],
        # Branch FPs
        "bfp_x":         [], "bfp_charpoly":    [], "bfp_stab":      [],
        "bfp_hopf":      [], "bfp_zeroeig":      [], "bfp_resid":     [],
    }


def save_shard(shard, out_dir, shard_idx, n_cases_in_shard):
    """Write one shard npz file in the Corpus C format."""
    d = shard
    n_dag    = len(d["dag_n_sccs"])
    n_scc    = len(d["scc_n"])
    n_iedge  = len(d["iedge_src_scc"])
    n_branch = len(d["branch_scc"])
    n_link   = len(d["link_parent_branch"])
    n_bfp    = len(d["bfp_x"])

    sizes = np.array([n_dag, n_scc, n_iedge, n_branch, n_link, n_bfp],
                     dtype=np.int64)

    def stack(lst, dtype, shape_suffix=()):
        if len(lst) == 0:
            return np.zeros((0,) + shape_suffix, dtype=dtype)
        return np.array(lst, dtype=dtype)

    # Build sentinel pointer arrays
    def make_ptr(count_list):
        counts = np.array(count_list, dtype=np.int64)
        ptr = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        return ptr

    scc_branch_counts   = [0] * n_scc
    branch_fp_counts    = [0] * n_branch
    branch_link_counts  = [0] * n_branch
    dag_scc_counts      = [0] * n_dag
    dag_edge_counts     = [0] * n_dag

    # These are already encoded in the lists; rebuild from recorded info
    # We store branch->scc mapping and count:
    for bi, sc in enumerate(d["branch_scc"]):
        scc_branch_counts[sc] += 1
    for bi, cnt in enumerate(d.get("_branch_fp_counts", [])):
        branch_fp_counts[bi] = cnt
    for bi, cnt in enumerate(d.get("_branch_link_counts", [])):
        branch_link_counts[bi] = cnt
    for si, di in enumerate(d["scc_dag_id"]):
        dag_scc_counts[di] += 1
    for ei, di in enumerate(d["_iedge_dag_id"]):
        dag_edge_counts[di] += 1

    scc_branch_ptr  = make_ptr(scc_branch_counts)
    branch_fp_ptr   = make_ptr(branch_fp_counts)
    branch_link_ptr = make_ptr(branch_link_counts)
    dag_scc_ptr     = make_ptr(dag_scc_counts)
    dag_edge_ptr    = make_ptr(dag_edge_counts)

    arrays = {
        "_sizes": sizes,
        "schema_version": np.int64(SCHEMA_VERSION),
        # DAG
        "dag_n_sccs":    stack(d["dag_n_sccs"],    np.int32),
        "dag_n_edges":   stack(d["dag_n_edges"],   np.int32),
        "dag_family":    stack(d["dag_family"],    np.int32),
        "dag_shape":     stack(d["dag_shape"],     np.int32),
        "dag_depth":     stack(d["dag_depth"],     np.int32),
        "dag_max_fanin": stack(d["dag_max_fanin"], np.int32),
        "dag_n_reconv":  stack(d["dag_n_reconv"],  np.int32),
        "dag_n_roots":   stack(d["dag_n_roots"],   np.int32),
        "dag_all_solved":stack(d["dag_all_solved"],np.bool_),
        "dag_scc_ptr":   dag_scc_ptr,
        "dag_edge_ptr":  dag_edge_ptr,
        # SCC
        "scc_n":         stack(d["scc_n"],         np.int32),
        "scc_tc":        stack(d["scc_tc"],         np.int32),
        "scc_dag_id":    stack(d["scc_dag_id"],    np.int32),
        "scc_topo_rank": stack(d["scc_topo_rank"], np.int32),
        "scc_depth":     stack(d["scc_depth"],     np.int32),
        "scc_n_parents": stack(d["scc_n_parents"], np.int32),
        "scc_truncated": stack(d["scc_truncated"], np.bool_),
        "scc_sign":      stack(d["scc_sign"],      np.float32, (N_MAX, N_MAX)),
        "scc_P":         stack(d["scc_P"],         np.float32, (N_MAX,)),
        "scc_K":         stack(d["scc_K"],         np.float32, (N_MAX,)),
        "scc_ep_T":      stack(d["scc_ep_T"],      np.float32, (N_MAX, N_MAX)),
        "scc_ep_nh":     stack(d["scc_ep_nh"],     np.float32, (N_MAX, N_MAX)),
        "scc_ep_F":      stack(d["scc_ep_F"],      np.float32, (N_MAX, N_MAX)),
        "scc_branch_ptr":scc_branch_ptr,
        # Inter-SCC edges
        "iedge_src_scc": stack(d["iedge_src_scc"], np.int32),
        "iedge_src_gene":stack(d["iedge_src_gene"],np.int32),
        "iedge_tgt_scc": stack(d["iedge_tgt_scc"], np.int32),
        "iedge_tgt_gene":stack(d["iedge_tgt_gene"],np.int32),
        "iedge_T":       stack(d["iedge_T"],       np.float32),
        "iedge_nh":      stack(d["iedge_nh"],      np.float32),
        "iedge_F":       stack(d["iedge_F"],       np.float32),
        # Branch
        "branch_scc":    stack(d["branch_scc"],    np.int32),
        "branch_key":    stack(d["branch_key"],    np.int64),
        "branch_cons":   stack(d["branch_cons"],   np.bool_),
        "branch_bsum":   stack(d["branch_bsum"],   np.float32),
        "branch_u":      stack(d["branch_u"],      np.float32, (N_MAX,)),
        "branch_fp_ptr": branch_fp_ptr,
        "branch_link_ptr":branch_link_ptr,
        # Links
        "link_parent_branch": stack(d["link_parent_branch"], np.int32),
        "link_parent_fp":     stack(d["link_parent_fp"],     np.int32),
        # Branch FPs
        "bfp_x":        stack(d["bfp_x"],       np.float32, (N_MAX,)),
        "bfp_charpoly": stack(d["bfp_charpoly"],np.float32, (N_MAX,)),
        "bfp_stab":     stack(d["bfp_stab"],    np.int32),
        "bfp_hopf":     stack(d["bfp_hopf"],    np.bool_),
        "bfp_zeroeig":  stack(d["bfp_zeroeig"], np.bool_),
        "bfp_resid":    stack(d["bfp_resid"],   np.float32),
    }

    fname = out_dir / f"corpus_c_shard_{shard_idx:04d}.npz"
    np.savez_compressed(fname, **arrays)
    return fname


def add_composition_case(shard, dag_idx, scc_offset, branch_offset, link_offset, bfp_offset,
                         adj_a, par_a, T_mat_a, nh_mat_a, F_mat_a,
                         adj_b, par_b,
                         cross_edges, T6, nh6, F6,
                         fps_6, stabs_6,
                         tc_a, tc_b, n_cross):
    """Add one composition case (SCC_A root + SCC_B branches) to the shard dict.

    Returns updated (scc_offset, branch_offset, link_offset, bfp_offset,
                     n_branches_added, n_bfps_added).
    """
    STABLE_CLASSES = {0, 1}  # stable node, stable spiral

    # ------------------------------------------------------------------ DAG
    # 2 SCCs, n_cross inter-SCC edges, depth 2 (A at level 0, B at level 1)
    shard["dag_n_sccs"].append(2)
    shard["dag_n_edges"].append(n_cross)
    shard["dag_family"].append(0)   # chain-like
    shard["dag_shape"].append(0)
    shard["dag_depth"].append(2)
    shard["dag_max_fanin"].append(1)
    shard["dag_n_reconv"].append(0)
    shard["dag_n_roots"].append(1)
    shard["dag_all_solved"].append(True)

    # ------------------------------------------------------------------ SCCs
    scc_a_idx = scc_offset
    scc_b_idx = scc_offset + 1

    def make_scc_arrays(adj3, par, tc, dag_id, topo_rank, depth, n_parents):
        sign_pad  = np.zeros((N_MAX, N_MAX), np.float32)
        T_pad     = np.ones((N_MAX, N_MAX),  np.float32)
        nh_pad    = np.ones((N_MAX, N_MAX),  np.float32)
        F_pad     = np.ones((N_MAX, N_MAX),  np.float32)
        P_pad     = np.ones(N_MAX,           np.float32)
        K_pad     = np.ones(N_MAX,           np.float32)
        sign_pad[:3, :3] = adj3
        T_pad[:3, :3]    = par["T"][0].astype(np.float32)
        nh_pad[:3, :3]   = par["nh"][0].astype(np.float32)
        F_pad[:3, :3]    = par["F"][0].astype(np.float32)
        P_pad[:3] = par["P"][0].astype(np.float32)
        K_pad[:3] = par["K"][0].astype(np.float32)
        return dict(n=3, tc=tc, dag_id=dag_id, topo_rank=topo_rank, depth=depth,
                    n_parents=n_parents, truncated=False,
                    sign=sign_pad, P=P_pad, K=K_pad, T=T_pad, nh=nh_pad, F=F_pad)

    s_a = make_scc_arrays(adj_a, par_a, tc_a, dag_idx, 0, 0, 0)
    s_b = make_scc_arrays(adj_b, par_b, tc_b, dag_idx, 1, 1, 1)

    for s in [s_a, s_b]:
        shard["scc_n"].append(s["n"])
        shard["scc_tc"].append(s["tc"])
        shard["scc_dag_id"].append(s["dag_id"])
        shard["scc_topo_rank"].append(s["topo_rank"])
        shard["scc_depth"].append(s["depth"])
        shard["scc_n_parents"].append(s["n_parents"])
        shard["scc_truncated"].append(s["truncated"])
        shard["scc_sign"].append(s["sign"])
        shard["scc_P"].append(s["P"])
        shard["scc_K"].append(s["K"])
        shard["scc_ep_T"].append(s["T"])
        shard["scc_ep_nh"].append(s["nh"])
        shard["scc_ep_F"].append(s["F"])
    shard["scc_dag_id"].extend([dag_idx] * 0)   # already done above

    # ------------------------------------------------------------------ Inter-SCC edges
    shard.setdefault("_iedge_dag_id", [])
    for (sa, tb, sg) in cross_edges:
        shard["iedge_src_scc"].append(scc_a_idx)
        shard["iedge_src_gene"].append(int(sa))
        shard["iedge_tgt_scc"].append(scc_b_idx)
        shard["iedge_tgt_gene"].append(int(tb))
        shard["iedge_T"].append(float(T6[sa, 3 + tb]))
        shard["iedge_nh"].append(float(nh6[sa, 3 + tb]))
        shard["iedge_F"].append(float(F6[sa, 3 + tb]))
        shard["_iedge_dag_id"].append(dag_idx)

    # ------------------------------------------------------------------ SCC_A root branch
    # Identify stable FPs of SCC_A from the 6-gene GT (unique x[:3] vectors).
    if len(fps_6) == 0:
        # No GT FPs — skip this case entirely
        return scc_offset, branch_offset, link_offset, bfp_offset, 0, 0

    # All unique 3-gene projections for SCC_A
    fps_a_all  = fps_6[:, :3]
    fps_b_all  = fps_6[:, 3:]
    stabs_a    = stabs_6
    stabs_b    = stabs_6

    # Deduplicate SCC_A FPs
    a_seen = {}
    for ri in range(len(fps_6)):
        key = tuple(np.round(fps_a_all[ri], 4))
        if key not in a_seen:
            a_seen[key] = ri

    a_unique_fps   = [fps_6[ri, :3] for ri in a_seen.values()]
    a_unique_stabs = [stabs_6[ri]   for ri in a_seen.values()]

    # Build SCC_A root branch
    u_a_pad = np.ones(N_MAX, np.float32)  # root, no forcing
    shard["branch_scc"].append(scc_a_idx)
    shard["branch_key"].append(0)
    shard["branch_cons"].append(True)   # index sum from GT is by definition correct
    shard["branch_bsum"].append(1.0)
    shard["branch_u"].append(u_a_pad)

    branch_a_idx = branch_offset
    n_a_fps = len(a_unique_fps)
    shard.setdefault("_branch_fp_counts", [])
    shard.setdefault("_branch_link_counts", [])
    shard["_branch_fp_counts"].append(n_a_fps)
    shard["_branch_link_counts"].append(0)  # root has no links

    for ri in range(n_a_fps):
        fp_pad = np.zeros(N_MAX, np.float32)
        fp_pad[:3] = a_unique_fps[ri]
        cp_pad = np.zeros(N_MAX, np.float32)   # charpoly not available; zero is fine
        shard["bfp_x"].append(fp_pad)
        shard["bfp_charpoly"].append(cp_pad)
        shard["bfp_stab"].append(int(a_unique_stabs[ri]))
        shard["bfp_hopf"].append(False)
        shard["bfp_zeroeig"].append(False)
        shard["bfp_resid"].append(0.0)

    branch_offset += 1
    bfp_offset    += n_a_fps

    # ------------------------------------------------------------------ SCC_B branches
    # One branch per stable SCC_A FP.
    stable_a_fps = [(ri, a_unique_fps[ri]) for ri in range(n_a_fps)
                    if int(a_unique_stabs[ri]) in STABLE_CLASSES]

    n_b_branches_added = 0
    n_b_bfps_added     = 0

    for fp_a_slot, x_a in stable_a_fps:
        # Compute forcing vector u for SCC_B
        u_b = compute_u_b(x_a, cross_edges, T6, nh6, F6)
        u_b_pad = np.ones(N_MAX, np.float32)
        u_b_pad[:3] = u_b.astype(np.float32)

        # Gather SCC_B FPs compatible with this SCC_A FP
        # (match on x_a coords within tolerance)
        compatible_b_fps   = []
        compatible_b_stabs = []
        for ri in range(len(fps_6)):
            if np.max(np.abs(fps_6[ri, :3] - x_a)) < 1e-3:
                compatible_b_fps.append(fps_6[ri, 3:])
                compatible_b_stabs.append(int(stabs_6[ri]))

        if len(compatible_b_fps) == 0:
            continue

        shard["branch_scc"].append(scc_b_idx)
        shard["branch_key"].append(fp_a_slot)
        shard["branch_cons"].append(True)
        shard["branch_bsum"].append(1.0)
        shard["branch_u"].append(u_b_pad)

        shard["_branch_fp_counts"].append(len(compatible_b_fps))
        shard["_branch_link_counts"].append(1)

        # Link: this branch's parent is branch_a_idx, fp slot fp_a_slot
        shard["link_parent_branch"].append(branch_a_idx)
        shard["link_parent_fp"].append(fp_a_slot)
        link_offset += 1

        for fp_b, stab_b in zip(compatible_b_fps, compatible_b_stabs):
            fp_pad = np.zeros(N_MAX, np.float32)
            fp_pad[:3] = fp_b
            cp_pad = np.zeros(N_MAX, np.float32)
            shard["bfp_x"].append(fp_pad)
            shard["bfp_charpoly"].append(cp_pad)
            shard["bfp_stab"].append(stab_b)
            shard["bfp_hopf"].append(False)
            shard["bfp_zeroeig"].append(False)
            shard["bfp_resid"].append(0.0)

        branch_offset      += 1
        bfp_offset         += len(compatible_b_fps)
        n_b_branches_added += 1
        n_b_bfps_added     += len(compatible_b_fps)

    scc_offset += 2
    return scc_offset, branch_offset, link_offset, bfp_offset, 1 + n_b_branches_added, n_a_fps + n_b_bfps_added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_cache", required=True,
                    help="Path to Corpus A cache dir (contains g_n.npy etc.)")
    ap.add_argument("--out_dir",      required=True,
                    help="Output dir for .npz shard files")
    ap.add_argument("--n_cases",      type=int, default=50000)
    ap.add_argument("--n_cross",      type=int, default=2,
                    help="Number of A→B cross edges per composition")
    ap.add_argument("--shard_size",   type=int, default=1000,
                    help="Number of composition cases per shard")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--no_cuda",      action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus cache...")
    C = load_corpus_cache(args.corpus_cache)

    three_gene_samples = [si for si in range(C["n_s"])
                          if int(C["g_n"][int(C["s_graph"][si])]) == 3]
    print(f"Found {len(three_gene_samples)} 3-gene samples in corpus")
    if len(three_gene_samples) < 2:
        print("Not enough 3-gene samples.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    shard         = empty_shard()
    shard_idx     = 0
    cases_in_shard = 0
    dag_offset    = 0
    scc_offset    = 0
    branch_offset = 0
    link_offset   = 0
    bfp_offset    = 0

    total_cases = total_errors = total_branches = total_bfps = 0

    for case_i in range(args.n_cases):
        si_a, si_b = rng.choice(len(three_gene_samples), size=2, replace=False)
        si_a = three_gene_samples[si_a]
        si_b = three_gene_samples[si_b]

        adj_a, par_a, _, _, _, T_a, nh_a, F_a = build_3gene_params(C, si_a)
        adj_b, par_b, _, _, _, T_b, nh_b, F_b = build_3gene_params(C, si_b)

        cross_edges = []
        used = set()
        for _ in range(args.n_cross):
            for attempt in range(20):
                sa = int(rng.integers(0, 3))
                tb = int(rng.integers(0, 3))
                if (sa, tb) not in used:
                    used.add((sa, tb))
                    sg = float(rng.choice([-1.0, 1.0]))
                    cross_edges.append((sa, tb, sg))
                    break

        adj6, par6, T6, nh6, F6 = compose_6gene(adj_a, par_a, adj_b, par_b, cross_edges, rng)

        try:
            fps_6, stabs_6 = solve_6gene(par6, device)
        except Exception as exc:
            total_errors += 1
            if total_errors <= 5:
                print(f"  case {case_i}: solver error: {exc}")
            continue

        tc_a = thomas_class(adj_a)
        tc_b = thomas_class(adj_b)

        (scc_offset, branch_offset, link_offset, bfp_offset,
         n_branches, n_bfps) = add_composition_case(
            shard, dag_offset, scc_offset, branch_offset, link_offset, bfp_offset,
            adj_a, par_a, T_a, nh_a, F_a,
            adj_b, par_b,
            cross_edges, T6, nh6, F6,
            fps_6, stabs_6,
            tc_a, tc_b, len(cross_edges)
        )

        if n_branches == 0:
            continue  # case had no FPs, not added

        dag_offset     += 1
        cases_in_shard += 1
        total_cases    += 1
        total_branches += n_branches
        total_bfps     += n_bfps

        # Flush shard
        if cases_in_shard >= args.shard_size:
            fname = save_shard(shard, out_dir, shard_idx, cases_in_shard)
            print(f"  shard {shard_idx:04d} saved: {cases_in_shard} cases, "
                  f"{len(shard['branch_scc'])} branches, {len(shard['bfp_x'])} bfps -> {fname.name}")
            shard_idx     += 1
            cases_in_shard = 0
            dag_offset    = 0
            scc_offset    = 0
            branch_offset = 0
            link_offset   = 0
            bfp_offset    = 0
            shard         = empty_shard()

        if (case_i + 1) % 2000 == 0:
            print(f"  {case_i+1}/{args.n_cases}  cases={total_cases}  "
                  f"branches={total_branches}  bfps={total_bfps}  errors={total_errors}")

    # Flush final shard
    if cases_in_shard > 0:
        fname = save_shard(shard, out_dir, shard_idx, cases_in_shard)
        print(f"  shard {shard_idx:04d} saved: {cases_in_shard} cases -> {fname.name}")

    print(f"\nDone. {total_cases} cases, {total_branches} branches, "
          f"{total_bfps} bfps, {total_errors} errors")
    print(f"Shards written to: {out_dir}")
    print(f"\nNext: run build_cache_c() to merge shards, then point train_layer3.py "
          f"at the merged cache via --cache_dir_c.")


if __name__ == "__main__":
    main()

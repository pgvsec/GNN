"""Generate a composition eval dataset: pairs of 3-gene SCCs joined with
directed cross-SCC edges, solved for ground-truth fixed points.

Each test case:
  - SCC_A: nodes 0..2  (drawn from Corpus A 3-gene entries)
  - SCC_B: nodes 3..5  (drawn from Corpus A 3-gene entries)
  - Cross edges: a small number of A→B directed edges (no B→A, so two SCCs
    remain distinct in the condensation DAG)
  - Ground-truth FPs: solved numerically with the same solver used for Corpus A

Output (--out_dir):
  adj.npy         [N_cases, 6, 6]  signed adjacency
  P.npy           [N_cases, 6]
  K.npy           [N_cases, 6]
  T.npy           [N_cases, 6, 6]
  nh.npy          [N_cases, 6, 6]
  F.npy           [N_cases, 6, 6]
  fp_state.npy    [total_fps, 6]
  fp_stab.npy     [total_fps]       uint8
  fp_ptr.npy      [N_cases]         start index into fp_state for each case
  n_fp.npy        [N_cases]         int32

Usage (from REPO_ROOT, on gilbreth):
  PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp python exhaust_gen/gen_composition_eval.py \
      --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A/cache \
      --out_dir      /scratch/gilbreth/gupt1042/runs/composition_eval \
      --n_cases      500 \
      --n_cross      2 \
      --seed         42
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Config for the solver (mirrors exhaust_gen/generate.py defaults)
# ---------------------------------------------------------------------------
SCFG = SimpleNamespace(
    eps_box=0.05,
    newton_steps=200,
    tol=1e-12,           # match solver.py production default (was 1e-10)
    accept_tol=1e-9,     # match solver.py production default (was 1e-6, 1000× looser)
    seeds=[96, 128, 160, 192, 224, 256],  # match solver.py per-n seed counts (was 0,8,16,32,64,96)
    max_corners=256,
    rounds=10,           # more rounds for n=6 high-FP cases (was 8)
    confirm=True,
)


def load_corpus_cache(cache_dir):
    cache_dir = Path(cache_dir)
    def mm(name):
        return np.load(cache_dir / f"{name}.npy", mmap_mode="r")
    g_n        = mm("g_n")
    g_edge_ptr = mm("g_edge_ptr")
    g_split    = mm("g_split")
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
    N_MAX = s_P.shape[1] if s_P.ndim > 1 else 10
    return dict(n_g=n_g, n_e=n_e, n_s=n_s, n_ep=n_ep, n_fp=n_fp, N_MAX=N_MAX,
                g_n=g_n, g_edge_ptr=g_edge_ptr, g_split=g_split,
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
    """Return (adj [3,3], par dict with P,K,T,nh,F [1,3] / [1,3,3]) for sample si."""
    gi = int(C["s_graph"][si])
    ep0, ep1 = get_edge_ptr(C, gi)
    e_src = C["e_src"][ep0:ep1].astype(int)
    e_tgt = C["e_tgt"][ep0:ep1].astype(int)
    e_sgn = C["e_sgn"][ep0:ep1].astype(float)

    adj = np.zeros((3, 3), dtype=np.float32)
    for s, t, sg in zip(e_src, e_tgt, e_sgn):
        adj[s, t] = sg

    ep_s0, ep_s1, _, _ = get_sample_ptrs(C, si)
    N_MAX = C["N_MAX"]

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

    par = {
        "P":   P[None],           # [1, 3]
        "K":   K[None],
        "adj": adj[None],         # [1, 3, 3]
        "T":   T_mat[None],
        "nh":  nh_mat[None],
        "F":   F_mat[None],
    }
    return adj, par


def compose_6gene(adj_a, par_a, adj_b, par_b, cross_edges, rng):
    """Combine two 3-gene blocks into one 6-gene system.

    cross_edges: list of (src_in_A, tgt_in_B, sign) — added as A→B edges only.
    Parameters for cross edges are sampled fresh (phase-1 style).
    Returns adj6 [6,6], par6 with P,K,T,nh,F arrays shaped [1,6] / [1,6,6].
    """
    adj6 = np.zeros((6, 6), dtype=np.float32)
    adj6[:3, :3] = adj_a
    adj6[3:, 3:] = adj_b
    for (sa, tb, sg) in cross_edges:
        adj6[sa, 3 + tb] = sg

    P6  = np.concatenate([par_a["P"][0],  par_b["P"][0]])            # [6]
    K6  = np.concatenate([par_a["K"][0],  par_b["K"][0]])
    T6  = np.ones((6, 6), dtype=np.float64)
    nh6 = np.ones((6, 6), dtype=np.float64)
    F6  = np.ones((6, 6), dtype=np.float64)

    T6[:3, :3]  = par_a["T"][0];  T6[3:, 3:]  = par_b["T"][0]
    nh6[:3, :3] = par_a["nh"][0]; nh6[3:, 3:] = par_b["nh"][0]
    F6[:3, :3]  = par_a["F"][0];  F6[3:, 3:]  = par_b["F"][0]

    # sample parameters for cross edges
    for (sa, tb, sg) in cross_edges:
        tgt = 3 + tb
        P_tgt = P6[tgt]; K_tgt = K6[tgt]
        # place T near the expression range of the source gene
        P_src = P6[sa]; K_src = K6[sa]
        middle = (P_src / K_src)
        T6[sa, tgt]  = float(np.clip(middle * 10.0 ** rng.uniform(-0.8, 0.8), 1e-3, 1e9))
        nh6[sa, tgt] = float(rng.choice([1, 2, 3, 4, 5]))
        if sg > 0:
            F6[sa, tgt] = float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        else:
            F6[sa, tgt] = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))

    par6 = {
        "P":   P6[None],
        "K":   K6[None],
        "adj": adj6[None],
        "T":   T6[None],
        "nh":  nh6[None],
        "F":   F6[None],
    }
    return adj6, par6


def solve_one(par6, device):
    """Solve fixed points for a single 6-gene draw.

    Returns (fp_state [M,6], fp_stab [M], brouwer_certified bool).
    brouwer_certified=True means: index sum=1, all FPs hyperbolic, confirmed round found nothing new.
    """
    import torch
    from solver import solve_draws
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    res = solve_draws(par6, SCFG, n=6, gen=gen, device=device)
    counts = res["counts"]   # [1]
    certified = bool(res["cons"][0])
    if counts[0] == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.uint8), certified
    states = res["state"].astype(np.float32)   # [M, 6]
    stabs  = res["stab"].astype(np.uint8)       # [M]
    return states, stabs, certified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_cache", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--n_cases",      type=int, default=500)
    ap.add_argument("--n_cross",      type=int, default=2,
                    help="Number of cross-SCC edges to add (A→B only)")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--no_cuda",      action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print("Loading corpus cache...")
    C = load_corpus_cache(args.corpus_cache)

    # Collect all 3-gene sample indices (test split or all)
    three_gene_samples = [si for si in range(C["n_s"])
                          if int(C["g_n"][int(C["s_graph"][si])]) == 3]
    print(f"Found {len(three_gene_samples)} 3-gene samples in corpus")
    if len(three_gene_samples) < 2:
        print("Not enough 3-gene samples.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_adj   = []
    all_P     = []
    all_K     = []
    all_T     = []
    all_nh    = []
    all_F     = []
    all_fps   = []
    all_stab  = []
    fp_ptr    = []
    n_fp_arr  = []
    n_cert_arr = []   # 1 if Brouwer-certified, 0 otherwise
    fp_cursor = 0
    errors    = 0
    n_even_gt = 0

    for case_i in range(args.n_cases):
        # sample two distinct 3-gene samples
        si_a, si_b = rng.choice(len(three_gene_samples), size=2, replace=False)
        si_a = three_gene_samples[si_a]
        si_b = three_gene_samples[si_b]

        adj_a, par_a = build_3gene_params(C, si_a)
        adj_b, par_b = build_3gene_params(C, si_b)

        # choose random cross edges: A nodes 0-2 → B nodes 0-2, sign ±1
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

        adj6, par6 = compose_6gene(adj_a, par_a, adj_b, par_b, cross_edges, rng)

        try:
            fps, stabs, certified = solve_one(par6, device)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  case {case_i}: solver error: {exc}")
            continue

        n_fp = len(fps)
        if n_fp % 2 == 0 and n_fp > 0:
            n_even_gt += 1

        all_adj.append(adj6)
        all_P.append(par6["P"][0])
        all_K.append(par6["K"][0])
        all_T.append(par6["T"][0])
        all_nh.append(par6["nh"][0])
        all_F.append(par6["F"][0])
        fp_ptr.append(fp_cursor)
        n_fp_arr.append(n_fp)
        n_cert_arr.append(int(certified))
        fp_cursor += n_fp
        if n_fp > 0:
            all_fps.append(fps)
            all_stab.append(stabs)

        if (case_i + 1) % 50 == 0:
            n_done = len(n_fp_arr)
            cert_frac = sum(n_cert_arr) / max(n_done, 1)
            print(f"  {case_i+1}/{args.n_cases}  fp_cursor={fp_cursor}  "
                  f"errors={errors}  even_gt={n_even_gt}  certified={cert_frac:.1%}")

    n_cases = len(all_adj)
    cert_frac = sum(n_cert_arr) / max(n_cases, 1)
    n_fp_arr_np = np.array(n_fp_arr, dtype=np.int32)
    n_even = int((n_fp_arr_np % 2 == 0).sum())
    print(f"\nGenerated {n_cases} cases ({errors} errors), {fp_cursor} total FPs")
    print(f"  Brouwer-certified: {sum(n_cert_arr)}/{n_cases} ({cert_frac:.1%})")
    print(f"  Even GT counts (Poincare-Hopf violation): {n_even}/{n_cases} ({n_even/max(n_cases,1):.1%})")
    from collections import Counter
    ct = Counter(n_fp_arr_np.tolist())
    for k in sorted(ct):
        print(f"    gt={k}: {ct[k]} cases {'  *** EVEN (solver artifact)' if k % 2 == 0 else ''}")

    np.save(out_dir / "adj.npy",      np.array(all_adj,  dtype=np.float32))
    np.save(out_dir / "P.npy",        np.array(all_P,    dtype=np.float64))
    np.save(out_dir / "K.npy",        np.array(all_K,    dtype=np.float64))
    np.save(out_dir / "T.npy",        np.array(all_T,    dtype=np.float64))
    np.save(out_dir / "nh.npy",       np.array(all_nh,   dtype=np.float64))
    np.save(out_dir / "F.npy",        np.array(all_F,    dtype=np.float64))
    np.save(out_dir / "fp_ptr.npy",   np.array(fp_ptr,   dtype=np.int64))
    np.save(out_dir / "n_fp.npy",     n_fp_arr_np)
    np.save(out_dir / "n_cert.npy",   np.array(n_cert_arr, dtype=np.uint8))
    if all_fps:
        np.save(out_dir / "fp_state.npy", np.concatenate(all_fps,  axis=0))
        np.save(out_dir / "fp_stab.npy",  np.concatenate(all_stab, axis=0))
    else:
        np.save(out_dir / "fp_state.npy", np.zeros((0, 6), dtype=np.float32))
        np.save(out_dir / "fp_stab.npy",  np.zeros(0,      dtype=np.uint8))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

"""Generalized composition eval dataset generator.

Supports arbitrary SCC sizes and linear chains of 2 or 3 SCCs, enabling
four experiments:

  E1 — in-distribution size variation (e.g. 2+4, 4+2, 1+5)
       --n_a 2 --n_b 4       (total n=6, different split from canonical 3+3)

  E2 — scale extrapolation to n>6 (e.g. 3+4, 4+4)
       --n_a 3 --n_b 4       (total n=7, larger than training max)
       --n_a 4 --n_b 4       (total n=8)

  E3 — topology holdout (same size, OOD graph topologies)
       --n_a 3 --n_b 3 --split test

  E4 — deep DAG (three-SCC linear chain A→B→C)
       --n_a 3 --n_b 3 --n_c 3      (n=9, two levels of forcing)
       --n_a 2 --n_b 3 --n_c 4      (n=9, asymmetric chain)

Each test case:
  SCC_A : nodes 0 .. n_a-1           (root, no upstream forcing)
  SCC_B : nodes n_a .. n_a+n_b-1     (forced by A via A→B cross edges)
  SCC_C : nodes n_a+n_b .. N-1       (optional; forced by B via B→C cross edges)

Cross-edge parameters (T, nh, F) are sampled fresh for each case.
Ground-truth FPs are solved numerically on the combined system.

Usage:
  PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP python exhaust_gen/gen_composition_eval_general.py \\
      --corpus_cache $SCRATCH/corpus_exhaust_A/cache \\
      --out_dir      $SCRATCH/runs/composition_eval_E2_3p4 \\
      --n_a 3 --n_b 4 \\
      --n_cases 500 --n_cross_ab 2 --seed 42
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Solver config (mirrors canonical eval settings)
# ---------------------------------------------------------------------------
SCFG = SimpleNamespace(
    eps_box=0.05,
    newton_steps=200,
    tol=1e-12,
    accept_tol=1e-9,
    seeds=[96, 128, 160, 192, 224, 256],
    max_corners=256,
    rounds=10,
    confirm=True,
)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_cache(cache_dir):
    cache_dir = Path(cache_dir)
    def mm(name):
        return np.load(cache_dir / f"{name}.npy", mmap_mode="r")
    return dict(
        g_n        = mm("g_n"),
        g_edge_ptr = mm("g_edge_ptr"),
        g_split    = mm("g_split"),
        e_src      = mm("e_src"),
        e_tgt      = mm("e_tgt"),
        e_sgn      = mm("e_sgn"),
        s_graph    = mm("s_graph"),
        s_P        = mm("s_P"),
        s_K        = mm("s_K"),
        s_ep_ptr   = mm("s_ep_ptr"),
        s_fp_ptr   = mm("s_fp_ptr"),
        s_n_fp     = mm("s_n_fp"),
        ep_T       = mm("ep_T"),
        ep_nh      = mm("ep_nh"),
        ep_F       = mm("ep_F"),
        fp_state   = mm("fp_state"),
        fp_stab    = mm("fp_stab"),
        n_g        = len(mm("g_n")),
        n_e        = len(mm("e_src")),
        n_s        = len(mm("s_graph")),
        n_ep       = len(mm("ep_T")),
        n_fp       = len(mm("fp_stab")),
        N_MAX      = mm("s_P").shape[1] if mm("s_P").ndim > 1 else 10,
    )


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


def build_scc_params(C, si):
    """Return (adj [n,n], par dict) for an n-gene sample si from Corpus A.

    Works for any n (2-6). Returns the intra-SCC signed adjacency and per-gene
    / per-edge kinetic parameters.
    """
    gi  = int(C["s_graph"][si])
    n   = int(C["g_n"][gi])
    ep0, ep1 = get_edge_ptr(C, gi)
    e_src = C["e_src"][ep0:ep1].astype(int)
    e_tgt = C["e_tgt"][ep0:ep1].astype(int)
    e_sgn = C["e_sgn"][ep0:ep1].astype(float)

    adj = np.zeros((n, n), dtype=np.float32)
    for s, t, sg in zip(e_src, e_tgt, e_sgn):
        adj[s, t] = sg

    ep_s0, ep_s1, _, _ = get_sample_ptrs(C, si)
    P   = C["s_P"][si, :n].astype(np.float64)
    K   = C["s_K"][si, :n].astype(np.float64)
    T_mat  = np.ones((n, n), dtype=np.float64)
    nh_mat = np.ones((n, n), dtype=np.float64)
    F_mat  = np.ones((n, n), dtype=np.float64)
    ep_T  = C["ep_T"][ep_s0:ep_s1]
    ep_nh = C["ep_nh"][ep_s0:ep_s1]
    ep_F  = C["ep_F"][ep_s0:ep_s1]
    for i, (s, t) in enumerate(zip(e_src, e_tgt)):
        T_mat[s, t]  = float(ep_T[i])
        nh_mat[s, t] = float(ep_nh[i])
        F_mat[s, t]  = float(ep_F[i])

    par = {
        "P":   P[None],          # [1, n]
        "K":   K[None],
        "adj": adj[None],        # [1, n, n]
        "T":   T_mat[None],
        "nh":  nh_mat[None],
        "F":   F_mat[None],
    }
    return adj, par


def _sample_cross_edge_params(P6, K6, offset_src, offset_tgt, sa, tb, sg, rng):
    """Sample Hill parameters for one cross-SCC edge sa (in block A) → tb (in block B).

    sa, tb are LOCAL indices within their respective SCC blocks.
    offset_src, offset_tgt are the starting gene indices of the two blocks.
    Returns (T_val, nh_val, F_val).
    """
    gi = offset_src + sa
    gj = offset_tgt + tb
    P_src = P6[gi]; K_src = K6[gi]
    middle = P_src / K_src
    T_val  = float(np.clip(middle * 10.0 ** rng.uniform(-0.8, 0.8), 1e-3, 1e9))
    nh_val = float(rng.choice([1, 2, 3, 4, 5]))
    if sg > 0:
        F_val = float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
    else:
        F_val = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))
    return T_val, nh_val, F_val


def compose_sccs(scc_params, cross_edge_lists, rng):
    """Compose K SCC blocks into a single combined GRN.

    scc_params   : list of (adj_k [n_k, n_k], par_k) for k=0..K-1
    cross_edge_lists : list of K-1 edge lists; cross_edge_lists[k] is the list
                       of (sa, tb, sign) edges from block k to block k+1.
                       sa, tb are LOCAL indices within their respective blocks.

    Returns (adj_N [N, N], par_N dict) with N = sum(n_k).
    """
    K = len(scc_params)
    assert len(cross_edge_lists) == K - 1, "need K-1 cross-edge lists for K blocks"

    ns    = [par["P"].shape[1] for _, par in scc_params]
    N     = sum(ns)
    offsets = [sum(ns[:k]) for k in range(K)]

    adj_N  = np.zeros((N, N), dtype=np.float32)
    P_N    = np.zeros(N,       dtype=np.float64)
    K_N    = np.zeros(N,       dtype=np.float64)
    T_N    = np.ones((N, N),   dtype=np.float64)
    nh_N   = np.ones((N, N),   dtype=np.float64)
    F_N    = np.ones((N, N),   dtype=np.float64)

    # Fill intra-SCC blocks
    for k, (adj_k, par_k) in enumerate(scc_params):
        o = offsets[k]
        n = ns[k]
        adj_N[o:o+n, o:o+n] = adj_k
        P_N[o:o+n]           = par_k["P"][0]
        K_N[o:o+n]           = par_k["K"][0]
        T_N[o:o+n, o:o+n]   = par_k["T"][0]
        nh_N[o:o+n, o:o+n]  = par_k["nh"][0]
        F_N[o:o+n, o:o+n]   = par_k["F"][0]

    # Fill cross-SCC edges
    for k, edges in enumerate(cross_edge_lists):
        o_src = offsets[k]
        o_tgt = offsets[k + 1]
        for (sa, tb, sg) in edges:
            gi = o_src + sa
            gj = o_tgt + tb
            adj_N[gi, gj] = sg
            T_val, nh_val, F_val = _sample_cross_edge_params(
                P_N, K_N, o_src, o_tgt, sa, tb, sg, rng)
            T_N[gi, gj]  = T_val
            nh_N[gi, gj] = nh_val
            F_N[gi, gj]  = F_val

    par_N = {
        "P":   P_N[None],
        "K":   K_N[None],
        "adj": adj_N[None],
        "T":   T_N[None],
        "nh":  nh_N[None],
        "F":   F_N[None],
    }
    return adj_N, par_N


def sample_cross_edges(n_src, n_tgt, n_cross, rng):
    """Sample n_cross distinct directed edges src→tgt with random ±1 signs."""
    edges = []
    used  = set()
    for _ in range(n_cross):
        for _ in range(50):
            sa = int(rng.integers(0, n_src))
            tb = int(rng.integers(0, n_tgt))
            if (sa, tb) not in used:
                used.add((sa, tb))
                sg = float(rng.choice([-1.0, 1.0]))
                edges.append((sa, tb, sg))
                break
    return edges


def solve_combined(par_N, n_total, device):
    """Numerically solve the combined system. Returns (fps [M,N], stabs [M], certified)."""
    import torch
    from solver import solve_draws
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    res = solve_draws(par_N, SCFG, n=n_total, gen=gen, device=device)
    certified = bool(res["cons"][0])
    if int(res["counts"][0]) == 0:
        return (np.zeros((0, n_total), dtype=np.float32),
                np.zeros(0, dtype=np.uint8),
                certified)
    return (res["state"].astype(np.float32),
            res["stab"].astype(np.uint8),
            certified)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus_cache", required=True,
                    help="Path to Corpus A cache directory")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--n_a",  type=int, default=3,
                    help="Gene count of SCC A (root). Default 3.")
    ap.add_argument("--n_b",  type=int, default=3,
                    help="Gene count of SCC B (forced by A). Default 3.")
    ap.add_argument("--n_c",  type=int, default=None,
                    help="Gene count of SCC C (forced by B). Omit for 2-SCC cases.")
    ap.add_argument("--n_cases",    type=int, default=500)
    ap.add_argument("--n_cross_ab", type=int, default=2,
                    help="Cross edges A→B. Default 2.")
    ap.add_argument("--n_cross_bc", type=int, default=2,
                    help="Cross edges B→C (only used when --n_c is set). Default 2.")
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "val", "test"],
                    help="Restrict SCC draws to this Corpus A topology split. "
                         "'test' = topologies never used during L1/L3 training (E3).")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    n_chain = 3 if args.n_c is not None else 2
    sizes   = [args.n_a, args.n_b] + ([args.n_c] if args.n_c else [])
    N_total = sum(sizes)
    print(f"Device: {device}")
    print(f"Experiment: {'+'.join(str(s) for s in sizes)} = {N_total}-gene graph  "
          f"({'3-SCC chain' if n_chain == 3 else '2-SCC'})")
    print(f"Cross edges: A→B={args.n_cross_ab}"
          + (f"  B→C={args.n_cross_bc}" if n_chain == 3 else ""))
    print(f"Split: {args.split}  |  Cases: {args.n_cases}  |  Seed: {args.seed}")

    print("Loading corpus cache...")
    C = load_corpus_cache(args.corpus_cache)

    # Index samples by size and (optionally) split
    split_code = {"train": 1, "val": 2, "test": 3}.get(args.split)
    def get_samples_for_n(n):
        return [si for si in range(C["n_s"])
                if int(C["g_n"][int(C["s_graph"][si])]) == n and
                (split_code is None or int(C["g_split"][int(C["s_graph"][si])]) == split_code)]

    pools = {}
    for n in set(sizes):
        pools[n] = get_samples_for_n(n)
        print(f"  {len(pools[n])} samples with n={n} (split={args.split})")
        if len(pools[n]) == 0:
            print(f"  ERROR: no samples found for n={n}, split={args.split}. Exiting.")
            sys.exit(1)

    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_adj  = []
    all_P    = []; all_K  = []
    all_T    = []; all_nh = []; all_F = []
    all_fps  = []; all_stab = []
    fp_ptr   = []; n_fp_arr = []; n_cert_arr = []
    fp_cursor = 0
    errors    = 0
    n_even_gt = 0

    for case_i in range(args.n_cases):
        # Draw one sample per SCC (with replacement allowed across blocks)
        scc_params = []
        for n in sizes:
            si = int(rng.choice(pools[n]))
            adj_k, par_k = build_scc_params(C, si)
            scc_params.append((adj_k, par_k))

        # Sample cross edges
        cross_ab = sample_cross_edges(sizes[0], sizes[1], args.n_cross_ab, rng)
        cross_edge_lists = [cross_ab]
        if n_chain == 3:
            cross_bc = sample_cross_edges(sizes[1], sizes[2], args.n_cross_bc, rng)
            cross_edge_lists.append(cross_bc)

        adj_N, par_N = compose_sccs(scc_params, cross_edge_lists, rng)

        try:
            fps, stabs, certified = solve_combined(par_N, N_total, device)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  case {case_i}: solver error: {exc}")
            continue

        n_fp = len(fps)
        if n_fp % 2 == 0 and n_fp > 0:
            n_even_gt += 1

        all_adj.append(adj_N)
        all_P.append(par_N["P"][0])
        all_K.append(par_N["K"][0])
        all_T.append(par_N["T"][0])
        all_nh.append(par_N["nh"][0])
        all_F.append(par_N["F"][0])
        fp_ptr.append(fp_cursor)
        n_fp_arr.append(n_fp)
        n_cert_arr.append(int(certified))
        fp_cursor += n_fp
        if n_fp > 0:
            all_fps.append(fps)
            all_stab.append(stabs)

        if (case_i + 1) % 50 == 0:
            n_done    = len(n_fp_arr)
            cert_frac = sum(n_cert_arr) / max(n_done, 1)
            print(f"  {case_i+1}/{args.n_cases}  fp_cursor={fp_cursor}  "
                  f"errors={errors}  even_gt={n_even_gt}  certified={cert_frac:.1%}")

    n_cases  = len(all_adj)
    cert_frac = sum(n_cert_arr) / max(n_cases, 1)
    n_fp_arr_np = np.array(n_fp_arr, dtype=np.int32)
    print(f"\nGenerated {n_cases} cases ({errors} errors), {fp_cursor} total GT FPs")
    print(f"  Brouwer-certified: {sum(n_cert_arr)}/{n_cases} ({cert_frac:.1%})")
    n_even = int((n_fp_arr_np % 2 == 0).sum())
    print(f"  Even GT counts (Poincaré-Hopf violation): {n_even}/{n_cases}")
    ct = Counter(n_fp_arr_np.tolist())
    for k in sorted(ct):
        print(f"    gt={k}: {ct[k]} cases")

    np.save(out_dir / "adj.npy",    np.array(all_adj,  dtype=np.float32))
    np.save(out_dir / "P.npy",      np.array(all_P,    dtype=np.float64))
    np.save(out_dir / "K.npy",      np.array(all_K,    dtype=np.float64))
    np.save(out_dir / "T.npy",      np.array(all_T,    dtype=np.float64))
    np.save(out_dir / "nh.npy",     np.array(all_nh,   dtype=np.float64))
    np.save(out_dir / "F.npy",      np.array(all_F,    dtype=np.float64))
    np.save(out_dir / "fp_ptr.npy", np.array(fp_ptr,   dtype=np.int64))
    np.save(out_dir / "n_fp.npy",   n_fp_arr_np)
    np.save(out_dir / "n_cert.npy", np.array(n_cert_arr, dtype=np.uint8))
    if all_fps:
        np.save(out_dir / "fp_state.npy", np.concatenate(all_fps,  axis=0))
        np.save(out_dir / "fp_stab.npy",  np.concatenate(all_stab, axis=0))
    else:
        np.save(out_dir / "fp_state.npy", np.zeros((0, N_total), dtype=np.float32))
        np.save(out_dir / "fp_stab.npy",  np.zeros(0,             dtype=np.uint8))

    # Save metadata so the eval context is self-documenting
    meta = {
        "n_a": args.n_a, "n_b": args.n_b, "n_c": args.n_c,
        "N_total": N_total, "n_chain": n_chain,
        "split": args.split, "n_cross_ab": args.n_cross_ab,
        "n_cross_bc": args.n_cross_bc if n_chain == 3 else None,
        "n_cases_requested": args.n_cases, "n_cases_generated": n_cases,
        "seed": args.seed,
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

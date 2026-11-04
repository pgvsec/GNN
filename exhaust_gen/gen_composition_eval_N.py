"""Generate a composition eval dataset with arbitrary SCC sizes na + nb.

Generalizes gen_composition_eval.py to support any SCC sizes (3–8 genes each).
Both SCCs are drawn from the same corpus cache (which must contain graphs of
the requested sizes).

Usage (from REPO_ROOT):
  PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp python exhaust_gen/gen_composition_eval_N.py \
      --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A/cache \
      --out_dir      /scratch/gilbreth/gupt1042/runs/generalization/e3_4p5 \
      --na 4 --nb 5 \
      --n_cases 500 \
      --n_cross 2 \
      --seed 42
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def build_ngene_params(C, si, n):
    """Return (adj [n,n], par dict) for sample si with n genes."""
    gi = int(C["s_graph"][si])
    ep0, ep1 = get_edge_ptr(C, gi)
    e_src = C["e_src"][ep0:ep1].astype(int)
    e_tgt = C["e_tgt"][ep0:ep1].astype(int)
    e_sgn = C["e_sgn"][ep0:ep1].astype(float)

    adj = np.zeros((n, n), dtype=np.float32)
    for s, t, sg in zip(e_src, e_tgt, e_sgn):
        if s < n and t < n:
            adj[s, t] = sg

    ep_s0, ep_s1, _, _ = get_sample_ptrs(C, si)

    P  = C["s_P"][si, :n].astype(np.float64)
    K  = C["s_K"][si, :n].astype(np.float64)
    T_mat  = np.ones((n, n), dtype=np.float64)
    nh_mat = np.ones((n, n), dtype=np.float64)
    F_mat  = np.ones((n, n), dtype=np.float64)
    ep_T  = C["ep_T"][ep_s0:ep_s1]
    ep_nh = C["ep_nh"][ep_s0:ep_s1]
    ep_F  = C["ep_F"][ep_s0:ep_s1]
    for i, (s, t) in enumerate(zip(e_src, e_tgt)):
        if s < n and t < n:
            T_mat[s, t]  = float(ep_T[i])
            nh_mat[s, t] = float(ep_nh[i])
            F_mat[s, t]  = float(ep_F[i])

    par = {
        "P":   P[None],
        "K":   K[None],
        "adj": adj[None],
        "T":   T_mat[None],
        "nh":  nh_mat[None],
        "F":   F_mat[None],
    }
    return adj, par


def compose_Ngene(adj_a, par_a, adj_b, par_b, na, nb, cross_edges, rng):
    """Combine SCC_A (na genes) and SCC_B (nb genes) into one (na+nb)-gene system.

    cross_edges: list of (src_in_A, tgt_in_B, sign) — A→B only.
    """
    n = na + nb
    adjN = np.zeros((n, n), dtype=np.float32)
    adjN[:na, :na] = adj_a
    adjN[na:, na:] = adj_b
    for (sa, tb, sg) in cross_edges:
        adjN[sa, na + tb] = sg

    PN  = np.concatenate([par_a["P"][0],  par_b["P"][0]])
    KN  = np.concatenate([par_a["K"][0],  par_b["K"][0]])
    TN  = np.ones((n, n), dtype=np.float64)
    nhN = np.ones((n, n), dtype=np.float64)
    FN  = np.ones((n, n), dtype=np.float64)

    TN[:na, :na]  = par_a["T"][0];  TN[na:, na:]  = par_b["T"][0]
    nhN[:na, :na] = par_a["nh"][0]; nhN[na:, na:] = par_b["nh"][0]
    FN[:na, :na]  = par_a["F"][0];  FN[na:, na:]  = par_b["F"][0]

    for (sa, tb, sg) in cross_edges:
        tgt = na + tb
        P_src = PN[sa]; K_src = KN[sa]
        middle = (P_src / K_src)
        TN[sa, tgt]  = float(np.clip(middle * 10.0 ** rng.uniform(-0.8, 0.8), 1e-3, 1e9))
        nhN[sa, tgt] = float(rng.choice([1, 2, 3, 4, 5]))
        if sg > 0:
            FN[sa, tgt] = float(np.exp(rng.uniform(np.log(10.0), np.log(100.0))))
        else:
            FN[sa, tgt] = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))

    parN = {
        "P":   PN[None],
        "K":   KN[None],
        "adj": adjN[None],
        "T":   TN[None],
        "nh":  nhN[None],
        "F":   FN[None],
    }
    return adjN, parN


def solve_one(parN, n_total, device):
    import torch
    from solver import solve_draws
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    res = solve_draws(parN, SCFG, n=n_total, gen=gen, device=device)
    counts = res["counts"]
    certified = bool(res["cons"][0])
    if counts[0] == 0:
        return np.zeros((0, n_total), dtype=np.float32), np.zeros(0, dtype=np.uint8), certified
    states = res["state"].astype(np.float32)
    stabs  = res["stab"].astype(np.uint8)
    return states, stabs, certified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_cache", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--na",           type=int, required=True, help="Genes in SCC_A (root)")
    ap.add_argument("--nb",           type=int, required=True, help="Genes in SCC_B (forced)")
    ap.add_argument("--n_cases",      type=int, default=500)
    ap.add_argument("--n_cross",      type=int, default=2)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--split",        default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--no_cuda",      action="store_true")
    args = ap.parse_args()

    na, nb = args.na, args.nb
    n_total = na + nb

    import torch
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}  SCC sizes: {na}+{nb} = {n_total} genes total")

    print("Loading corpus cache...")
    C = load_corpus_cache(args.corpus_cache)

    split_code = {"train": 1, "val": 2, "test": 3}.get(args.split)

    def collect_samples(n_genes):
        samples = [si for si in range(C["n_s"])
                   if int(C["g_n"][int(C["s_graph"][si])]) == n_genes and
                   (split_code is None or int(C["g_split"][int(C["s_graph"][si])]) == split_code)]
        print(f"Found {len(samples)} {n_genes}-gene samples in corpus")
        if len(samples) < 2:
            print(f"ERROR: not enough {n_genes}-gene samples in corpus. "
                  f"Check --corpus_cache points to a corpus with {n_genes}-gene graphs.")
            sys.exit(1)
        return samples

    samples_a = collect_samples(na)
    samples_b = collect_samples(nb) if nb != na else samples_a

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
    n_cert_arr = []
    fp_cursor = 0
    errors    = 0
    n_even_gt = 0

    for case_i in range(args.n_cases):
        si_a = samples_a[int(rng.integers(0, len(samples_a)))]
        if nb == na:
            # sample two distinct indices from the same pool
            idx_a, idx_b = rng.choice(len(samples_a), size=2, replace=False)
            si_a = samples_a[idx_a]
            si_b = samples_b[idx_b]
        else:
            si_a = samples_a[int(rng.integers(0, len(samples_a)))]
            si_b = samples_b[int(rng.integers(0, len(samples_b)))]

        try:
            adj_a, par_a = build_ngene_params(C, si_a, na)
            adj_b, par_b = build_ngene_params(C, si_b, nb)

            cross_edges = []
            used = set()
            for _ in range(args.n_cross):
                for attempt in range(20):
                    sa = int(rng.integers(0, na))
                    tb = int(rng.integers(0, nb))
                    if (sa, tb) not in used:
                        used.add((sa, tb))
                        sg = float(rng.choice([-1.0, 1.0]))
                        cross_edges.append((sa, tb, sg))
                        break

            adjN, parN = compose_Ngene(adj_a, par_a, adj_b, par_b, na, nb, cross_edges, rng)
            fps, stabs, certified = solve_one(parN, n_total, device)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  case {case_i}: error: {exc}")
            continue

        n_fp = len(fps)
        if n_fp % 2 == 0 and n_fp > 0:
            n_even_gt += 1

        all_adj.append(adjN)
        all_P.append(parN["P"][0])
        all_K.append(parN["K"][0])
        all_T.append(parN["T"][0])
        all_nh.append(parN["nh"][0])
        all_F.append(parN["F"][0])
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
    print(f"  Even GT counts: {n_even}/{n_cases} ({n_even/max(n_cases,1):.1%})")
    from collections import Counter
    ct = Counter(n_fp_arr_np.tolist())
    for k in sorted(ct):
        print(f"    gt={k}: {ct[k]} cases")

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
        np.save(out_dir / "fp_state.npy", np.zeros((0, n_total), dtype=np.float32))
        np.save(out_dir / "fp_stab.npy",  np.zeros(0,            dtype=np.uint8))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

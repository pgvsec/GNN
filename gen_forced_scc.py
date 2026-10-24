"""Forced-SCC training samples: Corpus A subnetworks under constant upstream input.

In a feed-forward composition a downstream SCC receives a constant multiplier u on the
production of each gene with an inter-SCC edge, so its fixed points are those of the
autonomous SCC with production P * u. Corpus A draws P from [0.01, 100], while composition
inputs reach u ~ 1e-2 .. 1e2 (fold changes of 10-100 for activation, 0.01-0.2 for
repression), so forced SCCs often sit outside the range Layer I was trained on. This script
draws Corpus A *train-split* SCCs, applies a composition-like input to one or two genes, and
solves the result with the same solver and settings as Corpus A. Output samples carry the
full Layer I labels (positions, characteristic polynomials, stability) in the item format of
data.load_sample, so Layer I trains on them with its full loss.

Input model (mirrors exhaust_gen/gen_composition_eval.py cross edges): each forced gene gets
one or two Hill factors H = (F r + 1) / (r + 1), with F log-uniform in [10, 100] (activation)
or [0.01, 0.2] (repression) and the upstream occupancy r log-uniform in [1e-3, 1e3].

  python gen_forced_scc.py --corpus_a_cache .../corpus_exhaust_A_v2/cache \\
      --out .../forced_scc/forced_s1.npz --n_samples 200000 --seed 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data import N_MAX, SPLIT_TRAIN, open_cache
from solver import SolverConfig, solve_draws


def forcing(rng, n):
    """Multiplier u [n] for one draw: one or two forced genes, one or two Hill factors each."""
    u = np.ones(n)
    for g in rng.choice(n, size=min(n, int(rng.integers(1, 3))), replace=False):
        for _ in range(int(rng.integers(1, 3))):
            F = np.exp(rng.uniform(np.log(10.0), np.log(100.0))) if rng.random() < 0.5 \
                else np.exp(rng.uniform(np.log(0.01), np.log(0.2)))
            r = 10.0 ** rng.uniform(-3, 3)
            u[g] *= (F * r + 1.0) / (r + 1.0)
    return u


def source_params(a, s):
    """Dense parameters of Corpus A sample s: n, P, K [n]; adj, T, nh, F [n, n]; has_pos_fb."""
    g = int(a["s_graph"][s])
    n = int(a["g_n"][g])
    e0, ne = int(a["g_edge_ptr"][g]), int(a["g_n_edges"][g])
    p0 = int(a["s_ep_ptr"][s])
    src = np.asarray(a["e_src"][e0:e0 + ne], np.int64)
    tgt = np.asarray(a["e_tgt"][e0:e0 + ne], np.int64)
    adj = np.zeros((n, n))
    T, nh, F = np.ones((n, n)), np.ones((n, n)), np.ones((n, n))
    adj[src, tgt] = a["e_sgn"][e0:e0 + ne]
    T[src, tgt] = a["ep_T"][p0:p0 + ne]
    nh[src, tgt] = a["ep_nh"][p0:p0 + ne]
    F[src, tgt] = a["ep_F"][p0:p0 + ne]
    return (n, np.asarray(a["s_P"][s, :n], np.float64), np.asarray(a["s_K"][s, :n], np.float64),
            adj, T, nh, F, bool(a["g_pos_fb"][g]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus_a_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_samples", type=int, default=200000, help="draws attempted")
    ap.add_argument("--n_min", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=6)
    ap.add_argument("--max_fp", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    scfg = SolverConfig()
    a = open_cache(args.corpus_a_cache)
    s_graph = np.asarray(a["s_graph"])
    g_n = np.asarray(a["g_n"])[s_graph]
    split = np.asarray(a["g_split"])[s_graph]
    n_fp = np.asarray(a["s_n_fp"])
    pool = np.nonzero((split == SPLIT_TRAIN) & (g_n >= args.n_min) & (g_n <= args.n_max) & (n_fp >= 1))[0]
    print(f"{pool.size} Corpus A train-split source samples (n {args.n_min}..{args.n_max})", flush=True)
    picks = rng.choice(pool, size=args.n_samples, replace=True)

    out = {k: [] for k in ("n", "k", "P", "K", "xmax", "adj", "T", "nh", "F", "has_pos_fb",
                           "fp_state", "fp_charpoly", "fp_stab", "u_logmax")}
    kept, t0 = 0, time.time()
    for n in range(args.n_min, args.n_max + 1):
        idx = picks[g_n[picks] == n]
        for b0 in range(0, idx.size, args.batch):
            rows = [source_params(a, int(s)) for s in idx[b0:b0 + args.batch]]
            us = np.stack([forcing(rng, n) for _ in rows])
            par = {"P": np.stack([r[1] for r in rows]) * us, "K": np.stack([r[2] for r in rows]),
                   "adj": np.stack([r[3] for r in rows]), "T": np.stack([r[4] for r in rows]),
                   "nh": np.stack([r[5] for r in rows]), "F": np.stack([r[6] for r in rows])}
            for k in ("P", "K", "T", "F"):                       # same float32 rounding as Corpus A
                par[k] = par[k].astype(np.float32).astype(np.float64)
            res = solve_draws(par, scfg, n, gen, device)
            counts = res["counts"]
            ends = np.cumsum(counts)
            for i, r in enumerate(rows):
                c = int(counts[i])
                if not (res["cons"][i] and 1 <= c <= args.max_fp):
                    continue
                f1 = int(ends[i])
                f0 = f1 - c

                def pad1(x):
                    o = np.ones(N_MAX, np.float32)
                    o[:n] = x
                    return o

                def pad2(x, fill):
                    o = np.full((N_MAX, N_MAX), fill, np.float32)
                    o[:n, :n] = x
                    return o

                out["n"].append(n)
                out["k"].append(c)
                out["P"].append(pad1(par["P"][i]))
                out["K"].append(pad1(par["K"][i]))
                out["xmax"].append(pad1(res["xmax"][i]))
                out["adj"].append(pad2(par["adj"][i], 0.0))
                out["T"].append(pad2(par["T"][i], 1.0))
                out["nh"].append(pad2(par["nh"][i], 1.0))
                out["F"].append(pad2(par["F"][i], 1.0))
                out["has_pos_fb"].append(r[7])
                st = np.zeros((args.max_fp, N_MAX), np.float32)
                cp = np.zeros((args.max_fp, N_MAX), np.float32)
                sb = np.full(args.max_fp, -1, np.int64)
                st[:c, :n] = res["state"][f0:f1, :n]
                cp[:c, :n] = np.nan_to_num(res["charpoly"][f0:f1, :n])
                sb[:c] = res["stab"][f0:f1]
                out["fp_state"].append(st)
                out["fp_charpoly"].append(cp)
                out["fp_stab"].append(sb)
                out["u_logmax"].append(float(np.abs(np.log10(us[i])).max()))
                kept += 1
            print(f"  n={n} {b0 + len(rows)}/{idx.size}  kept {kept}  {time.time() - t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: np.asarray(v) for k, v in out.items()})
    ks = np.asarray(out["k"])
    print(f"saved {kept} samples to {args.out}; k distribution "
          f"{dict(zip(*np.unique(ks, return_counts=True)))}")


if __name__ == "__main__":
    main()

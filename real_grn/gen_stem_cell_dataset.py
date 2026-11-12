"""Generate ground-truth fixed-point datasets for the stem cell differentiation network.

Pluripotency → Hematopoiesis → Lymphocyte activation
Three-SCC depth-3 chain:
  SCC₁ {OCT4, SOX2, NANOG}  →  SCC₂ {GATA1, GATA2, KLF1, SPI1, TAL1}  →  SCC₃ {BATF, IRF4}

The ground truth is computed on the FULL 10-gene ODE system (not per-SCC).
Three signaling contexts × 1 network = 3 test cases.

Saves standard .npy format expected by eval_composition2.py / eval_stem_cell.py:
  adj.npy    [3, 10, 10]
  P.npy      [3, 10]
  K.npy      [3, 10]
  T.npy      [3, 10, 10]
  nh.npy     [3, 10, 10]
  F.npy      [3, 10, 10]
  fp_state.npy  [total_fps, 10]
  fp_stab.npy   [total_fps]       uint8
  fp_ptr.npy    [3]               int32
  n_fp.npy      [3]               int32

Usage (from REPO_ROOT):
  python real_grn/gen_stem_cell_dataset.py --out_base /path/to/output --no_cuda
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from solver import solve_draws
from real_grn.stem_cell_params import (
    CONTEXTS, ALL_GENES, SCC1_GENES, SCC2_GENES, SCC3_GENES,
    build_full_network,
)

_SCFG = SimpleNamespace(
    eps_box=0.05,
    newton_steps=200,
    tol=1e-12,
    accept_tol=1e-9,
    seeds=(96, 128, 160, 192, 224, 256, 288, 320),  # 8 seed counts for 10-gene system
    max_corners=512,
    rounds=15,
    confirm=True,
)

_STAB_NAMES = [
    "stable-node", "stable-spiral",
    "unstable-node", "unstable-spiral",
    "saddle", "non-hyperbolic",
]


def _solve_one(P, K, adj, T, nh, F, device):
    par = {
        "P":   P[None].astype(np.float64),
        "K":   K[None].astype(np.float64),
        "adj": adj[None].astype(np.float32),
        "T":   T[None].astype(np.float64),
        "nh":  nh[None].astype(np.float64),
        "F":   F[None].astype(np.float64),
    }
    n = len(P)
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    return solve_draws(par, _SCFG, n=n, gen=gen, device=device)


def generate(out_base, device, verbose=True):
    out_dir = Path(out_base) / "stem_cell"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(ALL_GENES)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Stem cell differentiation network  ({n_genes} genes, 3 SCCs)")
        print(f"  SCC₁: {SCC1_GENES}")
        print(f"  SCC₂: {SCC2_GENES}")
        print(f"  SCC₃: {SCC3_GENES}")
        print(f"{'='*65}")

    all_adj, all_P, all_K, all_T, all_nh, all_F = [], [], [], [], [], []
    all_states, all_stab = [], []
    fp_ptr, n_fp_arr = [], []
    fp_cursor = 0

    for ctx_name, ctx_info in CONTEXTS.items():
        P, K, adj, T, nh, F = build_full_network(ctx_name)
        peff = {g: P[i] for i, g in enumerate(ALL_GENES)}

        if verbose:
            print(f"\n  Context: {ctx_name}  ({ctx_info['label']})")
            print(f"  P: " + "  ".join(f"{g}={v:.2f}" for g, v in peff.items()))

        res = _solve_one(P, K, adj, T, nh, F, device)
        n_fp   = int(res["counts"][0])
        states = res["state"].astype(np.float32) if n_fp > 0 else np.zeros((0, n_genes), np.float32)
        stabs  = res["stab"].astype(np.uint8)    if n_fp > 0 else np.zeros(0, np.uint8)
        cert   = bool(res["cons"][0])
        bsum   = int(res["bsum"][0])
        n_stable = int((stabs <= 1).sum()) if n_fp > 0 else 0

        if verbose:
            print(f"  → {n_fp} fixed points  ({n_stable} stable)  "
                  f"Brouwer sum={bsum}  certified={cert}")
            nat = {g: P[i] / K[i] for i, g in enumerate(ALL_GENES)}
            for j in range(n_fp):
                on = [ALL_GENES[k] for k in range(n_genes)
                      if states[j, k] > 0.5 * nat[ALL_GENES[k]]]
                print(f"    FP{j+1} [{_STAB_NAMES[stabs[j]]}]: ON≈{on}")

        all_adj.append(adj[None])
        all_P.append(P[None])
        all_K.append(K[None])
        all_T.append(T[None])
        all_nh.append(nh[None])
        all_F.append(F[None])
        all_states.append(states)
        all_stab.append(stabs)
        fp_ptr.append(fp_cursor)
        n_fp_arr.append(n_fp)
        fp_cursor += n_fp

    np.save(out_dir / "adj.npy",     np.concatenate(all_adj,  axis=0).astype(np.float32))
    np.save(out_dir / "P.npy",       np.concatenate(all_P,    axis=0).astype(np.float64))
    np.save(out_dir / "K.npy",       np.concatenate(all_K,    axis=0).astype(np.float64))
    np.save(out_dir / "T.npy",       np.concatenate(all_T,    axis=0).astype(np.float64))
    np.save(out_dir / "nh.npy",      np.concatenate(all_nh,   axis=0).astype(np.float64))
    np.save(out_dir / "F.npy",       np.concatenate(all_F,    axis=0).astype(np.float64))
    if all_states:
        np.save(out_dir / "fp_state.npy", np.concatenate(all_states, axis=0))
        np.save(out_dir / "fp_stab.npy",  np.concatenate(all_stab,   axis=0))
    else:
        np.save(out_dir / "fp_state.npy", np.zeros((0, n_genes), np.float32))
        np.save(out_dir / "fp_stab.npy",  np.zeros(0, np.uint8))
    np.save(out_dir / "fp_ptr.npy",  np.array(fp_ptr,   dtype=np.int32))
    np.save(out_dir / "n_fp.npy",    np.array(n_fp_arr, dtype=np.int32))

    _write_summary(out_dir)

    if verbose:
        print(f"\n  Saved → {out_dir}")
        print(f"  Cases: {len(CONTEXTS)}  |  Total FPs: {fp_cursor}")

    return out_dir


def _write_summary(out_dir):
    n_fp_arr = np.load(out_dir / "n_fp.npy")
    fp_state = np.load(out_dir / "fp_state.npy")
    fp_stab  = np.load(out_dir / "fp_stab.npy")
    fp_ptr   = np.load(out_dir / "fp_ptr.npy")
    n_genes  = len(ALL_GENES)

    summary = {
        "network": "Pluripotency → Hematopoiesis → Lymphocyte activation",
        "gene_order": ALL_GENES,
        "scc1": SCC1_GENES, "scc2": SCC2_GENES, "scc3": SCC3_GENES,
        "n_cases": int(len(n_fp_arr)),
        "contexts": list(CONTEXTS.keys()),
        "cases": [],
    }
    nat_P = None
    P_all = np.load(out_dir / "P.npy")
    K_all = np.load(out_dir / "K.npy")

    for i, ctx_name in enumerate(CONTEXTS.keys()):
        fp0, fp1 = int(fp_ptr[i]), int(fp_ptr[i]) + int(n_fp_arr[i])
        states = fp_state[fp0:fp1]
        stabs  = fp_stab[fp0:fp1]
        nat    = P_all[i] / K_all[i]
        fps_info = []
        for j in range(len(states)):
            on = [ALL_GENES[k] for k in range(n_genes) if states[j, k] > 0.5 * nat[k]]
            fps_info.append({
                "stability": _STAB_NAMES[int(stabs[j])],
                "on_genes":  on,
                "state":     [round(float(x), 4) for x in states[j]],
            })
        summary["cases"].append({
            "context":   ctx_name,
            "n_fp":      int(n_fp_arr[i]),
            "n_stable":  int((stabs <= 1).sum()) if len(stabs) > 0 else 0,
            "fixed_points": fps_info,
        })

    with open(out_dir / "ground_truth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="/scratch/gilbreth/gupt1042/runs/real_grn")
    ap.add_argument("--no_cuda",  action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    generate(args.out_base, device)


if __name__ == "__main__":
    main()

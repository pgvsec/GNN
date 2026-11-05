"""Generate ground-truth fixed-point datasets for the Drosophila gap gene network.

Saves datasets in the standard eval format expected by eval_composition2.py:
  adj.npy    [N_cases, n_genes, n_genes]
  P.npy      [N_cases, n_genes]
  K.npy      [N_cases, n_genes]
  T.npy      [N_cases, n_genes, n_genes]
  nh.npy     [N_cases, n_genes, n_genes]
  F.npy      [N_cases, n_genes, n_genes]
  fp_state.npy  [total_fps, n_genes]
  fp_stab.npy   [total_fps]  uint8
  fp_ptr.npy    [N_cases]    int32  (start index into fp_state)
  n_fp.npy      [N_cases]    int32

Each AP position is one test case.  Three positions × two models = 6 datasets.

Usage (from REPO_ROOT):
  python real_grn/gen_gap_gene_dataset.py --out_base /path/to/output --no_cuda
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# resolve imports from the repo root regardless of cwd
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from solver import solve_draws
from real_grn.gap_gene_params import (
    AP_POSITIONS,
    MODEL_A_GENE_NAMES, build_model_a,
    MODEL_B_GENE_NAMES, build_model_b,
)

# ---------------------------------------------------------------------------
# Solver config: generous seeds for a 4-gene problem
# ---------------------------------------------------------------------------
_SCFG = SimpleNamespace(
    eps_box=0.05,
    newton_steps=200,
    tol=1e-12,
    accept_tol=1e-9,
    seeds=(96, 128, 160, 192, 224, 256),  # covers n=2..7
    max_corners=256,
    rounds=12,
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


def _generate_model(model_name, out_base, device, verbose=True):
    """Generate and save datasets for all AP positions for one model."""
    build_fn   = build_model_a if model_name == "A" else build_model_b
    gene_names = MODEL_A_GENE_NAMES if model_name == "A" else MODEL_B_GENE_NAMES
    out_dir    = Path(out_base) / f"gap_gene_model_{model_name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_adj, all_P, all_K, all_T, all_nh, all_F = [], [], [], [], [], []
    all_states, all_stab = [], []
    fp_ptr, n_fp_arr = [], []
    fp_cursor = 0

    model_title = ("Model A — Reinitz/Manu 1-SCC"
                   if model_name == "A" else
                   "Model B — Papatsenko-Levine 2-SCC")
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {model_title}")
        print(f"  Genes (node order): {gene_names}")
        print(f"{'='*60}")

    for ap_name in AP_POSITIONS:
        P, K, adj, T, nh, F = build_fn(ap_name)
        peff = {g: P[i] for i, g in enumerate(gene_names)}

        if verbose:
            label = AP_POSITIONS[ap_name]["label"]
            print(f"\n  AP position: {ap_name}  ({label})")
            print(f"  P (effective production): " +
                  "  ".join(f"{g}={v:.2f}" for g, v in peff.items()))

        res = _solve_one(P, K, adj, T, nh, F, device)
        n_fp   = int(res["counts"][0])
        states = res["state"].astype(np.float32) if n_fp > 0 else np.zeros((0, len(P)), np.float32)
        stabs  = res["stab"].astype(np.uint8)    if n_fp > 0 else np.zeros(0, np.uint8)
        cert   = bool(res["cons"][0])
        bsum   = int(res["bsum"][0])
        n_stable = int((stabs <= 1).sum())

        if verbose:
            print(f"  → {n_fp} fixed points  ({n_stable} stable)  "
                  f"Brouwer sum={bsum}  certified={cert}")
            for j in range(n_fp):
                on = [gene_names[k] for k, v in enumerate(states[j]) if v > 0.5 * (P[k] / K[k])]
                print(f"    FP{j+1} [{_STAB_NAMES[stabs[j]]}]: "
                      f"ON≈{on}  "
                      f"x={[f'{states[j,k]:.3f}' for k in range(len(gene_names))]}")

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

    # Save
    np.save(out_dir / "adj.npy",      np.concatenate(all_adj,   axis=0).astype(np.float32))
    np.save(out_dir / "P.npy",        np.concatenate(all_P,     axis=0).astype(np.float64))
    np.save(out_dir / "K.npy",        np.concatenate(all_K,     axis=0).astype(np.float64))
    np.save(out_dir / "T.npy",        np.concatenate(all_T,     axis=0).astype(np.float64))
    np.save(out_dir / "nh.npy",       np.concatenate(all_nh,    axis=0).astype(np.float64))
    np.save(out_dir / "F.npy",        np.concatenate(all_F,     axis=0).astype(np.float64))
    if all_states:
        np.save(out_dir / "fp_state.npy", np.concatenate(all_states, axis=0))
        np.save(out_dir / "fp_stab.npy",  np.concatenate(all_stab,   axis=0))
    else:
        np.save(out_dir / "fp_state.npy", np.zeros((0, len(gene_names)), np.float32))
        np.save(out_dir / "fp_stab.npy",  np.zeros(0, np.uint8))
    np.save(out_dir / "fp_ptr.npy",   np.array(fp_ptr,    dtype=np.int32))
    np.save(out_dir / "n_fp.npy",     np.array(n_fp_arr,  dtype=np.int32))

    if verbose:
        print(f"\n  Saved → {out_dir}")
        print(f"  Cases: {len(AP_POSITIONS)}  |  Total FPs: {fp_cursor}")

    # Save a human-readable summary
    _write_summary(out_dir, model_name, gene_names)
    return out_dir


def _write_summary(out_dir, model_name, gene_names):
    """Write a JSON-style ground-truth summary for quick reference."""
    import json
    adj_all   = np.load(out_dir / "adj.npy")
    n_fp_arr  = np.load(out_dir / "n_fp.npy")
    fp_state  = np.load(out_dir / "fp_state.npy")
    fp_stab   = np.load(out_dir / "fp_stab.npy")
    fp_ptr    = np.load(out_dir / "fp_ptr.npy")

    summary = {
        "model": f"Model {model_name}",
        "gene_order": gene_names,
        "n_cases": int(len(n_fp_arr)),
        "ap_positions": list(AP_POSITIONS.keys()),
        "cases": [],
    }

    for i, ap_name in enumerate(AP_POSITIONS.keys()):
        fp0, fp1 = int(fp_ptr[i]), int(fp_ptr[i]) + int(n_fp_arr[i])
        states = fp_state[fp0:fp1]
        stabs  = fp_stab[fp0:fp1]
        fps_info = []
        for j in range(len(states)):
            fps_info.append({
                "stability": _STAB_NAMES[int(stabs[j])],
                "state": [round(float(x), 4) for x in states[j]],
            })
        summary["cases"].append({
            "ap_position": ap_name,
            "n_fp": int(n_fp_arr[i]),
            "n_stable": int((stabs <= 1).sum()),
            "fixed_points": fps_info,
        })

    with open(out_dir / "ground_truth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    choices=["A", "B", "both"], default="both",
                    help="Which model topology to generate")
    ap.add_argument("--out_base", default="/scratch/gilbreth/gupt1042/runs/real_grn",
                    help="Parent directory for dataset output")
    ap.add_argument("--no_cuda",  action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    models = ["A", "B"] if args.model == "both" else [args.model]
    for m in models:
        _generate_model(m, args.out_base, device)


if __name__ == "__main__":
    main()

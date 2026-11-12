"""Evaluate GNN predictions on the stem cell differentiation network.

Pluripotency → Hematopoiesis → Lymphocyte activation
Three-SCC depth-3 chain (10 genes):
  SCC₁ {OCT4, SOX2, NANOG}  →  SCC₂ {GATA1, GATA2, KLF1, SPI1, TAL1}  →  SCC₃ {BATF, IRF4}

The GNN processes this as:
  L1 → SCC₁  (root SCC, no upstream conditioning)
  L3 → SCC₂  (forced by SCC₁ FPs)
  L3 → SCC₃  (forced by SCC₁ × SCC₂ combined upstream FPs)

This is a DAG depth = 3, which is out-of-distribution for the training set
(Corpus C max depth = 2: L1 + single L3).  The GRNPredictor.predict() handles
it via topological-sort over the SCC DAG.

Usage (from REPO_ROOT):
  python real_grn/eval_stem_cell.py \\
      --l1_ckpt $SCRATCH/runs/e2e_v10/l1_best.pt \\
      --l3_ckpt $SCRATCH/runs/e2e_v10/l3_best.pt \\
      --data_base /path/to/real_grn \\
      --refine_steps 0        # 0 = pure ML
      --refine_steps 15       # 15 = ML + Newton
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

from real_grn.stem_cell_params import (
    CONTEXTS, ALL_GENES, SCC1_GENES, SCC2_GENES, SCC3_GENES,
)


# ---------------------------------------------------------------------------
# Metric helpers (same as eval_gap_gene.py)
# ---------------------------------------------------------------------------
def _fp_hd(pred_fps, gt_fps):
    if len(gt_fps) == 0:
        return float("nan")
    if len(pred_fps) == 0:
        return 0.0
    hits = 0
    for gfp in gt_fps:
        dists = [
            np.max(np.abs(
                np.log10(np.clip(pfp, 1e-12, None)) -
                np.log10(np.clip(gfp, 1e-12, None))
            ))
            for pfp in pred_fps
        ]
        if min(dists) <= 0.5:
            hits += 1
    return hits / len(gt_fps)


def _stable_acc(pred_stab, gt_stab):
    STABLE = {0, 1}
    if len(gt_stab) == 0:
        return float("nan")
    gt_s = sorted([int(s in STABLE) for s in gt_stab])
    pr_s = sorted([int(s in STABLE) for s in pred_stab])
    n = max(len(gt_s), len(pr_s))
    gt_s += [0] * (n - len(gt_s))
    pr_s += [0] * (n - len(pr_s))
    return sum(a == b for a, b in zip(gt_s, pr_s)) / n


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------
def _load_l1(path, device):
    from model import SccFixedPointNet
    ck  = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m   = SccFixedPointNet(cfg)
    w   = ck.get("ema", ck.get("model"))
    if any(k.startswith("backbone.") for k in w):
        w = {k[len("backbone."):]: v for k, v in w.items() if k.startswith("backbone.")}
    m.load_state_dict(w)
    return m.to(device).eval(), cfg


def _load_l3(path, l1_model, device):
    from model import SccFixedPointNet
    from model_c import CorpusCNet
    ck  = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    backbone = SccFixedPointNet(cfg)
    m = CorpusCNet(backbone, cfg)
    m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def eval_stem_cell(data_base, l1_ckpt, l3_ckpt, refine_steps, device, verbose=True):
    from assemble import GRNPredictor

    n_genes  = len(ALL_GENES)
    data_dir = Path(data_base) / "stem_cell"

    if not data_dir.exists():
        print(f"  Dataset not found: {data_dir}")
        print(f"  Run gen_stem_cell_dataset.py first.")
        return None

    adj_all  = np.load(data_dir / "adj.npy")
    P_all    = np.load(data_dir / "P.npy")
    K_all    = np.load(data_dir / "K.npy")
    T_all    = np.load(data_dir / "T.npy")
    nh_all   = np.load(data_dir / "nh.npy")
    F_all    = np.load(data_dir / "F.npy")
    fp_ptr   = np.load(data_dir / "fp_ptr.npy")
    n_fp_arr = np.load(data_dir / "n_fp.npy")
    fp_state = np.load(data_dir / "fp_state.npy")
    fp_stab  = np.load(data_dir / "fp_stab.npy")

    l1_model, _ = _load_l1(l1_ckpt, device)
    l3_model, _ = _load_l3(l3_ckpt, l1_model, device)
    predictor   = GRNPredictor(l1_model, l3_model, device=device, refine_steps=refine_steps)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Stem cell: Pluripotency → Hematopoiesis → Lymphocyte  "
              f"(refine_steps={refine_steps})")
        print(f"  Genes: {ALL_GENES}")
        print(f"  SCC₁={SCC1_GENES}  SCC₂={SCC2_GENES}  SCC₃={SCC3_GENES}")
        print(f"  DAG depth = 3  (L1→L3→L3; OOD: trained on max depth 2)")
        print(f"{'='*70}")
        hdr = f"  {'Context':<12} {'GT':>4} {'Pred':>5} {'count_exact':>11} {'fp_hd':>7} {'stable_acc':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

    rows = []
    for i, ctx_name in enumerate(CONTEXTS.keys()):
        adj = adj_all[i]
        P   = P_all[i]
        K   = K_all[i]
        T   = T_all[i]
        nh  = nh_all[i]
        F   = F_all[i]

        fp0, fp1  = int(fp_ptr[i]), int(fp_ptr[i]) + int(n_fp_arr[i])
        gt_states = fp_state[fp0:fp1]
        gt_stabs  = fp_stab[fp0:fp1]
        n_fp_gt   = len(gt_states)

        result     = predictor.predict(adj, P, K, T, nh, F)
        pred_fps   = result.get("fps_refined", result.get("global_fps", np.zeros((0, n_genes))))
        pred_stabs = result.get("global_stab", np.zeros(0, dtype=np.uint8))
        n_fp_pred  = int(result.get("n_fps", 0))

        count_exact = int(n_fp_pred == n_fp_gt)
        hd          = _fp_hd(pred_fps, gt_states)
        sa          = _stable_acc(pred_stabs, gt_stabs)

        if verbose:
            print(f"  {ctx_name:<12} {n_fp_gt:>4} {n_fp_pred:>5} "
                  f"{count_exact:>11} {hd:>7.3f} {sa:>10.3f}")

        rows.append({
            "context": ctx_name,
            "n_fp_gt": n_fp_gt, "n_fp_pred": n_fp_pred,
            "count_exact": count_exact,
            "fp_hd": hd, "stable_acc": sa,
        })

        if verbose:
            nat = {g: P[j] / K[j] for j, g in enumerate(ALL_GENES)}
            for j, (pfp, ps) in enumerate(zip(pred_fps, pred_stabs)):
                on_pred = [ALL_GENES[k] for k in range(n_genes)
                           if pfp[k] > 0.5 * nat[ALL_GENES[k]]]
                print(f"    pred FP{j+1}: ON≈{on_pred}")
            if n_fp_gt > 0:
                print(f"    ground truth ({n_fp_gt} FPs):")
                for j, (gfp, gs) in enumerate(zip(gt_states, gt_stabs)):
                    on_gt = [ALL_GENES[k] for k in range(n_genes)
                             if gfp[k] > 0.5 * nat[ALL_GENES[k]]]
                    print(f"      GT FP{j+1}: ON≈{on_gt}")

    n = len(rows)
    if n > 0:
        ce = sum(r["count_exact"] for r in rows) / n
        hd_vals = [r["fp_hd"] for r in rows if not np.isnan(r["fp_hd"])]
        sa_vals = [r["stable_acc"] for r in rows if not np.isnan(r["stable_acc"])]
        hd = sum(hd_vals) / len(hd_vals) if hd_vals else float("nan")
        sa = sum(sa_vals) / len(sa_vals) if sa_vals else float("nan")
        if verbose:
            print(f"\n  Mean over {n} contexts: "
                  f"count_exact={ce:.3f}  fp_hd={hd:.3f}  stable_acc={sa:.3f}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_ckpt",      required=True)
    ap.add_argument("--l3_ckpt",      required=True)
    ap.add_argument("--data_base",    required=True)
    ap.add_argument("--refine_steps", type=int, default=0)
    ap.add_argument("--no_cuda",      action="store_true")
    ap.add_argument("--out_json",     default=None)
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    print(f"L1: {args.l1_ckpt}")
    print(f"L3: {args.l3_ckpt}")
    print(f"Refine steps: {args.refine_steps} "
          f"({'pure ML' if args.refine_steps == 0 else 'ML + Newton'})")

    rows = eval_stem_cell(args.data_base, args.l1_ckpt, args.l3_ckpt,
                          args.refine_steps, device)

    if args.out_json and rows:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()

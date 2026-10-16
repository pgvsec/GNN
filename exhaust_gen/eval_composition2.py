"""Evaluate GRNPredictor on the composition eval dataset produced by
gen_composition_eval.py.

Usage (from REPO_ROOT):
  PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp python exhaust_gen/eval_composition2.py \
      --l1_ckpt   /scratch/gilbreth/gupt1042/runs/exhaust_layer1/best.pt \
      --l3_ckpt   /scratch/gilbreth/gupt1042/runs/exhaust_layer3/best.pt \
      --eval_dir  /scratch/gilbreth/gupt1042/runs/composition_eval \
      --out_csv   /scratch/gilbreth/gupt1042/runs/eval_composition2.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device).eval()
    return model, cfg


def load_l3(path, l1_model, device):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = CorpusCNet(l1_model, cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device).eval()
    return model, cfg


def fp_within_half_decade(pred_fps, gt_fps):
    if len(gt_fps) == 0:
        return float("nan")
    if len(pred_fps) == 0:
        return 0.0
    hits = 0
    for gfp in gt_fps:
        dists = [np.max(np.abs(np.log10(np.clip(pfp, 1e-12, None))
                               - np.log10(np.clip(gfp,  1e-12, None))))
                 for pfp in pred_fps]
        if min(dists) <= 0.5:
            hits += 1
    return hits / len(gt_fps)


def stable_binary_acc(pred_stab, gt_stab):
    STABLE = {0, 1}
    if len(gt_stab) == 0:
        return float("nan")
    gt_s  = sorted([int(s in STABLE) for s in gt_stab])
    pr_s  = sorted([int(s in STABLE) for s in pred_stab])
    n = max(len(gt_s), len(pr_s))
    gt_s  += [0] * (n - len(gt_s))
    pr_s  += [0] * (n - len(pr_s))
    return sum(a == b for a, b in zip(gt_s, pr_s)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_ckpt",  required=True)
    ap.add_argument("--l3_ckpt",  required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_csv",  default="eval_composition2.csv")
    ap.add_argument("--no_cuda",  action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print("Loading models...")
    l1_model, _ = load_l1(args.l1_ckpt, device)
    l3_model, _ = load_l3(args.l3_ckpt, l1_model, device)

    from assemble import GRNPredictor
    predictor = GRNPredictor(l1_model, l3_model, device=device, refine_steps=5,
                             thorough_seeds=128, thorough_rounds=6)

    d = Path(args.eval_dir)
    adj_all   = np.load(d / "adj.npy")       # [N, 6, 6]
    P_all     = np.load(d / "P.npy")         # [N, 6]
    K_all     = np.load(d / "K.npy")
    T_all     = np.load(d / "T.npy")
    nh_all    = np.load(d / "nh.npy")
    F_all     = np.load(d / "F.npy")
    fp_ptr    = np.load(d / "fp_ptr.npy")    # [N]
    n_fp_arr  = np.load(d / "n_fp.npy")      # [N]
    fp_state  = np.load(d / "fp_state.npy")  # [total, 6]
    fp_stab   = np.load(d / "fp_stab.npy")   # [total]

    N_cases = len(adj_all)
    n_total_fp = len(fp_state)
    print(f"Eval set: {N_cases} cases, {n_total_fp} total GT FPs")

    rows = []
    n_total = n_count_exact = n_fp_hit = n_fp_total_w = 0
    errors = 0

    for i in range(N_cases):
        adj = adj_all[i]
        P   = P_all[i].astype(np.float32)
        K   = K_all[i].astype(np.float32)
        T   = T_all[i].astype(np.float32)
        nh  = nh_all[i].astype(np.float32)
        F   = F_all[i].astype(np.float32)

        fp0 = int(fp_ptr[i])
        fp1 = fp0 + int(n_fp_arr[i])
        gt_fps  = fp_state[fp0:fp1]
        gt_stab = fp_stab[fp0:fp1]
        n_fp_gt = int(n_fp_arr[i])

        try:
            with torch.no_grad():
                result = predictor.predict(adj, P, K, T, nh, F, verbose=False,
                                           thorough=True)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  case {i}: {exc}")
            continue

        # ML-only predictions
        pred_fps  = result.get("fps_refined", result["global_fps"])
        pred_stab = result["global_stab"]
        n_pred_ml = result["n_fps"]

        # Thorough solver predictions
        th_fps   = result.get("thorough_fps",  pred_fps)
        th_stab  = result.get("thorough_stab", pred_stab)
        n_pred_th = result.get("n_fps_thorough", n_pred_ml)

        ce_ml = int(n_pred_ml == n_fp_gt)
        hd_ml = fp_within_half_decade(pred_fps, gt_fps)
        sa_ml = stable_binary_acc(pred_stab, gt_stab)

        ce_th = int(n_pred_th == n_fp_gt)
        hd_th = fp_within_half_decade(th_fps, gt_fps)
        sa_th = stable_binary_acc(th_stab, gt_stab)

        rows.append({"case": i, "n_fp_gt": n_fp_gt,
                     "n_fp_pred_ml": n_pred_ml, "n_fp_pred_thorough": n_pred_th,
                     "count_exact_ml": ce_ml, "count_exact_thorough": ce_th,
                     "fp_hd_ml": hd_ml, "fp_hd_thorough": hd_th,
                     "stable_acc_ml": sa_ml, "stable_acc_thorough": sa_th,
                     "n_converged": result.get("n_converged", -1)})

        n_total += 1
        n_count_exact += ce_th   # report thorough as primary
        if not np.isnan(hd_th):
            n_fp_hit      += hd_th * n_fp_gt
            n_fp_total_w  += n_fp_gt

    print(f"\n{'='*60}")
    print(f"Evaluated {n_total} cases ({errors} errors)")
    if n_total > 0:
        print(f"count_exact (thorough): {n_count_exact/n_total:.4f}  ({n_count_exact}/{n_total})")
    if n_fp_total_w > 0:
        print(f"fp_hd (thorough)      : {n_fp_hit/n_fp_total_w:.4f}")
    print(f"{'='*60}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

"""Evaluate GRNPredictor (ML only) on the composition eval dataset produced by
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
    weights = ck.get("ema", ck.get("model"))
    # CorpusBNet/CorpusCNet checkpoints nest the backbone under "backbone.*";
    # strip that prefix so we can load it directly into SccFixedPointNet.
    if any(k.startswith("backbone.") for k in weights):
        weights = {k[len("backbone."):]: v
                   for k, v in weights.items() if k.startswith("backbone.")}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, cfg


def load_l3(path, l1_model, device):
    from model import SccFixedPointNet
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    # Use a fresh backbone — do NOT pass l1_model directly, because load_state_dict
    # updates the backbone in-place and would overwrite l1_model's trained count_head.
    backbone = SccFixedPointNet(cfg)
    model = CorpusCNet(backbone, cfg)
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
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--no_cuda",  action="store_true")
    ap.add_argument("--dedup", default="linear",
                    choices=["linear", "log", "none", "fill"],
                    help="Slot deduplication mode. 'fill' trusts the ML count head exactly "
                         "(disables rescue and parity_drop). Default: linear.")
    ap.add_argument("--no_rescue",      action="store_true",
                    help="Disable Newton-rescue of extra slots outside the count head's k.")
    ap.add_argument("--no_parity_drop", action="store_true",
                    help="Disable Poincaré-Hopf parity enforcement.")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Inference opts: dedup={args.dedup}  rescue={not args.no_rescue}  "
          f"parity_drop={not args.no_parity_drop}")

    print("Loading models...")
    l1_model, _ = load_l1(args.l1_ckpt, device)
    l3_model, _ = load_l3(args.l3_ckpt, l1_model, device)
    # L3's backbone count_head may be poorly trained (e.g. initialized from a
    # checkpoint with a different count_head architecture and then fine-tuned at
    # a tiny backbone_lr that gave it almost no gradient).  Copy L1's properly-
    # trained count_head into the L3 backbone so count prediction is reliable.
    # The inputs to StructuralCountHead (slot exist logits, graph emb) are the
    # same structure for both root and forced SCCs, so the transfer is valid.
    if args.l1_ckpt != args.l3_ckpt:
        with torch.no_grad():
            l3_model.backbone.count_head.load_state_dict(
                l1_model.count_head.state_dict())
            l3_model.backbone.stable_count_head.load_state_dict(
                l1_model.stable_count_head.state_dict())
        print("Patched L3 backbone count_head from L1.")

    from assemble import GRNPredictor, InferenceOptions
    opts = InferenceOptions(
        dedup=args.dedup,
        rescue=not args.no_rescue,
        parity_drop=not args.no_parity_drop,
    )
    predictor = GRNPredictor(l1_model, l3_model, device=device, refine_steps=5, opts=opts)

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
    if args.max_cases is not None:
        N_cases = min(N_cases, args.max_cases)
    n_total_fp = len(fp_state)
    print(f"Eval set: {N_cases} cases, {n_total_fp} total GT FPs")

    rows = []
    n_total = n_count_exact = n_fp_hit = n_fp_total_w = 0
    n_stab_hit = n_stab_total = 0
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
                result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  case {i}: {exc}")
            continue

        pred_fps  = result.get("fps_refined", result["global_fps"])
        pred_stab = result["global_stab"]
        n_pred    = result["n_fps"]

        ce = int(n_pred == n_fp_gt)
        hd = fp_within_half_decade(pred_fps, gt_fps)
        sa = stable_binary_acc(pred_stab, gt_stab)

        rows.append({"case": i, "n_fp_gt": n_fp_gt,
                     "n_fp_pred": n_pred,
                     "count_exact": ce,
                     "fp_hd": hd,
                     "stable_acc": sa,
                     "n_converged": result.get("n_converged", -1)})

        n_total += 1
        n_count_exact += ce
        if not np.isnan(hd):
            n_fp_hit     += hd * n_fp_gt
            n_fp_total_w += n_fp_gt
        if not np.isnan(sa):
            n_stab_hit   += sa
            n_stab_total += 1

    print(f"\n{'='*60}")
    print(f"Evaluated {n_total} cases ({errors} errors)")
    if n_total > 0:
        print(f"count_exact: {n_count_exact/n_total:.4f}  ({n_count_exact}/{n_total})")
    if n_fp_total_w > 0:
        print(f"fp_hd      : {n_fp_hit/n_fp_total_w:.4f}  (GT-FP-count weighted)")
    if n_stab_total > 0:
        print(f"stable_acc : {n_stab_hit/n_stab_total:.4f}  (mean per case)")
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

"""Per-branch count accuracy of a Layer I checkpoint against how far the branch's production
rates P * u fall outside the Corpus A range [0.01, 100] (teacher-forced: exact branch
labels and inputs, count from the count head, no Newton step).

  python diag_forcing_range.py --l1_ckpt L1 --eval_dir DIR
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from composition_eval import attach_labels, load_cases
from finetune_l1_forced import CountView
from model import SccFixedPointNet
from train_count_net import BranchCount, predict

P_LO, P_HI = 0.01, 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_ckpt", required=True, nargs="+")
    ap.add_argument("--eval_dir", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branches = BranchCount(attach_labels(load_cases(args.eval_dir)))
    out_dec = np.array([max(0.0, np.log10(it["P"][:it["n"]].max() / P_HI),
                            np.log10(P_LO / it["P"][:it["n"]].min())) for it in branches.items])
    bins = [(0, 1e-9, "inside"), (1e-9, 0.5, "0-0.5 dec out"), (0.5, 1, "0.5-1"),
            (1, 2, "1-2"), (2, np.inf, ">2")]
    for path in args.l1_ckpt:
        ck = torch.load(path, map_location=device, weights_only=False)
        net = SccFixedPointNet(SimpleNamespace(**ck["cfg"]))
        net.load_state_dict(ck.get("ema", ck.get("model")))
        pred, k, _ = predict(CountView(net.to(device)), branches, device)
        ok = (pred == k).numpy()
        print(f"\n{path}: {len(ok)} branches, accuracy {ok.mean():.4f}")
        for lo, hi, name in bins:
            m = (out_dec >= lo) & (out_dec < hi)
            if m.any():
                print(f"  {name:>14}: n={m.sum():5d}  acc {ok[m].mean():.4f}  "
                      f"(k>=3: {ok[m & (k.numpy() >= 3)].mean() if (m & (k.numpy() >= 3)).any() else float('nan'):.4f})")


if __name__ == "__main__":
    main()

"""Diagnostic: trace the assembly pipeline on a handful of composition cases."""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m = SccFixedPointNet(cfg); m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.to(device).eval()

def load_l3(path, l1, device):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m = CorpusCNet(l1, cfg); m.load_state_dict(ck.get("ema", ck.get("model")))
    return m.to(device).eval()

ap = argparse.ArgumentParser()
ap.add_argument("--l1_ckpt", required=True)
ap.add_argument("--l3_ckpt", required=True)
ap.add_argument("--eval_dir", required=True)
ap.add_argument("--n_cases", type=int, default=10)
args = ap.parse_args()

device = torch.device("cpu")
l1 = load_l1(args.l1_ckpt, device)
l3 = load_l3(args.l3_ckpt, l1, device)

d = Path(args.eval_dir)
adj_all  = np.load(d/"adj.npy")
P_all    = np.load(d/"P.npy")
K_all    = np.load(d/"K.npy")
T_all    = np.load(d/"T.npy")
nh_all   = np.load(d/"nh.npy")
F_all    = np.load(d/"F.npy")
n_fp_arr = np.load(d/"n_fp.npy")

from assemble import decompose_grn, run_inference, assemble_global_fps, GRNPredictor

predictor = GRNPredictor(l1, l3, device=device, refine_steps=5)
predictor.verbose_scc = True  # enable per-SCC logit prints

for i in range(args.n_cases):
    adj = adj_all[i]; P = P_all[i].astype(np.float32)
    K   = K_all[i].astype(np.float32); T = T_all[i].astype(np.float32)
    nh  = nh_all[i].astype(np.float32); F = F_all[i].astype(np.float32)
    n_fp_gt = int(n_fp_arr[i])

    dag = decompose_grn(adj, P, K, T, nh, F)
    scc_sizes = [s.n for s in dag.sccs]
    print(f"\n=== Case {i} | GT n_fp={n_fp_gt} | SCCs={len(dag.sccs)} sizes={scc_sizes} roots={dag.roots} ===")

    with torch.no_grad():
        result = predictor.predict(adj, P, K, T, nh, F, verbose=True)

    print(f"  => n_fp_pred={result['n_fps']}")

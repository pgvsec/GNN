"""Diagnostic: trace NaN source in composition pipeline."""
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
ap.add_argument("--corpus_a_cache", default=None, help="Corpus A cache dir for sanity check")
args = ap.parse_args()

device = torch.device("cpu")
l1 = load_l1(args.l1_ckpt, device)
l3 = load_l3(args.l3_ckpt, l1, device)

# --- print model parameter norms to verify loading ---
p_norms = [p.norm().item() for p in l1.parameters()]
print(f"L1 param norms (first 5): {p_norms[:5]}")
p_norms3 = [p.norm().item() for p in l3.parameters()]
print(f"L3 param norms (first 5): {p_norms3[:5]}")

d = Path(args.eval_dir)
adj_all  = np.load(d/"adj.npy")
P_all    = np.load(d/"P.npy")
K_all    = np.load(d/"K.npy")
T_all    = np.load(d/"T.npy")
nh_all   = np.load(d/"nh.npy")
F_all    = np.load(d/"F.npy")
n_fp_arr = np.load(d/"n_fp.npy")

print(f"\neval_dir shapes: adj={adj_all.shape} P={P_all.shape} T={T_all.shape}")
print(f"P range: [{P_all.min():.4f}, {P_all.max():.4f}]")
print(f"K range: [{K_all.min():.4f}, {K_all.max():.4f}]")
print(f"T range: [{T_all.min():.4f}, {T_all.max():.4f}]  (any zero? {(T_all==0).any()})")
print(f"nh range: [{nh_all.min():.4f}, {nh_all.max():.4f}]")
print(f"F range: [{F_all.min():.4f}, {F_all.max():.4f}]")
print(f"adj nonzero values: {np.unique(adj_all[adj_all != 0])}")

from assemble import decompose_grn, _make_scc_batch, _compute_forcing
import itertools

for i in range(3):
    adj = adj_all[i]; P = P_all[i].astype(np.float32)
    K   = K_all[i].astype(np.float32); T = T_all[i].astype(np.float32)
    nh  = nh_all[i].astype(np.float32); F = F_all[i].astype(np.float32)

    dag = decompose_grn(adj, P, K, T, nh, F)
    print(f"\n=== Case {i} | GT n_fp={int(n_fp_arr[i])} ===")

    for scc in dag.sccs:
        u = np.ones(scc.n, dtype=np.float32)
        if scc.parents:
            # Use first parent FP = zero (placeholder) — just to check batch NaNs
            u_contrib = _compute_forcing(dag, scc.parents[0], scc.scc_id,
                                         np.zeros(dag.sccs[scc.parents[0]].n),
                                         adj, T, nh, F)
            u *= u_contrib

        batch = _make_scc_batch(scc, u, device)

        print(f"  SCC {scc.scc_id} (n={scc.n}, parents={scc.parents})")
        print(f"    u = {u.tolist()}")
        for k, v in batch.items():
            if torch.is_tensor(v):
                has_nan = torch.isnan(v).any().item()
                has_inf = torch.isinf(v).any().item()
                print(f"    batch[{k}]: shape={list(v.shape)}  nan={has_nan}  inf={has_inf}  "
                      f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")

        with torch.no_grad():
            is_root = len(scc.parents) == 0
            if is_root:
                out = l1(batch)
            else:
                out, _ = l3(batch, freeze_backbone=True)

        for ok, ov in out.items():
            if torch.is_tensor(ov):
                has_nan = torch.isnan(ov).any().item()
                print(f"    out[{ok}]: nan={has_nan}  range=[{ov.min().item():.4f}, {ov.max().item():.4f}]")

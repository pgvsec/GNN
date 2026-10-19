"""Evaluate L1 (SccFixedPointNet) standalone count accuracy on Corpus A shards.

Reports count_exact broken down by true k, so we can see whether L1's
count head is already failing on k=3 in Corpus A (bad training) or only
in the composition context (distribution shift).

Usage:
  python eval_l1_counts.py \
    --ckpt  /path/to/layer1_v10/best.pt \
    --shards /path/to/corpus_exhaust_A_v2/shards \
    --max_shards 20
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_l1(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    from model import SccFixedPointNet
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device).eval()
    return model, cfg


def make_batch(d, idx, device):
    """Build a single-sample batch from a shard, sample index idx."""
    from topology import feedback_loops

    ep0 = int(d["s_ep_ptr"][idx])
    ep1 = int(d["s_ep_ptr"][idx + 1])

    n = int(d["g_n"][d["s_graph"][idx]])
    P     = torch.tensor(d["s_P"][idx, :n],    dtype=torch.float32, device=device)
    K     = torch.tensor(d["s_K"][idx, :n],    dtype=torch.float32, device=device)
    xmax  = torch.tensor(d["s_xmax"][idx, :n], dtype=torch.float32, device=device)

    # adjacency [n, n]
    g_idx  = int(d["s_graph"][idx])
    e0     = int(d["g_edge_ptr"][g_idx])
    e1     = int(d["g_edge_ptr"][g_idx + 1])
    src    = d["e_src"][e0:e1]
    tgt    = d["e_tgt"][e0:e1]
    sgn    = d["e_sgn"][e0:e1]
    adj_np = np.zeros((n, n), dtype=np.int8)
    for s, t, sg in zip(src, tgt, sgn):
        adj_np[t, s] = sg  # convention: adj[target, source]

    adj = torch.tensor(adj_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,n,n]

    # edge parameters: T, nh, F — one per edge-parameter slot
    T_ep  = torch.tensor(d["ep_T"][ep0:ep1],  dtype=torch.float32, device=device)
    nh_ep = torch.tensor(d["ep_nh"][ep0:ep1], dtype=torch.float32, device=device)
    F_ep  = torch.tensor(d["ep_F"][ep0:ep1],  dtype=torch.float32, device=device)

    # Build full [n, n] parameter matrices padded to N_MAX
    from data import N_MAX
    T_mat  = torch.ones(1, N_MAX, N_MAX, device=device)
    nh_mat = torch.ones(1, N_MAX, N_MAX, device=device) * 2.0
    F_mat  = torch.ones(1, N_MAX, N_MAX, device=device) * 2.0
    ep_count = min(len(T_ep), len(src))
    for ei in range(ep_count):
        s, t = int(src[ei]), int(tgt[ei])
        T_mat[0, t, s]  = T_ep[ei]
        nh_mat[0, t, s] = nh_ep[ei]
        F_mat[0, t, s]  = F_ep[ei]

    node_mask = torch.zeros(1, N_MAX, dtype=torch.bool, device=device)
    node_mask[0, :n] = True

    # Pad P, K, xmax to N_MAX
    P_pad    = torch.zeros(1, N_MAX, device=device); P_pad[0, :n]    = P
    K_pad    = torch.zeros(1, N_MAX, device=device); K_pad[0, :n]    = K
    xmax_pad = torch.zeros(1, N_MAX, device=device); xmax_pad[0, :n] = xmax

    adj_pad  = torch.zeros(1, N_MAX, N_MAX, device=device)
    adj_pad[0, :n, :n] = adj[0]

    # has_pos_fb
    try:
        fb = feedback_loops(adj_np)
        has_pos = bool(fb[0]) if fb is not None else True
    except Exception:
        has_pos = True
    has_pos_fb = torch.tensor([has_pos], dtype=torch.bool, device=device)

    # u = ones (autonomous)
    u = torch.ones(1, N_MAX, device=device)

    return {
        "P":         P_pad,
        "K":         K_pad,
        "xmax":      xmax_pad,
        "adj":       adj_pad,
        "T":         T_mat,
        "nh":        nh_mat,
        "F":         F_mat,
        "u":         u,
        "n":         torch.tensor([n], device=device),
        "node_mask": node_mask,
        "has_pos_fb": has_pos_fb,
    }


def predict_k(model, batch):
    """Return predicted k (odd only)."""
    with torch.no_grad():
        out = model(batch)
    logit = out["count_logit"][0]  # [max_fp+1]
    even = torch.arange(logit.shape[0], device=logit.device) % 2 == 0
    logit = logit.clone(); logit[even] = float("-inf")
    return int(logit.argmax().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--shards",     required=True)
    ap.add_argument("--max_shards", type=int, default=20)
    ap.add_argument("--max_per_shard", type=int, default=500)
    ap.add_argument("--no_cuda",    action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")
    print("Loading model...")
    model, cfg = load_l1(args.ckpt, device)

    shard_dir = Path(args.shards)
    shards = sorted(shard_dir.glob("*.npz"))[:args.max_shards]
    print(f"Evaluating on {len(shards)} shards (max {args.max_per_shard} samples each)...")

    # correct[k] = list of 0/1
    results = defaultdict(list)
    total = 0
    errors = 0

    for shard_path in shards:
        d = np.load(shard_path)
        n_samples = len(d["s_n_fp"])
        n_use = min(n_samples, args.max_per_shard)

        for idx in range(n_use):
            k_true = int(d["s_n_fp"][idx])
            try:
                batch = make_batch(d, idx, device)
                k_pred = predict_k(model, batch)
                results[k_true].append(int(k_pred == k_true))
                total += 1
            except Exception as e:
                errors += 1

    print(f"\nEvaluated {total} samples ({errors} errors skipped)")
    print(f"\n{'k_true':>7}  {'n':>6}  {'count_exact':>12}  {'pred_k_dist'}")
    print("-" * 70)

    overall_correct = 0
    overall_total = 0
    for k in sorted(results):
        vals = results[k]
        acc = sum(vals) / len(vals)
        overall_correct += sum(vals)
        overall_total += len(vals)
        print(f"  k={k:3d}   {len(vals):6d}  {acc:12.4f}")

    print(f"\n  Overall: {overall_correct}/{overall_total} = "
          f"{overall_correct/overall_total:.4f}")


if __name__ == "__main__":
    main()

"""Fine-tune L1 count head on Corpus A shards with k_gt-proportional loss.

The composition eval bottleneck is L1 under-predicting k for multistable
SCCs (k=3,5,7...).  L1 was trained with flat CE loss where k=1 (56% of
data) dominated.  This script re-trains only l1.count_head and
l1.stable_count_head on Corpus A shards with loss weight = k_gt,
giving multistable SCCs proportionally more gradient signal.

All backbone parameters stay frozen.

Usage:
  python finetune_l1_on_corpusA.py \\
    --ckpt   /path/to/layer1_v10/best.pt \\
    --shards /path/to/corpus_exhaust_A_v2/shards \\
    --out_dir /path/to/l1_count_ft_v1 \\
    --epochs 10 --lr 3e-4
"""
import argparse
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_l1(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    from model import SccFixedPointNet
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device)
    return model, cfg, ck["cfg"]


def freeze_backbone(model):
    """Freeze everything except count_head and stable_count_head."""
    for name, p in model.named_parameters():
        if "count_head" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_shards(shard_dir: Path, max_shards=None):
    shards = sorted(shard_dir.glob("*.npz"))
    if max_shards:
        shards = shards[:max_shards]
    return shards


def make_batch(d, idx, device, N_MAX):
    """Build a single-sample L1 batch from a shard at index idx."""
    from topology import feedback_loops

    g_idx = int(d["s_graph"][idx])
    n     = int(d["g_n"][g_idx])

    ep0 = int(d["s_ep_ptr"][idx])
    ep1 = int(d["s_ep_ptr"][idx + 1])
    e0  = int(d["g_edge_ptr"][g_idx])
    e1  = int(d["g_edge_ptr"][g_idx + 1])

    src = d["e_src"][e0:e1]; tgt = d["e_tgt"][e0:e1]; sgn = d["e_sgn"][e0:e1]

    adj_np = np.zeros((n, n), dtype=np.float32)
    for s, t, sg in zip(src, tgt, sgn):
        adj_np[t, s] = float(sg)

    T_mat  = torch.ones(1, N_MAX, N_MAX, device=device)
    nh_mat = torch.ones(1, N_MAX, N_MAX, device=device) * 2.0
    F_mat  = torch.ones(1, N_MAX, N_MAX, device=device) * 2.0
    ep_count = min(ep1 - ep0, len(src))
    for ei in range(ep_count):
        s, t = int(src[ei]), int(tgt[ei])
        if s < N_MAX and t < N_MAX:
            T_mat[0, t, s]  = float(d["ep_T"][ep0 + ei])
            nh_mat[0, t, s] = float(d["ep_nh"][ep0 + ei])
            F_mat[0, t, s]  = float(d["ep_F"][ep0 + ei])

    P    = torch.tensor(d["s_P"][idx, :n],    dtype=torch.float32, device=device)
    K    = torch.tensor(d["s_K"][idx, :n],    dtype=torch.float32, device=device)
    xmax = torch.tensor(d["s_xmax"][idx, :n], dtype=torch.float32, device=device)

    node_mask = torch.zeros(1, N_MAX, dtype=torch.bool, device=device)
    node_mask[0, :n] = True

    P_p = torch.zeros(1, N_MAX, device=device);    P_p[0, :n]    = P
    K_p = torch.zeros(1, N_MAX, device=device);    K_p[0, :n]    = K
    xmax_p = torch.zeros(1, N_MAX, device=device); xmax_p[0, :n] = xmax
    adj_p = torch.zeros(1, N_MAX, N_MAX, device=device)
    adj_p[0, :n, :n] = torch.tensor(adj_np, device=device)

    try:
        fb = feedback_loops(adj_np)
        has_pos = bool(fb[0]) if fb is not None else True
    except Exception:
        has_pos = True
    has_pos_fb = torch.tensor([has_pos], dtype=torch.bool, device=device)

    u = torch.ones(1, N_MAX, device=device)

    return {
        "P": P_p, "K": K_p, "xmax": xmax_p, "adj": adj_p,
        "T": T_mat, "nh": nh_mat, "F": F_mat, "u": u,
        "n": torch.tensor([n], device=device),
        "node_mask": node_mask,
        "has_pos_fb": has_pos_fb,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def odd_ce_loss(logit, k_gt, weight=1.0):
    """Weighted CE over odd k values only."""
    max_count = logit.shape[-1] - 1
    even = torch.arange(logit.shape[-1], device=logit.device) % 2 == 0
    l = logit.clone(); l[even] = float("-inf")
    k_c = min(int(k_gt), max_count)
    if k_c % 2 == 0:
        k_c = min(k_c + 1, max_count)
    tgt = torch.tensor([k_c], dtype=torch.long, device=logit.device)
    return weight * F.cross_entropy(l.unsqueeze(0), tgt)


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(shards, model, device, N_MAX, optimizer=None,
              max_per_shard=None, base_weight=3.0):
    is_train = optimizer is not None
    total_loss = 0.0; n_correct = 0; n_total = 0; n_err = 0

    shard_list = list(shards)
    if is_train:
        random.shuffle(shard_list)

    for shard_path in shard_list:
        d = np.load(shard_path)
        n_samples = len(d["s_n_fp"])
        indices = list(range(n_samples))
        if is_train:
            random.shuffle(indices)
        if max_per_shard:
            indices = indices[:max_per_shard]

        for idx in indices:
            k_gt = int(d["s_n_fp"][idx])
            if k_gt == 0:
                continue
            try:
                batch = make_batch(d, idx, device, N_MAX)
            except Exception:
                n_err += 1; continue

            try:
                if is_train:
                    out = model(batch)
                else:
                    with torch.no_grad():
                        out = model(batch)

                count_logit = out["count_logit"][0]
                # weight = base_weight * k_gt / 3  (k=1→1×, k=3→3×, k=5→5×, ...)
                w = base_weight * max(1, k_gt) / 3.0
                loss = odd_ce_loss(count_logit, k_gt, weight=w)

                if is_train and loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()

                total_loss += float(loss.item()) / w  # log unweighted loss

                # Accuracy
                even = torch.arange(count_logit.shape[0]) % 2 == 0
                cl = count_logit.detach().clone(); cl[even] = float("-inf")
                k_pred = int(cl.argmax().item())
                n_correct += int(k_pred == k_gt)
                n_total += 1
            except Exception:
                n_err += 1

    return {
        "loss": total_loss / max(n_total, 1),
        "count_exact": n_correct / max(n_total, 1),
        "n": n_total,
        "errors": n_err,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          required=True)
    ap.add_argument("--shards",        required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--epochs",        type=int,   default=10)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--wd",            type=float, default=1e-4)
    ap.add_argument("--train_shards",  type=int,   default=None,
                    help="Number of shards to use for training (rest for val)")
    ap.add_argument("--max_per_shard", type=int,   default=None)
    ap.add_argument("--base_weight",   type=float, default=3.0,
                    help="Loss weight scale: w = base_weight * k_gt / 3")
    ap.add_argument("--no_cuda",       action="store_true")
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")

    print("Loading model...")
    model, cfg, cfg_dict = load_l1(args.ckpt, device)
    freeze_backbone(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    from data import N_MAX
    print(f"N_MAX = {N_MAX}")

    shard_dir = Path(args.shards)
    all_shards = sorted(shard_dir.glob("*.npz"))
    n_train = args.train_shards or max(1, int(len(all_shards) * 0.9))
    train_shards = all_shards[:n_train]
    val_shards   = all_shards[n_train:] or all_shards[-5:]
    print(f"Shards: {len(train_shards)} train / {len(val_shards)} val")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    best_val_ce = -1.0; log_rows = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train CE':>9}  "
          f"{'Val Loss':>9}  {'Val CE':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = run_epoch(train_shards, model, device, N_MAX,
                       optimizer=optimizer,
                       max_per_shard=args.max_per_shard,
                       base_weight=args.base_weight)

        model.eval()
        va = run_epoch(val_shards, model, device, N_MAX,
                       max_per_shard=args.max_per_shard,
                       base_weight=args.base_weight)

        scheduler.step()
        print(f"{epoch:>6}  {tr['loss']:>10.4f}  {tr['count_exact']:>9.4f}  "
              f"{va['loss']:>9.4f}  {va['count_exact']:>8.4f}")

        log_rows.append({"epoch": epoch, "train_loss": tr["loss"],
                         "train_ce": tr["count_exact"],
                         "val_loss": va["loss"], "val_ce": va["count_exact"]})
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader(); w.writerows(log_rows)

        if va["count_exact"] > best_val_ce:
            best_val_ce = va["count_exact"]
            torch.save({"cfg": cfg_dict,
                        "model": model.state_dict(),
                        "ema":   model.state_dict()},
                       out_dir / "l1_best.pt")
            print(f"         ↑ new best val CE = {best_val_ce:.4f}  (saved)")

    print(f"\nBest val count_exact: {best_val_ce:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

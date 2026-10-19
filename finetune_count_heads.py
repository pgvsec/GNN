"""End-to-end fine-tuning of count heads on composition-level data.

Only three small MLPs are updated:
  - l1_model.count_head          (StructuralCountHead — root SCC standalone)
  - l1_model.stable_count_head   (StructuralCountHead — root SCC stable count)
  - l3_model.forcing_count_head  (ForcingAwareCountHead — forced SCC)

All backbone parameters remain frozen.  Gradients flow from the per-SCC
count cross-entropy loss back through only these heads.

Per-SCC GT counts are extracted directly from the composition GT fixed points:
  - Root SCCs: count unique FP patterns in that SCC's gene columns.
  - Forced SCCs: for each unique parent-FP pattern, count unique child patterns;
    target is the modal child count across all parent branches.

This addresses the cascade under-prediction: L1 was trained on standalone
SCCs and never learned that multistable root SCCs need k=3 in composition
contexts.  End-to-end gradient signal teaches it to predict higher k.

Usage (from REPO_ROOT):
  python finetune_count_heads.py \\
    --l1_ckpt  /path/to/layer1_v10/best.pt \\
    --l3_ckpt  /path/to/layer3_v12/best.pt \\
    --train_dirs /path/to/composition_eval_v3 /path/to/composition_eval_v4 \\
    --val_dir    /path/to/composition_eval_v5 \\
    --out_dir    /path/to/count_ft_v1 \\
    --epochs 30 --lr 3e-4 --batch_size 64
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device)
    return model, cfg, ck["cfg"]


def load_l3(path, l1_model, device):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = CorpusCNet(l1_model, cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")), strict=False)
    model.to(device)
    return model, cfg, ck["cfg"]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_count_heads(l1_model, l3_model):
    """Unfreeze only the three count head MLPs."""
    for p in l1_model.count_head.parameters():
        p.requires_grad_(True)
    for p in l1_model.stable_count_head.parameters():
        p.requires_grad_(True)
    for p in l3_model.forcing_count_head.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

def load_corpus(dirpath: Path):
    d = dirpath
    return {
        "adj":      np.load(d / "adj.npy"),
        "P":        np.load(d / "P.npy"),
        "K":        np.load(d / "K.npy"),
        "T":        np.load(d / "T.npy"),
        "nh":       np.load(d / "nh.npy"),
        "F":        np.load(d / "F.npy"),
        "fp_ptr":   np.load(d / "fp_ptr.npy"),
        "n_fp":     np.load(d / "n_fp.npy"),
        "fp_state": np.load(d / "fp_state.npy"),
        "fp_stab":  np.load(d / "fp_stab.npy"),
    }


def gather_cases(corpus_dirs):
    """Merge multiple corpus directories into a flat list of case dicts."""
    cases = []
    for d in corpus_dirs:
        corp = load_corpus(Path(d))
        N = len(corp["adj"])
        for i in range(N):
            fp0 = int(corp["fp_ptr"][i])
            fp1 = fp0 + int(corp["n_fp"][i])
            cases.append({
                "adj":      corp["adj"][i],
                "P":        corp["P"][i].astype(np.float32),
                "K":        corp["K"][i].astype(np.float32),
                "T":        corp["T"][i].astype(np.float32),
                "nh":       corp["nh"][i].astype(np.float32),
                "F":        corp["F"][i].astype(np.float32),
                "fp_state": corp["fp_state"][fp0:fp1].astype(np.float32),
                "fp_stab":  corp["fp_stab"][fp0:fp1],
                "n_fp_gt":  int(corp["n_fp"][i]),
            })
    return cases


# ---------------------------------------------------------------------------
# 3. Per-SCC GT count extraction
# ---------------------------------------------------------------------------

def _unique_rows(arr: np.ndarray, rtol: float = 0.02) -> int:
    """Count unique rows in arr [n_fp, n_genes] using relative tolerance."""
    if arr.shape[0] == 0:
        return 0
    rounded = np.round(arr / (np.abs(arr).max(axis=0, keepdims=True) + 1e-30) / rtol) * rtol
    return len(np.unique(rounded, axis=0))


def extract_scc_gt_counts(dag, fp_state: np.ndarray):
    """
    For each SCC, return the GT FP count as seen from composition GT data.

    Root SCCs:
      k = number of unique patterns in fp_state[:, scc.genes]

    Forced SCCs:
      Group GT FPs by which root-SCC pattern they belong to, then count
      unique child patterns per group.  Target = modal count across groups.

    Returns dict: scc_id -> (gt_k: int, gt_fps_per_group: list[np.ndarray])
    where gt_fps_per_group is a list of x_local arrays for each distinct
    forcing context (used later to build GT forcing batches for L3).
    """
    result = {}

    # Process in topological order (roots first)
    for scc in dag.sccs:
        genes = scc.genes
        scc_vals = fp_state[:, genes]   # [n_fp_gt, n_scc_genes]

        if len(scc.parents) == 0:
            # Root: all GT FPs seen from same (autonomous) context
            k = _unique_rows(scc_vals)
            # Cluster GT FPs into k groups
            fps = _cluster_unique(scc_vals)
            result[scc.scc_id] = (k, [fp for fp in fps])
        else:
            # Forced: group by parent FP combination
            # Collect parent gene indices
            parent_genes_list = []
            for pid in scc.parents:
                parent_genes_list.extend(dag.sccs[pid].genes)
            parent_genes = sorted(set(parent_genes_list))

            if len(parent_genes) == 0 or fp_state.shape[0] == 0:
                result[scc.scc_id] = (1, [scc_vals[0]] if fp_state.shape[0] > 0 else [])
                continue

            # Cluster GT FPs by parent pattern
            parent_vals = fp_state[:, parent_genes]  # [n_fp, n_parent_genes]
            groups = _group_by_pattern(parent_vals, scc_vals)

            # Count unique child patterns per group
            child_counts = [_unique_rows(g) for g in groups if len(g) > 0]
            if not child_counts:
                result[scc.scc_id] = (1, [])
                continue
            # Modal count (most common) as the training target
            from collections import Counter
            modal_k = Counter(child_counts).most_common(1)[0][0]
            # Collect one representative GT FP per group (first unique)
            rep_fps = []
            for g in groups:
                if len(g) > 0:
                    rep_fps.append(_cluster_unique(g)[0])
            result[scc.scc_id] = (modal_k, rep_fps)

    return result


def _cluster_unique(arr: np.ndarray, rtol: float = 0.02) -> list:
    """Return list of representative rows, one per unique cluster."""
    if arr.shape[0] == 0:
        return []
    scale = np.abs(arr).max(axis=0, keepdims=True) + 1e-30
    rounded = np.round(arr / scale / rtol) * rtol
    seen = set()
    reps = []
    for i, row in enumerate(rounded):
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            reps.append(arr[i])
    return reps


def _group_by_pattern(parent_vals: np.ndarray, child_vals: np.ndarray,
                       rtol: float = 0.05) -> list:
    """Group child_vals rows by their corresponding parent_vals pattern."""
    scale = np.abs(parent_vals).max(axis=0, keepdims=True) + 1e-30
    rounded = np.round(parent_vals / scale / rtol) * rtol
    groups: dict = {}
    for i, row in enumerate(rounded):
        key = tuple(row.tolist())
        if key not in groups:
            groups[key] = []
        groups[key].append(child_vals[i])
    return [np.stack(v) for v in groups.values()]


# ---------------------------------------------------------------------------
# 4. Batch construction helpers
# ---------------------------------------------------------------------------

def _make_l1_batch(scc, device):
    """Build a single-sample batch for L1 (autonomous, u=ones), with has_pos_fb."""
    from topology import feedback_loops
    from assemble import _make_scc_batch
    u = np.ones(scc.n, dtype=np.float32)
    batch = _make_scc_batch(scc, u, device)
    try:
        fb = feedback_loops(scc.adj_intra)
        has_pos = bool(fb[0]) if fb is not None else True
    except Exception:
        has_pos = True
    batch["has_pos_fb"] = torch.tensor([has_pos], dtype=torch.bool, device=device)
    return batch


def _make_l3_batch(scc, parent_x_local: np.ndarray, parent_scc_id: int,
                   dag, global_T: np.ndarray, global_nh: np.ndarray,
                   global_F: np.ndarray, global_adj: np.ndarray, device):
    """Build a single-sample batch for L3 with GT forcing from a parent FP."""
    from assemble import _make_scc_batch, _compute_forcing
    from topology import feedback_loops
    u = _compute_forcing(dag, parent_scc_id, scc.scc_id,
                         parent_x_local, global_adj, global_T, global_nh, global_F)
    batch = _make_scc_batch(scc, u, device)
    # tc for forced SCCs (needed by ForcingAwareCountHead)
    try:
        fb = feedback_loops(scc.adj_intra)
        if fb is None:
            tc = 3
        elif fb[0] and fb[1]:
            tc = 3
        elif fb[0]:
            tc = 2
        else:
            tc = 1
    except Exception:
        tc = 3
    batch["tc"] = torch.tensor([tc], dtype=torch.long, device=device)
    batch["has_pos_fb"] = torch.tensor([fb[0] if fb is not None else True],
                                        dtype=torch.bool, device=device)
    return batch


# ---------------------------------------------------------------------------
# 5. Per-case loss computation
# ---------------------------------------------------------------------------

def _odd_ce_loss(count_logit: torch.Tensor, k_gt: int, max_count: int) -> torch.Tensor:
    """Cross-entropy over odd k values only. k_gt must be a positive integer."""
    # Mask even indices to -inf so softmax ignores them
    even = torch.arange(count_logit.shape[-1], device=count_logit.device) % 2 == 0
    logit = count_logit.clone()
    logit[..., even] = float("-inf")
    # Clamp k_gt to valid range
    k_clamped = min(k_gt, max_count)
    if k_clamped % 2 == 0:
        k_clamped = min(k_clamped + 1, max_count)
    target = torch.tensor([k_clamped], dtype=torch.long, device=count_logit.device)
    return F.cross_entropy(logit.unsqueeze(0) if logit.dim() == 1 else logit, target)


def compute_case_loss(case, dag, l1_model, l3_model, device, scc_gt_counts,
                      multistable_weight: float = 1.0):
    """
    Compute count head loss for one composition case.

    multistable_weight: multiplier on loss for SCCs where k_gt >= 3 (the
    primary failure mode — L1 predicts k=1 for multistable SCCs).

    Returns scalar loss tensor and a dict of per-SCC predictions for metrics.
    """
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0
    pred_counts = {}

    for scc in dag.sccs:
        scc_id = scc.scc_id
        if scc_id not in scc_gt_counts:
            continue
        gt_k, gt_fps = scc_gt_counts[scc_id]

        if len(scc.parents) == 0:
            # Root SCC: use L1
            batch = _make_l1_batch(scc, device)
            out = l1_model(batch)
            count_logit = out["count_logit"][0]   # [max_fp+1]
            max_count = count_logit.shape[0] - 1
            loss = _odd_ce_loss(count_logit, gt_k, max_count)
            # Up-weight multistable SCCs — these are the primary failure mode
            w = multistable_weight if gt_k >= 3 else 1.0
            total_loss = total_loss + w * loss
            n_terms += 1
            # Predicted k for metrics
            even = torch.arange(count_logit.shape[0]) % 2 == 0
            cl = count_logit.detach().clone()
            cl[even] = float("-inf")
            pred_counts[scc_id] = int(cl.argmax().item())
        else:
            # Forced SCC: use L3 with GT forcing from each parent branch
            # Use first parent in topological order
            pid = scc.parents[0]
            parent_fps = scc_gt_counts.get(pid, (1, []))
            _, parent_gt_fps = parent_fps

            if not parent_gt_fps:
                continue

            for parent_x in parent_gt_fps:
                batch = _make_l3_batch(scc, parent_x, pid, dag,
                                       case["T"], case["nh"], case["F"], case["adj"],
                                       device)
                out_cond, _ = l3_model(batch, freeze_backbone=True)
                count_logit = out_cond["count_logit"][0]
                max_count = count_logit.shape[0] - 1
                loss = _odd_ce_loss(count_logit, gt_k, max_count)
                total_loss = total_loss + loss
                n_terms += 1

            # Metrics: use first parent FP for prediction
            if parent_gt_fps:
                batch = _make_l3_batch(scc, parent_gt_fps[0], pid, dag,
                                       case["T"], case["nh"], case["F"], case["adj"],
                                       device)
                with torch.no_grad():
                    out_cond, _ = l3_model(batch, freeze_backbone=True)
                count_logit = out_cond["count_logit"][0].detach()
                even = torch.arange(count_logit.shape[0]) % 2 == 0
                cl = count_logit.clone()
                cl[even] = float("-inf")
                pred_counts[scc_id] = int(cl.argmax().item())

    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=False), pred_counts
    return total_loss / n_terms, pred_counts


# ---------------------------------------------------------------------------
# 6. Epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(cases, dag_cache, l1_model, l3_model, device,
              optimizer=None, max_cases=None, multistable_weight=1.0):
    """Run one training or validation epoch. Returns metrics dict."""
    is_train = optimizer is not None

    total_loss = 0.0
    n_exact = 0
    n_total = 0
    errors = 0

    indices = list(range(len(cases)))
    if is_train:
        random.shuffle(indices)
    if max_cases is not None:
        indices = indices[:max_cases]

    for i in indices:
        case = cases[i]
        dag = dag_cache[i]
        if dag is None:
            continue

        # Extract per-SCC GT counts
        try:
            scc_gt_counts = extract_scc_gt_counts(dag, case["fp_state"])
        except Exception as e:
            errors += 1
            continue

        try:
            if is_train:
                loss, pred_counts = compute_case_loss(
                    case, dag, l1_model, l3_model, device, scc_gt_counts,
                    multistable_weight=multistable_weight)
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in list(l1_model.parameters()) +
                         list(l3_model.forcing_count_head.parameters())
                         if p.requires_grad], 1.0)
                    optimizer.step()
                total_loss += float(loss.item())
            else:
                with torch.no_grad():
                    loss, pred_counts = compute_case_loss(
                        case, dag, l1_model, l3_model, device, scc_gt_counts)
                total_loss += float(loss.item())
        except Exception as e:
            errors += 1
            continue

        # Composition count_exact: predicted global count = product of SCC counts
        pred_global = 1
        for scc in dag.sccs:
            pred_global *= pred_counts.get(scc.scc_id, 1)
        n_exact += int(pred_global == case["n_fp_gt"])
        n_total += 1

    return {
        "loss": total_loss / max(n_total, 1),
        "count_exact": n_exact / max(n_total, 1),
        "n": n_total,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 7. DAG pre-computation (cache DAGs to avoid repeated decomposition)
# ---------------------------------------------------------------------------

def precompute_dags(cases):
    from assemble import decompose_grn
    dags = []
    for c in cases:
        try:
            dag = decompose_grn(c["adj"], c["P"], c["K"], c["T"], c["nh"], c["F"])
            dags.append(dag)
        except Exception:
            dags.append(None)
    return dags


# ---------------------------------------------------------------------------
# 8. Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, tag: str, l1_model, l3_model,
                    l1_cfg_dict, l3_cfg_dict, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save L1 count heads
    torch.save({
        "cfg": l1_cfg_dict,
        "model": l1_model.state_dict(),
    }, out_dir / f"l1_{tag}.pt")
    # Save L3 forcing_count_head only
    torch.save({
        "cfg": l3_cfg_dict,
        "forcing_count_head": l3_model.forcing_count_head.state_dict(),
        "metrics": metrics,
    }, out_dir / f"l3_count_head_{tag}.pt")


def save_best_full(out_dir: Path, l1_model, l3_model, l1_cfg_dict, l3_cfg_dict):
    """Save full model state dicts for best checkpoint (to use in eval)."""
    torch.save({"cfg": l1_cfg_dict, "model": l1_model.state_dict(),
                "ema": l1_model.state_dict()},
               out_dir / "l1_best.pt")
    torch.save({"cfg": l3_cfg_dict, "model": l3_model.state_dict(),
                "ema": l3_model.state_dict()},
               out_dir / "l3_best.pt")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1_ckpt",    required=True)
    ap.add_argument("--l3_ckpt",    required=True)
    ap.add_argument("--train_dirs", required=True, nargs="+")
    ap.add_argument("--val_dir",    required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--wd",         type=float, default=1e-4)
    ap.add_argument("--max_train",  type=int,   default=None,
                    help="Cap training cases per epoch (for quick tests)")
    ap.add_argument("--multistable_weight", type=float, default=3.0,
                    help="Loss multiplier for SCCs with k_gt>=3 (multistable). "
                         "Default 3.0 — these are 62%% of errors.")
    ap.add_argument("--no_cuda",    action="store_true")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    l1_model, l1_cfg, l1_cfg_dict = load_l1(args.l1_ckpt, device)
    l3_model, l3_cfg, l3_cfg_dict = load_l3(args.l3_ckpt, l1_model, device)

    # Freeze everything, then unfreeze only count heads
    freeze_all(l1_model)
    freeze_all(l3_model)
    unfreeze_count_heads(l1_model, l3_model)

    trainable = sum(p.numel() for p in list(l1_model.parameters()) +
                    list(l3_model.parameters()) if p.requires_grad)
    total     = sum(p.numel() for p in list(l1_model.parameters()) +
                    list(l3_model.parameters()))
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load data
    print("Loading data...")
    train_cases = gather_cases(args.train_dirs)
    val_cases   = gather_cases([args.val_dir])
    print(f"  Train: {len(train_cases)} cases | Val: {len(val_cases)} cases")

    print("Pre-computing SCC DAGs...")
    train_dags = precompute_dags(train_cases)
    val_dags   = precompute_dags(val_cases)

    # Optimizer: only trainable parameters
    trainable_params = ([p for p in l1_model.parameters() if p.requires_grad] +
                        [p for p in l3_model.forcing_count_head.parameters()])
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.out_dir)
    log_path = out_dir / "log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_ce = -1.0
    log_rows = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train CE':>9}  "
          f"{'Val Loss':>9}  {'Val CE':>8}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        l1_model.train()
        l3_model.forcing_count_head.train()
        train_m = run_epoch(train_cases, train_dags, l1_model, l3_model, device,
                            optimizer=optimizer, max_cases=args.max_train,
                            multistable_weight=args.multistable_weight)

        l1_model.eval()
        l3_model.forcing_count_head.eval()
        val_m = run_epoch(val_cases, val_dags, l1_model, l3_model, device)

        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
        scheduler.step()

        print(f"{epoch:>6}  {train_m['loss']:>10.4f}  "
              f"{train_m['count_exact']:>9.4f}  "
              f"{val_m['loss']:>9.4f}  "
              f"{val_m['count_exact']:>8.4f}  "
              f"{lr_now:>8.2e}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_ce": train_m["count_exact"],
            "val_loss": val_m["loss"],
            "val_ce": val_m["count_exact"],
            "lr": lr_now,
        })
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        if val_m["count_exact"] > best_val_ce:
            best_val_ce = val_m["count_exact"]
            save_best_full(out_dir, l1_model, l3_model, l1_cfg_dict, l3_cfg_dict)
            print(f"         ↑ new best val count_exact = {best_val_ce:.4f}  (saved)")

        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(out_dir, f"ep{epoch:03d}", l1_model, l3_model,
                            l1_cfg_dict, l3_cfg_dict, val_m)

    print(f"\nBest val count_exact: {best_val_ce:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()

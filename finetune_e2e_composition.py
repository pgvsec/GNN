"""End-to-end composition-level fine-tuning of L1 + L3.

SUPERVISION CHANGE (vs finetune_count_heads.py):
  OLD: CE(logit_scc, k_gt_scc) independently per SCC
       → L1 never sees gradient from composition outcomes
  NEW: CE(P(N_FP), n_fp_gt) at composition level
       → L1's count_head now gets gradient from COMPOSITION errors

HOW IT WORKS:
  1. For each SCC, compute P(k) = softmax over odd logit indices (differentiable).
  2. Compose all per-SCC P(k) into P(N_FP) via multiplicative convolution:
       P(N_FP = k1*k2) = ΣΣ P(k_root=k1) P(k_child=k2)   [2-SCC]
     implemented with scatter_add — fully differentiable through both L1 and L3.
  3. Minimize CE(P(N_FP), n_fp_gt) for the whole composition.

GRADIENT FLOW:
  ∂CE/∂logit_root  → L1.count_head
  ∂CE/∂logit_child → L3.forcing_count_head
  (backbones frozen by default; --lr_backbone > 0 to unfreeze with reduced LR)

FORCING:
  Two modes, controlled by --p_predicted (default 0.5):
    p_predicted=0.0 : GT parent FPs always (original "Option A" — teacher forcing)
    p_predicted=1.0 : predicted parent FPs always (closes train-test gap)
    0 < p < 1       : scheduled sampling — use GT with prob (1-p), predicted with prob p

  Predicted FPs are obtained by decoding L1's slot positions (no Newton refinement).
  This directly addresses the train-test discrepancy: at inference the forcing signal
  is built from predicted (not GT) parent FPs, so training with p_predicted > 0
  makes the child SCC robust to approximate forcing signals.

OPTIONAL REGULARISATION (--scc_reg_weight > 0):
  Adds a per-SCC CE term alongside the composition CE.  This prevents the
  composition loss alone from finding degenerate solutions that get N_FP right
  by luck while being wrong on individual SCCs.

Usage (from REPO_ROOT):
  python finetune_e2e_composition.py \\
    --l1_ckpt  checkpoints/layer1_v10/best.pt \\
    --l3_ckpt  checkpoints/layer3_v12/best.pt \\
    --train_dirs data/composition_eval_v3 data/composition_eval_v4 \\
    --val_dir    data/composition_eval_v5 \\
    --out_dir    checkpoints/e2e_v1 \\
    --epochs 40 --lr 1e-4 --scc_reg_weight 0.3
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
# 1. Model loading  (identical to finetune_count_heads.py)
# ---------------------------------------------------------------------------

def _remap_count_head_keys(sd: dict) -> dict:
    """
    Remap old-format count head keys (count_head.net.*) to the current
    format (count_head.mlp.net.*).  No-op if keys are already current.
    """
    mapping = {}
    for k in list(sd.keys()):
        for head in ("count_head", "stable_count_head", "forcing_count_head"):
            prefix = head + ".net."
            if k.startswith(prefix):
                new_k = head + ".mlp." + k[len(head) + 1:]
                mapping[k] = new_k
                break
    if not mapping:
        return sd
    new_sd = {mapping.get(k, k): v for k, v in sd.items()}
    return new_sd


def load_l1(path, device):
    from model import SccFixedPointNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    sd = _remap_count_head_keys(ck.get("ema", ck.get("model")))
    # Drop count head keys that don't match current architecture shape so that
    # strict=False can load the backbone cleanly; heads are trained from scratch.
    head_prefixes = ("count_head.", "stable_count_head.")
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if not any(k.startswith(p) for p in head_prefixes)
                or (k in model_sd and model_sd[k].shape == v.shape)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    non_head_missing = [k for k in missing
                        if not any(k.startswith(p) for p in head_prefixes)]
    if non_head_missing:
        raise RuntimeError(f"Missing backbone keys in L1 checkpoint: {non_head_missing}")
    model.to(device)
    return model, cfg, ck["cfg"]


def load_l3(path, l1_model, device, use_v2_count_head: bool = False):
    from model_c import CorpusCNet
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ck["cfg"])
    if use_v2_count_head:
        cfg_dict["use_v2_count_head"] = True
    cfg = SimpleNamespace(**cfg_dict)
    model = CorpusCNet(l1_model, cfg)
    sd = _remap_count_head_keys(ck.get("ema", ck.get("model")))
    # Drop forcing_count_head weights that don't match current architecture shape.
    # strict=False skips missing/unexpected keys but still errors on shape mismatches,
    # so we filter before loading (same pattern as load_l1 for count_head).
    model_sd = model.state_dict()
    sd_filtered = {k: v for k, v in sd.items()
                   if k not in model_sd or model_sd[k].shape == v.shape}
    model.load_state_dict(sd_filtered, strict=False)
    model.to(device)
    return model, cfg, cfg_dict


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _make_trainable_groups(l1_model, l3_model, lr_head, lr_backbone,
                           lr_pos: float = 0.0, lr_decoder: float = 0.0):
    """
    Return optimizer param groups.

    lr_head=0 + lr_backbone=0 + lr_pos>0 → pos_only mode:
      only l1_model.pos_head is trainable; count heads and backbone stay frozen.
      This keeps exist_logit unchanged so count head inputs never shift.

    lr_decoder>0 → enable SlotDecoder + exist_head training.
      Requires ForcingAwareCountHeadV2 (cfg.use_v2_count_head=True) so that
      decoder changes don't break count head calibration.
    """
    groups = []
    seen   = set()

    if lr_head > 0.0:
        head_params = (
            list(l1_model.count_head.parameters()) +
            list(l1_model.stable_count_head.parameters()) +
            list(l3_model.forcing_count_head.parameters())
        )
        for p in head_params:
            p.requires_grad_(True)
            seen.add(id(p))
        groups.append({"params": head_params, "lr": lr_head})

    if lr_backbone > 0.0:
        backbone_l1 = [p for n, p in l1_model.named_parameters()
                       if not any(n.startswith(h) for h in
                                  ("count_head", "stable_count_head"))]
        backbone_l3 = [p for n, p in l3_model.named_parameters()
                       if not n.startswith("forcing_count_head")]
        # Deduplicate by tensor id: CorpusCNet shares the L1 backbone by reference,
        # so backbone_l3 contains the same parameter objects as backbone_l1.
        backbone_params = []
        for p in backbone_l1 + backbone_l3:
            if id(p) not in seen:
                seen.add(id(p))
                backbone_params.append(p)
        for p in backbone_params:
            p.requires_grad_(True)
        groups.append({"params": backbone_params, "lr": lr_backbone})

    if lr_pos > 0.0:
        # Only L1's pos_head: decoder/exist_head/backbone stay frozen so
        # exist_logit values (count_head inputs) never change.
        pos_params = [p for p in l1_model.pos_head.parameters()
                      if id(p) not in seen]
        for p in pos_params:
            p.requires_grad_(True)
            seen.add(id(p))
        groups.append({"params": pos_params, "lr": lr_pos})

    if lr_decoder > 0.0:
        # SlotDecoder + exist_head from L1 backbone (shared with L3 backbone).
        # Safe only with ForcingAwareCountHeadV2 (no exist_logit dependency).
        dec_names = ("decoder.", "exist_head.")
        dec_params = [p for n, p in l1_model.named_parameters()
                      if any(n.startswith(d) for d in dec_names)
                      and id(p) not in seen]
        for p in dec_params:
            p.requires_grad_(True)
            seen.add(id(p))
        if dec_params:
            groups.append({"params": dec_params, "lr": lr_decoder})

    if not groups:
        raise ValueError(
            "No trainable params: lr, lr_backbone, lr_pos, and lr_decoder are all 0")

    return groups


# ---------------------------------------------------------------------------
# 2. Dataset loading  (identical to finetune_count_heads.py)
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
# 3. Per-SCC GT count extraction  (identical to finetune_count_heads.py)
# ---------------------------------------------------------------------------

def _unique_rows(arr: np.ndarray, rtol: float = 0.02) -> int:
    if arr.shape[0] == 0:
        return 0
    rounded = np.round(arr / (np.abs(arr).max(axis=0, keepdims=True) + 1e-30) / rtol) * rtol
    return len(np.unique(rounded, axis=0))


def _cluster_unique(arr: np.ndarray, rtol: float = 0.02) -> list:
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
    scale = np.abs(parent_vals).max(axis=0, keepdims=True) + 1e-30
    rounded = np.round(parent_vals / scale / rtol) * rtol
    groups: dict = {}
    for i, row in enumerate(rounded):
        key = tuple(row.tolist())
        if key not in groups:
            groups[key] = []
        groups[key].append(child_vals[i])
    return [np.stack(v) for v in groups.values()]


def extract_scc_gt_counts(dag, fp_state: np.ndarray):
    """
    Returns dict: scc_id -> (gt_k: int, gt_fps: list[np.ndarray])
    where gt_fps is a list of representative FP arrays for each distinct forcing context.
    """
    result = {}
    from collections import Counter

    for scc in dag.sccs:
        genes = scc.genes
        scc_vals = fp_state[:, genes]

        if len(scc.parents) == 0:
            k = _unique_rows(scc_vals)
            fps = _cluster_unique(scc_vals)
            result[scc.scc_id] = (k, fps)
        else:
            parent_genes_list = []
            for pid in scc.parents:
                parent_genes_list.extend(dag.sccs[pid].genes)
            parent_genes = sorted(set(parent_genes_list))

            if len(parent_genes) == 0 or fp_state.shape[0] == 0:
                result[scc.scc_id] = (1, [scc_vals[0]] if fp_state.shape[0] > 0 else [])
                continue

            parent_vals = fp_state[:, parent_genes]
            groups = _group_by_pattern(parent_vals, scc_vals)
            child_counts = [_unique_rows(g) for g in groups if len(g) > 0]
            if not child_counts:
                result[scc.scc_id] = (1, [])
                continue
            modal_k = Counter(child_counts).most_common(1)[0][0]
            rep_fps = [_cluster_unique(g)[0] for g in groups if len(g) > 0]
            result[scc.scc_id] = (modal_k, rep_fps)

    return result


# ---------------------------------------------------------------------------
# 4. Batch construction helpers  (identical to finetune_count_heads.py)
# ---------------------------------------------------------------------------

def _make_l1_batch(scc, device):
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
                   dag, global_T, global_nh, global_F, global_adj, device):
    from assemble import _make_scc_batch, _compute_forcing
    from topology import feedback_loops
    u = _compute_forcing(dag, parent_scc_id, scc.scc_id,
                         parent_x_local, global_adj, global_T, global_nh, global_F)
    batch = _make_scc_batch(scc, u, device)
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
        fb = None
    batch["tc"] = torch.tensor([tc], dtype=torch.long, device=device)
    batch["has_pos_fb"] = torch.tensor(
        [fb[0] if fb is not None else True], dtype=torch.bool, device=device)
    return batch


def _cat_batches(batches: list) -> dict:
    """Stack a list of single-sample batch dicts into one batched dict along dim 0."""
    out = {}
    for k in batches[0]:
        v = batches[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.cat([b[k] for b in batches], dim=0)
        else:
            out[k] = v  # scalar / non-tensor metadata: take first sample's value
    return out


def _pos_chamfer_loss(out: dict, batch: dict, gt_fps: list, scc_n: int):
    """
    Log-space Chamfer loss: for each GT FP, the nearest predicted slot must be close.

    For each GT fixed point, finds the minimum L∞ distance (max over genes, in
    log10 space) to any predicted slot.  Averaging those minima gives a loss that
    encourages at least one slot to land near every GT FP — direct gradient to the
    pos decoder without requiring any count-specific structure.

    Only called for root SCCs (L1 forward, backbone not frozen) so gradients reach
    the pos decoder.

    Returns a scalar tensor, or None if gt_fps is empty or an error occurs.
    """
    if not gt_fps:
        return None
    try:
        from model import decode_position
        pos   = out["pos"].float()        # [1, K, N_MAX]
        xmax  = batch["xmax"].float()     # [1, N_MAX]
        x_pred = decode_position(pos, xmax.unsqueeze(1))  # [1, K, N_MAX]
        x_pred_scc = x_pred[0, :, :scc_n]                # [K, scc_n]

        gt_arr = np.stack(gt_fps)         # [n_gt, scc_n]
        gt_t   = torch.tensor(gt_arr, dtype=torch.float32, device=pos.device)  # [n_gt, scc_n]

        # L∞ in log10 space: for each (gt_i, slot_j), max over genes of |log10(pred_j) - log10(gt_i)|
        x_log  = x_pred_scc.clamp_min(1e-12).log() / math.log(10)     # [K, scc_n]  log10
        gt_log = gt_t.clamp_min(1e-12).log()        / math.log(10)     # [n_gt, scc_n]

        diff           = (x_log.unsqueeze(0) - gt_log.unsqueeze(1)).abs()  # [n_gt, K, scc_n]
        max_gene_dist  = diff.max(dim=-1).values                           # [n_gt, K]
        min_slot_dist  = max_gene_dist.min(dim=-1).values                  # [n_gt]

        return min_slot_dist.mean()
    except Exception:
        return None


def _predicted_fps_from_l1(out: dict, batch: dict, scc_n: int) -> list:
    """Decode L1's predicted FP positions for the parent SCC.

    Selects the top-k slots (k from count head argmax over odd indices) by
    existence logit and returns their decoded coordinates as numpy arrays.
    No Newton refinement — raw slot-decoder positions are sufficient to give
    the child SCC a realistic (imperfect) forcing signal for robustness training.

    Returns list of np.ndarray [scc_n], one per predicted FP, or [] on any failure.
    """
    try:
        from model import decode_position
        count_logit = out["count_logit"][0].float()   # [max_count+1]
        even = torch.arange(count_logit.shape[-1],
                            device=count_logit.device) % 2 == 0
        k_hat = int(count_logit.masked_fill(even, float("-inf")).argmax().item())
        if k_hat == 0:
            return []

        exist_logit = out["exist_logit"][0].float()   # [K]
        top_idx = exist_logit.argsort(descending=True)[:k_hat]

        # pos [1, K, N_MAX], xmax [1, N_MAX]
        xmax = batch["xmax"]                          # [1, N_MAX]
        pos  = out["pos"]                             # [1, K, N_MAX]
        x    = decode_position(pos, xmax.unsqueeze(1))  # [1, K, N_MAX]

        fps = []
        for j in top_idx:
            coords = x[0, int(j.item()), :scc_n].detach().cpu().numpy().astype(np.float32)
            fps.append(coords)
        return fps
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 5. Composition CE loss (NEW)
# ---------------------------------------------------------------------------

def _odd_k_vals(max_k: int, device) -> torch.Tensor:
    """Return integer tensor [1, 3, 5, ..., max_k_odd]."""
    return torch.arange(1, max_k + 1, step=2, dtype=torch.long, device=device)


def odd_probs(count_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute P(k) over odd-valued k from a count_logit vector.

    Returns:
      probs:  [n_odd_k] probability tensor (differentiable)
      k_vals: [n_odd_k] integer tensor of the corresponding k values
    """
    max_k = count_logit.shape[-1] - 1
    odd_idx = torch.arange(1, max_k + 1, step=2, device=count_logit.device)
    logits_odd = count_logit[..., odd_idx]  # select odd positions
    probs = F.softmax(logits_odd, dim=-1)   # [n_odd_k]
    return probs, odd_idx.long()


def multiplicative_compose(prob_list: list[torch.Tensor],
                            k_vals_list: list[torch.Tensor],
                            max_n: int = 128) -> torch.Tensor:
    """
    Compute P(N_FP = n) via iterative multiplicative convolution.

    N_FP = k_1 * k_2 * ... * k_M  (product formula — each SCC multiplies through)

    Algorithm: start with P(N=1) = 1.  For each SCC with distribution P(k):
      P_new(N * k) += P_old(N) * P(k)   for all current N, all k in k_vals
    implemented with broadcast outer product + scatter_add (fully differentiable).

    Args:
      prob_list:  list of [n_odd_k_i] probability tensors, one per SCC
      k_vals_list: list of [n_odd_k_i] integer tensors of k values
      max_n: maximum N_FP to track (probability mass above max_n is dropped)

    Returns:
      dist: [max_n+1] probability tensor, where dist[n] = P(N_FP = n)
    """
    device = prob_list[0].device
    dtype  = prob_list[0].dtype

    dist = torch.zeros(max_n + 1, device=device, dtype=dtype)
    dist[1] = 1.0  # identity: P(N=1) = 1 before any SCC processed

    for probs, k_vals in zip(prob_list, k_vals_list):
        n_range = torch.arange(max_n + 1, device=device, dtype=torch.long)  # [max_n+1]

        # Outer product: shape [max_n+1, n_k]
        # outer_n[i, j] = n_range[i] * k_vals[j]  (integer products)
        outer_n    = n_range.unsqueeze(1) * k_vals.unsqueeze(0)   # [max_n+1, n_k]
        outer_prob = dist.unsqueeze(1)    * probs.unsqueeze(0)    # [max_n+1, n_k]

        # Clamp indices; contributions beyond max_n are lost (negligible for max_n=128)
        idx = outer_n.clamp(max=max_n).reshape(-1).long()
        val = outer_prob.reshape(-1)

        new_dist = torch.zeros(max_n + 1, device=device, dtype=dtype)
        new_dist.scatter_add_(0, idx, val)
        dist = new_dist

    return dist


def composition_ce_loss(n_fp_dist: torch.Tensor, n_fp_gt: int,
                         max_n: int = 128) -> torch.Tensor:
    """
    Cross-entropy between predicted N_FP distribution and ground truth N_FP.

    n_fp_dist: [max_n+1] probability vector (output of multiplicative_compose)
    n_fp_gt:   integer ground-truth total fixed-point count
    """
    # Log-probabilities (avoid log(0))
    log_probs = torch.log(n_fp_dist + 1e-12)  # [max_n+1]
    target = min(n_fp_gt, max_n)
    return -log_probs[target]


def _odd_ce_loss(count_logit: torch.Tensor, k_gt: int) -> torch.Tensor:
    """Per-SCC CE over odd k only (used for optional regularisation term)."""
    max_count = count_logit.shape[-1] - 1
    even = torch.arange(count_logit.shape[-1], device=count_logit.device) % 2 == 0
    logit = count_logit.clone()
    logit[..., even] = float("-inf")
    k_clamped = min(k_gt, max_count)
    if k_clamped % 2 == 0:
        k_clamped = min(k_clamped + 1, max_count)
    target = torch.tensor([k_clamped], dtype=torch.long, device=count_logit.device)
    return F.cross_entropy(logit.unsqueeze(0) if logit.dim() == 1 else logit, target)


# ---------------------------------------------------------------------------
# 6. Per-case end-to-end loss
# ---------------------------------------------------------------------------

def compute_e2e_loss(case, dag, l1_model, l3_model, device, scc_gt_counts,
                     scc_reg_weight: float = 0.0,
                     max_n: int = 128,
                     p_predicted: float = 0.0,
                     pos_reg_weight: float = 0.0,
                     freeze_l3_backbone: bool = True) -> tuple[torch.Tensor, dict]:
    """
    Compute end-to-end composition CE loss for one case.

    Returns (loss, pred_counts) where pred_counts maps scc_id -> predicted k.
    """
    prob_list   = []   # per-SCC P(k) distributions
    k_vals_list = []   # corresponding k value tensors
    scc_reg     = torch.tensor(0.0, device=device)
    n_reg_terms = 0
    pos_loss    = torch.tensor(0.0, device=device)
    n_pos_terms = 0
    pred_counts = {}
    pred_fps    = {}   # scc_id -> list[np.ndarray] of predicted FP coords

    for scc in dag.sccs:
        scc_id = scc.scc_id
        if scc_id not in scc_gt_counts:
            continue
        gt_k, gt_fps = scc_gt_counts[scc_id]

        if len(scc.parents) == 0:
            # Root SCC → L1
            batch = _make_l1_batch(scc, device)
            out = l1_model(batch)
            count_logit = out["count_logit"][0]   # [max_count+1]

            probs, k_vals = odd_probs(count_logit)
            prob_list.append(probs)
            k_vals_list.append(k_vals)

            if scc_reg_weight > 0.0:
                w = max(1, gt_k) / 3.0
                scc_reg = scc_reg + w * _odd_ce_loss(count_logit, gt_k)
                n_reg_terms += 1

            if pos_reg_weight > 0.0 and gt_fps:
                pl = _pos_chamfer_loss(out, batch, gt_fps, scc.n)
                if pl is not None:
                    pos_loss = pos_loss + pl
                    n_pos_terms += 1

            with torch.no_grad():
                pred_counts[scc_id] = int(k_vals[probs.argmax()].item())
                # Store predicted FP positions for child SCC forcing
                if p_predicted > 0.0:
                    pred_fps[scc_id] = _predicted_fps_from_l1(out, batch, scc.n)

        else:
            # Forced SCC → L3
            pid = scc.parents[0]
            _, parent_gt_fps = scc_gt_counts.get(pid, (1, []))
            if not parent_gt_fps:
                # Fallback: assume k=1 with certainty
                k_max  = 9
                k_vals = _odd_k_vals(k_max, device)
                probs  = torch.zeros(len(k_vals), device=device, dtype=torch.float32)
                probs[0] = 1.0   # P(k=1) = 1
                prob_list.append(probs)
                k_vals_list.append(k_vals)
                pred_counts[scc_id] = 1
                continue

            # Scheduled sampling: use predicted parent FPs with prob p_predicted.
            # Predicted FPs close the train-test gap (inference uses predicted, not GT).
            # Fall back to GT if no prediction was stored (topological order issue).
            if p_predicted > 0.0 and random.random() < p_predicted:
                parent_branch_fps = pred_fps.get(pid)
                if not parent_branch_fps:
                    parent_branch_fps = parent_gt_fps
            else:
                parent_branch_fps = parent_gt_fps

            branch_probs_list = []
            for parent_x in parent_branch_fps:
                batch = _make_l3_batch(scc, parent_x, pid, dag,
                                       case["T"], case["nh"], case["F"],
                                       case["adj"], device)
                out_cond, _ = l3_model(batch, freeze_backbone=freeze_l3_backbone)
                count_logit = out_cond["count_logit"][0]
                probs_branch, k_vals = odd_probs(count_logit)
                branch_probs_list.append(probs_branch)

                if scc_reg_weight > 0.0:
                    scc_reg = scc_reg + _odd_ce_loss(count_logit, gt_k)
                    n_reg_terms += 1

                # L3 Chamfer: direct position gradient to forced-SCC decoder.
                # Only useful when backbone is unfrozen (gradients can flow).
                if pos_reg_weight > 0.0 and gt_fps and not freeze_l3_backbone:
                    pl = _pos_chamfer_loss(out_cond, batch, gt_fps, scc.n)
                    if pl is not None:
                        pos_loss = pos_loss + pl
                        n_pos_terms += 1

            # Average across branches (marginalise over parent FPs under uniform weight)
            probs_avg = torch.stack(branch_probs_list).mean(dim=0)  # [n_odd_k]
            prob_list.append(probs_avg)
            k_vals_list.append(k_vals)

            with torch.no_grad():
                pred_counts[scc_id] = int(k_vals[probs_avg.argmax()].item())

    if not prob_list:
        return torch.tensor(0.0, device=device, requires_grad=False), pred_counts

    # Compose all per-SCC distributions → P(N_FP)
    n_fp_dist = multiplicative_compose(prob_list, k_vals_list, max_n=max_n)

    # Composition CE: supervision at the composition level
    comp_loss = composition_ce_loss(n_fp_dist, case["n_fp_gt"], max_n=max_n)

    # Optional per-SCC regularisation
    if scc_reg_weight > 0.0 and n_reg_terms > 0:
        comp_loss = comp_loss + scc_reg_weight * (scc_reg / n_reg_terms)

    # Position supervision: Chamfer loss on root-SCC slot positions vs GT FPs
    if pos_reg_weight > 0.0 and n_pos_terms > 0:
        comp_loss = comp_loss + pos_reg_weight * (pos_loss / n_pos_terms)

    return comp_loss, pred_counts


# ---------------------------------------------------------------------------
# 7. DAG pre-computation
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


def precompute_gt_counts(cases, dag_cache):
    """Pre-extract scc_gt_counts for every case so run_epoch avoids per-step numpy work."""
    result = []
    for case, dag in zip(cases, dag_cache):
        if dag is None:
            result.append(None)
            continue
        try:
            result.append(extract_scc_gt_counts(dag, case["fp_state"]))
        except Exception:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# 8. Epoch helpers
# ---------------------------------------------------------------------------

def _compute_chunk_loss(chunk_data, l1_model, l3_model, device,
                        scc_reg_weight, max_n, p_predicted, pos_reg_weight,
                        freeze_l3_backbone):
    """
    Batch L1 then L3 forward passes for a chunk of (case, dag, scc_gt_counts) triples.

    Instead of N sequential batch_size=1 GPU calls, issues at most 2 kernel launches
    (one L1 pass for all root SCCs in the chunk, one L3 pass for all forced-SCC
    branches).  This eliminates the kernel-launch overhead that causes an A100 and an
    A10 to run at identical speed when the per-step batch size is 1.

    Returns: (loss, pred_counts_list)
      loss              — mean composition CE across valid cases (scalar tensor)
      pred_counts_list  — list[dict], one per chunk entry, scc_id → predicted k
    """
    B = len(chunk_data)

    # ── Phase 1: collect all root-SCC (L1) inputs ────────────────────────────
    l1_entries = []   # (chunk_idx, scc_id, scc, gt_k, gt_fps)
    l1_raw     = []

    for ci, (case, dag, scc_gt_counts) in enumerate(chunk_data):
        for scc in dag.sccs:
            if len(scc.parents) == 0 and scc.scc_id in scc_gt_counts:
                gt_k, gt_fps = scc_gt_counts[scc.scc_id]
                l1_raw.append(_make_l1_batch(scc, device))
                l1_entries.append((ci, scc.scc_id, scc, gt_k, gt_fps))

    if not l1_raw:
        dummy = torch.tensor(0.0, device=device, requires_grad=False)
        return dummy, [{}] * B

    batched_l1_out = l1_model(_cat_batches(l1_raw))

    # ── Phase 2: extract per-case L1 results ─────────────────────────────────
    per_case = [{
        'prob_list':   [],
        'k_vals_list': [],
        'scc_reg':     torch.tensor(0.0, device=device),
        'n_reg':       0,
        'pos_loss':    torch.tensor(0.0, device=device),
        'n_pos':       0,
        'pred_fps':    {},
        'pred_counts': {},
    } for _ in range(B)]

    for li, (ci, scc_id, scc, gt_k, gt_fps) in enumerate(l1_entries):
        count_logit = batched_l1_out["count_logit"][li]
        probs, k_vals = odd_probs(count_logit)
        pc = per_case[ci]
        pc['prob_list'].append(probs)
        pc['k_vals_list'].append(k_vals)

        if scc_reg_weight > 0.0:
            w = max(1, gt_k) / 3.0
            pc['scc_reg'] = pc['scc_reg'] + w * _odd_ce_loss(count_logit, gt_k)
            pc['n_reg'] += 1

        need_out_i = (pos_reg_weight > 0.0 and bool(gt_fps)) or (p_predicted > 0.0)
        if need_out_i:
            out_i = {k: v[li:li+1] for k, v in batched_l1_out.items()
                     if isinstance(v, torch.Tensor)}
            raw_i = l1_raw[li]

            if pos_reg_weight > 0.0 and gt_fps:
                pl = _pos_chamfer_loss(out_i, raw_i, gt_fps, scc.n)
                if pl is not None:
                    pc['pos_loss'] = pc['pos_loss'] + pl
                    pc['n_pos'] += 1

            with torch.no_grad():
                pc['pred_counts'][scc_id] = int(k_vals[probs.argmax()].item())
                if p_predicted > 0.0:
                    pc['pred_fps'][scc_id] = _predicted_fps_from_l1(out_i, raw_i, scc.n)
        else:
            with torch.no_grad():
                pc['pred_counts'][scc_id] = int(k_vals[probs.argmax()].item())

    # ── Phase 3: collect all forced-SCC (L3) inputs ──────────────────────────
    l3_entries = []   # (chunk_idx, scc_id, scc, gt_k, gt_fps)
    l3_raw     = []

    for ci, (case, dag, scc_gt_counts) in enumerate(chunk_data):
        pc = per_case[ci]
        for scc in dag.sccs:
            scc_id = scc.scc_id
            if len(scc.parents) == 0 or scc_id not in scc_gt_counts:
                continue
            pid = scc.parents[0]
            gt_k, gt_fps = scc_gt_counts[scc_id]
            _, par_gt_fps = scc_gt_counts.get(pid, (1, []))

            if not par_gt_fps:
                k_max  = 9
                k_vals = _odd_k_vals(k_max, device)
                pr     = torch.zeros(len(k_vals), device=device, dtype=torch.float32)
                pr[0]  = 1.0
                pc['prob_list'].append(pr)
                pc['k_vals_list'].append(k_vals)
                pc['pred_counts'][scc_id] = 1
                continue

            if p_predicted > 0.0 and random.random() < p_predicted:
                branch_fps = pc['pred_fps'].get(pid) or par_gt_fps
            else:
                branch_fps = par_gt_fps

            for px in branch_fps:
                batch = _make_l3_batch(scc, px, pid, dag,
                                       case["T"], case["nh"], case["F"],
                                       case["adj"], device)
                l3_raw.append(batch)
                l3_entries.append((ci, scc_id, scc, gt_k, gt_fps))

    if l3_raw:
        batched_l3_out, _ = l3_model(_cat_batches(l3_raw),
                                      freeze_backbone=freeze_l3_backbone)

        l3_accum = {}   # (ci, scc_id) -> {'probs_list', 'k_vals', 'gt_k'}

        for li, (ci, scc_id, scc, gt_k, gt_fps) in enumerate(l3_entries):
            count_logit  = batched_l3_out["count_logit"][li]
            probs_branch, k_vals = odd_probs(count_logit)
            pc = per_case[ci]

            if scc_reg_weight > 0.0:
                pc['scc_reg'] = pc['scc_reg'] + _odd_ce_loss(count_logit, gt_k)
                pc['n_reg'] += 1

            if pos_reg_weight > 0.0 and gt_fps and not freeze_l3_backbone:
                out_i = {k: v[li:li+1] for k, v in batched_l3_out.items()
                         if isinstance(v, torch.Tensor)}
                pl = _pos_chamfer_loss(out_i, l3_raw[li], gt_fps, scc.n)
                if pl is not None:
                    pc['pos_loss'] = pc['pos_loss'] + pl
                    pc['n_pos'] += 1

            key = (ci, scc_id)
            if key not in l3_accum:
                l3_accum[key] = {'probs_list': [], 'k_vals': k_vals,
                                  'gt_k': gt_k, 'scc': scc}
            l3_accum[key]['probs_list'].append(probs_branch)

        for (ci, scc_id), acc in l3_accum.items():
            pc = per_case[ci]
            probs_avg = torch.stack(acc['probs_list']).mean(dim=0)
            pc['prob_list'].append(probs_avg)
            pc['k_vals_list'].append(acc['k_vals'])
            with torch.no_grad():
                pc['pred_counts'][scc_id] = int(acc['k_vals'][probs_avg.argmax()].item())

    # ── Phase 4: composition CE per case ─────────────────────────────────────
    total_loss       = torch.tensor(0.0, device=device)
    n_valid          = 0
    pred_counts_list = []

    for ci, (case, dag, scc_gt_counts) in enumerate(chunk_data):
        pc = per_case[ci]
        if not pc['prob_list']:
            pred_counts_list.append({})
            continue

        n_fp_dist = multiplicative_compose(pc['prob_list'], pc['k_vals_list'], max_n=max_n)
        comp_loss = composition_ce_loss(n_fp_dist, case["n_fp_gt"], max_n=max_n)

        if scc_reg_weight > 0.0 and pc['n_reg'] > 0:
            comp_loss = comp_loss + scc_reg_weight * pc['scc_reg'] / pc['n_reg']
        if pos_reg_weight > 0.0 and pc['n_pos'] > 0:
            comp_loss = comp_loss + pos_reg_weight * pc['pos_loss'] / pc['n_pos']

        total_loss = total_loss + comp_loss
        n_valid   += 1
        pred_counts_list.append(pc['pred_counts'])

    loss = total_loss / max(n_valid, 1)
    return loss, pred_counts_list


def run_epoch(cases, dag_cache, l1_model, l3_model, device,
              optimizer=None, max_cases=None,
              scc_reg_weight=0.0, max_n=128, p_predicted=0.0,
              pos_reg_weight=0.0,
              freeze_l3_backbone=True,
              gt_counts_cache=None,
              accumulate=32,
              scaler=None):
    is_train = optimizer is not None

    total_loss = 0.0
    n_exact    = 0
    n_total    = 0
    errors     = 0

    # Pre-build trainable param list once (not per step)
    if is_train:
        all_params = [p for p in
                      list(l1_model.parameters()) + list(l3_model.parameters())
                      if p.requires_grad]

    use_amp = (scaler is not None) and (device.type == "cuda")

    indices = list(range(len(cases)))
    if is_train:
        random.shuffle(indices)
    if max_cases is not None:
        indices = indices[:max_cases]

    # Pre-filter valid cases; pay extraction cost once outside the hot loop
    valid_items = []
    for i in indices:
        case = cases[i]
        dag  = dag_cache[i]
        if dag is None:
            continue
        if gt_counts_cache is not None:
            scc_gt_counts = gt_counts_cache[i]
        else:
            try:
                scc_gt_counts = extract_scc_gt_counts(dag, case["fp_state"])
            except Exception:
                errors += 1
                continue
        if scc_gt_counts is None:
            errors += 1
            continue
        valid_items.append((case, dag, scc_gt_counts))

    # Chunk size: accumulate cases per optimizer step (training) or 64 (val)
    chunk_size = accumulate if is_train else 64

    if is_train:
        optimizer.zero_grad()

    for chunk_start in range(0, len(valid_items), chunk_size):
        chunk = valid_items[chunk_start:chunk_start + chunk_size]

        try:
            if is_train:
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss, pred_counts_list = _compute_chunk_loss(
                            chunk, l1_model, l3_model, device,
                            scc_reg_weight, max_n, p_predicted, pos_reg_weight,
                            freeze_l3_backbone)
                else:
                    loss, pred_counts_list = _compute_chunk_loss(
                        chunk, l1_model, l3_model, device,
                        scc_reg_weight, max_n, p_predicted, pos_reg_weight,
                        freeze_l3_backbone)

                if loss.requires_grad:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                total_loss += float(loss.item()) * len(chunk)

                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            else:
                with torch.no_grad():
                    loss, pred_counts_list = _compute_chunk_loss(
                        chunk, l1_model, l3_model, device,
                        scc_reg_weight=0.0, max_n=max_n, p_predicted=0.0,
                        pos_reg_weight=0.0, freeze_l3_backbone=True)
                total_loss += float(loss.item()) * len(chunk)

        except Exception:
            errors += len(chunk)
            continue

        for (case, dag, scc_gt_counts), pred_counts in zip(chunk, pred_counts_list):
            if not pred_counts:
                continue
            pred_global = 1
            for scc in dag.sccs:
                pred_global *= pred_counts.get(scc.scc_id, 1)
            n_exact += int(pred_global == case["n_fp_gt"])
            n_total += 1

    return {
        "loss":        total_loss / max(n_total, 1),
        "count_exact": n_exact    / max(n_total, 1),
        "n":           n_total,
        "errors":      errors,
    }


# ---------------------------------------------------------------------------
# 9. Checkpoint saving
# ---------------------------------------------------------------------------

def save_best_full(out_dir: Path, l1_model, l3_model, l1_cfg_dict, l3_cfg_dict):
    torch.save({"cfg": l1_cfg_dict, "model": l1_model.state_dict(),
                "ema": l1_model.state_dict()},
               out_dir / "l1_best.pt")
    torch.save({"cfg": l3_cfg_dict, "model": l3_model.state_dict(),
                "ema": l3_model.state_dict()},
               out_dir / "l3_best.pt")


def save_checkpoint(out_dir: Path, tag: str, l1_model, l3_model,
                    l1_cfg_dict, l3_cfg_dict, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": l1_cfg_dict, "model": l1_model.state_dict()},
               out_dir / f"l1_{tag}.pt")
    torch.save({"cfg": l3_cfg_dict, "model": l3_model.state_dict(),
                "metrics": metrics},
               out_dir / f"l3_{tag}.pt")


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end composition-level fine-tuning (L1 + L3)")
    ap.add_argument("--l1_ckpt",    required=True)
    ap.add_argument("--l3_ckpt",    required=True)
    ap.add_argument("--train_dirs", required=True, nargs="+")
    ap.add_argument("--val_dir",    required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--epochs",         type=int,   default=40)
    ap.add_argument("--lr",             type=float, default=1e-4,
                    help="Learning rate for count heads")
    ap.add_argument("--lr_pos",         type=float, default=0.0,
                        help="LR for l1 pos_head only (pos-only isolation mode). "
                             "Set --lr 0 and --lr_backbone 0 to keep count heads and "
                             "backbone frozen while training only the slot pos MLP.")
    ap.add_argument("--lr_backbone",    type=float, default=0.0,
                    help="LR for backbone parameters (0 = frozen). "
                         "Recommended: 1e-5 if needed.")
    ap.add_argument("--lr_decoder",     type=float, default=0.0,
                    help="LR for L1 SlotDecoder + exist_head (0 = frozen). "
                         "Requires --use_v2_count_head; otherwise decoder gradient "
                         "shifts exist_logit and breaks count calibration. "
                         "Recommended: 1e-5.")
    ap.add_argument("--use_v2_count_head", action="store_true",
                    help="Use ForcingAwareCountHeadV2 (no exist_logit input). "
                         "Required when --lr_decoder > 0 to protect count calibration. "
                         "Reinitialises forcing_count_head from scratch.")
    ap.add_argument("--wd",             type=float, default=1e-4)
    ap.add_argument("--scc_reg_weight", type=float, default=0.3,
                    help="Weight on per-SCC CE regularisation term. "
                         "0 = pure composition CE, 1 = equal weight. "
                         "Default 0.3 keeps individual-SCC accuracy while "
                         "composition CE provides the primary gradient.")
    ap.add_argument("--max_n",          type=int,   default=128,
                    help="Maximum N_FP to track in composition distribution. "
                         "128 covers all 2-SCC compositions (max 9*9=81).")
    ap.add_argument("--p_predicted",     type=float, default=0.5,
                    help="Fraction of forced-SCC training calls that use predicted "
                         "(not GT) parent FPs to build the forcing signal. "
                         "0 = always GT (original teacher forcing), "
                         "1 = always predicted (no teacher forcing). "
                         "Default 0.5 (scheduled sampling).")
    ap.add_argument("--pos_reg_weight", type=float, default=0.0,
                    help="Weight on log-space Chamfer position loss for root SCC slots. "
                         "0 = disabled (default). With --lr_pos > 0 (pos-only mode) "
                         "set this to 1.0. With --lr_backbone > 0 set to 0.1.")
    ap.add_argument("--accumulate",     type=int,   default=32,
                    help="Gradient accumulation steps before each optimizer.step(). "
                         "Higher = fewer optimizer steps per epoch = faster. Default 32.")
    ap.add_argument("--no_amp",        action="store_true",
                    help="Disable automatic mixed precision (float16 forward). "
                         "AMP is enabled by default on CUDA and gives ~2x speedup.")
    ap.add_argument("--max_train",      type=int,   default=None)
    ap.add_argument("--no_cuda",        action="store_true")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}")

    if args.lr_decoder > 0.0 and not args.use_v2_count_head:
        raise ValueError(
            "--lr_decoder requires --use_v2_count_head.  Without V2 head, "
            "decoder gradient shifts exist_logit and breaks count calibration.")

    print("Loading models...")
    l1_model, l1_cfg, l1_cfg_dict = load_l1(args.l1_ckpt, device)
    l3_model, l3_cfg, l3_cfg_dict = load_l3(args.l3_ckpt, l1_model, device,
                                             use_v2_count_head=args.use_v2_count_head)

    freeze_all(l1_model)
    freeze_all(l3_model)
    pos_only_mode = (args.lr == 0.0 and args.lr_backbone == 0.0 and args.lr_pos > 0.0)
    param_groups = _make_trainable_groups(l1_model, l3_model,
                                          lr_head=args.lr,
                                          lr_backbone=args.lr_backbone,
                                          lr_pos=args.lr_pos,
                                          lr_decoder=args.lr_decoder)

    trainable = sum(p.numel() for g in param_groups for p in g["params"])
    total     = sum(p.numel() for p in list(l1_model.parameters()) +
                    list(l3_model.parameters()))
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    if pos_only_mode:
        print(f"  Mode: pos-only isolation — only l1_model.pos_head trained")
        print(f"  pos_head LR: {args.lr_pos:.2e}")
    elif args.lr_decoder > 0.0:
        print(f"  Mode: decoder unfreeze — SlotDecoder + exist_head trainable")
        print(f"  Count head: {'V2 (no exist_logit)' if args.use_v2_count_head else 'V1'}")
        print(f"  decoder LR: {args.lr_decoder:.2e}")
    elif args.lr_backbone > 0.0:
        print(f"  Backbone LR: {args.lr_backbone:.2e}  (backbone unfrozen)")
    else:
        print("  Backbone frozen — only count heads trained")
    if args.use_v2_count_head:
        print("  ForcingAwareCountHeadV2: forcing_count_head re-initialised from scratch")
    print(f"  Composition CE max_n: {args.max_n}")
    print(f"  Per-SCC regularisation weight: {args.scc_reg_weight}")
    print(f"  Position supervision weight: {args.pos_reg_weight}")
    print(f"  Teacher forcing p_predicted: {args.p_predicted:.2f}  "
          f"({'always GT' if args.p_predicted == 0.0 else 'always predicted' if args.p_predicted == 1.0 else 'scheduled sampling'})")

    print("Loading data...")
    train_cases = gather_cases(args.train_dirs)
    val_cases   = gather_cases([args.val_dir])
    print(f"  Train: {len(train_cases)} cases | Val: {len(val_cases)} cases")

    print("Pre-computing SCC DAGs...")
    train_dags = precompute_dags(train_cases)
    val_dags   = precompute_dags(val_cases)

    print("Pre-computing GT counts (avoids per-epoch numpy work)...")
    train_gt_counts = precompute_gt_counts(train_cases, train_dags)
    val_gt_counts   = precompute_gt_counts(val_cases,   val_dags)

    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"  Gradient accumulation: {args.accumulate} steps")
    print(f"  Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
    # Cosine annealing: anneal over the primary LR (pos_head LR in pos_only mode)
    primary_lr = args.lr_pos if pos_only_mode else args.lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=primary_lr * 0.01)

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"

    # pos_only: track lowest val pos_loss; otherwise highest val count_exact
    best_val_metric = float("inf") if pos_only_mode else -1.0
    log_rows = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train CE':>9}  "
          f"{'Val Loss':>9}  {'Val CE':>8}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        if pos_only_mode:
            l1_model.eval()   # backbone, count heads stay in eval
            l1_model.pos_head.train()
            l3_model.eval()
        elif args.lr_backbone > 0.0:
            l1_model.train()
            l3_model.train()
        else:
            l1_model.count_head.train()
            l1_model.stable_count_head.train()
            l3_model.forcing_count_head.train()

        train_m = run_epoch(train_cases, train_dags, l1_model, l3_model, device,
                            optimizer=optimizer, max_cases=args.max_train,
                            scc_reg_weight=args.scc_reg_weight, max_n=args.max_n,
                            p_predicted=args.p_predicted,
                            pos_reg_weight=args.pos_reg_weight,
                            freeze_l3_backbone=(args.lr_backbone == 0.0),
                            gt_counts_cache=train_gt_counts,
                            accumulate=args.accumulate,
                            scaler=scaler)

        l1_model.eval()
        l3_model.eval()
        # In pos_only mode pass pos_reg_weight so val_loss reflects pos quality
        val_pos_w = args.pos_reg_weight if pos_only_mode else 0.0
        val_m = run_epoch(val_cases, val_dags, l1_model, l3_model, device,
                          max_n=args.max_n,
                          pos_reg_weight=val_pos_w,
                          gt_counts_cache=val_gt_counts)

        lr_now = optimizer.param_groups[-1]["lr"]
        scheduler.step()

        print(f"{epoch:>6}  {train_m['loss']:>10.4f}  "
              f"{train_m['count_exact']:>9.4f}  "
              f"{val_m['loss']:>9.4f}  "
              f"{val_m['count_exact']:>8.4f}  "
              f"{lr_now:>8.2e}")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "train_ce":   train_m["count_exact"],
            "val_loss":   val_m["loss"],
            "val_ce":     val_m["count_exact"],
            "lr":         lr_now,
        })
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        if pos_only_mode:
            is_better = val_m["loss"] < best_val_metric
        else:
            is_better = val_m["count_exact"] > best_val_metric

        if is_better:
            best_val_metric = val_m["loss"] if pos_only_mode else val_m["count_exact"]
            save_best_full(out_dir, l1_model, l3_model, l1_cfg_dict, l3_cfg_dict)
            metric_label = "val pos_loss" if pos_only_mode else "val count_exact"
            print(f"         ↑ new best {metric_label} = {best_val_metric:.4f}  (saved)")

        if epoch % 5 == 0:
            save_checkpoint(out_dir, f"ep{epoch:03d}", l1_model, l3_model,
                            l1_cfg_dict, l3_cfg_dict, val_m)

    best_label = "val pos_loss" if pos_only_mode else "val count_exact"
    print(f"\nBest {best_label}: {best_val_metric:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()

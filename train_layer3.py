"""Entry point for Layer III: training Φ_ψ + A_η + U_ω on Corpus C (SCC-DAG).

Usage:
    python train_layer3.py --mode train \\
        --layer2_ckpt /path/to/layer2/best.pt \\
        --cache_dir_c  /path/to/corpusC/cache \\
        --out_dir      /path/to/runs/layer3

    python train_layer3.py --mode eval  \\
        --layer2_ckpt /path/to/layer2/best.pt \\
        --cache_dir_c /path/to/corpusC/cache \\
        --out_dir     /path/to/runs/layer3 --eval_split val

Any Config field can be overridden on the command line (--field value).

Design notes:
  - The Layer II best.pt is loaded as a warm start.  The backbone (E_θ + D_φ)
    is taken from the Layer II EMA weights (already frozen in Layer II), but the
    NEW modules Φ_ψ, A_η, U_ω are also loaded from Layer II — so Layer III resumes
    from where Layer II left off rather than re-initializing.
  - freeze_backbone=True by default: the backbone stays frozen; only Φ_ψ/A_η/U_ω
    are updated.  Setting freeze_backbone=False fine-tunes everything.
  - CorpusCNet.forward synthesizes port tensors from branch_u at forward time, so
    no changes to the loss or metric code are needed.
  - Corpus C does not store xmax; CorpusCNet computes it as P/K on the fly.
  - Corpus C does not store has_pos_fb; the thomas loss is therefore disabled
    (has_pos_fb defaults to False in CorpusCNet.forward).
  - The val split in Corpus C uses only_solved=True by default (DAGs where every SCC
    has at least one branch), since unsolved branches have k=0 and are filtered out.

Sections
  1. Config
  2. Utilities (EMA, optimizer, schedule — mirrors train_layer2.py)
  3. Evaluation
  4. Checkpoints and logging
  5. Modes: train, eval
"""
import argparse
import contextlib
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from data_c import make_loader_c, describe_dataset_c
from losses import (CalibrationTable, MetricAccumulator, SetCriterion,
                    batch_metrics)
from model import SccFixedPointNet
from model_b import CorpusBNet
from model_c import CorpusCNet


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # mode and paths
    mode:           str  = "train"
    layer2_ckpt:    str  = ""                 # path to Layer II best.pt (required)
    cache_dir_c:    str  = "corpusC/cache"    # Corpus C cache directory
    out_dir:        str  = "runs/layer3"
    ckpt:           str  = ""                 # eval: override checkpoint path
    eval_split:     str  = "val"
    eval_max_samples: int = 0
    only_solved_val: bool = True              # val: restrict to dag_all_solved

    # data
    max_fp:          int   = 16
    max_val_samples: int   = 20000
    sampler_power:   float = 0.5
    num_workers:     int   = 8
    batch_size:      int   = 128
    seed:            int   = 0

    # schedule
    steps_per_epoch: int   = 2000
    epochs:          int   = 60
    log_every:       int   = 100
    lr:              float = 5e-5            # lower than Layer II: fine-tuning
    weight_decay:    float = 0.05
    warmup_steps:    int   = 300
    min_lr_frac:     float = 0.02
    grad_clip:       float = 1.0
    ema_decay:       float = 0.999
    refine_steps:    int   = 3
    amp:             bool  = True
    resume:          bool  = True

    # backbone / Layer I model (must match the Layer II checkpoint config)
    d_model:       int   = 128
    n_gnn_layers:  int   = 4
    n_attn_layers: int   = 1
    n_dec_layers:  int   = 4
    n_heads:       int   = 8
    dropout:       float = 0.1
    max_fp_model:  int   = 16

    # Layer II/III new modules (must match the Layer II checkpoint config)
    m_dim:         int   = 128
    n_u_layers:    int   = 2
    freeze_backbone: bool = True
    max_ports_c:   int   = 8                 # port slots for Corpus C

    # matching / loss weights (same as Layer II defaults)
    cost_exist: float = 1.0
    cost_pos:   float = 5.0
    cost_cp:    float = 2.0
    cost_stab:  float = 1.0
    w_exist:    float = 1.0
    w_pos:      float = 5.0
    w_cp:       float = 2.0
    w_stab:     float = 1.0
    w_count:    float = 2.0
    w_count_kscale: float = 2.0   # instance weight = k_true * kscale + 1; 0 = flat
    sampler_kpower: float = 0.5   # upsample high-k branches; 0 = off
    w_phys:     float = 1.0
    w_thomas:   float = 0.0    # disabled: Corpus C has no has_pos_fb
    eos_coef:   float = 0.3
    huber_beta:  float = 0.05
    gate_tau:    float = 0.02
    dup_tol:     float = 0.02


def as_bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    for f in fields(Config):
        if isinstance(f.default, bool):
            parser.add_argument(f"--{f.name}", type=as_bool, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return Config(**vars(parser.parse_args(argv)))


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_ctx(cfg, device):
    enabled = cfg.amp and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


class _BackboneCfg:
    def __init__(self, cfg):
        self.d_model       = cfg.d_model
        self.n_gnn_layers  = cfg.n_gnn_layers
        self.n_attn_layers = cfg.n_attn_layers
        self.n_dec_layers  = cfg.n_dec_layers
        self.n_heads       = cfg.n_heads
        self.dropout       = cfg.dropout
        self.max_fp        = cfg.max_fp_model


def load_layer2_checkpoint(cfg, device):
    """Load Layer II best.pt and reconstruct CorpusCNet with those weights.

    Strategy:
      1. Build backbone from saved config (or cfg fallback).
      2. Build CorpusCNet (same parameter layout as CorpusBNet).
      3. Load the EMA state dict from the Layer II checkpoint — all parameter
         names are identical between CorpusBNet and CorpusCNet.
      4. Freeze backbone if cfg.freeze_backbone.
    """
    if not cfg.layer2_ckpt:
        raise ValueError("--layer2_ckpt must be set to the Layer II best.pt path")

    ck = torch.load(cfg.layer2_ckpt, map_location=device, weights_only=False)
    saved_cfg = ck.get("cfg", {})

    # Use saved backbone dimensions if present; fall back to cfg.
    def get(key, default):
        return saved_cfg.get(key, default) if isinstance(saved_cfg, dict) else \
               getattr(saved_cfg, key, default)

    bk_cfg = _BackboneCfg(cfg)
    bk_cfg.d_model       = get("d_model",       cfg.d_model)
    bk_cfg.n_gnn_layers  = get("n_gnn_layers",  cfg.n_gnn_layers)
    bk_cfg.n_attn_layers = get("n_attn_layers",  cfg.n_attn_layers)
    bk_cfg.n_dec_layers  = get("n_dec_layers",   cfg.n_dec_layers)
    bk_cfg.n_heads       = get("n_heads",        cfg.n_heads)
    bk_cfg.dropout       = get("dropout",        cfg.dropout)
    bk_cfg.max_fp        = get("max_fp_model",   cfg.max_fp_model)

    backbone = SccFixedPointNet(bk_cfg).to(device)

    model = CorpusCNet(backbone, cfg).to(device)

    # Load Layer II EMA weights (names match exactly)
    weights = ck.get("ema", ck.get("model"))
    missing, unexpected = model.load_state_dict(
        {k: v.to(device) for k, v in weights.items()}, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys from Layer II ckpt: {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys from Layer II ckpt: {unexpected[:5]}")

    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

    print(f"Layer II weights loaded from {cfg.layer2_ckpt} (epoch {ck.get('epoch', '?')}), "
          f"backbone {'frozen' if cfg.freeze_backbone else 'fine-tuned'}", flush=True)
    return model


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    if not decay and not no_decay:
        raise RuntimeError("No trainable parameters. Check freeze_backbone and model.")
    groups = [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_params = sum(p.numel() for p in decay + no_decay)
    print(f"trainable parameters: {n_params / 1e6:.3f}M", flush=True)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def build_scheduler(opt, cfg):
    total = cfg.epochs * cfg.steps_per_epoch

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        prog = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        cos  = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class EMA:
    def __init__(self, model, decay):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @contextlib.contextmanager
    def applied(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(backup)


def describe_model(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters total {total / 1e6:.2f}M  trainable {trainable / 1e6:.2f}M", flush=True)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if n > 0:
            print(f"   {name:<20s} {n / 1e6:7.3f}M  trainable {t / 1e6:.3f}M", flush=True)


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loader_c(model, criterion, loader, device, cfg):
    """Evaluate CorpusCNet on a Corpus C loader."""
    model.eval()
    acc   = MetricAccumulator()
    calib = CalibrationTable()
    loss_sum, n_batches = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad(), autocast_ctx(cfg, device):
            out_cond, _out0 = model(batch, freeze_backbone=cfg.freeze_backbone)
        _, logs, aux = criterion(out_cond, batch)
        for k, v in logs.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + float(v)
        # Corpus C: use "tc" as phase grouping key (Thomas class)
        if "phase" not in batch:
            batch["phase"] = torch.zeros(batch["tc"].shape[0], dtype=torch.long, device=batch["tc"].device)
        acc.update(batch_metrics(out_cond, batch, aux, cfg.refine_steps), batch)
        calib.update(out_cond, aux, batch)
        n_batches += 1
    overall, grouped = acc.compute()
    losses = {k: v / max(n_batches, 1) for k, v in loss_sum.items()}
    return losses, overall, grouped, calib.compute()


def format_metrics(overall):
    keys = ["count_exact", "regime_acc", "stable_count_exact", "fp_recall", "stab_acc",
            "pos_decades", "brouwer_idx_acc", "brouwer_consistent", "thomas_violation",
            "refined_converged"]
    return " ".join(f"{k} {overall[k]:.4f}" for k in keys if k in overall)


# ---------------------------------------------------------------------------
# 4. Checkpoints and logging
# ---------------------------------------------------------------------------
def selection_score(overall):
    return overall.get("count_exact", 0.0) + overall.get("fp_recall", 0.0)


def save_checkpoint(path, model, ema, opt, sched, epoch, best, cfg):
    torch.save({
        "model": model.state_dict(),
        "ema":   ema.shadow,
        "opt":   opt.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "best":  best,
        "cfg":   asdict(cfg),
    }, path)


def append_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
# 5. Modes
# ---------------------------------------------------------------------------
def run_train(cfg):
    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_json(os.path.join(cfg.out_dir, "config.json"), asdict(cfg))

    # Data
    train_ds, train_loader = make_loader_c(cfg, train=True,  only_solved=False)
    val_ds,   val_loader   = make_loader_c(cfg, train=False, only_solved=cfg.only_solved_val)
    describe_dataset_c(train_ds, "train")
    describe_dataset_c(val_ds,   "val")
    print(f"device: {device}", flush=True)

    # Model (warm-started from Layer II)
    model = load_layer2_checkpoint(cfg, device)
    describe_model(model)

    criterion = SetCriterion(cfg)
    opt       = build_optimizer(model, cfg)
    sched     = build_scheduler(opt, cfg)
    ema       = EMA(model, cfg.ema_decay)

    start_epoch, best = 0, -float("inf")
    last_path = os.path.join(cfg.out_dir, "last.pt")
    best_path = os.path.join(cfg.out_dir, "best.pt")

    if cfg.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"], ck["best"]
        print(f"resumed at epoch {start_epoch}", flush=True)

    log_path = os.path.join(cfg.out_dir, "log.csv")
    scaler   = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    for epoch in range(start_epoch, cfg.epochs):
        if cfg.freeze_backbone:
            model.backbone.eval()
            model.phi_psi.train()
            model.a_eta.train()
            model.u_omega.train()
        else:
            model.train()

        window, n_window = {}, 0
        epoch_sum, n_epoch = {}, 0

        for it, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            if "phase" not in batch:
                batch["phase"] = torch.zeros(batch["tc"].shape[0], dtype=torch.long, device=batch["tc"].device)

            with autocast_ctx(cfg, device):
                out_cond, _out0 = model(batch, freeze_backbone=cfg.freeze_backbone)
                loss, logs, _   = criterion(out_cond, batch)

            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} it {it}; step skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)

            logs["grad_norm"] = grad_norm.detach()
            for k, v in logs.items():
                window[k]    = window.get(k, 0.0) + v
                epoch_sum[k] = epoch_sum.get(k, 0.0) + v
            n_window += 1
            n_epoch  += 1

            if (it + 1) % cfg.log_every == 0:
                msg = " ".join(f"{k} {(v / n_window).item():.4f}" for k, v in window.items())
                print(f"epoch {epoch + 1} it {it + 1}/{cfg.steps_per_epoch} "
                      f"lr {sched.get_last_lr()[0]:.2e} | {msg}", flush=True)
                window, n_window = {}, 0

        # Validation with EMA weights
        with ema.applied(model):
            val_losses, overall, _, calib = evaluate_loader_c(
                model, criterion, val_loader, device, cfg)

        row = {"epoch": epoch + 1, "lr": sched.get_last_lr()[0]}
        row.update({f"train/loss_{k}": (v / max(n_epoch, 1)).item()
                    for k, v in epoch_sum.items()})
        row.update({f"val/loss_{k}": v for k, v in val_losses.items()})
        row.update({f"val/{k}": v for k, v in overall.items()})
        row["val/exist_ece"] = calib["ece"]
        print(f"epoch {epoch + 1} | val loss {val_losses['total']:.4f} | "
              f"{format_metrics(overall)} | ece {calib['ece']:.4f}", flush=True)
        append_csv(log_path, row)

        score = selection_score(overall)
        if score > best:
            best = score
            save_checkpoint(best_path, model, ema, opt, sched, epoch + 1, best, cfg)
        save_checkpoint(last_path, model, ema, opt, sched, epoch + 1, best, cfg)


def run_eval(cfg):
    set_seed(cfg.seed)
    device = get_device()
    path = cfg.ckpt or os.path.join(cfg.out_dir, "best.pt")

    if cfg.ckpt:
        # Explicit override: load from a specific checkpoint
        model = load_layer2_checkpoint(cfg, device)
        ck    = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})
    else:
        # Default: treat best.pt as a Layer III checkpoint (not Layer II)
        ck = torch.load(path, map_location=device, weights_only=False)
        # Reconstruct from saved cfg if available
        saved_cfg = ck.get("cfg", {})
        if saved_cfg:
            for f in fields(cfg):
                if f.name in saved_cfg:
                    object.__setattr__(cfg, f.name, saved_cfg[f.name])
        model = load_layer2_checkpoint(cfg, device)
        model.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})

    model.eval()
    criterion = SetCriterion(cfg)

    only_solved = cfg.eval_split == "val" and cfg.only_solved_val
    ds, loader  = make_loader_c(cfg, train=False,
                                only_solved=only_solved,
                                max_samples=cfg.eval_max_samples)
    print(f"{cfg.eval_split}: {len(ds)} branches", flush=True)
    losses, overall, grouped, calib = evaluate_loader_c(model, criterion, loader, device, cfg)

    print(f"checkpoint {path} (epoch {ck.get('epoch', '?')})")
    print("loss    : " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()))
    print("metrics : " + format_metrics(overall))
    print(f"existence ECE {calib['ece']:.4f}")
    out_path = os.path.join(cfg.out_dir, f"eval_{cfg.eval_split}.json")
    write_json(out_path, {
        "checkpoint": path, "epoch": ck.get("epoch", "?"), "split": cfg.eval_split,
        "branches": len(ds), "loss": losses, "overall": overall,
        "by_group": grouped, "calibration": calib,
    })
    print(f"wrote {out_path}")


def main():
    cfg = parse_config()
    if cfg.mode == "train":
        run_train(cfg)
    elif cfg.mode == "eval":
        run_eval(cfg)
    else:
        raise ValueError("mode must be train or eval")


if __name__ == "__main__":
    main()

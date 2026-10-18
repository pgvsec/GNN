"""Entry point for Layer II/III: training Φ_ψ + A_η + U_ω on Corpus B.

Usage:
    python train_layer2.py --mode train \\
        --layer1_ckpt /path/to/layer1/best.pt \\
        --cache_dir_b  /path/to/corpusB/cache \\
        --out_dir      /path/to/runs/layer2

    python train_layer2.py --mode eval  \\
        --layer1_ckpt /path/to/layer1/best.pt \\
        --cache_dir_b /path/to/corpusB/cache \\
        --out_dir     /path/to/runs/layer2 --eval_split val

Any Config field can be overridden on the command line (--field value).

Design notes:
  - The backbone SccFixedPointNet is loaded from the Layer I EMA checkpoint and
    FROZEN by default (freeze_backbone=True). All gradient flow goes through
    Φ_ψ, A_η, U_ω only.
  - The physics loss (hill_residual) automatically uses the forced P because
    Corpus B absorbs u into P at generation time (apply_forcing). No changes to
    SetCriterion are needed: batch["P"] already contains P_forced.
  - U_ω gets the pair-embedding tensor from the backbone's GNN layers via
    CorpusBNet._run_backbone(), which re-runs the backbone sub-modules.
    This is slightly more expensive than caching but avoids storing large
    edge-embedding tensors across batches and is simpler to implement correctly.

Sections
  1. Config
  2. Utilities (EMA, optimizer, schedule, etc. — mirrors train.py)
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

from data import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from data_b import make_loader_b
from losses import (CalibrationTable, MetricAccumulator, SetCriterion,
                    batch_metrics, label_residual)
from model import SccFixedPointNet
from model_b import CorpusBNet


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # mode and paths
    mode:          str   = "train"          # train | eval
    layer1_ckpt:   str   = ""               # path to Layer I best.pt (required)
    cache_dir_b:   str   = "corpusB/cache"  # Corpus B cache directory
    out_dir:       str   = "runs/layer2"
    ckpt:          str   = ""               # eval: override checkpoint path
    eval_split:    str   = "val"            # train | val | test
    eval_max_samples: int = 0

    # data
    max_fp:          int   = 16
    max_val_samples: int   = 20000
    sampler_power:   float = 0.5
    num_workers:     int   = 8
    batch_size:      int   = 128

    # schedule
    steps_per_epoch: int   = 2000
    epochs:          int   = 60
    log_every:       int   = 100
    lr:              float = 1e-4             # lower than Layer I: new modules only
    weight_decay:    float = 0.05
    warmup_steps:    int   = 500
    min_lr_frac:     float = 0.02
    grad_clip:       float = 1.0
    ema_decay:       float = 0.999
    refine_steps:    int   = 3
    amp:             bool  = True
    seed:            int   = 0
    resume:          bool  = True

    # backbone / Layer I model (must match the saved checkpoint)
    d_model:       int   = 128
    n_gnn_layers:  int   = 4
    n_attn_layers: int   = 1
    n_dec_layers:  int   = 4
    n_heads:       int   = 8
    dropout:       float = 0.1
    max_fp_model:  int   = 16              # max_fp used when building SccFixedPointNet

    # Layer II/III new modules
    m_dim:         int   = 128             # message dimension (Φ_ψ → A_η → U_ω)
    n_u_layers:    int   = 2               # number of GNN layers in U_ω
    freeze_backbone: bool = True           # freeze E_θ + D_φ during training

    # matching cost weights (same as Layer I defaults)
    cost_exist: float = 1.0
    cost_pos:   float = 5.0
    cost_cp:    float = 2.0
    cost_stab:  float = 1.0

    # loss weights (same as Layer I defaults)
    w_exist:  float = 1.0
    w_pos:    float = 5.0
    w_cp:     float = 2.0
    w_stab:   float = 1.0
    w_count:  float = 3.0
    w_exist_cnt: float = 0.5      # soft-count consistency loss weight
    count_focal_gamma: float = 2.0  # focal reweighting of count CE; 0 = plain CE
    count_head_layers: int = 3    # MLP depth for count/stable_count heads
    w_phys:   float = 1.0
    w_thomas: float = 0.5
    eos_coef: float = 0.5
    huber_beta:  float = 0.05
    gate_tau:    float = 0.02
    dup_tol:     float = 0.02

    # max_fp alias used by SccFixedPointNet constructor
    @property
    def max_fp_for_backbone(self):
        return self.max_fp_model


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


SPLITS = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}


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
    """Minimal shim so SccFixedPointNet(cfg) works from our Config."""
    def __init__(self, cfg):
        self.d_model      = cfg.d_model
        self.n_gnn_layers = cfg.n_gnn_layers
        self.n_attn_layers= cfg.n_attn_layers
        self.n_dec_layers = cfg.n_dec_layers
        self.n_heads      = cfg.n_heads
        self.dropout      = cfg.dropout
        self.max_fp       = cfg.max_fp_model


def load_backbone(cfg, device):
    """Construct SccFixedPointNet and load Layer I EMA weights."""
    if not cfg.layer1_ckpt:
        raise ValueError("--layer1_ckpt must be set to the Layer I best.pt path")
    bk_cfg  = _BackboneCfg(cfg)
    backbone = SccFixedPointNet(bk_cfg).to(device)
    ck = torch.load(cfg.layer1_ckpt, map_location=device)
    # Layer I checkpoints store EMA weights under "ema" and raw weights under "model"
    weights = ck.get("ema", ck.get("model"))
    backbone.load_state_dict({k: v.to(device) for k, v in weights.items()})
    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
    print(f"backbone loaded from {cfg.layer1_ckpt} (epoch {ck.get('epoch', '?')}), "
          f"{'frozen' if cfg.freeze_backbone else 'fine-tuned'}", flush=True)
    return backbone


def build_optimizer(model, cfg):
    """AdamW on trainable parameters only (backbone excluded when frozen)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    if not decay and not no_decay:
        raise RuntimeError("No trainable parameters found. Check freeze_backbone and model.")
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
    """EMA over the full CorpusBNet (backbone weights frozen, so EMA only tracks new modules)."""

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
    total = sum(p.numel() for p in model.parameters())
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
def check_labels_b(loader, device):
    """Check ODE residual of Corpus B fixed points under forced parameters."""
    batch = to_device(next(iter(loader)), device)
    r = label_residual(batch)
    if r.numel() == 0:
        print("label check: no fixed points in the first batch", flush=True)
        return
    q99 = torch.quantile(r.float(), 0.99).item()
    frac = (r > 1e-3).float().mean().item()
    print(f"label check (forced ODE): |residual| median {r.median().item():.2e}  "
          f"p99 {q99:.2e}  fraction > 1e-3: {frac:.3f}", flush=True)
    if r.median().item() > 1e-3:
        print("WARNING: Corpus B fixed points do not satisfy the forced ODE.", flush=True)


@torch.no_grad()
def evaluate_loader_b(model, criterion, loader, device, cfg):
    """Evaluate CorpusBNet on a Corpus B loader; returns losses, metrics, calibration."""
    model.eval()
    acc   = MetricAccumulator()
    calib = CalibrationTable()
    loss_sum, n_batches = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_ctx(cfg, device):
            out_cond, _out0 = model(batch, freeze_backbone=cfg.freeze_backbone)
        _, logs, aux = criterion(out_cond, batch)
        for k, v in logs.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + float(v)
        # batch_metrics expects "phase" key — Corpus B has "stratum" in that role;
        # we alias it so the grouped metric table still works.
        if "phase" not in batch:
            batch["phase"] = batch["stratum"]
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
    train_ds, train_loader = make_loader_b(cfg, SPLIT_TRAIN, train=True)
    val_ds,   val_loader   = make_loader_b(cfg, SPLIT_VAL,   train=False)
    print(f"train: {len(train_ds)} samples  val: {len(val_ds)} samples", flush=True)
    print(f"device: {device}", flush=True)
    check_labels_b(val_loader, device)

    # Model
    backbone = load_backbone(cfg, device)
    model    = CorpusBNet(backbone, cfg).to(device)
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
    scaler   = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

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
            # Alias stratum → phase for MetricAccumulator group keys
            if "phase" not in batch:
                batch["phase"] = batch["stratum"]

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
            val_losses, overall, _, calib = evaluate_loader_b(
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
    if cfg.eval_split not in SPLITS:
        raise ValueError(f"eval_split must be one of {sorted(SPLITS)}")
    path = cfg.ckpt or os.path.join(cfg.out_dir, "best.pt")
    ck   = torch.load(path, map_location=device)

    backbone = load_backbone(cfg, device)
    model    = CorpusBNet(backbone, cfg).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})
    model.eval()
    criterion = SetCriterion(cfg)

    ds, loader = make_loader_b(cfg, SPLITS[cfg.eval_split], train=False,
                                max_samples=cfg.eval_max_samples)
    print(f"{cfg.eval_split}: {len(ds)} samples", flush=True)
    losses, overall, grouped, calib = evaluate_loader_b(model, criterion, loader, device, cfg)

    print(f"checkpoint {path} (epoch {ck.get('epoch', '?')})")
    print("loss    : " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()))
    print("metrics : " + format_metrics(overall))
    print(f"existence ECE {calib['ece']:.4f}")
    out_path = os.path.join(cfg.out_dir, f"eval_{cfg.eval_split}.json")
    write_json(out_path, {
        "checkpoint": path, "epoch": ck.get("epoch", "?"), "split": cfg.eval_split,
        "samples": len(ds), "loss": losses, "overall": overall,
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

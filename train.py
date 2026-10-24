"""Entry point for Layer I: cache construction, training and evaluation.

    python train.py --mode cache --corpus_dir corpusA --cache_dir corpusA/cache
    python train.py --mode train --cache_dir corpusA/cache --out_dir runs/layer1
    python train.py --mode eval  --cache_dir corpusA/cache --out_dir runs/layer1 --eval_split test
    python train.py --mode polish --cache_dir corpusA/cache --out_dir runs/layer1 --eval_split test

Any Config field can be overridden on the command line (--field value).

Sections
  1. Config
  2. Utilities: seeding, devices, optimizer, schedule, weight averaging
  3. Data checks and evaluation over a loader
  4. Checkpoints and logging
  5. Modes: cache, train, eval
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

from data import (SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, build_cache, describe_dataset,
                  make_loader)
from losses import CalibrationTable, MetricAccumulator, SetCriterion, batch_metrics, label_residual
from model import SccFixedPointNet, decode_position
from solver import (SolverConfig, cluster_roots, make_seeds, newton, prepare_params,
                    root_properties)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # mode and paths
    mode: str = "train"                  # cache | train | eval | polish
    corpus_dir: str = "corpusA"
    cache_dir: str = "corpusA/cache"
    out_dir: str = "runs/layer1"
    ckpt: str = ""                       # eval: checkpoint path, default out_dir/best.pt
    eval_split: str = "val"              # eval: train | val | test
    eval_max_samples: int = 0            # eval: 0 evaluates every sample of the split

    # data
    max_fp: int = 16                     # K_FP: decoder slots; samples with more FPs are dropped
    max_val_samples: int = 20000
    sampler_power: float = 0.5           # sample weight = count(n) ** -power
    num_workers: int = 8
    batch_size: int = 256

    # schedule
    steps_per_epoch: int = 2000
    epochs: int = 60
    log_every: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    min_lr_frac: float = 0.02
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    refine_steps: int = 3                # Newton iterations used by the refined_converged metric
    amp: bool = True
    seed: int = 0
    resume: bool = True

    # model
    d_model: int = 128
    n_gnn_layers: int = 4
    n_attn_layers: int = 1
    n_dec_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    # matching cost weights
    cost_exist: float = 1.0
    cost_pos: float = 5.0
    cost_cp: float = 2.0
    cost_stab: float = 1.0

    # loss weights
    w_exist: float = 1.0
    w_pos: float = 5.0
    w_cp: float = 2.0
    w_stab: float = 1.0
    w_count: float = 3.0
    w_count_kscale: float = 0.0   # instance weight = k_true * kscale + 1; 0 = flat (physically honest)
    w_exist_cnt: float = 0.5      # soft-count consistency: MSE(sigmoid(exist).sum, k_true) on consistent
    count_focal_gamma: float = 2.0  # focal reweighting of count CE: (1-pt)^gamma; 0 = plain CE
    count_head_layers: int = 3    # MLP depth for count/stable_count heads (was 2)
    sampler_kpower: float = 0.0   # upsample high-k branches; 0 = off
    w_phys: float = 1.0
    w_thomas: float = 0.5
    eos_coef: float = 0.5                # weight of no-object slots in the existence loss
    huber_beta: float = 0.05
    gate_tau: float = 0.02               # residual above which an unmatched slot is a true negative
    dup_tol: float = 0.02                # position distance below which an unmatched slot is a duplicate


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


def build_optimizer(model, cfg):
    """AdamW with no weight decay on biases and normalization parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def build_scheduler(opt, cfg):
    """Linear warmup then cosine decay to min_lr_frac of the peak, in optimizer steps."""
    total = cfg.epochs * cfg.steps_per_epoch

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        prog = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def describe_model(model):
    """Parameter count of every top-level submodule and in total."""
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters {total / 1e6:.2f}M", flush=True)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            print(f"   {name:<20s} {n / 1e6:7.3f}M  ({100.0 * n / total:5.1f}%)", flush=True)


class EMA:
    """Exponential moving average of the weights. Validation and checkpoint selection use the
    averaged weights, which are much steadier than raw weights under set-prediction training."""

    def __init__(self, model, decay):
        self.decay = decay
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


# ---------------------------------------------------------------------------
# 3. Data checks and evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_labels(loader, device):
    """ODE residual of the stored fixed points under the stored parameters, first batch.

    A median above ~1e-3 means the edge parameters are misaligned with the fixed points or the
    labels are not fixed points of the model being trained; training on them would be pointless."""
    batch = to_device(next(iter(loader)), device)
    r = label_residual(batch)
    if r.numel() == 0:
        print("label check: no fixed points in the first batch", flush=True)
        return
    q99 = torch.quantile(r.float(), 0.99).item()
    print(f"label check: |relative ODE residual| median {r.median().item():.2e}  p99 {q99:.2e}"
          f"  fraction above 1e-3: {(r > 1e-3).float().mean().item():.3f}", flush=True)
    if r.median().item() > 1e-3:
        print("WARNING: stored fixed points do not satisfy the ODE under the stored parameters.",
              flush=True)


@torch.no_grad()
def evaluate_loader(model, criterion, loader, device, cfg):
    """Losses, overall metrics, grouped metrics and the existence calibration over a loader."""
    model.eval()
    acc = MetricAccumulator()
    calib = CalibrationTable()
    loss_sum, n_batches = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_ctx(cfg, device):
            out = model(batch)
        _, logs, aux = criterion(out, batch)
        for k, v in logs.items():
            loss_sum[k] = loss_sum.get(k, 0.0) + float(v)
        acc.update(batch_metrics(out, batch, aux, cfg.refine_steps), batch)
        calib.update(out, aux, batch)
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
    """Score that picks best.pt: decoded count accuracy plus recall of the true fixed points.

    Validation loss is not used. On the large model it was lowest at epoch 18 and then rose by
    0.38 while count accuracy, recall and position error all kept improving: the likelihood
    overfits (the count head turns overconfident) long before the decisions stop improving."""
    return overall["count_exact"] + overall["fp_recall"]


def save_checkpoint(path, model, ema, opt, sched, epoch, best, cfg):
    torch.save({
        "model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
        "sched": sched.state_dict(), "epoch": epoch, "best": best, "cfg": asdict(cfg),
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
def run_cache(cfg):
    build_cache(cfg.corpus_dir, cfg.cache_dir, cfg.max_fp)


def run_train(cfg):
    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_json(os.path.join(cfg.out_dir, "config.json"), asdict(cfg))

    train_ds, train_loader = make_loader(cfg, SPLIT_TRAIN, train=True)
    val_ds, val_loader = make_loader(cfg, SPLIT_VAL, train=False)
    describe_dataset(train_ds, "train")
    describe_dataset(val_ds, "val")
    print(f"device {device}", flush=True)
    check_labels(val_loader, device)

    model = SccFixedPointNet(cfg).to(device)
    criterion = SetCriterion(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    ema = EMA(model, cfg.ema_decay)
    describe_model(model)

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
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        window, n_window = {}, 0
        epoch_sum, n_epoch = {}, 0
        for it, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with autocast_ctx(cfg, device):
                out = model(batch)
            loss, logs, _ = criterion(out, batch)
            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} iteration {it}; step skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)

            logs["grad_norm"] = grad_norm.detach()
            for k, v in logs.items():
                window[k] = window.get(k, 0.0) + v
                epoch_sum[k] = epoch_sum.get(k, 0.0) + v
            n_window += 1
            n_epoch += 1
            if (it + 1) % cfg.log_every == 0:
                msg = " ".join(f"{k} {(v / n_window).item():.4f}" for k, v in window.items())
                print(f"epoch {epoch + 1} it {it + 1}/{cfg.steps_per_epoch} "
                      f"lr {sched.get_last_lr()[0]:.2e} | {msg}", flush=True)
                window, n_window = {}, 0

        with ema.applied(model):
            val_losses, overall, _, calib = evaluate_loader(model, criterion, val_loader, device, cfg)
        row = {"epoch": epoch + 1, "lr": sched.get_last_lr()[0]}
        row.update({f"train/loss_{k}": (v / max(n_epoch, 1)).item() for k, v in epoch_sum.items()})
        row.update({f"val/loss_{k}": v for k, v in val_losses.items()})
        row.update({f"val/{k}": v for k, v in overall.items()})
        row["val/exist_ece"] = calib["ece"]
        print(f"epoch {epoch + 1} | val loss {val_losses['total']:.4f} | {format_metrics(overall)}"
              f" | ece {calib['ece']:.4f}", flush=True)
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
    ck = torch.load(path, map_location=device)

    model = SccFixedPointNet(cfg).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})
    criterion = SetCriterion(cfg)

    ds, loader = make_loader(cfg, SPLITS[cfg.eval_split], train=False, max_samples=cfg.eval_max_samples)
    describe_dataset(ds, cfg.eval_split)
    losses, overall, grouped, calib = evaluate_loader(model, criterion, loader, device, cfg)

    print(f"checkpoint {path} (epoch {ck['epoch']})")
    print("loss    : " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()))
    print("metrics : " + format_metrics(overall))
    print(f"existence ECE {calib['ece']:.4f}")
    out_path = os.path.join(cfg.out_dir, f"eval_{cfg.eval_split}.json")
    write_json(out_path, {
        "checkpoint": path, "epoch": ck["epoch"], "split": cfg.eval_split, "samples": len(ds),
        "loss": losses, "overall": overall, "by_group": grouped, "calibration": calib,
    })
    print(f"wrote {out_path}")


def score_sets(roots, truth, k, cons, meta):
    """Per-sample comparison of a refined root set with the labels, one row per sample:
    n, phase, true count, true points recovered, roots found, roots matching no label,
    certified. A root recovers a true point when every coordinate agrees within 0.1%."""
    rows = np.zeros((len(roots), 7))
    for b, r in enumerate(roots):
        t = truth[b, :k[b]]
        if r.shape[0] > 0:
            d = np.abs(np.log(np.maximum(t, 1e-300))[:, None, :] - np.log(r)[None, :, :]).max(axis=-1)
            hit, extra = int((d.min(axis=1) < 1e-3).sum()), int((d.min(axis=0) >= 1e-3).sum())
        else:
            hit, extra = 0, 0
        rows[b] = [meta[b, 0], meta[b, 1], k[b], hit, r.shape[0], extra, cons[b]]
    return rows


def summarize_sets(a):
    """set_exact, recall of true points, mean unlabeled roots per sample, certified share."""
    exact = ((a[:, 3] == a[:, 2]) & (a[:, 4] == a[:, 2])).mean()
    return exact, a[:, 3].sum() / max(a[:, 2].sum(), 1.0), a[:, 5].mean(), a[:, 6].mean()


def run_polish(cfg):
    """The network as a seeder for the solver's Newton method.

    For every sample, Newton (damped, log coordinates, 60 steps, the generator's solver) is
    started from the K predicted slot positions, converged points are merged, and the set is
    certified by the Brouwer index sum. The same is done from the same number of random
    seeds, so the table shows what the network adds at an equal budget. Existence
    probabilities and the count head are not used: every slot is a seed."""
    set_seed(cfg.seed)
    device = get_device()
    path = cfg.ckpt or os.path.join(cfg.out_dir, "best.pt")
    ck = torch.load(path, map_location=device)
    model = SccFixedPointNet(cfg).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["ema"].items()})
    model.eval()
    ds, loader = make_loader(cfg, SPLITS[cfg.eval_split], train=False, max_samples=cfg.eval_max_samples)
    describe_dataset(ds, cfg.eval_split)
    scfg = SolverConfig()
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    rows = {"network": [], "random": []}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with autocast_ctx(cfg, device):
                out = model(batch)
            pos = torch.nan_to_num(decode_position(out["pos"].float(), batch["xmax"].unsqueeze(1)), nan=1.0)
            for n in batch["n"].unique().tolist():
                sel = (batch["n"] == n).nonzero()[:, 0]
                par = {"P": batch["P"][sel][:, :n].double().cpu().numpy(),
                       "K": batch["K"][sel][:, :n].double().cpu().numpy(),
                       "adj": batch["adj"][sel][:, :n, :n].cpu().numpy().astype(np.int8)}
                for key in ("T", "nh", "F"):
                    par[key] = batch[key][sel][:, :n, :n].double().cpu().numpy()
                p = prepare_params(par, scfg.eps_box, device)
                box = p["xmax"].unsqueeze(1)
                starts = {"network": torch.minimum(torch.maximum(pos[sel][:, :, :n].double(), 1e-12 * box), box),
                          "random": make_seeds(p, pos.shape[1], gen, 0)}
                truth = batch["fp_state"][sel][:, :, :n].double().cpu().numpy()
                k = batch["k"][sel].cpu().numpy()
                meta = np.stack([np.full(sel.numel(), n), batch["phase"][sel].cpu().numpy()], axis=1)
                for name, x0 in starts.items():
                    x, r = newton(x0, p, scfg.newton_steps, scfg.tol)
                    ok, xs = (r < scfg.accept_tol).cpu().numpy(), x.cpu().numpy()
                    roots = [cluster_roots(xs[i][ok[i]]) if ok[i].any() else np.zeros((0, n))
                             for i in range(xs.shape[0])]
                    cons = root_properties(roots, p)["cons"]
                    rows[name].append(score_sets(roots, truth, k, cons, meta))

    print(f"checkpoint {path} (epoch {ck['epoch']}), {pos.shape[1]} seeds per sample")
    print("group           | seeds from | set_exact  recall  unlabeled/sample  certified")
    for label, col, values in [("all", None, [None])] + [("phase", 1, [0, 1, 2])] + [("n", 0, list(range(2, 11)))]:
        for v in values:
            for name in ("network", "random"):
                a = np.concatenate(rows[name])
                if col is not None:
                    a = a[a[:, col] == v]
                if a.shape[0] == 0:
                    continue
                s = summarize_sets(a)
                tag = label if v is None else f"{label} {v}"
                print(f"{tag:<15} | {name:<10} | {s[0]:9.3f} {s[1]:7.3f} {s[2]:16.3f} {s[3]:10.3f}   (samples {a.shape[0]})")


def main():
    cfg = parse_config()
    if cfg.mode == "cache":
        run_cache(cfg)
    elif cfg.mode == "train":
        run_train(cfg)
    elif cfg.mode == "eval":
        run_eval(cfg)
    elif cfg.mode == "polish":
        run_polish(cfg)
    else:
        raise ValueError("mode must be cache, train, eval or polish")


if __name__ == "__main__":
    main()
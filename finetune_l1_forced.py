"""Fine-tune Layer I on Corpus A plus forced-SCC samples, with its full set-prediction loss.

The decomposition makes every SCC of a feed-forward composition an autonomous SCC with
production P * u, but composition inputs push P * u up to ~2 decades outside the Corpus A
range. This fine-tune widens Layer I's training distribution to cover them
(gen_forced_scc.py), with the same loss as train.py (SetCriterion: existence, positions,
characteristic polynomial, stability, counts, physics residual), so the count is learned
together with the rest of the portrait instead of in isolation.

Selection: teacher-forced composition score on a tuning set (fraction of cases whose every
branch count, read from the count head with no Newton step, is right). The result is a
normal Layer I checkpoint (cfg / model / ema) usable as --l1_ckpt anywhere.

  python finetune_l1_forced.py --l1_ckpt .../layer1_v10/best.pt \\
      --corpus_a_cache .../corpus_exhaust_A_v2/cache --forced .../forced_s1.npz \\
      --tune_dir .../composition_train_strict_s2001 --out_dir .../layer1_forced_v1
"""
import argparse
import copy
import csv
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from composition_eval import attach_labels, load_cases
from data import SPLIT_TRAIN, SccDataset, collate
from losses import SetCriterion
from model import SccFixedPointNet
from train_count_net import BranchCount, branch_scores, collate_count, fmt_k


class ForcedSCC(Dataset):
    """gen_forced_scc.py output, as data.load_sample items."""

    def __init__(self, paths):
        parts = [np.load(p) for p in paths]
        self.d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}

    def __len__(self):
        return int(self.d["n"].shape[0])

    def __getitem__(self, i):
        d = self.d
        return {"n": int(d["n"][i]), "k": int(d["k"][i]),
                "P": d["P"][i], "K": d["K"][i], "xmax": d["xmax"][i],
                "adj": d["adj"][i], "T": d["T"][i], "nh": d["nh"][i], "F": d["F"][i],
                "fp_state": d["fp_state"][i], "fp_charpoly": d["fp_charpoly"][i],
                "fp_stab": d["fp_stab"][i], "has_pos_fb": bool(d["has_pos_fb"][i]),
                "consistent": True, "phase": 1}


class CountView(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        return self.net(batch)["count_logit"]


@torch.no_grad()
def tune_score(model, branches, device):
    from train_count_net import predict
    return branch_scores(*predict(CountView(model), branches, device), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1_ckpt", required=True)
    ap.add_argument("--corpus_a_cache", required=True)
    ap.add_argument("--forced", required=True, nargs="+", help="npz files from gen_forced_scc.py")
    ap.add_argument("--tune_dir", required=True, help="composition set for selection (not the test set)")
    ap.add_argument("--tune_cases", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mix", type=float, default=0.5, help="fraction of forced samples per batch")
    ap.add_argument("--kpower", type=float, default=0.0,
                    help="sample weight ~ n_fp ** kpower within each source (more high-count SCCs)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = SccFixedPointNet(cfg)
    model.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device)
    criterion = SetCriterion(cfg)

    a_train = SccDataset(args.corpus_a_cache, SPLIT_TRAIN, cfg.max_fp)
    forced = ForcedSCC(args.forced)
    train_ds = ConcatDataset([forced, a_train])
    def by_k(k, total):
        w = np.asarray(k, np.float64) ** args.kpower
        return total * w / w.sum()

    w = np.concatenate([by_k(forced.d["k"], args.mix), by_k(a_train.sample_nfp, 1 - args.mix)])
    loader = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, collate_fn=collate,
                        sampler=WeightedRandomSampler(torch.from_numpy(w),
                                                      args.steps_per_epoch * args.batch_size),
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    tune = BranchCount(attach_labels(load_cases(args.tune_dir, args.tune_cases)))
    print(f"forced samples {len(forced)} | Corpus A train {len(a_train)} | tune branches {len(tune)}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    amp = getattr(cfg, "amp", True) and device.type == "cuda"

    def report(m, tag):
        s, by_k = tune_score(m, tune, device)
        print(f"  [{tag}] tune case score {s:.4f} | {fmt_k(by_k)}", flush=True)
        return s

    best = report(model, "L1 original")
    torch.save({"cfg": ck["cfg"], "model": model.state_dict(), "ema": model.state_dict(),
                "tune_case_score": best, "epoch": 0}, out / "best.pt")
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, n_steps = {}, 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                outp = model(batch)
            loss, logs, _ = criterion(outp, batch)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg.ema_decay).add_(pm.detach(), alpha=1 - cfg.ema_decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_steps += 1
        msg = " ".join(f"{k} {v / max(n_steps, 1):.4f}" for k, v in sums.items())
        print(f"epoch {epoch} | {msg}", flush=True)
        s = report(ema, f"epoch {epoch} EMA")
        log.append({"epoch": epoch, "tune_case_score": s,
                    **{k: v / max(n_steps, 1) for k, v in sums.items()}})
        with open(out / "log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            wr.writeheader()
            wr.writerows(log)
        if s > best:
            best = s
            torch.save({"cfg": ck["cfg"], "model": ema.state_dict(), "ema": ema.state_dict(),
                        "tune_case_score": s, "epoch": epoch}, out / "best.pt")
            print(f"  saved best ({s:.4f})", flush=True)
    print(f"best tune case score {best:.4f}")


if __name__ == "__main__":
    main()

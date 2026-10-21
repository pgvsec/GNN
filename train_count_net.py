"""Train CountNet (count_net.py) on Corpus A plus composition branches.

Data
  Corpus A train split : autonomous SCCs with their fixed-point count (s_n_fp).
  Composition branches : every consistent branch of the --train_dirs cases, as an
                         autonomous SCC with production P * u (exact per-branch labels from
                         composition_labels). These carry the forcing range that composition
                         produces (u up to ~100x) and that Corpus A barely covers.
  Each batch draws a --mix fraction of composition branches.

Loss: cross-entropy over odd counts (a hyperbolic SCC has an odd number of fixed points;
samples with an even label are skipped). All Layer I weights train, the count head at
--head_lr and the rest at --lr; EMA weights are evaluated and saved.

Selection: fraction of validation composition cases whose every branch count is right
(the teacher-forced composition count_exact), on a --val_frac split of the train dirs with
the same seed as finetune_count_heads.py. The test dir is scored once at the end.

Usage (from REPO_ROOT):
  python train_count_net.py \\
    --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v10/best.pt \\
    --corpus_a_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \\
    --train_dirs .../composition_eval .../composition_eval_v3 ... \\
    --test_dir .../composition_eval_v5 --out_dir .../count_net_v1
Then: composition_eval.py ... --count_net .../count_net_v1/best.pt \\
        --variants cn:forced_model=l1,count_net=1,solver_fallback=0
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from assemble import _make_scc_batch
from composition_eval import attach_labels, load_cases
from count_net import CountNet
from data import N_MAX, SPLIT_TRAIN, SPLIT_VAL, SccDataset
from finetune_count_heads import clamp_odd, odd_logits

KEYS_1D = ("P", "K", "xmax")
KEYS_2D = ("adj", "T", "nh", "F")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class CorpusACount(Dataset):
    """Corpus A samples reduced to the count-network inputs."""

    def __init__(self, cache_dir, split, max_fp, max_samples=0, seed=0):
        self.ds = SccDataset(cache_dir, split, max_fp, max_samples=max_samples, seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        return {k: s[k] for k in ("n", "k", "has_pos_fb") + KEYS_1D + KEYS_2D} | {"case": -1}


class BranchCount(Dataset):
    """Composition branches as autonomous SCCs with P * u, padded to N_MAX."""

    def __init__(self, cases):
        self.items = []
        cpu = torch.device("cpu")
        for ci, c in enumerate(cases):
            for scc in c["dag"].sccs:
                for br in c["labels"][scc.scc_id]:
                    if not br.consistent:
                        continue
                    b = _make_scc_batch(scc, br.u, cpu, frame="forced", pad_to=N_MAX)
                    item = {k: b[k][0].numpy() for k in KEYS_1D + KEYS_2D}
                    item.update(n=scc.n, k=br.k, has_pos_fb=bool(b["has_pos_fb"][0]), case=ci)
                    self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_count(items):
    n = torch.tensor([b["n"] for b in items], dtype=torch.long)
    nb = int(n.max())
    out = {"n": n,
           "k": torch.tensor([b["k"] for b in items], dtype=torch.long),
           "case": torch.tensor([b["case"] for b in items], dtype=torch.long),
           "has_pos_fb": torch.tensor([b["has_pos_fb"] for b in items], dtype=torch.bool)}
    for key in KEYS_1D:
        out[key] = torch.from_numpy(np.stack([b[key] for b in items]))[:, :nb].contiguous()
    for key in KEYS_2D:
        out[key] = torch.from_numpy(np.stack([b[key] for b in items]))[:, :nb, :nb].contiguous()
    out["node_mask"] = torch.arange(nb).unsqueeze(0) < n.unsqueeze(1)
    return out


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, ds, device, bs=512):
    model.eval()
    preds, ks, cases = [], [], []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_count):
        b = to_device(batch, device)
        preds.append(odd_logits(model(b).float()).argmax(-1).cpu())
        ks.append(batch["k"])
        cases.append(batch["case"])
    return torch.cat(preds), torch.cat(ks), torch.cat(cases)


def branch_scores(pred, k, case, n_cases_ok):
    """(case score, per-k accuracy). Cases with an incomplete GT have no branches here, so
    the case score is over the cases that contribute branches."""
    ok = {}
    for p, kk, c in zip(pred.tolist(), k.tolist(), case.tolist()):
        ok[c] = ok.get(c, True) and p == kk
    by_k = {}
    for kk in sorted(set(k.tolist())):
        m = k == kk
        by_k[kk] = (float((pred[m] == kk).float().mean()), int(m.sum()))
    return (float(np.mean(list(ok.values()))) if ok else float("nan")), by_k


def fmt_k(by_k):
    return " ".join(f"k{k}:{a:.3f}({n})" for k, (a, n) in by_k.items() if n >= 5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1_ckpt", required=True)
    ap.add_argument("--corpus_a_cache", required=True)
    ap.add_argument("--train_dirs", required=True, nargs="+")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mix", type=float, default=0.5, help="fraction of composition branches per batch")
    ap.add_argument("--corpus_a_max", type=int, default=0, help="0 = whole train split")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model from the Layer I checkpoint
    ck = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = CountNet(cfg)
    model.net.load_state_dict(ck.get("ema", ck.get("model")))
    model.to(device)

    # Composition cases: same split as finetune_count_heads.py
    train_cases = []
    for d in args.train_dirs:
        train_cases += load_cases(d)
    perm = np.random.default_rng(args.seed).permutation(len(train_cases))
    n_val = int(round(args.val_frac * len(train_cases)))
    val_cases = [train_cases[i] for i in perm[:n_val]]
    train_cases = [train_cases[i] for i in perm[n_val:]]
    test_cases = load_cases(args.test_dir) if args.test_dir else []
    for cs in (train_cases, val_cases, test_cases):
        attach_labels(cs)
    br_train, br_val = BranchCount(train_cases), BranchCount(val_cases)
    br_test = BranchCount(test_cases) if test_cases else None
    a_train = CorpusACount(args.corpus_a_cache, SPLIT_TRAIN, cfg.max_fp, args.corpus_a_max, args.seed)
    a_val = CorpusACount(args.corpus_a_cache, SPLIT_VAL, cfg.max_fp, 20000, args.seed)
    print(f"branches: train {len(br_train)}  val {len(br_val)}  "
          f"test {0 if br_test is None else len(br_test)} | Corpus A train {len(a_train)}  val {len(a_val)}")

    train_ds = ConcatDataset([br_train, a_train])
    w = np.concatenate([np.full(len(br_train), args.mix / max(len(br_train), 1)),
                        np.full(len(a_train), (1 - args.mix) / max(len(a_train), 1))])
    sampler = WeightedRandomSampler(torch.from_numpy(w), args.steps_per_epoch * args.batch_size,
                                    replacement=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        collate_fn=collate_count, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)

    head = [p for n, p in model.named_parameters() if "count_head" in n and "stable" not in n]
    head_ids = {id(p) for p in head}
    rest = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=args.wd)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    def evaluate(m, tag):
        cs, by_k = branch_scores(*predict(m, br_val, device), len(val_cases))
        pa, ka, _ = predict(m, a_val, device)
        odd = ka % 2 == 1
        a_acc = float((pa[odd] == ka[odd]).float().mean())
        print(f"  [{tag}] val case score {cs:.4f} | Corpus A val {a_acc:.4f} | {fmt_k(by_k)}", flush=True)
        return cs, a_acc

    best, _ = evaluate(model, "L1 original")
    torch.save({"cfg": ck["cfg"], "model": model.state_dict(), "ema": model.state_dict(),
                "val_case_score": best, "epoch": 0}, out / "best.pt")
    log = []
    amp = bool(args.amp) and device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_steps = 0.0, 0
        for batch in loader:
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = odd_logits(model(b).float())
            keep = (b["k"] % 2 == 1).float()
            tgt = clamp_odd(b["k"], logits.shape[-1] - 1)
            loss = (F.cross_entropy(logits, tgt, reduction="none") * keep).sum() / keep.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            loss_sum += float(loss.detach())
            n_steps += 1
        print(f"epoch {epoch}  loss {loss_sum / max(n_steps, 1):.4f}", flush=True)
        cs, a_acc = evaluate(ema, f"epoch {epoch} EMA")
        log.append({"epoch": epoch, "loss": loss_sum / max(n_steps, 1), "val_case_score": cs,
                    "corpus_a_val": a_acc})
        with open(out / "log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            wr.writeheader()
            wr.writerows(log)
        if cs > best:
            best = cs
            torch.save({"cfg": ck["cfg"], "model": ema.state_dict(), "ema": ema.state_dict(),
                        "val_case_score": cs, "epoch": epoch}, out / "best.pt")
            print(f"  saved best ({cs:.4f})", flush=True)

    print(f"best val case score {best:.4f}")
    if br_test is not None:
        from count_net import load_count_net
        base = CountNet(cfg)
        base.net.load_state_dict(ck.get("ema", ck.get("model")))
        base.to(device)
        for tag, m in (("L1 original", base), ("CountNet best", load_count_net(out / "best.pt", device))):
            cs, by_k = branch_scores(*predict(m, br_test, device), len(test_cases))
            print(f"test [{tag}] case score {cs:.4f} | {fmt_k(by_k)}")


if __name__ == "__main__":
    main()

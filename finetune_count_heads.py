"""Fine-tune the count heads on exact per-branch labels from composition data.

What is trained
  l1.count_head          (StructuralCountHead)    — root SCCs (and forced SCCs if --forced_model l1)
  l3.forcing_count_head  (ForcingAwareCountHead)  — forced SCCs
Everything else stays frozen and in eval mode.

How
  1. Every composition case is decomposed into its SCC DAG and its GT fixed points are turned
     into exact per-branch labels (composition_labels.extract_branch_labels): one label per
     root SCC and one per (forced SCC, forcing context), each with its own count. Branches
     whose GT is visibly incomplete (even count, contradictory contexts) are dropped.
  2. Each branch is presented to the networks exactly as assemble.py presents it at
     inference (same batch builder, same frame, same Thomas class), with the forcing computed
     from the GT parent state. Forward hooks record the inputs of the count heads.
  3. Because the backbone is frozen and the heads read detached features, the heads are then
     trained on the cached inputs with ordinary minibatches. That makes a sweep over loss
     weightings and learning rates cheap; the configuration with the best validation score is
     kept.
  4. Selection score = fraction of validation cases whose every branch count is right under
     teacher-forced contexts. When every branch count is right the assembled global count is
     right, so this is the composition count_exact without parent-position error.

Optionally Corpus C branches are added to the L3 training set (--corpus_c_cache), and the
final heads are scored with the real assembly pipeline on the validation and test
directories (--eval_assemble).

Outputs in --out_dir
  count_heads.pt  head weights + options (use with composition_eval.py --heads)
  l1_best.pt, l3_best.pt  full checkpoints in the layout eval_composition2.py loads
  sweep.csv       validation score of every configuration

Usage (from REPO_ROOT):
  python finetune_count_heads.py \\
    --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v10/best.pt \\
    --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v12/best.pt \\
    --train_dirs .../composition_eval .../composition_eval_v3 .../composition_eval_v4 \\
    --val_frac 0.15 --test_dir .../composition_eval_v5 \\
    --out_dir .../count_ft_v3 --eval_assemble
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from assemble import InferenceOptions, _make_scc_batch
from composition_eval import (apply_heads, attach_labels, evaluate, load_cases, load_models,
                              print_table)


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------
class HeadInputs:
    """Records the positional inputs of a module's forward through a hook."""

    def __init__(self, module):
        self.rows = []
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.rows.append(tuple(a.detach().float().cpu() for a in args))

    def pop(self):
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        self.handle.remove()


class FeatureSet:
    """Cached head inputs with labels, for one head."""

    def __init__(self):
        self.inputs = []      # list of tuples of [1, ...] tensors (or [B, ...] for Corpus C)
        self.k = []
        self.case = []        # case index (-1 for Corpus C samples)

    def add(self, inputs, k, case):
        self.inputs.append(inputs)
        self.k.extend(k)
        self.case.extend(case)

    def tensors(self):
        if not self.inputs:
            return None
        cols = [torch.cat([row[i].reshape(row[i].shape[0], -1) if row[i].dim() > 1
                           else row[i].reshape(-1, 1) for row in self.inputs], 0)
                for i in range(len(self.inputs[0]))]
        return cols, torch.tensor(self.k, dtype=torch.long), torch.tensor(self.case, dtype=torch.long)


def _head_args(cols, idx, kind):
    """Reassemble head forward arguments from cached columns."""
    if kind == "l1":
        e, g, pf = (c[idx] for c in cols)
        return e, g, pf.reshape(-1)
    e, z, us, tc = (c[idx] for c in cols)
    return e, z, us, tc


@torch.no_grad()
def extract_composition(cases, l1, l3, opts, device, hooks, sets):
    """Run every labelled branch of every case through the model inference would use."""
    for ci, c in enumerate(cases):
        for scc in c["dag"].sccs:
            for br in c["labels"][scc.scc_id]:
                if not br.consistent:
                    continue
                use_l1 = br.is_root or opts.forced_model == "l1"
                frame = "forced" if use_l1 else opts.l3_frame
                batch = _make_scc_batch(scc, br.u, device, frame=frame)
                if not use_l1 and not opts.pass_tc:
                    batch.pop("tc")
                # An L3 forward also runs L1's heads, so clear both hooks around every call.
                hooks["l1"].pop()
                hooks["l3"].pop()
                if use_l1:
                    l1(batch)
                    sets["l1"].add(hooks["l1"].pop()[0], [br.k], [ci])
                else:
                    l3(batch, freeze_backbone=True)
                    sets["l3"].add(hooks["l3"].pop()[0], [br.k], [ci])


@torch.no_grad()
def extract_corpus_c(cache_dir, max_samples, l1, l3, opts, device, hooks, sets, batch_size=256):
    """Forced Corpus C branches through the same frame as inference."""
    from torch.utils.data import DataLoader
    from data_c import CorpusCDataset, apply_frame, collate_c
    ds = CorpusCDataset(cache_dir, l1.decoder.queries.num_embeddings, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_c,
                        num_workers=4)
    n = 0
    for batch in loader:
        forced = ~batch["is_root"]
        if not forced.any():
            continue
        batch = {k: v[forced] if torch.is_tensor(v) and v.shape[:1] == forced.shape else v
                 for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        k = batch["k"].tolist()
        hooks["l1"].pop()
        hooks["l3"].pop()
        if opts.forced_model == "l1":
            l1(apply_frame(batch, "forced"))
            sets["l1"].add(hooks["l1"].pop()[0], k, [-1] * len(k))
        else:
            b = apply_frame(batch, opts.l3_frame)
            if not opts.pass_tc:
                b = {kk: vv for kk, vv in b.items() if kk != "tc"}
            l3(b, freeze_backbone=True)
            sets["l3"].add(hooks["l3"].pop()[0], k, [-1] * len(k))
        n += len(k)
    return n


# ---------------------------------------------------------------------------
# 2. Head training on cached features
# ---------------------------------------------------------------------------
def odd_logits(logits):
    even = torch.arange(logits.shape[-1], device=logits.device) % 2 == 0
    return logits.masked_fill(even, float("-inf"))


def clamp_odd(k, max_count):
    k = k.clamp(1, max_count)
    return torch.where(k % 2 == 0, (k - 1).clamp_min(1), k)


def sample_weights(k, scheme, cap=10.0):
    k = k.float()
    if scheme == "none":
        w = torch.ones_like(k)
    elif scheme == "sqrt":
        w = k.sqrt()
    elif scheme == "linear":
        w = k
    elif scheme == "balanced":
        vals, inv, cnt = torch.unique(k, return_inverse=True, return_counts=True)
        w = (cnt.sum().float() / (cnt.float() * len(vals)))[inv]
    else:
        raise ValueError(scheme)
    w = w.clamp(max=cap * w.min().clamp_min(1e-6))
    return w / w.mean()


def predict(head, kind, cols, device, bs=4096):
    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, cols[0].shape[0], bs):
            idx = torch.arange(s, min(s + bs, cols[0].shape[0]))
            args = [a.to(device) for a in _head_args(cols, idx, kind)]
            out.append(odd_logits(head(*args).float()).argmax(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def train_head(head, kind, cols, k, w, device, epochs, lr, wd, bs, seed):
    torch.manual_seed(seed)
    head = copy.deepcopy(head).to(device)
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = k.shape[0]
    steps = max(1, epochs * ((n + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            args = [a.to(device) for a in _head_args(cols, idx, kind)]
            logits = odd_logits(head(*args).float())
            tgt = clamp_odd(k[idx].to(device), logits.shape[-1] - 1)
            loss = (F.cross_entropy(logits, tgt, reduction="none") * w[idx].to(device)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
    return head.eval()


def case_score(preds, case_ids_eval):
    """Fraction of cases whose every branch prediction is right, plus per-k branch accuracy."""
    ok = {c: True for c in case_ids_eval}
    by_k = {}
    for kind in ("l1", "l3"):
        if preds.get(kind) is None:
            continue
        p, k, case = preds[kind]
        for pi, ki, ci in zip(p.tolist(), k.tolist(), case.tolist()):
            if ci not in ok:
                continue
            hit = pi == ki
            ok[ci] &= hit
            a = by_k.setdefault(f"{kind}_k{ki}", [0, 0])
            a[0] += hit
            a[1] += 1
    score = float(np.mean(list(ok.values()))) if ok else float("nan")
    return score, {key: (v[0] / v[1], v[1]) for key, v in sorted(by_k.items())}


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1_ckpt", required=True)
    ap.add_argument("--l3_ckpt", required=True)
    ap.add_argument("--train_dirs", required=True, nargs="+")
    ap.add_argument("--val_dir", default=None,
                    help="validation set for selection; default: --val_frac of the train cases")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_dir", default=None, help="held out; scored once at the end")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--corpus_c_cache", default=None, help="add forced Corpus C branches to L3 training")
    ap.add_argument("--corpus_c_max", type=int, default=100000)
    # inference options, must match how the heads will be used
    ap.add_argument("--l3_frame", default="forced", choices=["forced", "raw"])
    ap.add_argument("--forced_model", default="l3", choices=["l3", "l1"])
    ap.add_argument("--pass_tc", type=int, default=1)
    # sweep
    ap.add_argument("--k_weights", nargs="+", default=["none", "sqrt", "linear", "balanced"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval_assemble", action="store_true",
                    help="score original and new heads with the full GRNPredictor at the end")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    opts = InferenceOptions(l3_frame=args.l3_frame, forced_model=args.forced_model,
                            pass_tc=bool(args.pass_tc))
    print(f"device {device}  options {opts}")

    l1, l3 = load_models(args.l1_ckpt, args.l3_ckpt, device)
    for p in list(l1.parameters()) + list(l3.parameters()):
        p.requires_grad_(False)

    # Data: train / val / test cases with exact branch labels
    train_cases = []
    for d in args.train_dirs:
        train_cases += load_cases(d)
    rng = np.random.default_rng(args.seed)
    if args.val_dir:
        val_cases = load_cases(args.val_dir)
    else:
        perm = rng.permutation(len(train_cases))
        n_val = int(round(args.val_frac * len(train_cases)))
        val_cases = [train_cases[i] for i in perm[:n_val]]
        train_cases = [train_cases[i] for i in perm[n_val:]]
    test_cases = load_cases(args.test_dir) if args.test_dir else []
    for name, cs in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
        attach_labels(cs)
        bad = sum(not c["gt_consistent"] for c in cs)
        print(f"{name}: {len(cs)} cases ({bad} with incomplete GT)")

    # Features
    hooks = {"l1": HeadInputs(l1.count_head), "l3": HeadInputs(l3.forcing_count_head)}
    feats = {}
    for name, cs in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
        sets = {"l1": FeatureSet(), "l3": FeatureSet()}
        extract_composition(cs, l1, l3, opts, device, hooks, sets)
        if name == "train" and args.corpus_c_cache:
            n_c = extract_corpus_c(args.corpus_c_cache, args.corpus_c_max, l1, l3, opts,
                                   device, hooks, sets)
            print(f"  + {n_c} Corpus C forced branches")
        feats[name] = {kind: s.tensors() for kind, s in sets.items()}
        print(f"  {name} branches: " + ", ".join(
            f"{kind}={0 if v is None else len(v[1])}" for kind, v in feats[name].items()))
    for h in hooks.values():
        h.close()

    def consistent_ids(cs):
        return [i for i, c in enumerate(cs) if c["gt_consistent"]]

    val_ids, test_ids = consistent_ids(val_cases), consistent_ids(test_cases)
    heads0 = {"l1": l1.count_head, "l3": l3.forcing_count_head}

    def score(heads, split, ids):
        preds = {}
        for kind, v in feats[split].items():
            if v is not None:
                cols, k, case = v
                preds[kind] = (predict(heads[kind], kind, cols, device), k, case)
        return case_score(preds, ids)

    base_val, base_val_k = score(heads0, "val", val_ids)
    print(f"\noriginal heads: val case score {base_val:.4f}")
    print("  per-branch accuracy by k:", {k: f"{a:.3f} (n={n})" for k, (a, n) in base_val_k.items()})

    # Sweep
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, best = [], (base_val, "original", heads0)
    for scheme, lr in itertools.product(args.k_weights, args.lrs):
        heads = {}
        for kind, v in feats["train"].items():
            if v is None:
                heads[kind] = heads0[kind]
                continue
            cols, k, _ = v
            heads[kind] = train_head(heads0[kind], kind, cols, k, sample_weights(k, scheme),
                                     device, args.epochs, lr, args.wd, args.batch_size, args.seed)
        s, by_k = score(heads, "val", val_ids)
        rows.append({"k_weight": scheme, "lr": lr, "val_case_score": s,
                     **{k: round(a, 4) for k, (a, _) in by_k.items()}})
        print(f"  {scheme:>8}  lr={lr:.0e}  val case score {s:.4f}")
        if s > best[0]:
            best = (s, f"{scheme}/lr={lr:g}", heads)
    with open(out / "sweep.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("k_weight", "lr", "val_case_score"), x))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    best_score, best_name, best_heads = best
    print(f"\nbest: {best_name}  val case score {best_score:.4f} (original {base_val:.4f})")
    if test_cases:
        t0, _ = score(heads0, "test", test_ids)
        t1, t1k = score(best_heads, "test", test_ids)
        print(f"test case score: original {t0:.4f} -> {t1:.4f}")
        print("  per-branch accuracy by k:", {k: f"{a:.3f} (n={n})" for k, (a, n) in t1k.items()})

    # Save
    state = {"l1_count_head": best_heads["l1"].state_dict(),
             "l3_count_head": best_heads["l3"].state_dict(),
             "opts": vars(opts), "selected": best_name, "val_case_score": best_score}
    torch.save(state, out / "count_heads.pt")
    l1_ck = torch.load(args.l1_ckpt, map_location="cpu", weights_only=False)
    l3_ck = torch.load(args.l3_ckpt, map_location="cpu", weights_only=False)
    apply_heads(l1, l3, state)
    torch.save({"cfg": l1_ck["cfg"], "model": l1.state_dict(), "ema": l1.state_dict()}, out / "l1_best.pt")
    torch.save({"cfg": l3_ck["cfg"], "model": l3.state_dict(), "ema": l3.state_dict()}, out / "l3_best.pt")
    print(f"saved to {out}")

    if args.eval_assemble:
        results = {}
        l1o, l3o = load_models(args.l1_ckpt, args.l3_ckpt, device)
        for split, cs in (("val", val_cases), ("test", test_cases)):
            if not cs:
                continue
            results[f"{split}_original"], _ = evaluate(l1o, l3o, cs, opts, device)
            results[f"{split}_finetuned"], _ = evaluate(l1, l3, cs, opts, device)
        print_table(results)
        with open(out / "assemble_eval.json", "w") as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic script for composition eval CSV output (no pandas required)."""
import argparse
import csv
import sys
from collections import defaultdict


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"Columns: {cols}\n")
        for row in reader:
            rows.append(row)

    if not rows:
        print("Empty CSV")
        return

    # Parse numeric fields
    def get(row, key, default=None):
        v = row.get(key, None)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    k_true  = [get(r, "k_true")  for r in rows]
    k_pred  = [get(r, "k_pred")  for r in rows]
    fp_hd   = [get(r, "fp_hd")   for r in rows]
    ce      = [get(r, "count_exact") for r in rows]
    n_sccs  = [get(r, "n_sccs")  for r in rows]

    N = len(rows)

    # ── 1. fp_hd by count_exact ──────────────────────────────────────────────
    print("=" * 60)
    print("1. fp_hd conditioned on count accuracy")
    print("=" * 60)
    hd_correct = [fp_hd[i] for i in range(N) if ce[i] is not None and fp_hd[i] is not None and ce[i] == 1.0]
    hd_wrong   = [fp_hd[i] for i in range(N) if ce[i] is not None and fp_hd[i] is not None and ce[i] == 0.0]
    print(f"  count correct (n={len(hd_correct)}): fp_hd mean={mean(hd_correct):.4f}  "
          f"min={min(hd_correct, default=float('nan')):.3f}  max={max(hd_correct, default=float('nan')):.3f}")
    print(f"  count wrong   (n={len(hd_wrong)}):   fp_hd mean={mean(hd_wrong):.4f}  "
          f"min={min(hd_wrong, default=float('nan')):.3f}  max={max(hd_wrong, default=float('nan')):.3f}")

    # ── 2. Count accuracy by n_sccs ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("2. Count accuracy by n_sccs")
    print("=" * 60)
    has_nsccs = any(v is not None for v in n_sccs)
    if not has_nsccs:
        print("  n_sccs column not present — showing overall only")
        correct = sum(1 for v in ce if v == 1.0)
        print(f"  Overall: {correct}/{N} = {correct/N:.4f}")
    else:
        buckets = defaultdict(list)
        for i in range(N):
            if ce[i] is None:
                continue
            ns = n_sccs[i]
            if ns is None:
                key = "?"
            elif ns >= 4:
                key = "4+"
            else:
                key = str(int(ns))
            buckets[key].append(ce[i])
        for key in sorted(buckets.keys()):
            vals = buckets[key]
            acc = mean(vals)
            print(f"  n_sccs={key}: n={len(vals):4d}  count_exact={acc:.4f}  "
                  f"correct={sum(1 for v in vals if v==1.0)}")

    # ── 3. Under vs over predictions ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("3. Count error direction (k_pred - k_true)")
    print("=" * 60)
    diffs = []
    for i in range(N):
        if k_true[i] is not None and k_pred[i] is not None and ce[i] == 0.0:
            diffs.append(int(k_pred[i]) - int(k_true[i]))
    if not diffs:
        print("  No count errors found.")
        return

    under = sum(1 for d in diffs if d < 0)
    over  = sum(1 for d in diffs if d > 0)
    total_err = len(diffs)
    print(f"  Total errors: {total_err}  under={under} ({under/total_err:.1%})  over={over} ({over/total_err:.1%})")
    print()
    # Distribution
    dist = defaultdict(int)
    for d in diffs:
        dist[d] += 1
    max_bar = 40
    max_cnt = max(dist.values())
    print(f"  {'diff':>6}  {'count':>5}  bar")
    for d in sorted(dist.keys()):
        cnt = dist[d]
        bar = "█" * int(cnt / max_cnt * max_bar)
        print(f"  {d:>6}  {cnt:>5}  {bar}")

    # Also show mean k_true for errors vs correct
    print()
    kt_err = [k_true[i] for i in range(N) if ce[i] == 0.0 and k_true[i] is not None]
    kt_ok  = [k_true[i] for i in range(N) if ce[i] == 1.0 and k_true[i] is not None]
    print(f"  mean k_true when wrong:   {mean(kt_err):.2f}")
    print(f"  mean k_true when correct: {mean(kt_ok):.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic script for composition eval CSV output.

Analyzes:
1. fp_hd conditioned on count_exact (correct vs wrong)
2. Count accuracy broken down by n_sccs
3. Fraction of count errors that are under- vs over-predictions
"""

import argparse
import sys
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Diagnose composition eval CSV output")
    parser.add_argument("--csv", required=True, help="Path to the composition eval CSV file")
    args = parser.parse_args()

    # Load CSV
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("CSV COLUMNS PRESENT")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        n_null = df[col].isna().sum()
        print(f"  {col!r:40s}  dtype={str(dtype):12s}  nulls={n_null}")
    print()

    # ------------------------------------------------------------------ #
    # Check required columns
    # ------------------------------------------------------------------ #
    required = {"k_true", "k_pred", "fp_hd", "count_exact"}
    missing = required - set(df.columns)
    if missing:
        print(f"WARNING: Missing expected columns: {missing}")
        print("Analysis may be partial.\n")

    # ------------------------------------------------------------------ #
    # Question 1: fp_hd conditioned on count_exact
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Q1: fp_hd CONDITIONED ON count_exact")
    print("=" * 60)
    if "fp_hd" in df.columns and "count_exact" in df.columns:
        df["_count_correct"] = df["count_exact"].astype(int)
        for label, mask in [("count_exact == 1 (correct)", df["_count_correct"] == 1),
                             ("count_exact == 0 (wrong)",   df["_count_correct"] == 0)]:
            sub = df.loc[mask, "fp_hd"].dropna()
            if len(sub) == 0:
                print(f"  {label}: no rows")
            else:
                print(f"  {label}:")
                print(f"    n       = {len(sub)}")
                print(f"    mean    = {sub.mean():.4f}")
                print(f"    median  = {sub.median():.4f}")
                print(f"    std     = {sub.std():.4f}")
                print(f"    min     = {sub.min():.4f}")
                print(f"    max     = {sub.max():.4f}")
    else:
        missing_q1 = {"fp_hd", "count_exact"} - set(df.columns)
        print(f"  SKIPPED — missing columns: {missing_q1}")
    print()

    # ------------------------------------------------------------------ #
    # Question 2: count accuracy by n_sccs
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Q2: COUNT ACCURACY BROKEN DOWN BY n_sccs")
    print("=" * 60)
    if "count_exact" in df.columns and "n_sccs" in df.columns:
        df["_n_sccs_bucket"] = df["n_sccs"].apply(lambda x: x if (pd.notna(x) and int(x) < 4) else "4+")
        bucket_order = ["1", "2", "3", "4+"]

        rows = []
        for bucket in bucket_order:
            mask = df["_n_sccs_bucket"].astype(str) == str(bucket)
            sub = df.loc[mask, "count_exact"].dropna()
            if len(sub) == 0:
                continue
            acc = sub.astype(int).mean()
            rows.append({
                "n_sccs": bucket,
                "n_rows": len(sub),
                "count_accuracy": acc,
                "n_correct": int(sub.astype(int).sum()),
                "n_wrong": int(len(sub) - sub.astype(int).sum()),
            })

        if rows:
            result_df = pd.DataFrame(rows)
            print(result_df.to_string(index=False))
        else:
            print("  No data after bucketing.")
    elif "count_exact" in df.columns and "n_sccs" not in df.columns:
        print("  NOTE: 'n_sccs' column not found. Showing overall count accuracy only.")
        acc = df["count_exact"].astype(int).mean()
        print(f"  Overall count accuracy: {acc:.4f}  (n={len(df)})")
    else:
        missing_q2 = {"count_exact"} - set(df.columns)
        print(f"  SKIPPED — missing columns: {missing_q2}")
    print()

    # ------------------------------------------------------------------ #
    # Question 3: under- vs over-predictions among count errors
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Q3: UNDER- vs OVER-PREDICTIONS AMONG COUNT ERRORS")
    print("=" * 60)
    if "k_true" in df.columns and "k_pred" in df.columns:
        # Errors are rows where k_pred != k_true
        errors = df[df["k_pred"].astype(int) != df["k_true"].astype(int)].copy()
        n_errors = len(errors)
        print(f"  Total count errors: {n_errors}  (out of {len(df)} rows)")

        if n_errors == 0:
            print("  No errors to analyze.")
        else:
            errors["_diff"] = errors["k_pred"].astype(int) - errors["k_true"].astype(int)
            n_under = (errors["_diff"] < 0).sum()
            n_over  = (errors["_diff"] > 0).sum()
            n_exact_diff_zero = (errors["_diff"] == 0).sum()  # shouldn't happen by definition

            print(f"  Under-predictions (k_pred < k_true): {n_under:5d}  ({100*n_under/n_errors:.1f}%)")
            print(f"  Over-predictions  (k_pred > k_true): {n_over:5d}  ({100*n_over/n_errors:.1f}%)")
            if n_exact_diff_zero > 0:
                print(f"  [Unexpected zero-diff errors:         {n_exact_diff_zero}]")

            print()
            print("  Distribution of prediction error (k_pred - k_true) among errors:")
            diff_counts = errors["_diff"].value_counts().sort_index()
            for diff_val, cnt in diff_counts.items():
                bar = "#" * min(int(cnt), 40)
                print(f"    {diff_val:+4d}: {cnt:5d}  {bar}")
    else:
        missing_q3 = {"k_true", "k_pred"} - set(df.columns)
        print(f"  SKIPPED — missing columns: {missing_q3}")
    print()
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

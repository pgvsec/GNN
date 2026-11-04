"""Filter a generated dataset to the cases where v8 (or any model) failed.

Usage:
    python filter_hard_negatives.py \
        --eval_csv  /path/to/eval_results.csv \
        --dataset_dir /path/to/source/dataset \
        --out_dir  /path/to/hard_neg_output

The eval CSV must have columns: case, count_exact  (produced by eval_composition2.py).
The dataset_dir must contain adj.npy, par.npy, K.npy, T.npy, nh.npy, F.npy,
fp_state.npy, fp_stab.npy, fp_ptr.npy, n_fp.npy.

Outputs the same set of .npy files in out_dir, containing only the cases
where count_exact == 0 (model predicted the wrong FP count).
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Filter dataset to hard-negative cases")
    ap.add_argument("--eval_csv",     required=True,
                    help="CSV output from eval_composition2.py")
    ap.add_argument("--dataset_dir",  required=True,
                    help="Source dataset directory (contains *.npy files)")
    ap.add_argument("--out_dir",      required=True,
                    help="Output directory for filtered dataset")
    ap.add_argument("--fp_hd_thresh", type=float, default=None,
                    help="Also include cases where fp_hd < this threshold "
                         "(in addition to count-wrong cases). Default: count-wrong only.")
    args = ap.parse_args()

    eval_csv  = Path(args.eval_csv)
    src_dir   = Path(args.dataset_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read failure indices from CSV ─────────────────────────────────────────
    fail_set = set()
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ce = float(row.get("count_exact", row.get("count_correct", 1)))
            fh = float(row.get("fp_hd", 1.0))
            case_idx = int(row["case"])
            if ce == 0:
                fail_set.add(case_idx)
            elif args.fp_hd_thresh is not None and fh < args.fp_hd_thresh:
                fail_set.add(case_idx)

    if not fail_set:
        print("No failures found in CSV — output directory will be empty.")
        return

    fail_idx = sorted(fail_set)
    fail_arr = np.array(fail_idx, dtype=np.int64)
    print(f"Hard negatives: {len(fail_idx)} / {fail_arr.max()+1} cases "
          f"({100*len(fail_idx)/(fail_arr.max()+1):.1f}%)")

    # ── Load and subset per-case arrays ──────────────────────────────────────
    for fname in ["adj.npy", "P.npy", "K.npy", "T.npy", "nh.npy", "F.npy"]:
        src = src_dir / fname
        if not src.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        arr = np.load(src)
        subset = arr[fail_arr]
        np.save(out_dir / fname, subset)
        print(f"  {fname}: {arr.shape} → {subset.shape}")

    # ── Subset FP arrays (variable-length, stored with ptr/n_fp) ─────────────
    for ptr_file, n_file, state_file, stab_file in [
        ("fp_ptr.npy", "n_fp.npy", "fp_state.npy", "fp_stab.npy"),
    ]:
        ptr_path   = src_dir / ptr_file
        n_fp_path  = src_dir / n_file
        state_path = src_dir / state_file
        stab_path  = src_dir / stab_file

        if not all(p.exists() for p in [ptr_path, n_fp_path, state_path]):
            print(f"  SKIP FP arrays (missing {ptr_file}/{n_file}/{state_file})")
            continue

        fp_ptr   = np.load(ptr_path)
        n_fp_arr = np.load(n_fp_path)
        fp_state = np.load(state_path)
        fp_stab  = np.load(stab_path) if stab_path.exists() else None

        new_state_list = []
        new_stab_list  = []
        new_n_fp       = []
        new_ptr        = []
        offset         = 0

        for ci in fail_idx:
            p0 = int(fp_ptr[ci])
            n  = int(n_fp_arr[ci])
            new_state_list.append(fp_state[p0:p0+n])
            if fp_stab is not None:
                new_stab_list.append(fp_stab[p0:p0+n])
            new_n_fp.append(n)
            new_ptr.append(offset)
            offset += n

        new_fp_state = np.concatenate(new_state_list) if new_state_list else np.empty((0,) + fp_state.shape[1:], dtype=fp_state.dtype)
        np.save(out_dir / state_file, new_fp_state)
        print(f"  {state_file}: {fp_state.shape} → {new_fp_state.shape}")

        if fp_stab is not None and new_stab_list:
            new_fp_stab = np.concatenate(new_stab_list)
            np.save(out_dir / stab_file, new_fp_stab)
            print(f"  {stab_file}: {fp_stab.shape} → {new_fp_stab.shape}")

        np.save(out_dir / n_file,   np.array(new_n_fp, dtype=n_fp_arr.dtype))
        np.save(out_dir / ptr_file, np.array(new_ptr,  dtype=fp_ptr.dtype))
        print(f"  {n_file}: {len(fail_idx)} entries, "
              f"total FPs={int(np.sum(new_n_fp))}")

    print(f"\nDone. Hard-negative dataset written to: {out_dir}")


if __name__ == "__main__":
    main()

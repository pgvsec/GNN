"""Merge two pre-built Corpus C caches into a single combined cache.

Usage:
    PYTHONPATH=/scratch/gilbreth/gupt1042/GNN \
    python merge_corpus_caches.py \
        --cache_a /scratch/gilbreth/gupt1042/corpus_c/cache \
        --cache_b /scratch/gilbreth/gupt1042/corpus_d/cache \
        --cache_out /scratch/gilbreth/gupt1042/corpus_cd/cache

The merge is index-safe: all sentinel pointer arrays and absolute-index
arrays in cache_b are shifted by the group totals of cache_a before
concatenation.
"""
import argparse
import json
import os

import numpy as np

from data_c import (ABS_INDEX_C, DATASET_KEYS_C, GROUP_ORDER_C,
                    SENTINEL_PTRS_C, SENTINEL_TGTS_C, _group_of_c)


def open_cache_raw(cache_dir):
    """Load all DATASET_KEYS_C arrays from a cache directory."""
    return {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
            for k in DATASET_KEYS_C}


def group_totals(a):
    """Extract group sizes from a loaded cache dict."""
    totals = {}
    for g in GROUP_ORDER_C:
        # Use the last element of any sentinel ptr whose source is this group,
        # or fall back to the length of a non-ptr array from the group.
        ptr_for_g = [k for k, src in SENTINEL_PTRS_C.items() if src == g]
        if ptr_for_g:
            # Sentinel ptr shape is (source_group + 1,): len - 1 = source group size.
            # Using [-1] would give the TARGET group total instead.
            totals[g] = len(a[ptr_for_g[0]]) - 1
        else:
            # find any array from this group
            candidates = [k for k in DATASET_KEYS_C
                          if k not in SENTINEL_PTRS_C and _group_of_c(k) == g]
            totals[g] = len(a[candidates[0]])
    return totals


def merge_caches_c(cache_a_dir, cache_b_dir, cache_out_dir):
    print(f"Loading cache A from {cache_a_dir}")
    a = open_cache_raw(cache_a_dir)
    print(f"Loading cache B from {cache_b_dir}")
    b = open_cache_raw(cache_b_dir)

    tot_a = group_totals(a)
    tot_b = group_totals(b)
    print("Cache A group sizes:", tot_a)
    print("Cache B group sizes:", tot_b)

    os.makedirs(cache_out_dir, exist_ok=True)

    for k in DATASET_KEYS_C:
        arr_a = a[k]
        arr_b = b[k]

        if k in SENTINEL_PTRS_C:
            # sentinel ptr: shape (src_group + 1,)
            # Merge by dropping leading 0 of B, adding offset = last value of A
            offset = int(arr_a[-1])
            merged = np.concatenate([arr_a, arr_b[1:] + offset])
        elif k in ABS_INDEX_C:
            # plain index into another group: add group offset from A
            ref_group = ABS_INDEX_C[k]
            offset = tot_a[ref_group]
            merged = np.concatenate([arr_a, arr_b + offset])
        else:
            merged = np.concatenate([arr_a, arr_b], axis=0)

        out_path = os.path.join(cache_out_dir, k + ".npy")
        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=merged.dtype, shape=merged.shape)
        mm[:] = merged
        mm.flush()
        del mm

    # Write meta.json
    merged_totals = {g: tot_a[g] + tot_b[g] for g in GROUP_ORDER_C}
    with open(os.path.join(cache_out_dir, "meta.json"), "w") as fh:
        json.dump(merged_totals, fh)

    print("\nMerged cache written to:", cache_out_dir)
    for g, n in merged_totals.items():
        print(f"  {g:10s}: {n:12d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_a",   required=True, help="First Corpus C cache dir")
    ap.add_argument("--cache_b",   required=True, help="Second Corpus C cache dir (e.g. Corpus D)")
    ap.add_argument("--cache_out", required=True, help="Output merged cache dir")
    args = ap.parse_args()
    merge_caches_c(args.cache_a, args.cache_b, args.cache_out)


if __name__ == "__main__":
    main()

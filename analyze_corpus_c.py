#!/usr/bin/env python3
"""
analyze_corpus_c.py  –  Corpus C health statistics
====================================================
Scans every .npz shard in corpusC/, aggregates arrays, and writes a
JSON summary that the companion HTML dashboard can load directly.

Usage
-----
  python analyze_corpus_c.py [--corpus_dir corpusC] [--out corpus_c_stats.json]

Output: corpus_c_stats.json (machine-readable; consumed by the dashboard).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ── constants matching the generator ──────────────────────────────────────────
N_MIN, N_MAX = 2, 10
TC_NAMES = {1: "TC_NEG", 2: "TC_POS", 3: "TC_BOTH"}
STAB_NAMES = {
    0: "stable node",
    1: "stable spiral",
    2: "unstable node",
    3: "unstable spiral",
    4: "saddle",
    5: "non-hyperbolic",
}
STAB_STABLE = {0, 1}   # classes counted as "stable attractor"
# Keep in sync with corpus_c_generator.FAMILIES / SHAPE_NAMES (not imported: this script
# deliberately needs only numpy).
FAMILY_NAMES = {0: "chain", 1: "tree", 2: "reconvergent", 3: "fanin", 4: "multiroot", 5: "random"}
SHAPE_NAMES = {0: "chain", 1: "tree", 2: "merge", 3: "reconvergent"}
ZEROEIG_NEAR_SN = 1e-3   # min|lambda|/rho below this counts as near a saddle-node


def load_shards(corpus_dir: Path) -> dict:
    """Load all .npz shards and concatenate arrays."""
    shards = sorted(corpus_dir.glob("corpus_c_shard_*.npz"))
    if not shards:
        sys.exit(f"No corpus_c_shard_*.npz files found in {corpus_dir}")

    print(f"Found {len(shards)} shards in {corpus_dir}")

    # We accumulate lists then concatenate
    acc = {
        # per-DAG
        "dag_n_sccs":    [],
        "dag_n_edges":   [],
        # per-DAG structure labels (schema v2; absent in v1 shards)
        "dag_family":    [],
        "dag_shape":     [],
        "dag_depth":     [],
        "dag_max_fanin": [],
        "dag_n_reconv":  [],
        "dag_n_roots":   [],
        "dag_all_solved": [],
        # per-SCC
        "scc_n":         [],
        "scc_tc":        [],
        "scc_topo_rank": [],
        "scc_dag_id":    [],   # local dag id – used for cross-joins only
        "scc_depth":     [],
        "scc_n_parents": [],
        "scc_truncated": [],
        # inter-SCC edges
        "iedge_src_scc": [],
        "iedge_tgt_scc": [],
        "iedge_T":       [],
        "iedge_nh":      [],
        "iedge_F":       [],
        # branches
        "branch_scc":    [],
        "branch_cons":   [],
        "branch_bsum":   [],
        # fixed points
        "bfp_stab":      [],
        "bfp_hopf":      [],
        "bfp_zeroeig":   [],
        "bfp_resid":     [],
    }

    shard_rows: list[dict] = []

    for shard_path in shards:
        z = np.load(shard_path, allow_pickle=False)

        n_dags  = z["dag_n_sccs"].shape[0]
        n_sccs  = z["scc_n"].shape[0]
        n_iedge = z["iedge_T"].shape[0]
        n_br    = z["branch_scc"].shape[0]
        n_bfp   = z["bfp_stab"].shape[0]

        shard_rows.append({
            "file":    shard_path.name,
            "n_dags":  int(n_dags),
            "n_sccs":  int(n_sccs),
            "n_iedge": int(n_iedge),
            "n_br":    int(n_br),
            "n_bfp":   int(n_bfp),
        })

        for key in acc:
            if key in z:
                acc[key].append(z[key])

    # concatenate
    cats: dict[str, np.ndarray] = {}
    for key, parts in acc.items():
        if parts:
            cats[key] = np.concatenate(parts, axis=0)
        else:
            cats[key] = np.array([])

    return cats, shard_rows


def histogram(arr: np.ndarray, lo: float, hi: float, n_bins: int = 30) -> dict:
    """Return a compact histogram dict {bins, counts}."""
    counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bins":   [round(float(c), 5) for c in centers],
        "counts": [int(c) for c in counts],
    }


def value_counts(arr: np.ndarray, label_map: dict | None = None) -> list[dict]:
    """Return sorted [{label, count}] for an integer array."""
    vals, cnts = np.unique(arr.astype(int), return_counts=True)
    out = []
    for v, c in zip(vals, cnts):
        label = label_map[int(v)] if (label_map and int(v) in label_map) else str(int(v))
        out.append({"label": label, "value": int(v), "count": int(c)})
    return out


def percentiles(arr: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    if len(arr) == 0:
        return {str(q): None for q in qs}
    return {str(q): round(float(np.percentile(arr, q)), 6) for q in qs}


def compute_stats(cats: dict, shard_rows: list[dict]) -> dict:
    """Compute all statistics from concatenated arrays."""
    dag_n_sccs  = cats["dag_n_sccs"]
    dag_n_edges = cats["dag_n_edges"]
    scc_n       = cats["scc_n"]
    scc_tc      = cats["scc_tc"]
    scc_topo    = cats["scc_topo_rank"]
    iedge_T     = cats["iedge_T"]
    iedge_nh    = cats["iedge_nh"]
    iedge_F     = cats["iedge_F"]
    bfp_stab    = cats["bfp_stab"]
    bfp_resid   = cats["bfp_resid"]
    bfp_hopf    = cats["bfp_hopf"]
    bfp_zeroeig = cats["bfp_zeroeig"]
    branch_scc  = cats["branch_scc"]

    n_dags  = int(len(dag_n_sccs))
    n_sccs  = int(len(scc_n))
    n_iedge = int(len(iedge_T))
    n_br    = int(len(branch_scc))
    n_bfp   = int(len(bfp_stab))

    # ── corpus overview ──────────────────────────────────────────────────────
    overview = {
        "n_shards": len(shard_rows),
        "n_dags":   n_dags,
        "n_sccs":   n_sccs,
        "n_inter_edges": n_iedge,
        "n_branches": n_br,
        "n_fixed_points": n_bfp,
    }

    # ── DAG structure ────────────────────────────────────────────────────────
    dag_stats = {
        "sccs_per_dag_hist": histogram(dag_n_sccs, 1, int(dag_n_sccs.max()) + 1, 20)
            if n_dags > 0 else None,
        "edges_per_dag_hist": histogram(dag_n_edges, 0, int(dag_n_edges.max()) + 1, 20)
            if n_dags > 0 else None,
        "sccs_per_dag_pct":  percentiles(dag_n_sccs) if n_dags > 0 else {},
        "edges_per_dag_pct": percentiles(dag_n_edges) if n_dags > 0 else {},
        "frac_single_scc": float(np.mean(dag_n_sccs == 1)) if n_dags > 0 else None,
        "frac_multi_scc":  float(np.mean(dag_n_sccs > 1)) if n_dags > 0 else None,
    }

    # ── SCC statistics ────────────────────────────────────────────────────────
    scc_stats = {
        "n_dist": value_counts(scc_n),
        "tc_dist": value_counts(scc_tc, TC_NAMES),
        "topo_rank_hist": histogram(scc_topo, 0, int(scc_topo.max()) + 1, 15)
            if n_sccs > 0 else None,
        # roots are SCCs without a parent; topo rank 0 is only one of them in multi-root DAGs
        "frac_root_sccs": (float(np.mean(cats["scc_n_parents"] == 0))
                           if len(cats["scc_n_parents"]) == n_sccs and n_sccs > 0
                           else (float(np.mean(scc_topo == 0)) if n_sccs > 0 else None)),
    }

    # SCC size × Thomas class cross-tab
    if n_sccs > 0:
        cross = {}
        for tc_val, tc_name in TC_NAMES.items():
            mask = scc_tc == tc_val
            counts_by_n = {}
            for n_val in range(N_MIN, N_MAX + 1):
                counts_by_n[str(n_val)] = int(np.sum(mask & (scc_n == n_val)))
            cross[tc_name] = counts_by_n
        scc_stats["n_x_tc_cross"] = cross

    # ── Inter-SCC edge statistics ─────────────────────────────────────────────
    if n_iedge > 0:
        # Hill threshold T
        T_hist = histogram(iedge_T, float(iedge_T.min()), float(iedge_T.max()), 30)
        # Hill cooperativity nh
        nh_hist = histogram(iedge_nh, float(iedge_nh.min()), float(iedge_nh.max()), 20)
        # fold-change F: F>1 = activation, F<1 = repression, F=1 = neutral
        F_hist = histogram(iedge_F, float(iedge_F.min()), float(iedge_F.max()), 30)
        frac_act = float(np.mean(iedge_F > 1.0))
        frac_rep = float(np.mean(iedge_F < 1.0))
    else:
        T_hist = nh_hist = F_hist = None
        frac_act = frac_rep = None

    iedge_stats = {
        "T_hist":      T_hist,
        "nh_hist":     nh_hist,
        "F_hist":      F_hist,
        "frac_activation": frac_act,
        "frac_repression": frac_rep,
        "edges_pct":   percentiles(iedge_T) if n_iedge > 0 else {},
    }

    # ── Branch / fixed-point health ──────────────────────────────────────────
    # Branches per SCC (count unique branch_scc entries per scc index)
    if n_br > 0 and n_sccs > 0:
        br_per_scc = np.bincount(branch_scc, minlength=n_sccs)
        br_per_scc_hist = histogram(br_per_scc, 0, int(br_per_scc.max()) + 1, 20)
    else:
        br_per_scc_hist = None

    stab_dist = value_counts(bfp_stab, STAB_NAMES) if n_bfp > 0 else []
    n_stable = int(np.sum(np.isin(bfp_stab, list(STAB_STABLE)))) if n_bfp > 0 else 0
    frac_stable = float(n_stable / n_bfp) if n_bfp > 0 else None

    resid_log = np.log10(bfp_resid + 1e-20) if n_bfp > 0 else np.array([])
    resid_hist = histogram(resid_log, float(resid_log.min()), float(resid_log.max()), 30) \
        if n_bfp > 0 else None

    # Hopf / zero-eigenvalue flags
    frac_hopf = float(np.mean(bfp_hopf > 0)) if n_bfp > 0 else None
    # zeroeig = min|lambda| / spectral radius (solver.py), so "> 0" is always true. Report the
    # fraction close to a saddle-node (zero eigenvalue) instead, plus the distribution.
    frac_zeig = float(np.mean(bfp_zeroeig < ZEROEIG_NEAR_SN)) if n_bfp > 0 else None

    fp_stats = {
        "stab_dist":         stab_dist,
        "frac_stable":       frac_stable,
        "n_stable":          n_stable,
        "resid_log10_hist":  resid_hist,
        "resid_pct":         percentiles(bfp_resid) if n_bfp > 0 else {},
        "frac_hopf":         frac_hopf,
        "frac_zero_eig":     frac_zeig,
        "zero_eig_pct":      percentiles(bfp_zeroeig) if n_bfp > 0 else {},
        "branches_per_scc_hist": br_per_scc_hist,
    }

    # ── Per-shard summary (for the shard health panel) ────────────────────────
    shards_sorted = sorted(shard_rows, key=lambda r: r["file"])

    # ── Schema v2: DAG shape / depth strata, completeness, truncation ────────
    structure_stats = {}
    if n_dags > 0 and len(cats["dag_shape"]) == n_dags:
        depth = cats["dag_depth"]
        structure_stats.update({
            "shape_dist":       value_counts(cats["dag_shape"], SHAPE_NAMES),
            "family_dist":      value_counts(cats["dag_family"], FAMILY_NAMES),
            "depth_dist":       value_counts(depth),
            "max_fanin_dist":   value_counts(cats["dag_max_fanin"]),
            "frac_reconvergent": float(np.mean(cats["dag_n_reconv"] > 0)),
            "frac_depth_ge_6":  float(np.mean(depth >= 6)),
            "frac_fanin_ge_3":  float(np.mean(cats["dag_max_fanin"] >= 3)),
            "frac_multi_root":  float(np.mean(cats["dag_n_roots"] > 1)),
            "frac_all_solved":  float(np.mean(cats["dag_all_solved"])),
        })
    if n_br > 0 and len(cats["branch_cons"]) == n_br:
        structure_stats["frac_branch_cons"] = float(np.mean(cats["branch_cons"]))
        structure_stats["bsum_dist"] = value_counts(cats["branch_bsum"])
    if n_sccs > 0 and len(cats["scc_truncated"]) == n_sccs:
        structure_stats["frac_scc_truncated"] = float(np.mean(cats["scc_truncated"]))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overview":     overview,
        "dag_stats":    dag_stats,
        "scc_stats":    scc_stats,
        "iedge_stats":  iedge_stats,
        "fp_stats":     fp_stats,
        "structure_stats": structure_stats,
        "shard_rows":   shards_sorted,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Corpus C and produce a JSON stats file.")
    parser.add_argument("--corpus_dir", default="corpusC",
                        help="Directory containing shard_*.npz files (default: corpusC)")
    parser.add_argument("--out", default="corpus_c_stats.json",
                        help="Output JSON path (default: corpus_c_stats.json)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        sys.exit(f"corpus_dir {corpus_dir!r} does not exist")

    t0 = time.time()
    cats, shard_rows = load_shards(corpus_dir)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    stats = compute_stats(cats, shard_rows)
    print(f"  Stats computed in {time.time()-t1:.2f}s")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"  Written → {out_path} ({out_path.stat().st_size // 1024} KB)")

    # Print a quick summary to stdout
    ov = stats["overview"]
    print(f"\n=== Corpus C overview ===")
    print(f"  Shards:         {ov['n_shards']}")
    print(f"  DAGs:           {ov['n_dags']:,}")
    print(f"  SCCs:           {ov['n_sccs']:,}")
    print(f"  Inter-edges:    {ov['n_inter_edges']:,}")
    print(f"  Branches:       {ov['n_branches']:,}")
    print(f"  Fixed points:   {ov['n_fixed_points']:,}")
    st = stats["structure_stats"]
    if st:
        pct = lambda d: "  ".join(f"{r['label']} {100 * r['count'] / max(ov['n_dags'], 1):.1f}%" for r in d)
        print(f"  Shapes:         {pct(st['shape_dist'])}")
        print(f"  Depth:          {pct(st['depth_dist'])}")
        print(f"  Depth >= 6:     {st['frac_depth_ge_6']*100:.1f}%   fan-in >= 3: {st['frac_fanin_ge_3']*100:.1f}%"
              f"   multi-root: {st['frac_multi_root']*100:.1f}%")
        print(f"  Unsolved DAGs:  {(1 - st['frac_all_solved'])*100:.1f}% (some SCC has no branch)")
        if "frac_branch_cons" in st:
            print(f"  Branches with complete FP set (Brouwer): {st['frac_branch_cons']*100:.1f}%")
        if "frac_scc_truncated" in st:
            print(f"  SCCs with subsampled branches: {st['frac_scc_truncated']*100:.2f}%")
    fp = stats["fp_stats"]
    if fp["frac_stable"] is not None:
        print(f"  Stable FPs:     {fp['n_stable']:,} ({fp['frac_stable']*100:.1f}%)")


if __name__ == "__main__":
    main()

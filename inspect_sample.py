#!/usr/bin/env python3
"""
inspect_sample.py  –  Extract one DAG from a Corpus C shard → JSON
====================================================================
Usage
-----
  python3 inspect_sample.py --shard corpusC/corpus_c_shard_0003.npz --dag 7
  python3 inspect_sample.py --shard corpusC/corpus_c_shard_0003.npz --dag 7 --out sample.json

Writes sample.json (default).  Drag it onto corpus_c_dashboard.html to inspect.

Omit --dag to list all DAG IDs in the shard and exit.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_MAX = 10

STAB_NAMES = {
    0: "stable node", 1: "stable spiral",
    2: "unstable node", 3: "unstable spiral",
    4: "saddle", 5: "non-hyperbolic",
}
TC_NAMES = {1: "TC_NEG", 2: "TC_POS", 3: "TC_BOTH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="Path to shard_NNN.npz")
    ap.add_argument("--dag",   type=int, default=None, help="DAG index within shard (0-based)")
    ap.add_argument("--out",   default="sample.json", help="Output JSON path")
    args = ap.parse_args()

    shard_path = Path(args.shard)
    if not shard_path.exists():
        sys.exit(f"Shard not found: {shard_path}")

    z = np.load(shard_path, allow_pickle=False)

    n_dags = int(z["dag_n_sccs"].shape[0])
    print(f"Shard: {shard_path.name}  |  {n_dags} DAGs")

    if args.dag is None:
        print(f"\nDAG IDs available: 0 .. {n_dags-1}")
        print("Re-run with --dag <ID> to extract one.")
        sys.exit(0)

    dag_id = args.dag
    if dag_id < 0 or dag_id >= n_dags:
        sys.exit(f"--dag must be 0..{n_dags-1}, got {dag_id}")

    # ── DAG-level offsets ─────────────────────────────────────────────────
    dag_scc_ptr  = z["dag_scc_ptr"]   # [n_dags+1]
    dag_edge_ptr = z["dag_edge_ptr"]  # [n_dags+1]

    scc_start = int(dag_scc_ptr[dag_id])
    scc_end   = int(dag_scc_ptr[dag_id + 1])
    n_sccs_dag = scc_end - scc_start

    edge_start = int(dag_edge_ptr[dag_id])
    edge_end   = int(dag_edge_ptr[dag_id + 1])

    print(f"DAG {dag_id}: {n_sccs_dag} SCCs, {edge_end-edge_start} inter-SCC edges")

    # ── SCC data ──────────────────────────────────────────────────────────
    scc_n_arr    = z["scc_n"][scc_start:scc_end]
    scc_tc_arr   = z["scc_tc"][scc_start:scc_end]
    scc_rank_arr = z["scc_topo_rank"][scc_start:scc_end]
    scc_P_arr    = z["scc_P"][scc_start:scc_end]   # [n_sccs, N_MAX]
    scc_K_arr    = z["scc_K"][scc_start:scc_end]   # [n_sccs, N_MAX]
    scc_sign_arr = z["scc_sign"][scc_start:scc_end] # [n_sccs, N_MAX, N_MAX]
    scc_branch_ptr = z["scc_branch_ptr"]            # [n_total_sccs+1]

    # branch / FP data
    branch_scc = z["branch_scc"]       # [n_branches]
    branch_key = z["branch_key"]       # [n_branches] uint64
    branch_fp_ptr = z["branch_fp_ptr"] # [n_branches+1]

    bfp_x       = z["bfp_x"]
    bfp_charpoly= z["bfp_charpoly"]
    bfp_stab    = z["bfp_stab"]
    bfp_hopf    = z["bfp_hopf"]
    bfp_zeroeig = z["bfp_zeroeig"]
    bfp_resid   = z["bfp_resid"]

    sccs = []
    for local_i in range(n_sccs_dag):
        global_i = scc_start + local_i
        n        = int(scc_n_arr[local_i])
        tc       = int(scc_tc_arr[local_i])
        rank     = int(scc_rank_arr[local_i])
        P        = scc_P_arr[local_i, :n].tolist()
        K        = scc_K_arr[local_i, :n].tolist()
        sign_mat = scc_sign_arr[local_i, :n, :n].tolist()

        # branches for this SCC
        br_start = int(scc_branch_ptr[global_i])
        br_end   = int(scc_branch_ptr[global_i + 1])

        branches = []
        for bi in range(br_start, br_end):
            fp_start = int(branch_fp_ptr[bi])
            fp_end   = int(branch_fp_ptr[bi + 1])
            fps = []
            for fi in range(fp_start, fp_end):
                fps.append({
                    "x":        bfp_x[fi, :n].tolist(),
                    "charpoly": bfp_charpoly[fi, :n].tolist(),
                    "stab":     int(bfp_stab[fi]),
                    "stab_name":STAB_NAMES.get(int(bfp_stab[fi]), "?"),
                    "hopf":     float(bfp_hopf[fi]),
                    "zeroeig":  float(bfp_zeroeig[fi]),
                    "resid":    float(bfp_resid[fi]),
                })
            branches.append({
                "branch_idx": bi,
                "branch_key": int(branch_key[bi]),
                "fps": fps,
            })

        sccs.append({
            "local_id":  local_i,
            "global_id": global_i,
            "n":         n,
            "tc":        tc,
            "tc_name":   TC_NAMES.get(tc, "?"),
            "topo_rank": rank,
            "P":         P,
            "K":         K,
            "sign_matrix": sign_mat,
            "branches":  branches,
        })

    # ── inter-SCC edges ───────────────────────────────────────────────────
    edges = []
    if edge_end > edge_start:
        iedge_src_scc  = z["iedge_src_scc"][edge_start:edge_end]
        iedge_src_gene = z["iedge_src_gene"][edge_start:edge_end]
        iedge_tgt_scc  = z["iedge_tgt_scc"][edge_start:edge_end]
        iedge_tgt_gene = z["iedge_tgt_gene"][edge_start:edge_end]
        iedge_T        = z["iedge_T"][edge_start:edge_end]
        iedge_nh       = z["iedge_nh"][edge_start:edge_end]
        iedge_F        = z["iedge_F"][edge_start:edge_end]

        for ei in range(edge_end - edge_start):
            src_local = int(iedge_src_scc[ei]) - scc_start
            tgt_local = int(iedge_tgt_scc[ei]) - scc_start
            F = float(iedge_F[ei])
            edges.append({
                "src_scc":  src_local,
                "src_gene": int(iedge_src_gene[ei]),
                "tgt_scc":  tgt_local,
                "tgt_gene": int(iedge_tgt_gene[ei]),
                "T":        float(iedge_T[ei]),
                "nh":       float(iedge_nh[ei]),
                "F":        F,
                "type":     "activation" if F > 1 else ("repression" if F < 1 else "neutral"),
            })

    out = {
        "shard":   shard_path.name,
        "dag_id":  dag_id,
        "n_sccs":  n_sccs_dag,
        "n_edges": len(edges),
        "sccs":    sccs,
        "edges":   edges,
    }

    out_path = Path(args.out)
    # JSON doesn't allow NaN — replace with null
    import math
    def sanitize(obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    out_path.write_text(json.dumps(sanitize(out), indent=2))
    n_fps = sum(len(b["fps"]) for s in sccs for b in s["branches"])
    print(f"  {len(edges)} inter-SCC edges, "
          f"{sum(len(s['branches']) for s in sccs)} branches, "
          f"{n_fps} fixed points")
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()

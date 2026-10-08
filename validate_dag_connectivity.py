#!/usr/bin/env python3
"""
validate_dag_connectivity.py  –  Corpus C DAG connectivity audit
================================================================
For every DAG in every shard, checks whether the inter-SCC edge graph
forms a single weakly connected component across all SCCs in that DAG.

A DAG with k > 1 weakly connected components is BROKEN: it bundles k
independent sub-graphs into one training sample, which corrupts the
compositional training signal.

Usage
-----
  python3 validate_dag_connectivity.py --corpus_dir corpusC
  python3 validate_dag_connectivity.py --corpus_dir corpusC --out dag_audit.json
  python3 validate_dag_connectivity.py --corpus_dir corpusC --examples 5

Output
------
  Prints a summary to stdout.
  Writes dag_audit.json with per-DAG breakdown and worst examples.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_wccs(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-Find: return list of weakly connected components (as node lists)."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for u, v in edges:
        union(u, v)

    groups = defaultdict(list)
    for i in range(n_nodes):
        groups[find(i)].append(i)
    return list(groups.values())


def audit_shard(shard_path: Path) -> list[dict]:
    """Return one record per DAG in this shard."""
    z = np.load(shard_path, allow_pickle=False)

    dag_scc_ptr  = z["dag_scc_ptr"]   # [n_dags+1]
    dag_edge_ptr = z["dag_edge_ptr"]  # [n_dags+1]
    n_dags = int(dag_scc_ptr.shape[0]) - 1

    iedge_src_scc = z["iedge_src_scc"]  # global SCC indices
    iedge_tgt_scc = z["iedge_tgt_scc"]

    records = []
    for dag_id in range(n_dags):
        scc_start  = int(dag_scc_ptr[dag_id])
        scc_end    = int(dag_scc_ptr[dag_id + 1])
        n_sccs     = scc_end - scc_start

        edge_start = int(dag_edge_ptr[dag_id])
        edge_end   = int(dag_edge_ptr[dag_id + 1])
        n_edges    = edge_end - edge_start

        # Build local edge list (0-based local SCC indices)
        local_edges = []
        for ei in range(edge_start, edge_end):
            src_local = int(iedge_src_scc[ei]) - scc_start
            tgt_local = int(iedge_tgt_scc[ei]) - scc_start
            local_edges.append((src_local, tgt_local))

        wccs = find_wccs(n_sccs, local_edges)
        n_components = len(wccs)

        # Classify isolated roots/leaves per component
        component_sizes = sorted([len(c) for c in wccs], reverse=True)

        # Count isolated SCCs (no inter-SCC edges at all)
        connected_sccs = set()
        for u, v in local_edges:
            connected_sccs.add(u)
            connected_sccs.add(v)
        n_isolated = n_sccs - len(connected_sccs)

        records.append({
            "shard":        shard_path.name,
            "dag_id":       dag_id,
            "n_sccs":       n_sccs,
            "n_edges":      n_edges,
            "n_components": n_components,
            "component_sizes": component_sizes,
            "n_isolated_sccs": n_isolated,
            "broken":       n_components > 1,
        })

    return records


def main():
    ap = argparse.ArgumentParser(description="Audit Corpus C DAG weak connectivity.")
    ap.add_argument("--corpus_dir", default="corpusC")
    ap.add_argument("--out", default="dag_audit.json")
    ap.add_argument("--examples", type=int, default=10,
                    help="Number of broken DAG examples to include in JSON output")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    shards = sorted(corpus_dir.glob("corpus_c_shard_*.npz"))
    if not shards:
        sys.exit(f"No corpus_c_shard_*.npz found in {corpus_dir}")

    print(f"Scanning {len(shards)} shards...")
    t0 = time.time()

    all_records = []
    for shard_path in shards:
        recs = audit_shard(shard_path)
        all_records.extend(recs)
        broken = sum(1 for r in recs if r["broken"])
        print(f"  {shard_path.name}: {len(recs)} DAGs, {broken} broken")

    elapsed = time.time() - t0

    # ── Aggregate statistics ──────────────────────────────────────────────────
    n_total      = len(all_records)
    n_broken     = sum(1 for r in all_records if r["broken"])
    n_multi_scc  = sum(1 for r in all_records if r["n_sccs"] > 1)
    n_broken_multi = sum(1 for r in all_records if r["broken"] and r["n_sccs"] > 1)

    # Component count distribution among broken DAGs
    comp_dist = defaultdict(int)
    for r in all_records:
        if r["broken"]:
            comp_dist[r["n_components"]] += 1

    # Isolated SCC counts
    total_sccs      = sum(r["n_sccs"] for r in all_records)
    total_isolated  = sum(r["n_isolated_sccs"] for r in all_records)

    # Worst examples (most components, then most SCCs)
    broken_examples = sorted(
        [r for r in all_records if r["broken"]],
        key=lambda r: (-r["n_components"], -r["n_sccs"])
    )[:args.examples]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CORPUS C DAG CONNECTIVITY AUDIT")
    print(f"{'='*55}")
    print(f"  Shards scanned:          {len(shards)}")
    print(f"  Total DAGs:              {n_total:,}")
    print(f"  Multi-SCC DAGs:          {n_multi_scc:,}")
    print(f"  Broken (disconnected):   {n_broken:,}  ({100*n_broken/n_total:.1f}% of all DAGs)")
    if n_multi_scc > 0:
        print(f"  Broken among multi-SCC:  {n_broken_multi:,}  ({100*n_broken_multi/n_multi_scc:.1f}% of multi-SCC DAGs)")
    print(f"  Total SCCs:              {total_sccs:,}")
    print(f"  Isolated SCCs (no edges):{total_isolated:,}  ({100*total_isolated/total_sccs:.1f}% of all SCCs)")
    print(f"\n  Component count distribution (broken DAGs only):")
    for k in sorted(comp_dist):
        print(f"    {k} components: {comp_dist[k]:,} DAGs")
    print(f"\n  Worst examples:")
    for r in broken_examples[:5]:
        print(f"    {r['shard']}  DAG {r['dag_id']:4d}:  "
              f"{r['n_sccs']} SCCs, {r['n_edges']} edges → "
              f"{r['n_components']} components {r['component_sizes']}")
    print(f"\n  Scan time: {elapsed:.1f}s")
    print(f"{'='*55}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_dir": str(corpus_dir),
        "summary": {
            "n_shards":           len(shards),
            "n_dags":             n_total,
            "n_multi_scc_dags":   n_multi_scc,
            "n_broken_dags":      n_broken,
            "frac_broken":        round(n_broken / n_total, 6) if n_total else 0,
            "n_broken_multi_scc": n_broken_multi,
            "frac_broken_multi":  round(n_broken_multi / n_multi_scc, 6) if n_multi_scc else 0,
            "total_sccs":         total_sccs,
            "total_isolated_sccs": total_isolated,
            "frac_isolated_sccs": round(total_isolated / total_sccs, 6) if total_sccs else 0,
            "component_count_dist": {str(k): v for k, v in sorted(comp_dist.items())},
        },
        "broken_examples": broken_examples,
        # Full per-DAG table (can be large — set broken_only=True to slim it)
        "all_dags": all_records,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten → {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

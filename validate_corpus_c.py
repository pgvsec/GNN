#!/usr/bin/env python3
"""validate_corpus_c.py - structural and physical audit of Corpus C shards (schema v2).

For every shard checks the contract a Stage 3 consumer relies on:

  pointers     all *_ptr arrays are monotone and end at the array lengths
  roots        a root SCC has exactly one branch and no parent links
  links        a non-root branch has one link per parent SCC, in parent order; each link
               points at a branch of that parent SCC and at a STABLE fixed point inside it
  forcing      branch_u equals the product of Hill signals recomputed from the linked parent
               fixed points and the inter-SCC edge table (this is what teacher forcing uses)
  lineage      the lineage rebuilt from the links is conflict-free and hashes to branch_key;
               branches of one SCC have distinct keys
  fixed points residual below FP_RESID_TOL, no non-hyperbolic points
  labels       dag_shape / depth / max_fanin / n_reconv / n_roots match a recomputation from
               the edge table; dag_all_solved matches the branch counts

Usage
  python3 validate_corpus_c.py --corpus_dir corpusC [--max_errors 20]
Exit status is 1 when any check fails.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from corpus_c_generator import (
    FP_RESID_TOL, SCHEMA_VERSION, SHAPE_NAMES, DagTopology, dag_labels, hill, lineage_hash,
)

STABLE = (0, 1)   # stab codes: stable node, stable spiral


class Report:
    def __init__(self, max_errors):
        self.max_errors = max_errors
        self.errors = Counter()
        self.shown = 0

    def fail(self, kind, msg):
        self.errors[kind] += 1
        if self.shown < self.max_errors:
            print(f"  FAIL [{kind}] {msg}")
            self.shown += 1


def check_shard(path, rep):
    z = np.load(path, allow_pickle=False)
    tag = path.name
    if "schema_version" not in z or int(z["schema_version"]) != SCHEMA_VERSION:
        rep.fail("schema", f"{tag}: schema_version is not {SCHEMA_VERSION}")
        return None

    n_dag, n_scc = z["dag_n_sccs"].shape[0], z["scc_n"].shape[0]
    n_br, n_bfp = z["branch_scc"].shape[0], z["bfp_stab"].shape[0]
    n_link = z["link_parent_branch"].shape[0]
    dag_ptr, scc_bptr = z["dag_scc_ptr"], z["scc_branch_ptr"]
    bfp_ptr, link_ptr = z["branch_fp_ptr"], z["branch_link_ptr"]
    for name, ptr, end in (("dag_scc_ptr", dag_ptr, n_scc), ("scc_branch_ptr", scc_bptr, n_br),
                           ("branch_fp_ptr", bfp_ptr, n_bfp), ("branch_link_ptr", link_ptr, n_link),
                           ("dag_edge_ptr", z["dag_edge_ptr"], z["iedge_T"].shape[0])):
        if ptr[0] != 0 or ptr[-1] != end or np.any(np.diff(ptr) < 0):
            rep.fail("pointers", f"{tag}: {name} is not monotone from 0 to {end}")
            return None

    sign_scc = z["branch_scc"]
    edges_into = defaultdict(list)                       # target scc -> iedge rows
    for e in range(z["iedge_T"].shape[0]):
        edges_into[int(z["iedge_tgt_scc"][e])].append(e)

    lineages = {}                                        # branch -> lineage dict
    stable_total = 0
    for d in range(n_dag):
        s0, s1 = int(dag_ptr[d]), int(dag_ptr[d + 1])
        parents = {s: [] for s in range(s0, s1)}
        for s in range(s0, s1):
            for e in edges_into[s]:
                p = int(z["iedge_src_scc"][e])
                if p not in parents[s]:
                    parents[s].append(p)

        for s in range(s0, s1):                          # SCCs are stored in topological order
            b0, b1 = int(scc_bptr[s]), int(scc_bptr[s + 1])
            n_par = len(parents[s])
            if int(z["scc_n_parents"][s]) != n_par:
                rep.fail("labels", f"{tag}: scc {s} scc_n_parents != edge table")
            if n_par == 0 and b1 - b0 != 1:
                rep.fail("roots", f"{tag}: root scc {s} has {b1 - b0} branches, expected 1")
            keys = set()
            for b in range(b0, b1):
                lk0, lk1 = int(link_ptr[b]), int(link_ptr[b + 1])
                if lk1 - lk0 != n_par:
                    rep.fail("links", f"{tag}: branch {b} has {lk1 - lk0} links, scc has {n_par} parents")
                    continue
                lin_parts, u_ref = [], np.ones(z["branch_u"].shape[1])
                for (pb, fi), p_scc in zip(zip(z["link_parent_branch"][lk0:lk1], z["link_parent_fp"][lk0:lk1]),
                                           parents[s]):
                    pb, fi = int(pb), int(fi)
                    if not (0 <= pb < n_br) or int(sign_scc[pb]) != p_scc:
                        rep.fail("links", f"{tag}: branch {b} link to branch {pb} not in parent scc {p_scc}")
                        break
                    row = int(bfp_ptr[pb]) + fi
                    if fi < 0 or row >= int(bfp_ptr[pb + 1]):
                        rep.fail("links", f"{tag}: branch {b} link fp {fi} outside parent branch {pb}")
                        break
                    if int(z["bfp_stab"][row]) not in STABLE:
                        rep.fail("links", f"{tag}: branch {b} is forced by a non-stable parent fp (row {row})")
                    lin = dict(lineages.get(pb, {}))
                    lin[p_scc - s0] = fi          # lineage keys are DAG-local SCC indices
                    lin_parts.append(lin)
                    x_src = z["bfp_x"][row]
                    for e in edges_into[s]:
                        if int(z["iedge_src_scc"][e]) == p_scc:
                            g = int(z["iedge_tgt_gene"][e])
                            u_ref[g] *= hill(float(x_src[int(z["iedge_src_gene"][e])]),
                                             float(z["iedge_T"][e]), float(z["iedge_nh"][e]),
                                             float(z["iedge_F"][e]))
                else:
                    merged, ok = {}, True
                    for lin in lin_parts:
                        for k_, v_ in lin.items():
                            if merged.setdefault(k_, v_) != v_:
                                ok = False
                    if not ok:
                        rep.fail("lineage", f"{tag}: branch {b} mixes two states of one ancestor")
                    lineages[b] = merged
                    if lineage_hash(merged) != int(z["branch_key"][b]):
                        rep.fail("lineage", f"{tag}: branch {b} branch_key does not match its lineage")
                    if int(z["branch_key"][b]) in keys:
                        rep.fail("lineage", f"{tag}: scc {s} has two branches with the same key")
                    keys.add(int(z["branch_key"][b]))
                    if not np.allclose(z["branch_u"][b], u_ref, rtol=1e-3, atol=1e-6):
                        rep.fail("forcing", f"{tag}: branch {b} branch_u differs from the recomputed Hill signal")
                    continue
                lineages.setdefault(b, {})
            if n_par == 0 and b1 > b0:
                lineages[b0] = {}

    if n_bfp:
        if float(z["bfp_resid"].max()) >= FP_RESID_TOL:
            rep.fail("fixed_points", f"{tag}: bfp_resid max {float(z['bfp_resid'].max()):.2e} >= tolerance")
        if np.any(z["bfp_stab"] == 5):
            rep.fail("fixed_points", f"{tag}: non-hyperbolic fixed points present")
    stable_total = int(np.isin(z["bfp_stab"], STABLE).sum())

    # DAG labels recomputed from the edge table
    for d in range(n_dag):
        s0, s1 = int(dag_ptr[d]), int(dag_ptr[d + 1])
        k = s1 - s0
        parents = [[] for _ in range(k)]
        e0, e1 = int(z["dag_edge_ptr"][d]), int(z["dag_edge_ptr"][d + 1])
        for e in range(e0, e1):
            i, j = int(z["iedge_src_scc"][e]) - s0, int(z["iedge_tgt_scc"][e]) - s0
            if i >= j:
                rep.fail("labels", f"{tag}: dag {d} edge {i}->{j} is not forward in topological order")
            if i not in parents[j]:
                parents[j].append(i)
        lab = dag_labels(DagTopology(k=k, parents=parents, inter_edges=[], topo_order=list(range(k))))
        for key, name in (("shape", "dag_shape"), ("depth", "dag_depth"), ("max_fanin", "dag_max_fanin"),
                          ("n_reconv", "dag_n_reconv"), ("n_roots", "dag_n_roots")):
            if int(z[name][d]) != lab[key]:
                rep.fail("labels", f"{tag}: dag {d} {name} stored {int(z[name][d])}, recomputed {lab[key]}")
        solved = all(int(scc_bptr[s + 1]) > int(scc_bptr[s]) for s in range(s0, s1))
        if bool(z["dag_all_solved"][d]) != solved:
            rep.fail("labels", f"{tag}: dag {d} dag_all_solved disagrees with the branch counts")

    return {
        "dags": n_dag, "sccs": n_scc, "branches": n_br, "fps": n_bfp, "stable_fps": stable_total,
        "cons": int(z["branch_cons"].sum()), "truncated_sccs": int(z["scc_truncated"].sum()),
        "unsolved_dags": int((~z["dag_all_solved"]).sum()),
        "shape": Counter(int(v) for v in z["dag_shape"]),
        "depth": Counter(int(v) for v in z["dag_depth"]),
        "fanin": Counter(int(v) for v in z["dag_max_fanin"]),
        "tc": Counter(int(v) for v in z["scc_tc"]),
        "branches_per_scc": Counter(int(v) for v in np.diff(scc_bptr)),
    }


def main():
    ap = argparse.ArgumentParser(description="Audit Corpus C shards (schema v2)")
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--max_errors", type=int, default=20, help="failures printed in full")
    args = ap.parse_args()

    shards = sorted(Path(args.corpus_dir).glob("corpus_c_shard_*.npz"))
    if not shards:
        sys.exit(f"no corpus_c_shard_*.npz under {args.corpus_dir}")
    rep = Report(args.max_errors)
    tot = Counter()
    dist = {k: Counter() for k in ("shape", "depth", "fanin", "tc", "branches_per_scc")}
    for path in shards:
        r = check_shard(path, rep)
        if r is None:
            continue
        for k, v in r.items():
            if k in dist:
                dist[k].update(v)
            else:
                tot[k] += v

    print(f"\n{len(shards)} shards  {tot['dags']} DAGs  {tot['sccs']} SCCs  {tot['branches']} branches  "
          f"{tot['fps']} fixed points ({tot['stable_fps']} stable)")
    if tot["branches"]:
        print(f"  branches with complete fixed-point set (Brouwer sum 1): {100 * tot['cons'] / tot['branches']:.1f}%")
    print(f"  SCCs with subsampled branches: {tot['truncated_sccs']}   DAGs with an unsolved SCC: {tot['unsolved_dags']}")
    n = max(tot["dags"], 1)
    print("  shape: " + "  ".join(f"{SHAPE_NAMES[k]} {100 * v / n:.1f}%" for k, v in sorted(dist["shape"].items())))
    print("  depth: " + "  ".join(f"{k}:{100 * v / n:.1f}%" for k, v in sorted(dist["depth"].items())))
    print("  max fan-in: " + "  ".join(f"{k}:{100 * v / n:.1f}%" for k, v in sorted(dist["fanin"].items())))
    ns = max(tot["sccs"], 1)
    print("  Thomas class (1 neg, 2 pos, 3 both): " + "  ".join(f"{k}:{100 * v / ns:.1f}%" for k, v in sorted(dist["tc"].items())))

    if rep.errors:
        print("\nFAILED: " + ", ".join(f"{k} x{v}" for k, v in sorted(rep.errors.items())))
        sys.exit(1)
    print("\nPASSED all structural and physical checks")


if __name__ == "__main__":
    main()

"""Overlap between composition datasets: identical cases and shared 3-gene SCC blocks.

A case is identified by its full parameter set (adjacency, P, K, T, nh, F rounded to 5
significant digits); a block by the parameters of one SCC with its intra-SCC edges. Use it
before trusting a test score: a test case that also sits in a training dir is memorised,
not predicted.

  python check_overlap.py --test DIR --train DIR [DIR ...]
"""
import argparse
from pathlib import Path

import numpy as np

from assemble import decompose_grn


def _key(*arrays):
    return tuple(np.concatenate([np.asarray(a, np.float64).ravel() for a in arrays]).round(5).tolist())


def keys(d):
    d = Path(d)
    a = {k: np.load(d / f"{k}.npy") for k in ("adj", "P", "K", "T", "nh", "F")}
    cases, blocks = [], []
    for i in range(len(a["adj"])):
        P = np.float64(a["P"][i])
        sig = lambda x: np.array([float(f"{v:.5g}") for v in np.ravel(x)])
        cases.append(_key(a["adj"][i], sig(P), sig(a["K"][i]), sig(a["T"][i]), a["nh"][i], sig(a["F"][i])))
        dag = decompose_grn(a["adj"][i], a["P"][i], a["K"][i], a["T"][i], a["nh"][i], a["F"][i])
        blocks.append([_key(s.adj_intra, sig(s.P), sig(s.K), sig(s.T), s.nh, sig(s.F)) for s in dag.sccs])
    return cases, blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True, nargs="+")
    args = ap.parse_args()
    t_cases, t_blocks = keys(args.test)
    all_c, all_b = set(), set()
    for d in args.train:
        c, b = keys(d)
        sc, sb = set(c), {k for bl in b for k in bl}
        all_c |= sc
        all_b |= sb
        print(f"{d}: identical cases {sum(k in sc for k in t_cases)}/{len(t_cases)}  "
              f"test cases sharing a block {sum(any(k in sb for k in bl) for bl in t_blocks)}/{len(t_cases)}")
    print(f"ALL TRAIN: identical cases {sum(k in all_c for k in t_cases)}/{len(t_cases)}  "
          f"test cases sharing a block {sum(any(k in all_b for k in bl) for bl in t_blocks)}/{len(t_cases)}")


if __name__ == "__main__":
    main()

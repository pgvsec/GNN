"""Composition evaluation with inference ablations and error attribution.

Runs GRNPredictor over a composition eval directory (gen_composition_eval.py layout) for one
or more inference variants and reports the three target metrics with the exact definitions
of exhaust_gen/eval_composition2.py:

  count_exact : fraction of cases whose predicted global FP count equals the GT count
  fp_hd       : fraction of GT FPs with a predicted FP within half a decade in every gene,
                weighted by the GT count of each case
  stable_acc  : per-case stable/unstable agreement (stable_binary_acc), averaged over cases

Every metric is also reported on the cases whose GT passes the consistency check of
composition_labels (no even branch, no contradictory branch), and errors are attributed to
the root SCC, to a forced SCC, or to both, using the exact per-branch labels.

Variants are given as name:key=value,... with keys from assemble.InferenceOptions:
  l3_frame=forced|raw  forced_model=l3|l1  pass_tc=0|1  solver_fallback=0|1
  rescue=0|1  dedup=linear|log|none|fill  dedup_tol=<decades>  refine=<Newton steps>  parity_drop=0|1  count_net=0|1 (needs --count_net)
  head-only count (pure ML): rescue=0,dedup=none,parity_drop=0

Usage (from REPO_ROOT, on the cluster):
  python composition_eval.py \\
      --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v10/best.pt \\
      --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v12/best.pt \\
      --eval_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v5 \\
      --out_dir /scratch/gilbreth/gupt1042/runs/ablate_v5 \\
      --variants old:pass_tc=0 tc:pass_tc=1 tc_raw:pass_tc=1,l3_frame=raw \\
                 l1_forced:forced_model=l1 tc_nofallback:pass_tc=1,solver_fallback=0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from assemble import INFER_STATS, GRNPredictor, InferenceOptions, decompose_grn
from composition_labels import _source_genes, case_consistent, extract_branch_labels

REPO = Path(__file__).resolve().parent


def _load_metric_fns():
    """fp_within_half_decade and stable_binary_acc from eval_composition2.py, loaded by file
    path so the definitions are literally the ones the reported numbers used."""
    spec = importlib.util.spec_from_file_location(
        "_eval_composition2", REPO / "exhaust_gen" / "eval_composition2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fp_within_half_decade, mod.stable_binary_acc


fp_within_half_decade, stable_binary_acc = _load_metric_fns()


# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------
def load_models(l1_ckpt, l3_ckpt, device, heads=None):
    """L1 and L3 from their checkpoints. `heads` optionally points to a file written by
    finetune_count_heads.py (count_heads.pt) whose count-head weights replace the originals."""
    from model import SccFixedPointNet
    from model_c import CorpusCNet
    ck1 = torch.load(l1_ckpt, map_location=device, weights_only=False)
    l1 = SccFixedPointNet(SimpleNamespace(**ck1["cfg"]))
    l1.load_state_dict(ck1.get("ema", ck1.get("model")))
    ck3 = torch.load(l3_ckpt, map_location=device, weights_only=False)
    l3 = CorpusCNet(l1, SimpleNamespace(**ck3["cfg"]))
    l3.load_state_dict(ck3.get("ema", ck3.get("model")), strict=False)
    if heads:
        apply_heads(l1, l3, torch.load(heads, map_location=device, weights_only=False))
    return l1.to(device).eval(), l3.to(device).eval()


def apply_heads(l1, l3, state):
    if "l1_count_head" in state:
        l1.count_head.load_state_dict(state["l1_count_head"])
    if "l3_count_head" in state:
        l3.forcing_count_head.load_state_dict(state["l3_count_head"])


def load_cases(eval_dir, max_cases=None):
    d = Path(eval_dir)
    a = {k: np.load(d / f"{k}.npy") for k in
         ("adj", "P", "K", "T", "nh", "F", "fp_ptr", "n_fp", "fp_state", "fp_stab")}
    n = len(a["adj"]) if max_cases is None else min(len(a["adj"]), max_cases)
    cases = []
    for i in range(n):
        f0 = int(a["fp_ptr"][i])
        f1 = f0 + int(a["n_fp"][i])
        cases.append({
            "idx": i,
            "adj": a["adj"][i],
            "P": a["P"][i].astype(np.float32), "K": a["K"][i].astype(np.float32),
            "T": a["T"][i].astype(np.float32), "nh": a["nh"][i].astype(np.float32),
            "F": a["F"][i].astype(np.float32),
            "fp_state": a["fp_state"][f0:f1], "fp_stab": a["fp_stab"][f0:f1],
            "n_fp_gt": int(a["n_fp"][i]),
        })
    return cases


def attach_labels(cases):
    """Adds dag, per-branch labels and the GT consistency flag to every case."""
    for c in cases:
        c["dag"] = decompose_grn(c["adj"], c["P"], c["K"], c["T"], c["nh"], c["F"])
        c["labels"] = extract_branch_labels(c["dag"], c["fp_state"], c["T"], c["nh"], c["F"])
        c["gt_consistent"] = case_consistent(c["labels"])
    return cases


def parse_variant(spec, count_net=None):
    name, _, kv = spec.partition(":")
    opts = InferenceOptions()
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        if k in ("pass_tc", "solver_fallback", "rescue", "parity_drop"):
            setattr(opts, k, v.lower() in ("1", "true", "yes"))
        elif k in ("l3_frame", "forced_model", "dedup"):
            setattr(opts, k, v)
        elif k == "dedup_tol":
            opts.dedup_tol = float(v)
        elif k == "refine":
            opts.refine_steps = int(v)
        elif k == "count_net":
            if v.lower() in ("1", "true", "yes"):
                if count_net is None:
                    raise ValueError(f"variant {spec!r} needs --count_net")
                opts.count_model = count_net
        else:
            raise ValueError(f"unknown option {k!r} in variant {spec!r}")
    return name, opts


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def _match(x, pts, tol_dec=0.5):
    """Index of the point in pts [m, n] within tol_dec decades of x in every gene, or -1."""
    if len(pts) == 0:
        return -1
    d = np.abs(np.log10(np.clip(pts, 1e-12, None)) - np.log10(np.clip(x, 1e-12, None))[None]).max(-1)
    j = int(d.argmin())
    return j if d[j] <= tol_dec else -1


def attribute(case, result):
    """('root', 'forced', 'both' or 'none') for the SCC-level count errors of one case.

    Root SCCs: predicted number of branch FPs vs the GT root count.
    Forced SCCs with one parent: for each predicted parent point that matches a GT parent
    point, the predicted number of child points vs the GT count of the branch that parent
    point forces. Unmatched parent points are themselves a root/parent error."""
    dag, labels, branches = case["dag"], case["labels"], result["all_branches"]
    root_bad = forced_bad = False
    for scc in dag.sccs:
        if not scc.parents:
            if len(branches.get(scc.scc_id, [])) != labels[scc.scc_id][0].k:
                root_bad = True
            continue
        if len(scc.parents) != 1:
            continue
        pid = scc.parents[0]
        gt_parent_pts = np.concatenate([b.x for b in labels[pid]], 0) if labels[pid] else np.zeros((0, 1))
        for pnode in branches.get(pid, []):
            j = _match(pnode.x_local, gt_parent_pts)
            if j < 0:
                continue
            # GT branch of the child under that parent point: match through the global state.
            x_glob = np.ones(dag.n_genes)
            x_glob[list(dag.sccs[pid].genes)] = gt_parent_pts[j]
            best, best_d = None, np.inf
            src = _source_genes(dag, scc.scc_id)
            for b in labels[scc.scc_id]:
                d = np.abs(np.log(np.clip(b.src_values, 1e-300, None)) -
                           np.log(np.clip(x_glob[src], 1e-300, None))).max()
                if d < best_d:
                    best, best_d = b, d
            if best is not None and len(pnode.children.get(scc.scc_id, [])) != best.k:
                forced_bad = True
    if root_bad and forced_bad:
        return "both"
    return "root" if root_bad else ("forced" if forced_bad else "none")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(l1, l3, cases, opts, device, refine_steps=5, verbose_every=0):
    """Metrics dict and per-case rows for one inference variant."""
    for k in INFER_STATS:
        INFER_STATS[k] = 0
    # refine_fps draws random restarts; seed so variants are compared on identical draws.
    torch.manual_seed(0)
    predictor = GRNPredictor(l1, l3, device=device, refine_steps=refine_steps, opts=opts)
    rows, errors = [], 0
    t0 = time.time()
    for n_done, c in enumerate(cases, 1):
        try:
            with torch.no_grad():
                r = predictor.predict(c["adj"], c["P"], c["K"], c["T"], c["nh"], c["F"],
                                      verbose=False)
        except Exception as exc:          # same policy as eval_composition2.py
            errors += 1
            if errors <= 5:
                print(f"  case {c['idx']}: {exc}")
            continue
        pred_fps = r.get("fps_refined", r["global_fps"])
        n_pred = int(r["n_fps"])
        rows.append({
            "case": c["idx"], "n_fp_gt": c["n_fp_gt"], "n_fp_pred": n_pred,
            "count_exact": int(n_pred == c["n_fp_gt"]),
            "fp_hd": fp_within_half_decade(pred_fps, c["fp_state"]),
            "stable_acc": stable_binary_acc(r["global_stab"], c["fp_stab"]),
            "n_sccs": len(c["dag"].sccs),
            "gt_consistent": int(c["gt_consistent"]),
            "blame": attribute(c, r) if n_pred != c["n_fp_gt"] else "none",
        })
        if verbose_every and n_done % verbose_every == 0:
            print(f"  {n_done}/{len(cases)}  {time.time() - t0:.0f}s", flush=True)
    m = summarize(rows)
    m.update({"errors": errors, **{f"n_{k}": v for k, v in INFER_STATS.items()}})
    return m, rows


def summarize(rows):
    def agg(rs):
        if not rs:
            return {"n": 0}
        ce = float(np.mean([r["count_exact"] for r in rs]))
        w = [(r["fp_hd"], r["n_fp_gt"]) for r in rs if not np.isnan(r["fp_hd"])]
        hd = sum(a * b for a, b in w) / max(sum(b for _, b in w), 1)
        sa = [r["stable_acc"] for r in rs if not np.isnan(r["stable_acc"])]
        return {"n": len(rs), "count_exact": ce, "fp_hd": hd,
                "stable_acc": float(np.mean(sa)) if sa else float("nan")}

    out = {"all": agg(rows), "gt_consistent": agg([r for r in rows if r["gt_consistent"]]),
           "count_right": agg([r for r in rows if r["count_exact"]])}
    wrong = [r for r in rows if not r["count_exact"]]
    out["blame"] = dict(Counter(r["blame"] for r in wrong))
    out["under"] = sum(r["n_fp_pred"] < r["n_fp_gt"] for r in wrong)
    out["over"] = sum(r["n_fp_pred"] > r["n_fp_gt"] for r in wrong)
    by_k = defaultdict(list)
    for r in rows:
        by_k[r["n_fp_gt"]].append(r["count_exact"])
    out["count_exact_by_gt_k"] = {k: (float(np.mean(v)), len(v)) for k, v in sorted(by_k.items())}
    return out


def print_table(results):
    print(f"\n{'variant':<18} {'count_exact':>11} {'fp_hd':>7} {'stable':>7} | "
          f"{'clean CE':>8} {'clean hd':>8} | {'under':>5} {'over':>4}  blame r/f/b    fallback  post up/down")
    for name, m in results.items():
        a, g, b = m["all"], m["gt_consistent"], m["blame"]
        print(f"{name:<18} {a['count_exact']:>11.4f} {a['fp_hd']:>7.4f} {a['stable_acc']:>7.4f} | "
              f"{g.get('count_exact', float('nan')):>8.4f} {g.get('fp_hd', float('nan')):>8.4f} | "
              f"{m['under']:>5} {m['over']:>4}  "
              f"{b.get('root', 0):>4}/{b.get('forced', 0)}/{b.get('both', 0):<5} "
              f"{m['n_solver_fallback']:>8}  {m['n_post_up']}/{m['n_post_down']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1_ckpt", required=True)
    ap.add_argument("--l3_ckpt", required=True)
    ap.add_argument("--heads", default=None, help="count_heads.pt from finetune_count_heads.py")
    ap.add_argument("--count_net", default=None, help="best.pt from train_count_net.py (count_net=1)")
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["default:"])
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--refine_steps", type=int, default=5)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    l1, l3 = load_models(args.l1_ckpt, args.l3_ckpt, device, args.heads)
    count_net = None
    if args.count_net:
        from count_net import load_count_net
        count_net = load_count_net(args.count_net, device)
    cases = attach_labels(load_cases(args.eval_dir, args.max_cases))
    n_bad = sum(not c["gt_consistent"] for c in cases)
    print(f"{len(cases)} cases, {n_bad} with visibly incomplete GT ({n_bad / max(len(cases), 1):.1%})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.variants:
        name, opts = parse_variant(spec, count_net)
        print(f"\n== {name}: {opts}", flush=True)
        m, rows = evaluate(l1, l3, cases, opts, device, args.refine_steps, verbose_every=100)
        results[name] = m
        if rows:
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(json.dumps(m, indent=1, default=str))
    print_table(results)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

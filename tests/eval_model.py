#!/usr/bin/env python3
"""Model evaluation harness.

Runs a structured battery of tests against real checkpoints and reports a
scorecard.  Designed to be fast (<5 min on CPU) so it can be re-run after
every training improvement.

Usage
-----
  python tests/eval_model.py --l1 best_1.pt --l3 best_layer3.pt
  python tests/eval_model.py --l1 best_1.pt --l3 best_layer3.pt --n-random 50
  python tests/eval_model.py --smoke          # random weights, sanity-only

Output
------
  Per-tier table + final scorecard printed to stdout.
  Pass --out results.json to also save raw numbers.

Test tiers
----------
  T1 - Curated biological primitives (known ground truth, hard constraints)
       toggle switch, activating chain, mutual activator, 3-SCC composite
  T2 - Random GRNs, GNN vs ground-truth solver
       count accuracy, position RMSE, stability accuracy, Newton convergence
  T3 - Brouwer certificate on all tiers (Σ ind must equal 1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _xmax(P, K, adj, F, eps=0.05):
    """Positively-invariant box upper bound."""
    N = len(P)
    x = (1 + eps) * P / K
    for i in range(N):
        for j in range(N):
            if adj[j, i] != 0:
                x[i] *= max(1.0, float(F[j, i]))
    return x


def _match_fps(pred, true, xmax):
    """Hungarian matching between predicted and true FP sets.

    Returns (matched_pred_idx, matched_true_idx, unmatched_pred, unmatched_true,
             mean_normalized_distance).
    All positions normalised by xmax before distance.
    """
    from scipy.optimize import linear_sum_assignment
    if len(pred) == 0 or len(true) == 0:
        return [], [], list(range(len(pred))), list(range(len(true))), float('nan')
    p = pred / (xmax + 1e-12)
    t = true / (xmax + 1e-12)
    cost = np.linalg.norm(p[:, None] - t[None, :], axis=-1)   # [np, nt]
    ri, ci = linear_sum_assignment(cost)
    dists = cost[ri, ci]
    n = min(len(pred), len(true))
    unmatched_pred = [i for i in range(len(pred)) if i not in ri[:n]]
    unmatched_true = [i for i in range(len(true)) if i not in ci[:n]]
    return list(ri), list(ci), unmatched_pred, unmatched_true, float(dists.mean())


def _load_models(l1_path, l3_path, device):
    from infer import load_models
    return load_models(l1_path, l3_path, device)


def _run_gnn(grn, l1, l3, device, refine_steps=5, newton_steps=30,
             newton_restarts=2, tol=1e-8):
    from assemble import GRNPredictor
    predictor = GRNPredictor(l1, l3, device=device, refine_steps=refine_steps,
                             newton_steps=newton_steps,
                             newton_restarts=newton_restarts, tol=tol)
    return predictor.predict(
        grn["adj"], grn["P"], grn["K"], grn["T"], grn["nh"], grn["F"],
        verbose=False,
    )


def _run_solver(grn, device, rounds=3):
    from solver import SolverConfig, solve_draws
    par = {k: grn[k][None] for k in ("adj", "P", "K", "T", "nh", "F")}
    scfg = SolverConfig(rounds=rounds)  # default seeds cover n=2..10
    N = grn["adj"].shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    result = solve_draws(par, scfg, N, gen, device)
    n = int(result["counts"][0])
    # FIELDS are flat arrays; for a single draw all n FPs are in [:n]
    return {
        "n_fps": n,
        "fps":   result["state"][:n],
        "stab":  result["stab"][:n],
        "bidx":  result["bidx"][:n],
        "bsum":  float(result["bsum"][0]),
        "converged": bool(result["cons"][0]),
    }


# ---------------------------------------------------------------------------
# Canonical GRN fixtures
# ---------------------------------------------------------------------------

def _toggle_switch():
    adj = np.array([[0, -1], [-1, 0]], dtype=np.float32)
    P  = np.array([2.0, 2.0], dtype=np.float32)
    K  = np.array([1.0, 1.0], dtype=np.float32)
    T  = np.ones((2, 2), dtype=np.float32) * 1.5
    nh = np.ones((2, 2), dtype=np.float32) * 4.0
    F  = np.where(adj < 0, 0.02, 1.0).astype(np.float32)
    return dict(adj=adj, P=P, K=K, T=T, nh=nh, F=F,
                name="toggle_switch", expected_n_fps=3, expected_n_stable=2)


def _activating_chain(n=4):
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    P  = np.ones(n, dtype=np.float32)
    K  = np.ones(n, dtype=np.float32)
    T  = np.where(adj != 0, 0.5, 1.0).astype(np.float32)
    nh = np.where(adj != 0, 2.0, 1.0).astype(np.float32)
    F  = np.where(adj != 0, 5.0, 1.0).astype(np.float32)
    return dict(adj=adj, P=P, K=K, T=T, nh=nh, F=F,
                name=f"activating_chain_n{n}", expected_n_fps=1, expected_n_stable=1)


def _mutual_activator():
    """Two genes activating each other — can be monostable or bistable."""
    adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
    P  = np.array([1.0, 1.0], dtype=np.float32)
    K  = np.array([1.0, 1.0], dtype=np.float32)
    T  = np.ones((2, 2), dtype=np.float32) * 0.5
    nh = np.ones((2, 2), dtype=np.float32) * 3.0
    F  = np.where(adj > 0, 8.0, 1.0).astype(np.float32)
    return dict(adj=adj, P=P, K=K, T=T, nh=nh, F=F,
                name="mutual_activator")


def _three_scc_composite():
    """3-gene toggle → 2-gene activating pair → 1-gene sink."""
    from demo_assemble import make_multistable_grn
    grn = make_multistable_grn(n_bistable_sccs=1, extra_scc_size=2, seed=7)
    grn["name"] = "three_scc_composite"
    return grn


# ---------------------------------------------------------------------------
# Tier 1: Curated primitives
# ---------------------------------------------------------------------------

def run_tier1(l1, l3, device, solver_device):
    """Hard constraints on canonical GRNs."""
    fixtures = [
        _toggle_switch(),
        _activating_chain(n=3),
        _activating_chain(n=5),
        _mutual_activator(),
        _three_scc_composite(),
    ]

    rows = []
    for grn in fixtures:
        name = grn.get("name", "?")
        t0 = time.time()
        result = _run_gnn(grn, l1, l3, device)
        elapsed = time.time() - t0

        n_pred = result["n_fps"]
        n_stable_pred = int(result["global_is_stable"].sum())
        n_converged = int(result["converged"].sum()) if n_pred > 0 else 0
        xmax = _xmax(grn["P"], grn["K"], grn["adj"], grn["F"])

        # Ground-truth solver
        sol = _run_solver(grn, solver_device)
        n_true = sol["n_fps"]
        bsum_true = sol["bsum"]

        # Brouwer check on GNN output (if we have stability info)
        # bidx = sgn det(-J) computed from predicted stability
        # We rely on the solver's bsum for the true certificate

        # Position RMSE for matched FPs
        if n_pred > 0 and n_true > 0:
            gnn_fps = result["global_fps"]
            true_fps = sol["fps"].numpy() if torch.is_tensor(sol["fps"]) else np.array(sol["fps"])
            _, _, _, _, dist = _match_fps(gnn_fps, true_fps, xmax)
        else:
            dist = float('nan')

        count_ok  = (n_pred == n_true)
        stable_ok = (n_stable_pred == int((sol["stab"] < 2).sum()))
        conv_pct  = 100 * n_converged / max(n_pred, 1)
        bsum_ok   = abs(bsum_true - 1.0) < 0.5     # solver certificate

        expected_n = grn.get("expected_n_fps")
        expected_s = grn.get("expected_n_stable")
        hard_pass  = True
        hard_notes = []
        if expected_n is not None and n_pred != expected_n:
            hard_pass = False
            hard_notes.append(f"count {n_pred}≠{expected_n}")
        if expected_s is not None and n_stable_pred != expected_s:
            hard_pass = False
            hard_notes.append(f"stable {n_stable_pred}≠{expected_s}")
        if not bsum_ok:
            hard_pass = False
            hard_notes.append(f"bsum={bsum_true:.2f}")

        rows.append(dict(
            name=name, n_pred=n_pred, n_true=n_true,
            count_ok=count_ok, n_stable_pred=n_stable_pred,
            stable_ok=stable_ok, conv_pct=conv_pct, dist=dist,
            hard_pass=hard_pass, hard_notes=", ".join(hard_notes),
            elapsed=elapsed,
        ))

    return rows


# ---------------------------------------------------------------------------
# Tier 2: Random GRNs vs solver
# ---------------------------------------------------------------------------

def run_tier2(l1, l3, device, solver_device, n_grns=20, seed=42):
    """GNN vs ground-truth solver on random GRNs."""
    from demo_assemble import make_synthetic_grn
    rng = np.random.default_rng(seed)

    count_hits = 0
    stab_hits  = 0
    conv_total = 0
    conv_denom = 0
    dists      = []
    bsum_ok    = 0
    rows       = []

    for i in range(n_grns):
        s = int(rng.integers(0, 10000))
        n_sccs  = int(rng.integers(1, 4))
        scc_sz  = int(rng.integers(2, 4))   # cap: n_sccs*scc_sz <= 9 (seeds cover n=2..10)
        if n_sccs * scc_sz > 9:
            scc_sz = 2
        grn = make_synthetic_grn(n_sccs=n_sccs, scc_size=scc_sz, seed=s)
        N = grn["adj"].shape[0]

        t0 = time.time()
        result = _run_gnn(grn, l1, l3, device)
        elapsed = time.time() - t0

        sol = _run_solver(grn, solver_device, rounds=3)

        n_pred  = result["n_fps"]
        n_true  = sol["n_fps"]
        xmax    = _xmax(grn["P"], grn["K"], grn["adj"], grn["F"])

        count_ok = (n_pred == n_true)
        count_hits += int(count_ok)

        if n_pred > 0 and n_true > 0:
            gnn_fps  = result["global_fps"]
            true_fps = sol["fps"].numpy() if torch.is_tensor(sol["fps"]) else np.array(sol["fps"])
            pi, ti, _, _, dist = _match_fps(gnn_fps, true_fps, xmax)
            dists.append(dist)

            # Stability accuracy on matched pairs
            pred_stab = result["global_stab"]
            true_stab = sol["stab"]
            for p_i, t_i in zip(pi, ti):
                stab_hits += int(int(pred_stab[p_i]) == int(true_stab[t_i]))
            stab_denom = len(pi)
        else:
            stab_denom = 0

        n_conv = int(result["converged"].sum()) if n_pred > 0 else 0
        conv_total += n_conv
        conv_denom += n_pred

        bsum_ok += int(abs(sol["bsum"] - 1.0) < 0.5)

        rows.append(dict(
            i=i, N=N, n_sccs=n_sccs, n_pred=n_pred, n_true=n_true,
            count_ok=count_ok, stab_denom=stab_denom,
            n_conv=n_conv, elapsed=elapsed,
        ))

    summary = dict(
        n_grns=n_grns,
        count_acc=count_hits / n_grns,
        stab_acc=stab_hits / max(sum(r["stab_denom"] for r in rows), 1),
        conv_rate=conv_total / max(conv_denom, 1),
        mean_dist=float(np.nanmean(dists)) if dists else float('nan'),
        bsum_pct=bsum_ok / n_grns,
    )
    return rows, summary


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
BOLD  = "\033[1m"

def _c(ok, s):
    return (GREEN if ok else RED) + s + RESET


def print_tier1(rows):
    print(f"\n{BOLD}=== Tier 1: Curated GRN Primitives ==={RESET}")
    hdr = f"{'Name':<28} {'pred':>4} {'true':>4} {'Δn':>3} {'stab':>5} {'conv%':>6} {'dist':>7} {'hard':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        delta = r["n_pred"] - r["n_true"]
        line = (
            f"{r['name']:<28} "
            f"{r['n_pred']:>4} "
            f"{r['n_true']:>4} "
            + _c(delta == 0, f"{delta:>+3}") + " "
            + _c(r["stable_ok"], f"{r['n_stable_pred']:>5}") + " "
            f"{r['conv_pct']:>5.0f}% "
            f"{r['dist']:>7.4f} "
            + _c(r["hard_pass"], f"{'PASS' if r['hard_pass'] else 'FAIL':>5}")
            + (f"  [{r['hard_notes']}]" if r["hard_notes"] else "")
        )
        print(line)
    n_hard = sum(r["hard_pass"] for r in rows)
    print(f"\nHard constraints: {_c(n_hard == len(rows), f'{n_hard}/{len(rows)} passed')}")


def print_tier2(summary):
    print(f"\n{BOLD}=== Tier 2: Random GRN Batch (n={summary['n_grns']}) ==={RESET}")
    metrics = [
        ("Count accuracy",    f"{100*summary['count_acc']:5.1f}%",  summary["count_acc"] > 0.70),
        ("Stability accuracy",f"{100*summary['stab_acc']:5.1f}%",   summary["stab_acc"]  > 0.70),
        ("Newton conv. rate", f"{100*summary['conv_rate']:5.1f}%",  summary["conv_rate"] > 0.50),
        ("Mean norm. dist",   f"{summary['mean_dist']:.4f}",        summary["mean_dist"] < 0.10),
        ("Brouwer cert. %",   f"{100*summary['bsum_pct']:5.1f}%",   summary["bsum_pct"]  > 0.90),
    ]
    for name, val, ok in metrics:
        print(f"  {_c(ok, '✓' if ok else '✗')} {name:<24} {val}")


def print_scorecard(t1_rows, t2_summary):
    n_hard = sum(r["hard_pass"] for r in t1_rows)
    t1_score = n_hard / len(t1_rows)
    t2_score = np.mean([
        t2_summary["count_acc"] > 0.70,
        t2_summary["stab_acc"]  > 0.70,
        t2_summary["conv_rate"] > 0.50,
        t2_summary["mean_dist"] < 0.10,
    ])
    overall = 0.4 * t1_score + 0.6 * t2_score
    print(f"\n{BOLD}=== Scorecard ==={RESET}")
    print(f"  T1 hard constraints : {100*t1_score:5.1f}%")
    print(f"  T2 metric thresholds: {100*t2_score:5.1f}%  (4 thresholds)")
    print(f"  Overall             : {100*overall:5.1f}%")
    grade = "A" if overall > 0.85 else "B" if overall > 0.70 else "C" if overall > 0.55 else "D"
    color = GREEN if overall > 0.70 else RED
    print(f"  Grade               : {color}{BOLD}{grade}{RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN model evaluation harness")
    parser.add_argument("--l1",       default=None)
    parser.add_argument("--l3",       default=None)
    parser.add_argument("--smoke",    action="store_true",
                        help="Use random weights (no checkpoints needed)")
    parser.add_argument("--n-random", type=int, default=20,
                        help="Number of random GRNs for Tier 2 (default 20)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--cpu",      action="store_true")
    parser.add_argument("--out",      default=None, help="Save raw results to JSON")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.smoke:
        args.l1 = None
        args.l3 = None

    t0 = time.time()
    l1, l3 = _load_models(args.l1, args.l3, device)

    print("\nRunning Tier 1 (curated primitives)...")
    t1_rows = run_tier1(l1, l3, device, device)
    print_tier1(t1_rows)

    print(f"\nRunning Tier 2 ({args.n_random} random GRNs vs solver)...")
    t2_rows, t2_summary = run_tier2(l1, l3, device, device,
                                    n_grns=args.n_random, seed=args.seed)
    print_tier2(t2_summary)

    print_scorecard(t1_rows, t2_summary)
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    if args.out:
        out = dict(tier1=t1_rows, tier2_summary=t2_summary)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

"""End-to-end demo: global fixed-point inference on a synthetic 50-gene GRN.

This script validates the full assembly pipeline:
  1. Generates a synthetic 50-gene GRN with a 5-SCC DAG structure
  2. Loads Layer I + Layer III models (or uses random weights for smoke testing)
  3. Runs GRNPredictor end-to-end
  4. Compares predicted FPs to ground-truth FPs from solve_draws()

Usage
  # Smoke test (no checkpoints needed):
  python demo_assemble.py --smoke

  # With real checkpoints:
  python demo_assemble.py --l1 artifacts/layer1/best.pt --l3 artifacts/layer3/best.pt

  # Use layer2 checkpoint as l3 fallback (same architecture):
  python demo_assemble.py --l1 artifacts/layer1/best.pt --l3 artifacts/layer2/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Make sure gnn root is on path
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# 1. Synthetic GRN generator
# ---------------------------------------------------------------------------

def make_synthetic_grn(
    n_sccs: int = 5,
    scc_size: int = 10,
    seed: int = 42,
    intra_density: float = 0.4,
    inter_density: float = 0.15,
) -> dict:
    """Generate a synthetic GRN with a controlled SCC-DAG structure.

    The graph has `n_sccs` SCCs of equal size `scc_size` arranged in a
    linear chain: SCC 0 → SCC 1 → ... → SCC (n_sccs-1).

    Parameters are drawn from biologically plausible ranges:
      P ~ LogUniform(0.1, 10)
      K ~ LogUniform(0.1, 2)
      T ~ LogUniform(0.1 * xmax, 0.9 * xmax)   (threshold inside the box)
      nh ~ Uniform(1.5, 4.0)
      F  ~ LogUniform(2, 10)  (activator fold-change)

    Returns
    -------
    dict with keys: adj, P, K, T, nh, F, gene_scc_map
      adj [N, N] signed: +1 = activator, -1 = repressor
      P, K [N]
      T, nh, F [N, N] (neutral = T=1, nh=1, F=1 where adj=0)
    """
    rng = np.random.default_rng(seed)
    N = n_sccs * scc_size

    adj = np.zeros((N, N), dtype=np.float32)
    T   = np.ones((N, N), dtype=np.float32)
    nh  = np.ones((N, N), dtype=np.float32)
    F   = np.ones((N, N), dtype=np.float32)

    gene_scc_map = np.repeat(np.arange(n_sccs), scc_size)

    # --- Intra-SCC edges (dense, mixed sign to create multistability) ---
    for s in range(n_sccs):
        start = s * scc_size
        genes = list(range(start, start + scc_size))
        for i in genes:
            for j in genes:
                if i == j:
                    continue
                if rng.random() < intra_density:
                    sign = 1.0 if rng.random() < 0.6 else -1.0
                    adj[i, j] = sign
                    T[i, j]   = rng.uniform(0.05, 0.95)     # will be scaled by xmax later
                    nh[i, j]  = rng.uniform(1.5, 4.0)
                    F[i, j]   = rng.uniform(1.5, 8.0) if sign > 0 else rng.uniform(0.05, 0.4)
        # Ensure at least one positive feedback loop in each SCC for multistability
        # (simple 2-gene positive feedback)
        g0, g1 = genes[0], genes[1]
        adj[g0, g1] = 1.0
        adj[g1, g0] = 1.0
        T[g0, g1]   = 0.5
        T[g1, g0]   = 0.5
        nh[g0, g1]  = 2.0
        nh[g1, g0]  = 2.0
        F[g0, g1]   = 5.0
        F[g1, g0]   = 5.0

    # --- Inter-SCC edges (DAG: SCC s -> SCC s+1 for s in 0..n_sccs-2) ---
    for s in range(n_sccs - 1):
        src_genes = list(range(s * scc_size, (s + 1) * scc_size))
        tgt_genes = list(range((s + 1) * scc_size, (s + 2) * scc_size))
        for j in tgt_genes:
            for i in src_genes:
                if rng.random() < inter_density:
                    sign = 1.0  # inter-SCC: activating (typical regulatory cascade)
                    adj[i, j] = sign
                    T[i, j]   = 0.5
                    nh[i, j]  = 2.0
                    F[i, j]   = 2.0

    # --- Per-gene parameters ---
    P = rng.uniform(0.5, 5.0, size=N).astype(np.float32)
    K = rng.uniform(0.2, 1.5, size=N).astype(np.float32)

    # Scale T by xmax estimate so thresholds are inside the box
    xmax_approx = P / K * np.maximum(1.0, F.max(axis=0))  # crude upper bound
    for i in range(N):
        for j in range(N):
            if adj[i, j] != 0:
                xmax_i = float(xmax_approx[i])
                # T is the threshold on gene i (the source): place it at a fraction of its range
                T[i, j] = float(rng.uniform(0.1, 0.9) * max(xmax_i, 0.1))

    return {
        "adj": adj,
        "P":   P,
        "K":   K,
        "T":   T,
        "nh":  nh,
        "F":   F,
        "gene_scc_map": gene_scc_map,
        "n_sccs": n_sccs,
        "scc_size": scc_size,
        "N": N,
    }


# ---------------------------------------------------------------------------
# 2. Model loading helpers
# ---------------------------------------------------------------------------

def _make_default_cfg(**overrides):
    """Minimal config matching the training defaults (from train.py / train_layer2.py)."""
    cfg = dict(
        # Layer I (SccFixedPointNet) — matches train.py defaults
        d_model=128,
        n_heads=8,
        n_gnn_layers=4,
        n_attn_layers=1,
        n_dec_layers=4,
        dropout=0.0,
        max_fp=16,
        # Layer II/III (CorpusBNet / CorpusCNet) extras
        d_port=32,
        n_port_layers=2,
        max_ports=8,
        max_ports_c=8,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def load_or_create_l1(path: str | None, device: torch.device):
    """Load L1 checkpoint or create random-weight model."""
    from model import SccFixedPointNet

    if path and Path(path).exists():
        print(f"Loading L1 from {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = SimpleNamespace(**ck["cfg"])
        model = SccFixedPointNet(cfg)
        model.load_state_dict(ck.get("ema", ck.get("model")))
    else:
        print("No L1 checkpoint found — using random weights (smoke test)")
        cfg = _make_default_cfg()
        model = SccFixedPointNet(cfg)

    model.to(device).eval()
    return model, cfg


def load_or_create_l3(path: str | None, l1_model, device: torch.device):
    """Load L3 checkpoint or create random-weight model."""
    from model_c import CorpusCNet

    if path and Path(path).exists():
        print(f"Loading L3 from {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = SimpleNamespace(**ck["cfg"])
        model = CorpusCNet(l1_model, cfg)
        model.load_state_dict(ck.get("ema", ck.get("model")))
    else:
        print("No L3 checkpoint found — using random weights (smoke test)")
        cfg = _make_default_cfg()
        model = CorpusCNet(l1_model, cfg)

    model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# 3. Ground-truth solver comparison
# ---------------------------------------------------------------------------

def solve_ground_truth(grn: dict, device: torch.device, verbose: bool = True) -> dict:
    """Run the exact Newton solver on the full 50-gene GRN."""
    from solver import solve_draws, prepare_params, SolverConfig

    N = grn["N"]
    par = {
        "P":   grn["P"][np.newaxis],      # [1, N]
        "K":   grn["K"][np.newaxis],
        "adj": grn["adj"][np.newaxis],
        "T":   grn["T"][np.newaxis],
        "nh":  grn["nh"][np.newaxis],
        "F":   grn["F"][np.newaxis],
    }

    if verbose:
        print(f"\n[Ground truth] Solving {N}-gene GRN with Newton solver...")
    t0 = time.time()

    # seeds[n-2] = number of random starts for a system of size n.
    # Default SolverConfig only covers n=2..10 (9 entries).
    # Extend to n=2..50 (49 entries) with increasing budgets for larger systems.
    seeds_extended = (
        96, 128, 160, 192, 224, 256, 288, 320, 352,   # n=2..10  (original)
        384, 416, 448, 480, 512, 544, 576, 608, 640,   # n=11..19
        672, 704, 736, 768, 800, 832, 864, 896, 928,   # n=20..28
        960, 992, 1024, 1056, 1088, 1120, 1152, 1184,  # n=29..36
        1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440,# n=37..44
        1472, 1504, 1536, 1568, 1600, 1632,             # n=45..50 (covers n=50 at index 48)
    )
    cfg = SolverConfig(seeds=seeds_extended, rounds=3)
    rng = np.random.default_rng(0)

    try:
        result = solve_draws(par, cfg, N, rng, device)
        elapsed = time.time() - t0

        # solve_draws returns a flat dict:
        #   counts [B]  — number of FPs per draw
        #   state  [total_fps, N]  — all FP positions concatenated across draws
        #   stab   [total_fps]     — stability class per FP
        #   (plus bsum, cons, xmax, seed_ok, rounds_used)
        counts = result["counts"]      # [B=1]
        n_gt   = int(counts[0])

        gt_fps   = []
        gt_stabs = []
        # FPs for draw 0 are the first counts[0] rows of the flat arrays
        if n_gt > 0:
            states = result["state"][:n_gt]    # [n_gt, N] numpy
            stabs  = result["stab"][:n_gt]     # [n_gt] numpy
            for k in range(n_gt):
                gt_fps.append(states[k])
                gt_stabs.append(int(stabs[k]))

        if verbose:
            print(f"  Solver found {n_gt} FP(s) in {elapsed:.1f}s")
        return {"fps": gt_fps, "stab": gt_stabs, "elapsed": elapsed}
    except Exception as e:
        import traceback
        if verbose:
            print(f"  Solver failed: {e}")
            traceback.print_exc()
        return {"fps": [], "stab": [], "elapsed": 0.0}


# ---------------------------------------------------------------------------
# 4. Comparison metrics
# ---------------------------------------------------------------------------

def compare_fps(pred_fps: np.ndarray, gt_fps: list, tol_log: float = 1.0) -> dict:
    """Compare predicted FPs to ground-truth FPs.

    A predicted FP matches a GT FP if the mean |log(pred/gt)| over all genes
    is below tol_log (1.0 = within 1 decade on average).

    Returns
    -------
    dict with:
      tp, fp_count, fn : true positives, false positives, false negatives
      recall, precision, f1
      per_gt_matched : list of bool
    """
    if len(gt_fps) == 0:
        return {"tp": 0, "fp_count": len(pred_fps), "fn": 0,
                "recall": float("nan"), "precision": 0.0, "f1": 0.0,
                "per_gt_matched": []}

    if len(pred_fps) == 0:
        return {"tp": 0, "fp_count": 0, "fn": len(gt_fps),
                "recall": 0.0, "precision": float("nan"), "f1": 0.0,
                "per_gt_matched": [False] * len(gt_fps)}

    n_gt   = len(gt_fps)
    n_pred = len(pred_fps)
    gt_arr = np.stack(gt_fps, axis=0)    # [n_gt, N]
    pr_arr = pred_fps                    # [n_pred, N]

    # Pairwise log-distance
    eps = 1e-12
    log_gt   = np.log(np.maximum(gt_arr, eps))   # [n_gt, N]
    log_pred = np.log(np.maximum(pr_arr, eps))   # [n_pred, N]
    dist = np.abs(log_gt[:, np.newaxis] - log_pred[np.newaxis])  # [n_gt, n_pred, N]
    mean_dist = dist.mean(axis=-1)                                 # [n_gt, n_pred]

    # Greedy matching (GT -> best Pred)
    per_gt_matched = [False] * n_gt
    matched_pred = set()
    tp = 0
    for i in range(n_gt):
        j_best = int(mean_dist[i].argmin())
        if mean_dist[i, j_best] < tol_log and j_best not in matched_pred:
            per_gt_matched[i] = True
            matched_pred.add(j_best)
            tp += 1

    fn         = n_gt - tp
    fp_count   = n_pred - len(matched_pred)
    recall     = tp / n_gt  if n_gt   > 0 else float("nan")
    precision  = tp / n_pred if n_pred > 0 else float("nan")
    f1 = (2 * recall * precision / (recall + precision)
          if recall + precision > 0 else 0.0)

    return {
        "tp": tp, "fp_count": fp_count, "fn": fn,
        "recall": recall, "precision": precision, "f1": f1,
        "per_gt_matched": per_gt_matched,
        "best_log_dist": float(mean_dist.min(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# 5. Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n{'='*65}")
    print(f" GRN Fixed-Point Assembly Pipeline — End-to-End Demo")
    print(f"{'='*65}")
    print(f"Device: {device}")

    # --- 1. Generate synthetic GRN ---
    print(f"\n[Step 1] Generating {args.n_sccs}-SCC × {args.scc_size}-gene GRN "
          f"({args.n_sccs * args.scc_size} genes total)")
    grn = make_synthetic_grn(
        n_sccs=args.n_sccs,
        scc_size=args.scc_size,
        seed=args.seed,
        intra_density=args.intra_density,
        inter_density=args.inter_density,
    )
    N = grn["N"]
    n_edges = int((grn["adj"] != 0).sum())
    print(f"  Genes: {N}, Edges: {n_edges}")

    # --- 2. Load models ---
    print(f"\n[Step 2] Loading models")
    l1_model, l1_cfg = load_or_create_l1(args.l1, device)
    l3_model, l3_cfg = load_or_create_l3(args.l3, l1_model, device)

    # --- 3. Run assembly pipeline ---
    print(f"\n[Step 3] Running assembly pipeline")
    from assemble import GRNPredictor, print_result_summary

    predictor = GRNPredictor(
        l1_model, l3_model,
        device=device,
        refine_steps=args.refine_steps,
        newton_steps=args.newton_steps,
        newton_restarts=args.newton_restarts,
        tol=args.tol,
    )

    t0 = time.time()
    result = predictor.predict(
        grn["adj"], grn["P"], grn["K"], grn["T"], grn["nh"], grn["F"],
        verbose=True,
    )
    t_pipeline = time.time() - t0
    print(f"\n  Pipeline time: {t_pipeline:.2f}s")

    print_result_summary(result)

    # --- 4. Ground-truth comparison (only if not smoke-only) ---
    if not args.smoke:
        print(f"\n[Step 4] Ground-truth comparison")
        gt = solve_ground_truth(grn, device, verbose=True)

        if result["n_fps"] > 0 and len(gt["fps"]) > 0:
            # Compare against Newton-refined predictions
            refined = result["fps_refined"] if result["n_converged"] > 0 else result["global_fps"]
            metrics = compare_fps(refined, gt["fps"])
            print(f"\n  === Comparison ===")
            print(f"  Predicted FPs  : {result['n_fps']}")
            print(f"  Ground-truth   : {len(gt['fps'])}")
            print(f"  True positives : {metrics['tp']}")
            print(f"  False positives: {metrics['fp_count']}")
            print(f"  False negatives: {metrics['fn']}")
            print(f"  Recall         : {metrics['recall']:.3f}")
            print(f"  Precision      : {metrics['precision']:.3f}")
            print(f"  F1             : {metrics['f1']:.3f}")
            print(f"  Avg log dist   : {metrics.get('best_log_dist', float('nan')):.3f} decades")

            # Stability comparison
            if len(gt["stab"]) > 0:
                gt_stable = sum(1 for s in gt["stab"] if s in (0, 1))
                pred_stable = int(result["global_is_stable"].sum())
                print(f"\n  GT stable FPs  : {gt_stable}/{len(gt['fps'])}")
                print(f"  Pred stable FPs: {pred_stable}/{result['n_fps']}")
        else:
            print("  (Cannot compare: one side has 0 FPs)")
            if len(gt["fps"]) == 0:
                print("  Note: ground-truth solver found no FPs — GRN may be trivially stable")
    else:
        print(f"\n[Step 4] Skipping ground-truth comparison (--smoke mode)")
        if result["n_fps"] > 0:
            print(f"  Pipeline produced {result['n_fps']} FP(s)")
            print(f"  First FP (first 10 genes): {result['global_fps'][0, :10].round(4)}")
        else:
            print("  Pipeline produced 0 FPs (expected for random weights)")

    print(f"\n{'='*65}")
    print(f" Demo complete!")
    print(f"{'='*65}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="GRN assembly pipeline demo")
    parser.add_argument("--l1", type=str, default=None,
                        help="Path to Layer I checkpoint (SccFixedPointNet)")
    parser.add_argument("--l3", type=str, default=None,
                        help="Path to Layer III checkpoint (CorpusCNet)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test only: skip ground-truth solver comparison")
    parser.add_argument("--n-sccs", type=int, default=5,
                        help="Number of SCCs in the synthetic GRN (default: 5)")
    parser.add_argument("--scc-size", type=int, default=10,
                        help="Genes per SCC (default: 10; max N_MAX=10)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--intra-density", type=float, default=0.3,
                        help="Density of intra-SCC edges")
    parser.add_argument("--inter-density", type=float, default=0.1,
                        help="Density of inter-SCC edges")
    parser.add_argument("--refine-steps", type=int, default=5,
                        help="Newton steps inside GNN (after each SCC inference)")
    parser.add_argument("--newton-steps", type=int, default=30,
                        help="Newton steps in global refinement")
    parser.add_argument("--newton-restarts", type=int, default=2,
                        help="Random restarts per FP in global Newton")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Newton convergence tolerance")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()

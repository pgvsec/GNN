#!/usr/bin/env python3
"""End-to-end GRN fixed-point inference.

Loads a Layer I (SccFixedPointNet) and Layer III (CorpusCNet) checkpoint and runs
the full SCC-DAG assembly pipeline on a user-supplied gene regulatory network.

Usage
-----
  # With real checkpoints:
  python infer.py --l1 best_1.pt --l3 best_3.pt --adj adj.npy

  # With all kinetic-parameter files:
  python infer.py --l1 best_1.pt --l3 best_3.pt \\
      --adj adj.npy --P P.npy --K K.npy --T T.npy --nh nh.npy --F F.npy

  # Smoke test on a synthetic 3-SCC × 4-gene GRN (random weights):
  python infer.py --smoke

  # Save numpy results:
  python infer.py --l1 best_1.pt --l3 best_3.pt --adj adj.npy --out results.npz

Input format
------------
  --adj   [N, N] float .npy  signed adjacency; nonzero = edge, sign = regulation type
  --P     [N]    float .npy  production rates         (default: all 1.0)
  --K     [N]    float .npy  degradation rates        (default: all 1.0)
  --T     [N, N] float .npy  Hill thresholds          (default: all 1.0)
  --nh    [N, N] float .npy  Hill coefficients        (default: all 2.0)
  --F     [N, N] float .npy  fold-changes             (default: 5 activator / 0.2 repressor)

Output keys
-----------
  n_fps            int            number of predicted global fixed points
  global_fps       [n_fps, N]     position vectors
  global_stab      [n_fps]        stability class (0=stable node, 1=stable spiral,
                                   2=unstable node, 3=unstable spiral, 4=saddle, 5=non-hyp)
  global_is_stable [n_fps]        True for stable nodes and spirals
  fps_refined      [n_fps, N]     after Newton refinement on the full GRN ODE
  residuals        [n_fps]        max relative ODE residual at each refined FP
  converged        [n_fps]        True when residual < tol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_params(adj: np.ndarray):
    """Biologically neutral kinetic parameters for a signed adjacency matrix."""
    N = adj.shape[0]
    P  = np.ones(N, dtype=np.float32)
    K  = np.ones(N, dtype=np.float32)
    T  = np.ones((N, N), dtype=np.float32)
    nh = np.full((N, N), 2.0, dtype=np.float32)
    # Default fold-change: 5× activator, 0.2× repressor, 1 where no edge
    F  = np.where(adj > 0, 5.0, np.where(adj < 0, 0.2, 1.0)).astype(np.float32)
    return P, K, T, nh, F


def _load_npy_or_default(path: str | None, default: np.ndarray) -> np.ndarray:
    if path is None:
        return default
    return np.load(path).astype(np.float32)


def _default_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        d_model=128, n_heads=8, n_gnn_layers=4, n_attn_layers=1,
        n_dec_layers=4, dropout=0.0, max_fp=16,
        d_port=32, n_port_layers=2, max_ports=8, max_ports_c=8,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(l1_path: str | None, l3_path: str | None, device: torch.device):
    """Load Layer I and Layer III models from checkpoints (or create random-weight models).

    Parameters
    ----------
    l1_path : path to Layer I (SccFixedPointNet) checkpoint, or None
    l3_path : path to Layer III (CorpusCNet) checkpoint, or None
    device  : torch device

    Returns
    -------
    (l1_model, l3_model) both in eval mode on device
    """
    from model import SccFixedPointNet
    from model_c import CorpusCNet

    if l1_path and Path(l1_path).exists():
        ck = torch.load(l1_path, map_location=device, weights_only=False)
        cfg = SimpleNamespace(**ck["cfg"])
        l1 = SccFixedPointNet(cfg)
        l1.load_state_dict(ck.get("ema", ck.get("model")))
        print(f"Loaded Layer I  from: {l1_path}")
    else:
        l1 = SccFixedPointNet(_default_cfg())
        if l1_path:
            print(f"Warning: Layer I checkpoint not found at {l1_path!r} — using random weights")
        else:
            print("Layer I: random weights (no checkpoint specified)")

    if l3_path and Path(l3_path).exists():
        ck = torch.load(l3_path, map_location=device, weights_only=False)
        cfg = SimpleNamespace(**ck["cfg"])
        l3 = CorpusCNet(l1, cfg)
        l3.load_state_dict(ck.get("ema", ck.get("model")))
        print(f"Loaded Layer III from: {l3_path}")
    else:
        l3 = CorpusCNet(l1, _default_cfg())
        if l3_path:
            print(f"Warning: Layer III checkpoint not found at {l3_path!r} — using random weights")
        else:
            print("Layer III: random weights (no checkpoint specified)")

    l1.to(device).eval()
    l3.to(device).eval()
    return l1, l3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    adj: np.ndarray,
    P: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    nh: np.ndarray,
    F: np.ndarray,
    l1_path: str | None = None,
    l3_path: str | None = None,
    device: torch.device | None = None,
    refine_steps: int = 5,
    newton_steps: int = 30,
    newton_restarts: int = 3,
    tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Load models and run the full GRN fixed-point inference pipeline.

    This is the main programmatic entry point. It wraps GRNPredictor from assemble.py.

    Parameters
    ----------
    adj  : [N, N] signed adjacency (nonzero = edge; sign = regulation type)
    P, K : [N] production and degradation rates
    T, nh, F : [N, N] Hill parameters indexed [source, target]; set neutral values
              (T=1, nh=1, F=1) for absent edges
    l1_path, l3_path : checkpoint paths (None → random weights for smoke testing)
    device  : torch device; None = auto (CUDA if available, else CPU)
    refine_steps : Newton steps inside each SCC inference call
    newton_steps : Newton steps in the global refinement pass
    newton_restarts : random restarts per FP in global Newton refinement
    tol     : Newton convergence tolerance for the "converged" flag
    verbose : print progress to stdout

    Returns
    -------
    dict from GRNPredictor.predict() — see assemble.py for the full list of keys.
    """
    from assemble import GRNPredictor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    l1, l3 = load_models(l1_path, l3_path, device)

    predictor = GRNPredictor(
        l1, l3,
        device=device,
        refine_steps=refine_steps,
        newton_steps=newton_steps,
        newton_restarts=newton_restarts,
        tol=tol,
    )
    return predictor.predict(adj, P, K, T, nh, F, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRN fixed-point inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--l1",  type=str, default=None, help="Layer I checkpoint (SccFixedPointNet)")
    parser.add_argument("--l3",  type=str, default=None, help="Layer III checkpoint (CorpusCNet)")
    parser.add_argument("--adj", type=str, default=None, help="Signed adjacency matrix .npy [N, N]")
    parser.add_argument("--P",   type=str, default=None, help="Production rates .npy [N]")
    parser.add_argument("--K",   type=str, default=None, help="Degradation rates .npy [N]")
    parser.add_argument("--T",   type=str, default=None, help="Hill thresholds .npy [N, N]")
    parser.add_argument("--nh",  type=str, default=None, help="Hill coefficients .npy [N, N]")
    parser.add_argument("--F",   type=str, default=None, help="Fold-changes .npy [N, N]")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test on a synthetic 3-SCC GRN (no adjacency file needed)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for smoke GRN (default 42)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results to .npz (keys: global_fps, global_stab, fps_refined, "
                             "residuals, converged)")
    parser.add_argument("--refine-steps",    type=int,   default=5)
    parser.add_argument("--newton-steps",    type=int,   default=30)
    parser.add_argument("--newton-restarts", type=int,   default=3)
    parser.add_argument("--tol",             type=float, default=1e-8)
    parser.add_argument("--cpu",    action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--quiet",  action="store_true", help="Suppress per-step progress output")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build or load the GRN
    if args.smoke:
        from demo_assemble import make_synthetic_grn
        grn = make_synthetic_grn(n_sccs=3, scc_size=4, seed=args.seed)
        adj, P, K, T, nh, F = grn["adj"], grn["P"], grn["K"], grn["T"], grn["nh"], grn["F"]
        print(f"Smoke GRN: {grn['N']} genes, {grn['n_sccs']} SCCs")
    elif args.adj is None:
        parser.error("Provide --adj or use --smoke")
    else:
        adj = np.load(args.adj).astype(np.float32)
        N = adj.shape[0]
        P0, K0, T0, nh0, F0 = _default_params(adj)
        P  = _load_npy_or_default(args.P,  P0)
        K  = _load_npy_or_default(args.K,  K0)
        T  = _load_npy_or_default(args.T,  T0)
        nh = _load_npy_or_default(args.nh, nh0)
        F  = _load_npy_or_default(args.F,  F0)
        print(f"GRN: {N} genes, {int((adj != 0).sum())} edges")

    result = run_inference(
        adj, P, K, T, nh, F,
        l1_path=args.l1,
        l3_path=args.l3,
        device=device,
        refine_steps=args.refine_steps,
        newton_steps=args.newton_steps,
        newton_restarts=args.newton_restarts,
        tol=args.tol,
        verbose=not args.quiet,
    )

    from assemble import print_result_summary
    print_result_summary(result)

    if args.out:
        out_path = Path(args.out)
        np.savez(
            out_path,
            global_fps    = result["global_fps"],
            global_stab   = result["global_stab"],
            fps_refined   = result["fps_refined"],
            residuals     = result["residuals"],
            converged     = result["converged"],
        )
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

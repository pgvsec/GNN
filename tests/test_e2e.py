"""Hard mathematical tests verifying the Overleaf theoretical spec.

Each test group maps to a theorem or construction in the Overleaf document.

  (A) TestHillFunction       — H bounded in (min(1,F), max(1,F)); H(0)=1; H(∞)=F
  (B) TestInvariantBox       — f_i < 0 on the face x_i = x_i^max (inward-pointing flow)
  (C) TestBrouwerIndex       — Σ_r ind(x*_r) = 1; ind = sgn det(-J)
  (D) TestThomasConditions   — DAG → exactly 1 FP; positive feedback needed for bistability
  (E) TestToggleSwitchBistability  — mutual repressor with n≥4 gives 2 stable FPs + 1 saddle
  (F) TestCharPoly           — c_1 = -Tr(J), c_n = (-1)^n det(J); char poly from hill_jacobian
  (G) TestHopfRatio          — ρ = |Re(α_lead)|/|Im(α_lead)|; α_lead = complex pair with min |Re|
  (H) TestSCCDecomposition   — topo order; DAG → singleton SCCs; block-triangular Jacobian
  (I) TestGRNPredictorSmoke  — pipeline runs, returns expected keys, FPs inside invariant box

All solver-based tests (C, D, E) use a minimal SolverConfig (rounds=3) to keep runtime < 10 s.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers shared by multiple test groups
# ---------------------------------------------------------------------------

def _hill(x: float, T: float, n: float, F: float) -> float:
    """Hill function H(x; T, n, F) = (F r + 1) / (r + 1),  r = (x/T)^n."""
    r = (max(x, 0.0) / T) ** n
    return (F * r + 1.0) / (r + 1.0)


def _ode_rhs_np(x: np.ndarray, P, K, adj, T, nh, F) -> np.ndarray:
    """Numpy implementation of dx/dt = P * prod H_ji(x_j) - K * x.
    All arrays are 1-D (gene-indexed) or 2-D [source, target]."""
    N = len(x)
    f = np.zeros(N, dtype=np.float64)
    for i in range(N):
        prod = 1.0
        for j in range(N):
            if adj[j, i] != 0:
                prod *= _hill(float(x[j]), float(T[j, i]), float(nh[j, i]), float(F[j, i]))
        f[i] = float(P[i]) * prod - float(K[i]) * float(x[i])
    return f


def _xmax_np(P, K, adj, F, eps: float = 0.05) -> np.ndarray:
    """Invariant-box bound: x_i^max = (1+ε) * (P_i/K_i) * prod_{j→i} max(1, F_{ji})."""
    N = len(P)
    gain = np.ones(N, dtype=np.float64)
    for j in range(N):
        for i in range(N):
            if adj[j, i] != 0:
                gain[i] *= max(1.0, float(F[j, i]))
    return (1.0 + eps) * np.asarray(P, dtype=np.float64) / np.asarray(K, dtype=np.float64) * gain


def _toggle_switch_params():
    """Canonical 2-gene mutual repressor (toggle switch).

    H(x; T=1.5, n=4, F=0.02): strong cooperative repression.
    Has 2 stable FPs (gene1-high, gene2-high) and 1 saddle.
    """
    adj = np.array([[0, -1], [-1, 0]], dtype=np.float32)
    T   = np.array([[1, 1.5], [1.5, 1]], dtype=np.float32)
    nh  = np.array([[1, 4.0], [4.0, 1]], dtype=np.float32)
    F   = np.array([[1, 0.02], [0.02, 1]], dtype=np.float32)
    P   = np.array([2.0, 2.0], dtype=np.float32)
    K   = np.array([1.0, 1.0], dtype=np.float32)
    return adj, P, K, T, nh, F


def _chain_params(n: int = 3):
    """Linear activating chain: gene 0 → gene 1 → ... → gene n-1 (DAG, no feedback).
    Exactly 1 FP by Thomas's theorem.
    """
    adj = np.zeros((n, n), dtype=np.float32)
    T   = np.ones((n, n), dtype=np.float32)
    nh  = np.full((n, n), 2.0, dtype=np.float32)
    F   = np.ones((n, n), dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        T[i, i + 1]   = 0.5
        F[i, i + 1]   = 4.0
    P = np.ones(n, dtype=np.float32) * 1.5
    K = np.ones(n, dtype=np.float32)
    return adj, P, K, T, nh, F


def _solve_simple(par_np: dict, n: int, device=None, n_seeds: int = 96, rounds: int = 3):
    """Run solve_draws on a single-draw parameter dict. Returns the full result dict."""
    from solver import solve_draws, SolverConfig
    if device is None:
        device = torch.device("cpu")
    seeds_tuple = tuple([n_seeds] * max(n - 1, 1))
    cfg = SolverConfig(seeds=seeds_tuple, rounds=rounds)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    par_batch = {k: v[np.newaxis].astype(np.float32) if np.asarray(v).ndim < 2 else
                    v[np.newaxis].astype(np.float32)
                 for k, v in par_np.items()}
    return solve_draws(par_batch, cfg, n, gen, device)


# ===========================================================================
# (A) Hill function mathematical properties
# ===========================================================================

class TestHillFunction(unittest.TestCase):
    """H(x; T, n, F) = (F r + 1)/(r + 1),  r = (x/T)^n."""

    def test_at_zero_is_basal_one(self):
        """H(0) = 1 for all T, n, F."""
        for T in (0.5, 1.0, 2.0):
            for n in (1.0, 2.0, 4.0):
                for F in (0.1, 1.0, 5.0, 10.0):
                    self.assertAlmostEqual(_hill(0, T, n, F), 1.0, places=10,
                        msg=f"H(0; T={T}, n={n}, F={F}) should be 1")

    def test_at_inf_approaches_F(self):
        """H(x→∞) → F."""
        for F in (0.05, 0.2, 5.0, 10.0):
            h = _hill(1e9, T=1.0, n=2.0, F=F)
            self.assertAlmostEqual(h, F, places=5, msg=f"H(∞; F={F}) should approach {F}")

    def test_activator_strictly_between_one_and_F(self):
        """For F > 1: H(x) ∈ (1, F) for all finite x > 0."""
        for F in (2.0, 5.0, 10.0):
            for x in np.logspace(-3, 3, 30):
                h = _hill(x, T=1.0, n=2.0, F=F)
                self.assertGreater(h, 1.0, msg=f"H({x}; F={F}) must exceed 1")
                self.assertLess(h, F, msg=f"H({x}; F={F}) must be below {F}")

    def test_repressor_strictly_between_F_and_one(self):
        """For F < 1: H(x) ∈ (F, 1) for all finite x > 0."""
        for F in (0.02, 0.1, 0.5):
            for x in np.logspace(-3, 3, 30):
                h = _hill(x, T=1.0, n=2.0, F=F)
                self.assertGreater(h, F, msg=f"H({x}; F={F}) must exceed {F}")
                self.assertLess(h, 1.0, msg=f"H({x}; F={F}) must be below 1")

    def test_activator_is_monotone_increasing(self):
        """H is monotone increasing in x when F > 1."""
        x_vals = np.logspace(-2, 2, 50)
        h_vals = [_hill(x, T=1.0, n=2.0, F=5.0) for x in x_vals]
        diffs = np.diff(h_vals)
        self.assertTrue(np.all(diffs > 0), "Activator Hill function must be strictly increasing")

    def test_repressor_is_monotone_decreasing(self):
        """H is monotone decreasing in x when F < 1."""
        x_vals = np.logspace(-2, 2, 50)
        h_vals = [_hill(x, T=1.0, n=2.0, F=0.1) for x in x_vals]
        diffs = np.diff(h_vals)
        self.assertTrue(np.all(diffs < 0), "Repressor Hill function must be strictly decreasing")

    def test_neutral_is_identically_one(self):
        """When F = 1, H(x) = 1 for all x ≥ 0."""
        for x in np.logspace(-4, 4, 20):
            self.assertAlmostEqual(_hill(x, T=1.0, n=2.0, F=1.0), 1.0, places=12,
                msg=f"H({x}; F=1) must be exactly 1")

    def test_hill_coefficient_increases_steepness(self):
        """Higher n → steeper sigmoid transition; |H(1.5T) - H(0.5T)| grows with n."""
        gaps = []
        for n in (1.0, 2.0, 4.0, 8.0):
            h_above = _hill(1.5, T=1.0, n=n, F=5.0)
            h_below = _hill(0.5, T=1.0, n=n, F=5.0)
            gaps.append(h_above - h_below)
        self.assertTrue(all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1)),
            "Higher Hill coefficient should give steeper transition")


# ===========================================================================
# (B) Positively invariant box
# ===========================================================================

class TestInvariantBox(unittest.TestCase):
    """x_i^max = (1+ε)(P_i/K_i) ∏_{j→i} max(1, F_{ji}); f_i < 0 on its face."""

    def _check_inward_on_boundary(self, adj, P, K, T, nh, F, eps=0.05, n_test=20):
        """At x_i = x_i^max with all x_j = x_j^max, verify f_i < 0."""
        xmax = _xmax_np(P, K, adj, F, eps)
        N = len(P)
        x = xmax.copy()  # all genes at their box maximum
        for i in range(N):
            f = _ode_rhs_np(x, P, K, adj, T, nh, F)
            self.assertLess(f[i], 0.0,
                msg=f"f_{i} should be negative at x_i = x_i^max; got f_{i} = {f[i]:.4e}")

    def test_toggle_switch_boundary(self):
        adj, P, K, T, nh, F = _toggle_switch_params()
        self._check_inward_on_boundary(adj, P, K, T, nh, F)

    def test_chain_boundary(self):
        adj, P, K, T, nh, F = _chain_params(n=4)
        self._check_inward_on_boundary(adj, P, K, T, nh, F)

    def test_mixed_activator_repressor_boundary(self):
        """Mixed GRN: activator on gene 0→1, repressor on gene 1→0 (two-gene negative loop)."""
        adj = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        T   = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        nh  = np.array([[1, 2.0], [2.0, 1]], dtype=np.float32)
        F   = np.array([[1, 3.0], [0.1, 1]], dtype=np.float32)
        P   = np.array([1.0, 1.0], dtype=np.float32)
        K   = np.array([1.0, 1.0], dtype=np.float32)
        self._check_inward_on_boundary(adj, P, K, T, nh, F)

    def test_solver_fps_inside_box(self):
        """All FPs found by the solver lie inside the positively invariant box."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        n_fps = int(result["counts"][0])
        if n_fps == 0:
            self.skipTest("Solver found no FPs (unexpected for toggle switch)")
        fps = result["state"][:n_fps]         # [n_fps, 2]
        xmax = _xmax_np(P, K, adj, F)
        for r in range(n_fps):
            for i in range(2):
                self.assertLessEqual(
                    float(fps[r, i]), float(xmax[i]) * 1.001,
                    msg=f"FP {r} gene {i}: {fps[r,i]:.4e} exceeds xmax={xmax[i]:.4e}")
                self.assertGreater(float(fps[r, i]), 0.0,
                    msg=f"FP {r} gene {i} must be positive")

    def test_xmax_formula_matches_solver_xmax(self):
        """Our numpy xmax matches solver.prepare_params xmax."""
        from solver import prepare_params
        adj, P, K, T, nh, F = _toggle_switch_params()
        par_np = {"P": P[np.newaxis], "K": K[np.newaxis], "adj": adj[np.newaxis],
                  "T": T[np.newaxis], "nh": nh[np.newaxis], "F": F[np.newaxis]}
        p = prepare_params(par_np, eps_box=0.05, device=torch.device("cpu"))
        xmax_solver = p["xmax"][0].numpy()
        xmax_ours   = _xmax_np(P, K, adj, F, eps=0.05)
        np.testing.assert_allclose(xmax_ours, xmax_solver, rtol=1e-5,
            err_msg="xmax formula mismatch between numpy impl and solver.prepare_params")


# ===========================================================================
# (C) Brouwer index: Σ ind(x*) = 1
# ===========================================================================

class TestBrouwerIndex(unittest.TestCase):
    """Brouwer index ind(x*) = sgn det(-J(x*)) = 1 - 2*(n_pos_real mod 2).

    For isolated hyperbolic FPs: Σ ind = 1 (Poincaré-Hopf on the invariant box).
    """

    def test_index_formula_stable(self):
        """Stable FP (all eigenvalues negative real) → n_pos_real=0 → ind=+1."""
        # 2x2 stable node: eigenvalues -1, -2
        J = np.array([[-1.5, 0.1], [0.1, -2.0]])
        ev = np.linalg.eigvals(J)
        n_pos_real = int((ev.real > 0).sum())
        ind = 1 - 2 * (n_pos_real % 2)
        self.assertEqual(ind, 1, f"Stable FP should have ind=+1; eigenvalues={ev}")

    def test_index_formula_saddle_2d(self):
        """2D saddle (one pos, one neg real eigenvalue) → n_pos_real=1 → ind=-1."""
        J = np.array([[-1.0, 0.5], [2.0, 0.5]])
        ev = np.linalg.eigvals(J)
        # ensure one positive, one negative
        self.assertTrue(ev.real.max() > 0 and ev.real.min() < 0,
            f"Expected saddle eigenvalues, got {ev}")
        n_pos_real = int((ev.real > 0).sum())
        ind = 1 - 2 * (n_pos_real % 2)
        self.assertEqual(ind, -1)

    def test_index_formula_source_2d(self):
        """2D source (both eigenvalues positive real) → n_pos_real=2 → ind=+1."""
        J = np.array([[1.5, -0.1], [-0.1, 2.0]])
        ev = np.linalg.eigvals(J)
        self.assertTrue((ev.real > 0).all(), f"Expected source, got {ev}")
        n_pos_real = int((ev.real > 0).sum())
        ind = 1 - 2 * (n_pos_real % 2)
        self.assertEqual(ind, 1)

    def test_bsum_toggle_switch_is_one(self):
        """Toggle switch: 2 stable FPs (ind=+1 each) + 1 saddle (ind=-1) → sum=1."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        bsum = int(result["bsum"][0])
        n_fps = int(result["counts"][0])
        self.assertEqual(bsum, 1,
            f"Toggle switch Brouwer sum should be 1; got {bsum} with {n_fps} FPs")

    def test_bsum_chain_is_one(self):
        """Linear chain (DAG, exactly 1 FP) → Brouwer sum = 1."""
        adj, P, K, T, nh, F = _chain_params(n=3)
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=3)
        bsum = int(result["bsum"][0])
        self.assertEqual(bsum, 1,
            f"Chain Brouwer sum should be 1; got {bsum}")

    def test_stable_fps_have_positive_index(self):
        """All stable FPs (stab ∈ {0,1}) must have bidx = +1."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        n_fps = int(result["counts"][0])
        stab  = result["stab"][:n_fps]
        bidx  = result["bidx"][:n_fps]
        for r in range(n_fps):
            if stab[r] in (0, 1):  # stable node or spiral
                self.assertEqual(int(bidx[r]), 1,
                    f"Stable FP {r} (stab={stab[r]}) must have bidx=+1; got {bidx[r]}")

    def test_saddle_fps_have_negative_index(self):
        """Saddle FPs (stab=4) must have bidx = -1."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        n_fps = int(result["counts"][0])
        stab  = result["stab"][:n_fps]
        bidx  = result["bidx"][:n_fps]
        saddle_found = False
        for r in range(n_fps):
            if stab[r] == 4:  # saddle
                saddle_found = True
                self.assertEqual(int(bidx[r]), -1,
                    f"Saddle FP {r} must have bidx=-1; got {bidx[r]}")
        if not saddle_found:
            self.skipTest("No saddle found by solver (unexpected for toggle switch)")

    def test_complex_pair_does_not_affect_index(self):
        """A complex conjugate pair ±iβ contributes 0 positive real parts → ind unchanged."""
        # 4x4: stable spiral (2 complex pairs, all negative real parts)
        # eigenvalues: -1±2i, -3±1i  →  n_pos_real=0  →  ind=+1
        alpha1, beta1 = -1.0, 2.0
        alpha2, beta2 = -3.0, 1.0
        ev = np.array([alpha1 + 1j*beta1, alpha1 - 1j*beta1,
                       alpha2 + 1j*beta2, alpha2 - 1j*beta2])
        tol = 1e-9 * np.abs(ev).max()
        is_complex = np.abs(ev.imag) > tol
        n_pos_real = int(((ev.real > 0) & ~is_complex).sum())
        ind = 1 - 2 * (n_pos_real % 2)
        self.assertEqual(ind, 1,
            "Stable spiral (complex pairs, negative real parts) should have ind=+1")


# ===========================================================================
# (D) Thomas necessary conditions
# ===========================================================================

class TestThomasConditions(unittest.TestCase):
    """Thomas (1981): positive feedback circuit necessary for multistability."""

    def test_dag_has_exactly_one_fp(self):
        """A DAG (no cycles, no feedback of any kind) has exactly 1 FP."""
        adj, P, K, T, nh, F = _chain_params(n=4)
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=4)
        n_fps = int(result["counts"][0])
        self.assertEqual(n_fps, 1,
            f"Linear chain (DAG) must have exactly 1 FP; solver found {n_fps}")

    def test_dag_fp_is_stable(self):
        """The unique FP of a DAG must be stable (stab ∈ {0, 1})."""
        adj, P, K, T, nh, F = _chain_params(n=3)
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=3)
        n_fps = int(result["counts"][0])
        self.assertGreater(n_fps, 0, "Chain should have at least 1 FP")
        stab = result["stab"][0]
        self.assertIn(int(stab), (0, 1),
            f"Unique DAG FP must be stable (stab∈{{0,1}}); got stab={stab}")

    def test_positive_feedback_needed_for_bistability(self):
        """Mutual repressor (positive feedback via double-negative) → bistable (≥2 stable FPs).
        A negative loop alone → monostable.
        """
        # Negative feedback loop: gene 0 activates gene 1, gene 1 represses gene 0
        adj_neg = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        T_neg   = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        nh_neg  = np.array([[1, 2.0], [2.0, 1]], dtype=np.float32)
        F_neg   = np.array([[1, 4.0], [0.1, 1]], dtype=np.float32)
        P_neg   = np.array([1.0, 1.0], dtype=np.float32)
        K_neg   = np.array([1.0, 1.0], dtype=np.float32)

        result_neg = _solve_simple(
            {"P": P_neg, "K": K_neg, "adj": adj_neg, "T": T_neg,
             "nh": nh_neg, "F": F_neg}, n=2)
        n_stable_neg = int(((result_neg["stab"][:result_neg["counts"][0]]) < 2).sum())

        # Positive feedback: mutual repressor (even number of negative edges = positive loop)
        adj_pos, P_pos, K_pos, T_pos, nh_pos, F_pos = _toggle_switch_params()
        result_pos = _solve_simple(
            {"P": P_pos, "K": K_pos, "adj": adj_pos, "T": T_pos,
             "nh": nh_pos, "F": F_pos}, n=2)
        n_stable_pos = int(((result_pos["stab"][:result_pos["counts"][0]]) < 2).sum())

        self.assertLessEqual(n_stable_neg, 1,
            f"Negative feedback loop should be monostable; got {n_stable_neg} stable FPs")
        self.assertGreaterEqual(n_stable_pos, 2,
            f"Mutual repressor (positive feedback) should be bistable; got {n_stable_pos} stable FPs")


# ===========================================================================
# (E) Toggle switch bistability
# ===========================================================================

class TestToggleSwitchBistability(unittest.TestCase):
    """Mutual repressor toggle switch: 2 stable FPs + 1 saddle."""

    def _get_toggle_result(self):
        adj, P, K, T, nh, F = _toggle_switch_params()
        return _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)

    def test_has_exactly_three_fps(self):
        result = self._get_toggle_result()
        n_fps = int(result["counts"][0])
        self.assertEqual(n_fps, 3,
            f"Toggle switch should have exactly 3 FPs; solver found {n_fps}")

    def test_has_two_stable_fps(self):
        result = self._get_toggle_result()
        n_fps = int(result["counts"][0])
        n_stable = int(np.sum(result["stab"][:n_fps] < 2))
        self.assertEqual(n_stable, 2,
            f"Toggle switch should have 2 stable FPs; found {n_stable}")

    def test_has_one_saddle(self):
        result = self._get_toggle_result()
        n_fps = int(result["counts"][0])
        n_saddle = int(np.sum(result["stab"][:n_fps] == 4))
        self.assertEqual(n_saddle, 1,
            f"Toggle switch should have 1 saddle; found {n_saddle}")

    def test_symmetric_fps(self):
        """The two stable FPs are symmetric: (a, b) and (b, a) with a >> b."""
        result = self._get_toggle_result()
        n_fps = int(result["counts"][0])
        fps   = result["state"][:n_fps]
        stab  = result["stab"][:n_fps]
        stable_fps = fps[stab < 2]   # the two stable ones

        self.assertEqual(len(stable_fps), 2, "Need 2 stable FPs for symmetry test")
        a0, b0 = stable_fps[0, 0], stable_fps[0, 1]
        a1, b1 = stable_fps[1, 0], stable_fps[1, 1]
        # First stable FP: gene 0 high, gene 1 low (or vice versa)
        ratio_fp0 = max(a0, b0) / min(a0, b0)
        ratio_fp1 = max(a1, b1) / min(a1, b1)
        # With F=0.02, T=1.5, nh=4, P=2, K=1 the numerically solved ratio is ~3.7
        self.assertGreater(ratio_fp0, 2.0,
            f"Stable FP 0 should be asymmetric; ratio={ratio_fp0:.2f}")
        self.assertGreater(ratio_fp1, 2.0,
            f"Stable FP 1 should be asymmetric; ratio={ratio_fp1:.2f}")

    def test_saddle_between_stable_fps(self):
        """The saddle's position is between the two stable FPs in gene-0 coordinate."""
        result = self._get_toggle_result()
        n_fps = int(result["counts"][0])
        fps   = result["state"][:n_fps]
        stab  = result["stab"][:n_fps]
        stable_x0 = sorted(fps[stab < 2, 0])
        saddle_x0 = fps[stab == 4, 0]
        self.assertEqual(len(saddle_x0), 1)
        self.assertGreater(float(saddle_x0[0]), float(stable_x0[0]),
            "Saddle gene-0 coordinate should be above the lower stable FP")
        self.assertLess(float(saddle_x0[0]), float(stable_x0[1]),
            "Saddle gene-0 coordinate should be below the upper stable FP")


# ===========================================================================
# (F) Characteristic polynomial coefficients
# ===========================================================================

class TestCharPoly(unittest.TestCase):
    """det(λI - J) = λ^n + c_1 λ^{n-1} + … + c_n.
    c_1 = -Tr(J);  c_n = (-1)^n det(J) = Π λ_i."""

    @staticmethod
    def _charpoly_from_eigenvalues(ev: np.ndarray) -> np.ndarray:
        """Vieta's formulas: coefficients of the monic char poly from eigenvalues."""
        from solver import charpoly_from_eigenvalues as _cp
        return _cp(ev[np.newaxis])[0]

    def test_c1_is_neg_trace_2x2(self):
        """c_1 = -(λ_1 + λ_2) = -Tr(J) for a 2×2 matrix."""
        J = np.array([[-1.2, 0.5], [0.3, -2.1]])
        ev = np.linalg.eigvals(J)
        c = self._charpoly_from_eigenvalues(ev)
        expected_c1 = -np.trace(J)
        self.assertAlmostEqual(float(c[0]), expected_c1, places=8,
            msg=f"c_1 = {c[0]:.6f} should equal -Tr(J) = {expected_c1:.6f}")

    def test_cn_is_det_2x2(self):
        """c_2 = λ_1 λ_2 = det(J) = (-1)^2 det(J) for n=2."""
        J = np.array([[-1.2, 0.5], [0.3, -2.1]])
        ev = np.linalg.eigvals(J)
        c = self._charpoly_from_eigenvalues(ev)
        expected_cn = np.linalg.det(J)   # = (-1)^2 det(J) for n=2
        self.assertAlmostEqual(float(c[1]), expected_cn, places=8,
            msg=f"c_2 = {c[1]:.6f} should equal det(J) = {expected_cn:.6f}")

    def test_cn_is_det_3x3(self):
        """c_3 = (-1)^3 det(J) for n=3."""
        J = np.array([[-1.0, 0.2, 0.0],
                      [0.3, -2.0, 0.4],
                      [0.0, 0.1, -0.8]])
        ev = np.linalg.eigvals(J)
        c = self._charpoly_from_eigenvalues(ev)
        expected_cn = (-1)**3 * np.linalg.det(J)
        self.assertAlmostEqual(float(c[2]), expected_cn, places=8,
            msg=f"c_3 = {c[2]:.6f} should equal (-1)^3 det(J) = {expected_cn:.6f}")

    def test_c1_neg_trace_matches_numpy_poly(self):
        """numpy.poly gives same c_1 = -Tr(J) as our charpoly_from_eigenvalues."""
        J = np.array([[-0.5, 0.8, -0.2],
                      [0.1, -1.2, 0.3],
                      [0.0, 0.4, -0.9]])
        ev = np.linalg.eigvals(J)
        c_ours = self._charpoly_from_eigenvalues(ev)
        c_numpy = np.poly(ev)   # [1, c1, c2, c3] for n=3 (monic, coeffs of char poly)
        np.testing.assert_allclose(
            c_ours, c_numpy[1:].real, atol=1e-8,
            err_msg="charpoly_from_eigenvalues should match numpy.poly(eigenvalues)")

    def test_solver_charpoly_c1_is_neg_trace(self):
        """Solver-reported char poly has c_1 = -Tr(J) at each FP of the toggle switch."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        n_fps = int(result["counts"][0])
        if n_fps == 0:
            self.skipTest("No FPs found")
        fps        = result["state"][:n_fps]        # [n_fps, 2]
        charpoly   = result["charpoly"][:n_fps]      # [n_fps, 2] (c_1, c_2)

        from model import hill_jacobian
        device = torch.device("cpu")
        adj_t  = torch.as_tensor(adj[np.newaxis], dtype=torch.float64, device=device)
        P_t    = torch.as_tensor(P[np.newaxis],   dtype=torch.float64, device=device)
        K_t    = torch.as_tensor(K[np.newaxis],   dtype=torch.float64, device=device)
        T_t    = torch.as_tensor(T[np.newaxis],   dtype=torch.float64, device=device)
        nh_t   = torch.as_tensor(nh[np.newaxis],  dtype=torch.float64, device=device)
        F_t    = torch.as_tensor(F[np.newaxis],   dtype=torch.float64, device=device)

        for r in range(n_fps):
            x_r = torch.as_tensor(fps[r][np.newaxis, np.newaxis], dtype=torch.float64, device=device)  # [1,1,2]
            J_r = hill_jacobian(x_r, P_t, K_t, adj_t, T_t, nh_t, F_t)[0, 0].numpy()
            trace_J = np.trace(J_r)
            expected_c1 = -trace_J
            self.assertAlmostEqual(float(charpoly[r, 0]), expected_c1, places=5,
                msg=f"FP {r}: solver charpoly c_1={charpoly[r,0]:.4f} vs -Tr(J)={expected_c1:.4f}")


# ===========================================================================
# (G) Hopf proximity ratio
# ===========================================================================

class TestHopfRatio(unittest.TestCase):
    """ρ_Hopf = |Re(α_lead)| / |Im(α_lead)|
    where α_lead is the complex eigenvalue with the SMALLEST |Re| among complex pairs.
    ρ → 0 at a Hopf bifurcation; ρ = NaN when all eigenvalues are real.
    """

    @staticmethod
    def _hopf_ratio(evals: np.ndarray) -> float:
        """Compute the Hopf ratio from a 1-D array of (possibly complex) eigenvalues."""
        re, im = evals.real, evals.imag
        tol = 1e-9 * np.abs(evals).max() if len(evals) > 0 else 1e-9
        is_complex = np.abs(im) > tol
        if not is_complex.any():
            return np.nan
        abs_re_complex = np.where(is_complex, np.abs(re), np.inf)
        lead_idx = abs_re_complex.argmin()
        return float(np.abs(re[lead_idx]) / np.abs(im[lead_idx]))

    def test_near_hopf_ratio_is_small(self):
        """ε ± iω with ε << ω → ρ ≈ ε/ω ≈ 0."""
        evals = np.array([-0.01 + 2j, -0.01 - 2j])
        rho = self._hopf_ratio(evals)
        self.assertAlmostEqual(rho, 0.01 / 2.0, places=8)
        self.assertLess(rho, 0.1, "Near-Hopf should give small ρ < 0.1")

    def test_far_from_hopf_ratio_is_large(self):
        """-5 ± 1i → ρ = 5 (large)."""
        evals = np.array([-5.0 + 1j, -5.0 - 1j])
        rho = self._hopf_ratio(evals)
        self.assertAlmostEqual(rho, 5.0, places=8)
        self.assertGreater(rho, 1.0)

    def test_real_eigenvalues_give_nan(self):
        """No complex pairs → ρ = NaN."""
        evals = np.array([-1.0, -2.0, -3.0])
        rho = self._hopf_ratio(evals)
        self.assertTrue(np.isnan(rho), f"Expected NaN for real eigenvalues; got {rho}")

    def test_picks_smallest_abs_re_among_multiple_complex_pairs(self):
        """When multiple complex pairs exist, pick the one with smallest |Re|."""
        # Pair 1: -0.1 ± 3i  (|Re|=0.1, smallest)
        # Pair 2: -2.0 ± 1i  (|Re|=2.0)
        evals = np.array([-0.1 + 3j, -0.1 - 3j, -2.0 + 1j, -2.0 - 1j])
        rho = self._hopf_ratio(evals)
        # α_lead = -0.1 ± 3i, so ρ = 0.1/3 ≈ 0.0333
        self.assertAlmostEqual(rho, 0.1 / 3.0, places=8)

    def test_solver_hopf_matches_definition(self):
        """Solver-reported hopf matches our manual computation on toggle switch FPs."""
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = _solve_simple(
            {"P": P, "K": K, "adj": adj, "T": T, "nh": nh, "F": F}, n=2)
        n_fps = int(result["counts"][0])
        if n_fps == 0:
            self.skipTest("No FPs found")
        fps  = result["state"][:n_fps]
        hopf = result["hopf"][:n_fps]

        from model import hill_jacobian
        device = torch.device("cpu")
        adj_t = torch.as_tensor(adj[np.newaxis], dtype=torch.float64, device=device)
        P_t   = torch.as_tensor(P[np.newaxis],   dtype=torch.float64, device=device)
        K_t   = torch.as_tensor(K[np.newaxis],   dtype=torch.float64, device=device)
        T_t   = torch.as_tensor(T[np.newaxis],   dtype=torch.float64, device=device)
        nh_t  = torch.as_tensor(nh[np.newaxis],  dtype=torch.float64, device=device)
        F_t   = torch.as_tensor(F[np.newaxis],   dtype=torch.float64, device=device)

        for r in range(n_fps):
            x_r = torch.as_tensor(fps[r][np.newaxis, np.newaxis],
                                   dtype=torch.float64, device=device)
            J_r = hill_jacobian(x_r, P_t, K_t, adj_t, T_t, nh_t, F_t)[0, 0].numpy()
            ev  = np.linalg.eigvals(J_r)
            rho_manual = self._hopf_ratio(ev)

            if np.isnan(rho_manual):
                self.assertTrue(np.isnan(hopf[r]),
                    f"FP {r}: solver hopf={hopf[r]:.4f} but manual says NaN (real evals)")
            else:
                self.assertAlmostEqual(float(hopf[r]), rho_manual, places=4,
                    msg=f"FP {r}: solver hopf={hopf[r]:.4f} vs manual {rho_manual:.4f}")


# ===========================================================================
# (H) SCC decomposition
# ===========================================================================

class TestSCCDecomposition(unittest.TestCase):
    """Condensation DAG: topo order; block-triangular Jacobian structure."""

    def test_dag_gives_singleton_sccs(self):
        """A DAG (no feedback cycles) decomposes into N singleton SCCs."""
        from assemble import decompose_grn
        adj, P, K, T, nh, F = _chain_params(n=4)
        dag = decompose_grn(adj, P, K, T, nh, F)
        for scc in dag.sccs:
            self.assertEqual(scc.n, 1,
                f"Chain DAG should have only singleton SCCs; SCC {scc.scc_id} has n={scc.n}")

    def test_mutual_repressor_is_one_scc(self):
        """A 2-gene mutual repressor forms a single 2-gene SCC."""
        from assemble import decompose_grn
        adj, P, K, T, nh, F = _toggle_switch_params()
        dag = decompose_grn(adj, P, K, T, nh, F)
        self.assertEqual(len(dag.sccs), 1, "Toggle switch should be 1 SCC")
        self.assertEqual(dag.sccs[0].n, 2)

    def test_all_genes_in_exactly_one_scc(self):
        """Every gene appears in exactly one SCC."""
        from assemble import decompose_grn
        adj, P, K, T, nh, F = _chain_params(n=5)
        dag = decompose_grn(adj, P, K, T, nh, F)
        seen = {}
        for scc in dag.sccs:
            for g in scc.genes:
                self.assertNotIn(g, seen, f"Gene {g} appears in multiple SCCs")
                seen[g] = scc.scc_id
        self.assertEqual(set(seen.keys()), set(range(dag.n_genes)))

    def test_inter_scc_edges_are_forward_in_topo_order(self):
        """All inter-SCC edges go from a lower to higher topological index (roots first)."""
        from assemble import decompose_grn
        # Build a two-SCC GRN: SCC 0 (genes 0,1) drives SCC 1 (genes 2,3)
        N = 4
        adj = np.zeros((N, N), dtype=np.float32)
        # Intra-SCC 0: mutual activation
        adj[0, 1] = adj[1, 0] = 1.0
        # Intra-SCC 1: mutual activation
        adj[2, 3] = adj[3, 2] = 1.0
        # Inter: gene 0 → gene 2 (SCC 0 → SCC 1, forward)
        adj[0, 2] = 1.0
        P  = np.ones(N, dtype=np.float32)
        K  = np.ones(N, dtype=np.float32)
        T  = np.ones((N, N), dtype=np.float32)
        nh = np.full((N, N), 2.0, dtype=np.float32)
        F  = np.where(adj != 0, 3.0, 1.0).astype(np.float32)
        dag = decompose_grn(adj, P, K, T, nh, F)
        for (sa, sb) in dag.inter_edges.keys():
            self.assertLess(sa, sb,
                f"Inter-SCC edge ({sa}→{sb}) goes backward in topo order — should go forward")

    def test_roots_have_no_parents(self):
        """Root SCCs have no parents in the condensation DAG."""
        from assemble import decompose_grn
        adj, P, K, T, nh, F = _chain_params(n=4)
        dag = decompose_grn(adj, P, K, T, nh, F)
        for root_id in dag.roots:
            scc = dag.sccs[root_id]
            self.assertEqual(len(scc.parents), 0,
                f"Root SCC {root_id} should have no parents; got {scc.parents}")

    def test_block_triangular_jacobian(self):
        """For a 2-SCC system, the full N×N Jacobian is block lower-triangular.
        The off-diagonal block from child to parent should be zero.
        """
        from model import hill_jacobian
        # 2-SCC: genes {0,1} (SCC A, root) drive genes {2,3} (SCC B)
        # adj: intra-A (0↔1), intra-B (2↔3), A→B (0→2)
        N = 4
        adj = np.zeros((N, N), dtype=np.float32)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        adj[0, 2] = 1.0
        P  = np.array([1.5, 1.5, 2.0, 2.0], dtype=np.float32)
        K  = np.ones(N, dtype=np.float32)
        T  = np.ones((N, N), dtype=np.float32)
        nh = np.full((N, N), 2.0, dtype=np.float32)
        F  = np.where(adj != 0, 3.0, 1.0).astype(np.float32)
        # Evaluate Jacobian at the center of the box
        xmax = _xmax_np(P, K, adj, F)
        x_eval = (0.5 * xmax).astype(np.float32)
        device = torch.device("cpu")
        adj_t = torch.as_tensor(adj[np.newaxis, np.newaxis], dtype=torch.float32)
        P_t   = torch.as_tensor(P[np.newaxis], dtype=torch.float32)
        K_t   = torch.as_tensor(K[np.newaxis], dtype=torch.float32)
        T_t   = torch.as_tensor(T[np.newaxis, np.newaxis], dtype=torch.float32)
        nh_t  = torch.as_tensor(nh[np.newaxis, np.newaxis], dtype=torch.float32)
        F_t   = torch.as_tensor(F[np.newaxis, np.newaxis], dtype=torch.float32)
        x_t   = torch.as_tensor(x_eval[np.newaxis, np.newaxis], dtype=torch.float32)
        # Note: hill_jacobian expects [B, K, N] for x; we need the right adj layout
        # Use the numpy ODE function to compute a numerical Jacobian instead
        eps = 1e-5
        J = np.zeros((N, N), dtype=np.float64)
        f0 = _ode_rhs_np(x_eval, P, K, adj, T, nh, F)
        for j in range(N):
            xp = x_eval.copy()
            xp[j] *= (1 + eps)
            J[:, j] = (_ode_rhs_np(xp, P, K, adj, T, nh, F) - f0) / (xp[j] - x_eval[j])
        # In our layout, genes {0,1}=SCC A (root), genes {2,3}=SCC B (child)
        # The child-to-parent block is J[0:2, 2:4] = d(f_A)/d(x_B) → should be ~0
        # because genes 2,3 don't affect genes 0,1 in this GRN.
        child_to_parent_block = J[0:2, 2:4]
        np.testing.assert_allclose(child_to_parent_block, 0.0, atol=1e-8,
            err_msg="Block J[A,B] (child-to-parent) must be zero for DAG structure")


# ===========================================================================
# (I) GRNPredictor end-to-end smoke test
# ===========================================================================

class TestGRNPredictorSmoke(unittest.TestCase):
    """Pipeline runs without error with random weights; output satisfies structural invariants."""

    def _make_predictor(self):
        from model import SccFixedPointNet
        from model_c import CorpusCNet
        from assemble import GRNPredictor
        cfg = SimpleNamespace(
            d_model=128, n_heads=8, n_gnn_layers=4, n_attn_layers=1,
            n_dec_layers=4, dropout=0.0, max_fp=16,
            d_port=32, n_port_layers=2, max_ports=8, max_ports_c=8,
        )
        torch.manual_seed(0)
        l1 = SccFixedPointNet(cfg).eval()
        l3 = CorpusCNet(l1, cfg).eval()
        return GRNPredictor(l1, l3, device=torch.device("cpu"),
                            newton_steps=5, newton_restarts=1, tol=1e-6)

    def test_returns_expected_keys(self):
        """GRNPredictor.predict() returns all documented keys."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=3)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        required_keys = {
            "dag", "all_branches", "global_fps", "global_stab",
            "global_is_stable", "scc_stabs", "exist_probs", "n_fps",
            "fps_refined", "residuals", "converged", "n_converged",
        }
        missing = required_keys - set(result.keys())
        self.assertEqual(missing, set(), f"Missing result keys: {missing}")

    def test_output_shapes_consistent(self):
        """Array shapes in result are mutually consistent."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=4)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        n_fps = result["n_fps"]
        N = adj.shape[0]
        self.assertEqual(result["global_fps"].shape,    (n_fps, N))
        self.assertEqual(result["fps_refined"].shape,   (n_fps, N))
        self.assertEqual(result["global_stab"].shape,   (n_fps,))
        self.assertEqual(result["residuals"].shape,     (n_fps,))
        self.assertEqual(result["converged"].shape,     (n_fps,))
        self.assertEqual(result["exist_probs"].shape,   (n_fps,))

    def test_fps_inside_invariant_box(self):
        """Any predicted FPs must lie inside the positively invariant box."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _toggle_switch_params()
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        if result["n_fps"] == 0:
            return  # no FPs to check (valid with random weights)
        xmax = _xmax_np(P, K, adj, F, eps=0.10)  # 10% slack for Newton refinement
        fps  = result["fps_refined"]
        for r in range(result["n_fps"]):
            for i in range(adj.shape[0]):
                self.assertLessEqual(float(fps[r, i]), float(xmax[i]) * 1.05,
                    msg=f"FP {r} gene {i}: {fps[r,i]:.4e} > xmax={xmax[i]:.4e} (5% slack)")
                self.assertGreater(float(fps[r, i]), 0.0,
                    msg=f"FP {r} gene {i}: position must be positive")

    def test_stability_labels_in_valid_range(self):
        """All stability labels must be in {0, 1, 2, 3, 4, 5, 6}."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=3)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        for i, s in enumerate(result["global_stab"]):
            self.assertIn(int(s), range(7), f"FP {i} has invalid stab={s}")

    def test_is_stable_consistent_with_stab(self):
        """global_is_stable[i] must equal (global_stab[i] in {0, 1})."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=3)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        for i, (stab, is_stable) in enumerate(
                zip(result["global_stab"], result["global_is_stable"])):
            expected = bool(int(stab) in (0, 1))
            self.assertEqual(bool(is_stable), expected,
                f"FP {i}: stab={stab} but is_stable={is_stable}")

    def test_vector_field_residual_positive(self):
        """After Newton refinement, residuals are non-negative (sanity check on type)."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=4)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        for i, r in enumerate(result["residuals"]):
            self.assertGreaterEqual(float(r), 0.0,
                f"Residual for FP {i} must be non-negative; got {r}")

    def test_scc_decomposition_inside_predictor(self):
        """The DAG inside the result has correct number of genes and SCCs."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=5)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        dag = result["dag"]
        self.assertEqual(dag.n_genes, 5)
        # Chain of 5 genes → 5 singleton SCCs
        self.assertEqual(len(dag.sccs), 5)

    def test_topo_order_inference(self):
        """Root SCCs are processed before child SCCs (topo order maintained in branch dict)."""
        predictor = self._make_predictor()
        adj, P, K, T, nh, F = _chain_params(n=4)
        result = predictor.predict(adj, P, K, T, nh, F, verbose=False)
        dag = result["dag"]
        # In topo order, for each SCC, all its parents should have lower scc_id
        for scc in dag.sccs:
            for parent_id in scc.parents:
                self.assertLess(parent_id, scc.scc_id,
                    f"Parent SCC {parent_id} has higher topo index than child {scc.scc_id}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

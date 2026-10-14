"""Tests for the GNN assembly pipeline.

Run on Gilbreth (conda activate 2025.06-py313/gnn):
    cd /path/to/GNN
    pytest tests/test_pipeline.py -v

Individual tests:
    pytest tests/test_pipeline.py::test_degenerate_check_zero_logits -v
    pytest tests/test_pipeline.py::test_degenerate_check_noise_in_high_slots -v
    pytest tests/test_pipeline.py::test_degenerate_check_real_signal -v
    pytest tests/test_pipeline.py::test_solver_fallback_bistable_toggle -v
    pytest tests/test_pipeline.py::test_decompose_grn_toggle -v
    pytest tests/test_pipeline.py::test_predictor_smoke_random_weights -v
    pytest tests/test_pipeline.py::test_shard_glob_pattern -v
    pytest tests/test_pipeline.py::test_corpus_b_sizes_length -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_out(count_logit, exist_logit):
    """Build a minimal model-output dict for _model_is_degenerate tests."""
    return {
        "count_logit": torch.tensor(count_logit, dtype=torch.float32).unsqueeze(0),
        "exist_logit": torch.tensor(exist_logit, dtype=torch.float32).unsqueeze(0),
    }


def _bistable_toggle_scc():
    """SccNode for a 2-gene mutual-repression toggle switch.

    P=2, K=1, T=1.5, nh=4, F=0.02 → exactly 2 stable FPs.
    """
    from assemble import SccNode

    n = 2
    adj  = np.array([[0, -1], [-1, 0]], dtype=np.float32)
    P    = np.array([2.0, 2.0], dtype=np.float32)
    K    = np.array([1.0, 1.0], dtype=np.float32)
    T    = np.ones((2, 2), dtype=np.float32) * 1.5
    nh   = np.ones((2, 2), dtype=np.float32) * 4.0
    F    = np.ones((2, 2), dtype=np.float32)
    F[0, 1] = 0.02
    F[1, 0] = 0.02

    return SccNode(
        scc_id=0, genes=[0, 1], n=n,
        adj_intra=adj, P=P, K=K, T=T, nh=nh, F=F,
        parents=[], children=[],
    )


# ---------------------------------------------------------------------------
# 1. _model_is_degenerate unit tests
# ---------------------------------------------------------------------------

class TestModelIsDegenerate:
    """Unit tests for _model_is_degenerate (does NOT need a trained model)."""

    def test_all_zero_logits_is_degenerate(self):
        from assemble import _model_is_degenerate

        # count_logit: index 1 is the only odd → k=1
        # exist_logit: all zeros → max of top-1 is 0.0 < 0.1
        out = _make_out(
            count_logit=[0.0, 0.0, 0.0, 0.0, 0.0],
            exist_logit=[0.0] * 8,
        )
        assert _model_is_degenerate(out), "all-zero logits should be degenerate"

    def test_uniform_half_probs_is_degenerate(self):
        from assemble import _model_is_degenerate

        # logit = 0 ↔ prob = 0.5 (uniform predictions)
        out = _make_out(
            count_logit=[0.0, 0.001, 0.0, 0.0, 0.0],
            exist_logit=[0.0] * 16,
        )
        assert _model_is_degenerate(out), "near-zero logits should be degenerate"

    def test_negative_noise_in_high_index_slots_still_degenerate(self):
        """Old bug: abs().max() check was bypassed by a strongly-NEGATIVE logit in a far slot.

        If slot 12 has logit=-5.0 (confidently absent), the old code saw abs()=5.0 >= 0.05
        and returned False (not degenerate), even though every selected slot (top-k by
        exist_logit) was near zero.  The new code uses topk(k) which picks the k most
        positive logits, so slot 12 with -5.0 is excluded and slot 0 with 0.02 is the top-1.
        """
        from assemble import _model_is_degenerate

        # count_logit: odd argmax → index 1 → k=1
        # exist_logit: the most positive slot (0) has logit=0.02 (near zero → degenerate);
        #              slot 12 has -5.0 (confidently absent; old abs() saw 5.0 and returned False)
        exist = [-0.01] * 16
        exist[0]  =  0.02   # most positive → decode_slots would pick this slot
        exist[12] = -5.0    # strongly absent; old code: abs()=5.0 ≥ 0.05 → not degenerate (bug)
        out = _make_out(
            count_logit=[0.0, 0.5, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 0.0],
            exist_logit=exist,
        )
        assert _model_is_degenerate(out), \
            "strongly-negative logit in a far slot should NOT rescue the degenerate check"

    def test_real_signal_not_degenerate(self):
        from assemble import _model_is_degenerate

        # count_logit: strong peak at index 3 (k=3 FPs)
        # exist_logit: top-3 slots all have logit > 2.0
        exist = [-1.0] * 16
        exist[0] = 3.5
        exist[1] = 2.8
        exist[2] = 2.1
        out = _make_out(
            count_logit=[0.0, 0.1, 0.0, 4.5, 0.0, 0.2, 0.0, 0.0, 0.0],
            exist_logit=exist,
        )
        assert not _model_is_degenerate(out), "strong signal should NOT be degenerate"

    def test_single_fp_prediction_not_degenerate(self):
        from assemble import _model_is_degenerate

        exist = [0.0] * 8
        exist[0] = 2.0   # confident k=1 prediction
        out = _make_out(
            count_logit=[0.0, 3.0, 0.0, 0.1, 0.0],
            exist_logit=exist,
        )
        assert not _model_is_degenerate(out), "single confident FP should not be degenerate"

    def test_k_clamps_to_1_when_all_even_slots_win(self):
        """Edge case: if masked_fill leaves all -inf (impossible since odd slots exist for max_fp>=1),
        we at least check that k=1 is the floor."""
        from assemble import _model_is_degenerate

        # count_logit has only 1 element (max_fp=0 edge case): falls back to k=1
        exist = [0.05]
        out = _make_out(
            count_logit=[10.0],     # only index 0 (even) → after mask, -inf everywhere → argmax=0 → k=0 → clamp→1
            exist_logit=exist,
        )
        # exist[0] = 0.05 < 0.1 → degenerate
        assert _model_is_degenerate(out)


# ---------------------------------------------------------------------------
# 2. Newton solver fallback for bistable toggle
# ---------------------------------------------------------------------------

class TestSolverFallback:
    """Tests that _solver_fallback_scc finds 2 stable FPs for a bistable toggle."""

    def test_bistable_toggle_finds_two_fps(self):
        from assemble import _solver_fallback_scc

        scc = _bistable_toggle_scc()
        u = np.ones(scc.n, dtype=np.float32)
        device = torch.device("cpu")

        fps = _solver_fallback_scc(scc, u, device)
        assert len(fps) >= 2, f"Expected ≥2 FPs for bistable toggle, got {len(fps)}"

    def test_bistable_toggle_fps_are_stable(self):
        """Both FPs of a clean toggle switch should be stable nodes (class 0 or 1)."""
        from assemble import _solver_fallback_scc
        from assemble import STABLE_CLASSES

        scc = _bistable_toggle_scc()
        u = np.ones(scc.n, dtype=np.float32)
        device = torch.device("cpu")

        fps = _solver_fallback_scc(scc, u, device)
        stable = [fp for fp in fps if fp[1] in STABLE_CLASSES]
        assert len(stable) >= 2, \
            f"Expected ≥2 stable FPs for bistable toggle, stable={len(stable)}/{len(fps)}"

    def test_bistable_toggle_fps_antisymmetric(self):
        """For a symmetric toggle g0↔g1, the two stable FPs should be mirror images:
        FP1 ≈ (high, low) and FP2 ≈ (low, high).  Check that x[0] > x[1] for one and vice versa."""
        from assemble import _solver_fallback_scc, STABLE_CLASSES

        scc = _bistable_toggle_scc()
        u = np.ones(scc.n, dtype=np.float32)
        device = torch.device("cpu")

        fps = _solver_fallback_scc(scc, u, device)
        stable_fps = [fp[0] for fp in fps if fp[1] in STABLE_CLASSES]
        assert len(stable_fps) >= 2

        sorted_fps = sorted(stable_fps, key=lambda x: x[0], reverse=True)
        fp_hi, fp_lo = sorted_fps[0], sorted_fps[1]
        assert fp_hi[0] > fp_hi[1], "In 'g0-high' FP, gene 0 should exceed gene 1"
        assert fp_lo[0] < fp_lo[1], "In 'g1-high' FP, gene 1 should exceed gene 0"


# ---------------------------------------------------------------------------
# 3. SCC decomposition unit tests
# ---------------------------------------------------------------------------

class TestDecomposeGrn:
    """Unit tests for decompose_grn."""

    def test_single_toggle_is_one_scc(self):
        """Two mutually repressing genes form one SCC."""
        from assemble import decompose_grn

        scc = _bistable_toggle_scc()
        adj = np.zeros((2, 2), dtype=np.float32)
        adj[0, 1] = -1.0; adj[1, 0] = -1.0
        dag = decompose_grn(adj, scc.P, scc.K, scc.T, scc.nh, scc.F)
        assert len(dag.sccs) == 1, f"Expected 1 SCC, got {len(dag.sccs)}"
        assert dag.sccs[0].n == 2

    def test_linear_chain_two_sccs(self):
        """Gene 0 → Gene 1 (no back edge) → two singleton SCCs with 0 as root."""
        from assemble import decompose_grn

        N = 2
        adj = np.array([[0, 1], [0, 0]], dtype=np.float32)
        P = np.ones(N, dtype=np.float32)
        K = np.ones(N, dtype=np.float32)
        T = np.ones((N, N), dtype=np.float32)
        nh = np.ones((N, N), dtype=np.float32)
        F  = np.ones((N, N), dtype=np.float32)
        F[0, 1] = 3.0

        dag = decompose_grn(adj, P, K, T, nh, F)
        assert len(dag.sccs) == 2, f"Expected 2 SCCs (two singletons), got {len(dag.sccs)}"
        assert len(dag.roots) == 1, f"Expected 1 root, got {len(dag.roots)}"

    def test_multistable_grn_scc_count(self):
        """decompose_grn on a multistable GRN should find >= n_designed_sccs SCCs.

        The actual count may exceed n_sccs because the root's random intra-edges
        may not form a single strongly-connected component, splitting it into several
        singleton/small SCCs.  We only assert the DAG is non-trivial and has a root.
        """
        from assemble import decompose_grn
        from demo_assemble import make_multistable_grn

        grn = make_multistable_grn(n_bistable_sccs=2, extra_scc_size=2, seed=0)
        dag = decompose_grn(
            grn["adj"], grn["P"], grn["K"], grn["T"], grn["nh"], grn["F"],
        )
        assert len(dag.sccs) >= grn["n_sccs"], \
            f"Expected >= {grn['n_sccs']} SCCs, got {len(dag.sccs)}"
        assert len(dag.roots) >= 1, "DAG should have at least one root"


# ---------------------------------------------------------------------------
# 4. GRNPredictor smoke test with random-weight models
# ---------------------------------------------------------------------------

class TestGRNPredictorSmoke:
    """Full pipeline smoke test: random-weight models → Newton fallback fires → ≥1 FP found."""

    @pytest.fixture(scope="class")
    def grn_small(self):
        """Small 2-SCC GRN: root monostable + 1 bistable toggle."""
        from demo_assemble import make_multistable_grn
        return make_multistable_grn(n_bistable_sccs=1, extra_scc_size=2, seed=7)

    @pytest.fixture(scope="class")
    def random_models(self):
        """Random-weight L1 + L3 models (no checkpoint needed)."""
        from types import SimpleNamespace
        from model import SccFixedPointNet
        from model_c import CorpusCNet

        cfg = SimpleNamespace(
            d_model=64, n_heads=4, n_gnn_layers=2, n_attn_layers=1,
            n_dec_layers=2, dropout=0.0, max_fp=8,
            d_port=16, n_port_layers=1, max_ports=4, max_ports_c=4,
        )
        device = torch.device("cpu")
        l1 = SccFixedPointNet(cfg).to(device).eval()
        l3 = CorpusCNet(l1, cfg).to(device).eval()
        return l1, l3, device

    def test_predictor_runs_without_crash(self, grn_small, random_models):
        from assemble import GRNPredictor

        l1, l3, device = random_models
        predictor = GRNPredictor(l1, l3, device=device, refine_steps=0)
        result = predictor.predict(
            grn_small["adj"], grn_small["P"], grn_small["K"],
            grn_small["T"], grn_small["nh"], grn_small["F"],
        )
        assert "global_fps" in result
        assert "n_fps" in result

    def test_predictor_finds_at_least_one_fp(self, grn_small, random_models):
        """Random-weight models should be detected as degenerate → Newton fallback fires
        and should find ≥1 FP for a bistable GRN."""
        from assemble import GRNPredictor

        l1, l3, device = random_models
        predictor = GRNPredictor(l1, l3, device=device, refine_steps=0)
        result = predictor.predict(
            grn_small["adj"], grn_small["P"], grn_small["K"],
            grn_small["T"], grn_small["nh"], grn_small["F"],
        )
        assert result["n_fps"] >= 1, \
            f"Newton fallback should find ≥1 FP, got {result['n_fps']}"

    def test_predictor_finds_two_fps_for_bistable(self, random_models):
        """A standalone (unforced) bistable toggle should yield 2 global FPs.

        We construct a minimal 2-SCC GRN: one monostable root singleton feeding
        a 2-gene bistable toggle with no forcing that kills the bistability.
        """
        from assemble import GRNPredictor, decompose_grn

        # Build the smallest GRN that guarantees 2 global FPs:
        # gene 0 (root, monostable) → gene 1, gene 2 (toggle, no inter-SCC feedback)
        N = 3
        adj = np.zeros((N, N), dtype=np.float32)
        P   = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        K   = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        T   = np.ones((N, N), dtype=np.float32)
        nh  = np.ones((N, N), dtype=np.float32)
        F   = np.ones((N, N), dtype=np.float32)
        # Toggle: genes 1 ↔ 2 mutual repression
        adj[1, 2] = -1.0; adj[2, 1] = -1.0
        T[1, 2] = 1.5; T[2, 1] = 1.5
        nh[1, 2] = 4.0; nh[2, 1] = 4.0
        F[1, 2] = 0.02; F[2, 1] = 0.02
        # Root → toggle: weak activating (should not kill bistability)
        adj[0, 1] = 1.0; adj[0, 2] = 1.0
        T[0, 1] = 0.5; T[0, 2] = 0.5
        nh[0, 1] = 2.0; nh[0, 2] = 2.0
        F[0, 1] = 1.5; F[0, 2] = 1.5

        l1, l3, device = random_models
        predictor = GRNPredictor(l1, l3, device=device, refine_steps=0)
        result = predictor.predict(adj, P, K, T, nh, F)
        assert result["n_fps"] >= 2, \
            f"Bistable toggle should yield ≥2 global FPs, got {result['n_fps']}"

    def test_global_fps_shape(self, grn_small, random_models):
        from assemble import GRNPredictor

        l1, l3, device = random_models
        predictor = GRNPredictor(l1, l3, device=device, refine_steps=0)
        result = predictor.predict(
            grn_small["adj"], grn_small["P"], grn_small["K"],
            grn_small["T"], grn_small["nh"], grn_small["F"],
        )
        if result["n_fps"] > 0:
            fps = result["global_fps"]
            assert fps.ndim == 2
            assert fps.shape[1] == grn_small["N"], \
                f"FP should have N={grn_small['N']} genes, got shape {fps.shape}"


# ---------------------------------------------------------------------------
# 5. Data pipeline smoke tests
# ---------------------------------------------------------------------------

class TestDataPipeline:
    """Smoke tests for corpus B/C shard glob patterns and _sizes metadata."""

    def test_corpus_b_shard_glob_pattern(self):
        """SHARD_GLOB_B should match shard_b_NNNN.npz but not shard_NNNN.npz."""
        import fnmatch
        from data_b import SHARD_GLOB_B

        assert fnmatch.fnmatch("shard_b_0000.npz", SHARD_GLOB_B), \
            f"SHARD_GLOB_B={SHARD_GLOB_B!r} should match shard_b_0000.npz"
        assert fnmatch.fnmatch("shard_b_9999.npz", SHARD_GLOB_B), \
            f"SHARD_GLOB_B={SHARD_GLOB_B!r} should match shard_b_9999.npz"
        assert not fnmatch.fnmatch("shard_0000.npz", SHARD_GLOB_B), \
            f"SHARD_GLOB_B={SHARD_GLOB_B!r} should NOT match corpus-A shard_0000.npz"

    def test_corpus_b_sizes_code_has_6_elements(self):
        """Check the _sizes assignment in corpus_b_generator has exactly 6 entries.

        We locate the out["_sizes"] = np.array([...]) block in the source and count
        the number of lines that contain a value expression (one per entry).
        """
        import inspect
        import corpus_b_generator

        lines = inspect.getsource(corpus_b_generator).splitlines()

        # Find the line: out["_sizes"] = np.array([
        start_idx = None
        for i, line in enumerate(lines):
            if 'out["_sizes"]' in line and 'np.array' in line:
                start_idx = i
                break
        assert start_idx is not None, \
            'Could not find out["_sizes"] = np.array([...]) in corpus_b_generator.py'

        # Count element lines: collect until the closing ], dtype= line
        n_elements = 0
        for line in lines[start_idx + 1:]:
            stripped = line.strip()
            if stripped.startswith(']'):
                break
            if stripped and not stripped.startswith('#'):
                n_elements += 1

        assert n_elements == 6, \
            f"corpus_b_generator _sizes should have 6 elements, found {n_elements}"

    def test_corpus_c_generator_ep_padding_is_ones(self):
        """corpus_c_generator: verify ep_T/ep_NH/ep_F neutral padding changed from zeros to ones.

        We check the source code for 'np.ones' rather than 'np.zeros' in the ep_T/ep_NH/ep_F
        initialization (structural test — no solver required).
        """
        import inspect
        import corpus_c_generator

        src = inspect.getsource(corpus_c_generator)
        # Find the block that initializes ep_T, ep_NH, ep_F
        # Expect something like: ep_T = np.ones((N_MAX, N_MAX), ...)
        for var in ("ep_T", "ep_NH", "ep_F"):
            # Find assignment lines for this variable
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{var}") and "=" in stripped and "np." in stripped:
                    assert "np.ones" in stripped, \
                        f"{var} neutral padding should use np.ones, found: {stripped!r}"
                    break


# ---------------------------------------------------------------------------
# 6. Model B FiLM injection test
# ---------------------------------------------------------------------------

class TestModelBFiLM:
    """Check that FiLMInjection doesn't crash on negative port-gene indices (from mask)."""

    def test_film_injection_masked_genes_dont_crash(self):
        """FiLMInjection(h, M, port_gene, port_mask) should not crash or NaN on masked genes."""
        from model_b import FiLMInjection

        m_dim, d_model = 16, 32
        B, N_MAX, K_ports = 2, 6, 3
        film = FiLMInjection(m_dim, d_model)

        h         = torch.randn(B, N_MAX, d_model)
        M         = torch.randn(B, m_dim)                                    # aggregated message
        port_gene = torch.tensor([[0, 2, -1], [1, -1, -1]], dtype=torch.long)
        port_mask = (port_gene >= 0)                                          # BoolTensor [B, L]

        out = film(h, M, port_gene, port_mask)
        assert out.shape == h.shape, f"FiLM output shape mismatch: {out.shape} vs {h.shape}"
        assert not torch.isnan(out).any(), "FiLM output contains NaN"
        assert not torch.isinf(out).any(), "FiLM output contains Inf"

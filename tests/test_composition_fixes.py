"""Regression tests for the composition inference / fine-tuning fixes.

  - the SCC batch carries tc, has_pos_fb and the physical (forced) parameters
  - Layer III is invariant to how far a batch is padded (portrait and forcing statistics)
  - the batched Corpus C path and the assembly path give the count head the same inputs
  - per-branch labels reproduce the ground-truth fixed-point count (sum formula)
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from assemble import SccNode, _make_scc_batch, decompose_grn, thomas_class
from composition_labels import case_consistent, enumerate_count, extract_branch_labels
from data import N_MAX

CFG = dict(d_model=32, n_heads=4, n_gnn_layers=2, n_attn_layers=1, n_dec_layers=2,
           dropout=0.0, max_fp=8, d_port=16, n_port_layers=1, max_ports=4, max_ports_c=4)


def _toggle_scc():
    adj = np.array([[0, -1, 0], [-1, 0, 1], [1, 0, 0]], dtype=np.float32)
    return SccNode(scc_id=1, genes=[3, 4, 5], n=3, adj_intra=adj,
                   P=np.array([2.0, 3.0, 1.5], np.float32), K=np.ones(3, np.float32),
                   T=np.full((3, 3), 0.7, np.float32), nh=np.full((3, 3), 2.0, np.float32),
                   F=np.where(adj > 0, 20.0, np.where(adj < 0, 0.05, 1.0)).astype(np.float32),
                   parents=[0])


@pytest.fixture(scope="module")
def models():
    from model import SccFixedPointNet
    from model_c import CorpusCNet
    torch.manual_seed(0)
    l1 = SccFixedPointNet(SimpleNamespace(**CFG)).eval()
    l3 = CorpusCNet(l1, SimpleNamespace(**CFG)).eval()
    return l1, l3


class TestSccBatch:
    def test_thomas_class(self):
        assert thomas_class(np.array([[0, 1], [1, 0]]))[0] == 2      # positive loop only
        assert thomas_class(np.array([[0, 1], [-1, 0]]))[0] == 1     # negative loop only
        tc, has_pos = thomas_class(_toggle_scc().adj_intra)
        assert (tc, has_pos) == (3, True)

    def test_batch_carries_tc_and_physics(self):
        scc = _toggle_scc()
        u = np.array([4.0, 1.0, 0.5], np.float32)
        raw = _make_scc_batch(scc, u, torch.device("cpu"), frame="raw")
        forced = _make_scc_batch(scc, u, torch.device("cpu"), frame="forced")
        assert int(raw["tc"]) == 3 and bool(raw["has_pos_fb"])
        assert raw["P"].shape == (1, 3)                               # trimmed to the SCC
        np.testing.assert_allclose(raw["P"][0].numpy(), scc.P)
        np.testing.assert_allclose(forced["P"][0].numpy(), scc.P * u, rtol=1e-6)
        for b in (raw, forced):
            np.testing.assert_allclose(b["P_phys"][0].numpy(), scc.P * u, rtol=1e-6)
        assert torch.allclose(forced["xmax"], forced["xmax_phys"])


class TestPaddingInvariance:
    def _head_inputs(self, l3, batch):
        rows = []
        h = l3.forcing_count_head.register_forward_hook(lambda m, a, o: rows.append(a))
        with torch.no_grad():
            l3(batch, freeze_backbone=True)
        h.remove()
        return rows[0]

    @pytest.mark.parametrize("frame", ["raw", "forced"])
    def test_l3_count_inputs_do_not_depend_on_padding(self, models, frame):
        _, l3 = models
        scc = _toggle_scc()
        u = np.array([1.0, 1.02, 30.0], np.float32)   # one weakly forced gene: log|u-1| < 0
        a = self._head_inputs(l3, _make_scc_batch(scc, u, torch.device("cpu"), frame=frame))
        b = self._head_inputs(l3, _make_scc_batch(scc, u, torch.device("cpu"), frame=frame,
                                                  pad_to=N_MAX))
        for x, y in zip(a, b):
            assert torch.allclose(x.float(), y.float(), atol=1e-5)

    def test_corpus_c_collate_matches_assembly_batch(self, models):
        import data_c
        _, l3 = models
        scc = _toggle_scc()
        u = np.array([3.0, 1.0, 0.2], np.float32)

        def pad(a, fill):
            return np.pad(a.astype(np.float32), [(0, N_MAX - 3)] * a.ndim, constant_values=fill)

        item = {"n": 3, "k": 1, "P": pad(scc.P, 1), "K": pad(scc.K, 1),
                "adj": pad(scc.adj_intra, 0), "T": pad(scc.T, 1), "nh": pad(scc.nh, 1),
                "F": pad(scc.F, 1), "u": pad(u, 1),
                "fp_state": np.zeros((8, N_MAX), np.float32),
                "fp_charpoly": np.zeros((8, N_MAX), np.float32),
                "fp_stab": np.full(8, -1, np.int64), "consistent": True, "tc": 3,
                "n_parents": 1, "is_root": False, "dag_all_solved": True}
        # A second, larger SCC in the batch forces padding of the first one.
        big = dict(item, n=5, adj=np.pad(np.ones((5, 5), np.float32) - np.eye(5, dtype=np.float32),
                                         [(0, N_MAX - 5)] * 2))
        for frame in ("raw", "forced"):
            batch = data_c.apply_frame(data_c.collate_c([item, big]), frame)
            a = self._head_inputs(l3, batch)
            b = self._head_inputs(l3, _make_scc_batch(scc, u, torch.device("cpu"), frame=frame))
            for x, y in zip(a, b):
                assert torch.allclose(x[0].float(), y[0].float(), atol=1e-4), frame


class TestBranchLabels:
    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls):
        """A few A->B compositions of random 3-gene SCCs, solved exactly."""
        import importlib.util
        from pathlib import Path
        from generate import sample_params
        from topology import strongly_connected
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "gce", root / "exhaust_gen" / "gen_composition_eval.py")
        gce = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gce)
        rng = np.random.default_rng(3)
        out = []
        while len(out) < 12:
            blocks = []
            for _ in range(2):
                while True:
                    s = np.where(rng.random((3, 3)) < 0.55,
                                 np.where(rng.random((3, 3)) < 0.6, 1, -1), 0).astype(np.int8)
                    if strongly_connected((s != 0)[None])[0]:
                        break
                blocks.append((s.astype(np.float32), sample_params(1, s[None], rng)))
            cross = [(int(rng.integers(3)), int(rng.integers(3)), int(rng.choice([-1, 1])))
                     for _ in range(2)]
            adj, par = gce.compose_6gene(blocks[0][0], blocks[0][1], blocks[1][0], blocks[1][1],
                                         cross, rng)
            fps, _, _ = gce.solve_one(par, torch.device("cpu"))
            if len(fps) % 2 == 1:
                out.append((adj, {k: v[0] for k, v in par.items()}, fps))
        return out

    def test_labels_reproduce_gt_count(self, cases):
        n_multi = 0
        for adj, par, fps in cases:
            dag = decompose_grn(adj, par["P"], par["K"], par["T"], par["nh"], par["F"])
            labels = extract_branch_labels(dag, fps, par["T"], par["nh"], par["F"])
            if case_consistent(labels):
                assert enumerate_count(dag, labels) == len(fps)
            n_multi += len(fps) > 1
        assert n_multi > 0   # the check must cover multistable compositions

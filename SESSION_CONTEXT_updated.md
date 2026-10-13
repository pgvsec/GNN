# GNN Project — Session Context for Handoff

**Last updated:** 2026-10-13  
**GitHub repo:** `pgvsec/GNN`  
**Active dev branch:** `claude/eloquent-cerf-l0zqbo` (HEAD `9fdc88a`)  
**Stable branch:** `main` (HEAD `3c500f8`) — Layer I + Layer II/III infrastructure  
**Cluster A (GPU training):** Purdue Gilbreth (`gilbreth.rcac.purdue.edu`)  
**Cluster B (CPU corpus work):** Purdue Negishi (`negishi.rcac.purdue.edu`)  
**Scratch path (Gilbreth):** `/scratch/gilbreth/gupt1042/`  
**Scratch path (Negishi):** `~/python/`  
**Conda env:** `gnn` (via `anaconda/2025.06-py313`)

---

## 1. What This Project Is

A **three-layer Graph Neural Network** for predicting equilibrium (fixed-point) structure of Gene Regulatory Networks (GRNs) modelled by Hill-function ODEs:

$$\dot{x}_i = P_i \prod_{j \to i} H_{ji}(x_j) - K^{\text{deg}}_i x_i, \quad H = \frac{F r + 1}{r + 1},\; r = \left(\frac{x_j}{T}\right)^n$$

The three layers are trained on three different corpora (A, B, C), with later layers conditioning on the output of earlier frozen ones.

---

## 2. Architecture Overview

### Layer I — Autonomous SCC Portrait (`model.py`, `train.py`)
- **Encoder:** `NodeEncoder` + `PairEncoder` → 4× `GraphNetBlock` (directed message-passing) → 1× `NodeAttentionBlock` → `GraphReadout`
- **Decoder (`D_φ`):** `SlotDecoder` (4 transformer layers) → heads for:
  - `exist_logit` [B, K] — slot existence
  - `stab_logit` [B, K, 7] — stability class (7 classes)
  - `pos` [B, K, N] — fixed-point position (log-encoded, decoded as fraction of `xmax`)
  - `charpoly` [B, K, N] — characteristic polynomial coefficients (symlog/rate-normalised)
  - `count_logit` [B, K+1] — number of fixed points
  - `stable_count_logit` [B, K+1] — number of stable fixed points
- **Key constants:** `N_MAX=10`, `K=max_fp=16`, `NUM_STABILITY=7`, `d_model=128`
- **Architecture class:** `SccFixedPointNet` in `model.py`

**PATCH:** `SccFixedPointNet.forward()` exposes three intermediate tensors for Layer II/III:
```python
"slot_emb":  s,   # branch tokens q^(r)   [B, K, d_model]
"graph_emb": g,   # z_j^0                 [B, d_model]
"node_emb":  h,   # h_jv^0               [B, N, d_model]
```
These three lines were added inside the `torch.autocast(..., enabled=False)` block at the end of `forward()` (just before the final `return out` statement).

### Layer II/III — Forced Response (`model_b.py`, `train_layer2.py`)

Takes a frozen Layer I backbone and adds three trainable modules (~0.67M new params vs 2.23M frozen):

1. **`PhiPsi` (Φ_ψ):** Per-port signal MLP.  
   Input: `[u_k (1) | z_j^0 (d) | h_{b_k}^0 (d) | log(T)/3, nh/5, log(F)/5 (3)]` → `m_k ∈ R^{m_dim}`

2. **`AEta` (A_η):** DeepSets aggregator (permutation-invariant over ports).  
   `rho`-MLP per element → masked mean → `phi`-MLP → LayerNorm → `M_j ∈ R^{m_dim}`

3. **`UOmega` (U_ω):** FiLM-conditioned GNN update.
   - Portrait summary: flattens `[exist_logit | stab_logit | pos_padded]` → MLP → `portrait ∈ R^d`
   - Node-level conditioning bias broadcast from `[M | portrait]`
   - `FiLMInjection`: generates `γ, β` from `M`; scatter-adds `(γ−1)·h_{gene} + β` at each active port gene
   - 2× `GraphNetBlock` refinement layers
   - Graph update: `z_cond = LayerNorm(z^0 + MLP([z^0 | mean(h_cond)]))`

4. **`CorpusBNet`:** Wrapper combining frozen backbone + Φ_ψ + A_η + U_ω.  
   Forward:
   - Re-runs backbone sub-modules via `_run_backbone()` → `z^0`, `h^0`, `edge_emb`, autonomous portrait
   - Gathers `h_b = h^0[:, port_gene, :]` (forced gene embeddings)
   - Φ_ψ messages → A_η aggregate `M` → U_ω condition → D_φ decode conditioned portrait

**Critical dimension fix:** `portrait_raw_dim = max_fp * (1 + NUM_STABILITY + N_MAX)` = `16*(1+7+10) = 288`. At runtime `pos [B, K, n]` where `n = actual graph size ≤ N_MAX=10`. Before flattening, `pos` is **zero-padded to N_MAX along the last dim** so the input to `portrait_proj` is always exactly 288:
```python
if pos.shape[-1] < _N_MAX:
    pad = torch.zeros(B, K_port, _N_MAX - pos.shape[-1], ...)
    pos_pad = torch.cat([pos, pad], dim=-1)
```

**Bug fix (commit `5a3d05c`):** `FiLMInjection.forward()` crashed on GPU because `port_gene` uses `-1` as a padding sentinel for inactive ports (fewer than `MAX_PORTS` active). GPU scatter/gather kernels reject negative indices even when the corresponding `active` mask is 0. Fix: clamp before indexing.
```python
# BEFORE (crashes on GPU):
h_k = h[torch.arange(B, device=h.device), gene_k]
idx = gene_k.view(B, 1, 1).expand(B, 1, d)

# AFTER (safe):
gene_k_safe = gene_k.clamp(min=0)   # -1 → 0; active=0 so inc is zero anyway
h_k = h[torch.arange(B, device=h.device), gene_k_safe]
idx = gene_k_safe.view(B, 1, 1).expand(B, 1, d)
```

---

## 3. Data Pipeline

### Corpus A (`data.py`) — Layer I training data
- Memory-mapped `.npy` cache under a single directory
- 5 key groups: `g_*` (graph), `e_*` (edge), `s_*` (sample), `ep_*` (edge-params), `fp_*` (fixed-points)
- Key constants in `data.py`: `N_MAX=10`, `NUM_STABILITY=7`, `SPLIT_TRAIN=1`, `SPLIT_VAL=2`, `SPLIT_TEST=3`
- `make_loader()` → returns `(dataset, DataLoader)` with `collate_fn` that trims to max active nodes/fps in batch

> ⚠️ **Branch discrepancy:** The `main` branch session context shows `SPLIT_TRAIN=0, SPLIT_VAL=1, SPLIT_TEST=2`. The `claude/eloquent-cerf-l0zqbo` branch `data.py` uses `SPLIT_TRAIN=1, SPLIT_VAL=2, SPLIT_TEST=3`. **Reconcile before merging branches.**

### Corpus B (`data_b.py`) — Layer II/III training data (forced single-SCC)

**NEW FILE** written on branch `claude/eloquent-cerf-l0zqbo`. Key design:

- Same shard format as Corpus A, plus a 6th `cb_*` group for variable-length port forcing arrays
- Extra pointer: `s_cb_ptr` → into `cb_port_*` arrays
- **6 groups:** `GROUP_ORDER_B = ("g", "e", "s", "ep", "fp", "cb")`
- **`MAX_PORTS = 3`** (cap on simultaneous forced genes in this branch)
- Shard files named `shard_b_NNNN.npz` (not `shard_NNNN.npz` like Corpus A); `SHARD_GLOB_B = "shard_b_[0-9][0-9][0-9][0-9].npz"` (commit `e0ce08e`)
- `_report_cache_b` fixed to index `cb_stratum` per-sample via `s_cb_ptr` (commit `3b6d0c2`):
  ```python
  stratum = np.asarray(a["cb_stratum"])[np.asarray(a["s_cb_ptr"])]
  ```
- `collate_b` produces `port_mask` (commit `c68e73d`):
  ```python
  "port_mask": (torch.arange(MAX_PORTS).unsqueeze(0) < np_.unsqueeze(1)),  # [B, MAX_PORTS]
  ```

> ⚠️ **Branch discrepancy:** The `main` branch `data_b.py` uses `MAX_PORTS=8`. **Reconcile before merging.** The Corpus B generator must match whichever value is chosen.

- `_group_of(key)`: resolves prefix by sorting GROUP_ORDER_B by length descending (avoids `"ep_T"` matching `"e"`)
- `build_cache_b(corpus_dir, cache_dir)`: **requires 6-element `_sizes`**; raises `ValueError` for old 5-element shards
- `load_sample_b(a, s, max_fp, max_ports=3)`: returns n, k, n_ports, P, K, xmax, adj, T, nh, F, fp_state, fp_charpoly, fp_stab, port_gene, port_u, port_F, port_T, port_nh, has_pos_fb, consistent, stratum, bif_type, bif_crossed, tc
- `ep_nh` stored as uint8 in shards; converted with `.astype(np.float32)` in `load_sample_b`
- **Corpus B absorbs forcing into P at generation time** (`apply_forcing` in `corpus_b_generator.py`): `P_forced[b_k] = P[b_k] * u_k`. So `batch["P"]` already contains `P_forced` and `SetCriterion` works unchanged.
- `has_pos_fb` comes from `g_pos_fb` topology key
- `CorpusBDataset`, `collate_b`, `make_loader_b(cfg, split, train)`, `describe_dataset_b`
- Val metrics: `batch["stratum"] → batch["phase"]` alias for `MetricAccumulator` group keys

### Corpus C (`data_c.py`, `model_c.py`) — Layer III training data (multi-SCC networks)

**NEW FILE** written on branch `claude/eloquent-cerf-l0zqbo`. Key design:

- SCC-DAG decomposition: genes partitioned into SCCs; condensation DAG provides forcing context
- **One sample = one branch** (one SCC under one set of parent FP choices)
- **6 groups:** `GROUP_ORDER_C = ("dag", "scc", "iedge", "branch", "link", "bfp")`
- **CSR sentinel pointer arrays** `[n+1]` shape (unlike A/B which use `[n]` start-of-block arrays):
  ```python
  SENTINEL_PTRS_C = {
      "dag_scc_ptr":     "scc",
      "dag_edge_ptr":    "iedge",
      "scc_branch_ptr":  "branch",
      "branch_fp_ptr":   "bfp",
      "branch_link_ptr": "link",
  }
  ```
- **Absolute index arrays** shifted during cache merge:
  ```python
  ABS_INDEX_C = {
      "link_parent_branch": "branch",
      "iedge_src_scc":      "scc",
      "iedge_tgt_scc":      "scc",
      "branch_scc":         "scc",
      "scc_dag_id":         "dag",
  }
  ```
- `build_cache_c`: sentinel arrays handled specially — allocates `(total+1)` shaped arrays, initializes `[0]=0`, writes `z[k][1:] + pos[tgt]` per shard (drops duplicate leading zero)
- `load_branch(a, b, max_fp)`: returns SCC intra-params (P, K, adj, T, nh, F), forcing vector `u` (N_MAX, 1.0 for unforced genes), FP arrays, consistent, tc, n_parents, is_root, dag_all_solved
- `CorpusCDataset`: per-branch, optionally filtered with `only_solved=True` (uses `dag_all_solved`)
- `collate_c`, `make_loader_c(cfg, train, only_solved=False)`, `describe_dataset_c`
- **Note:** Corpus C shards do not store `xmax`; can be recomputed from P, K, adj, F, u if needed

**Cache paths on cluster:**
- Corpus A: `/scratch/gilbreth/gupt1042/corpusA/cache`
- Corpus B shards: `~/python/corpusB_new/` on Negishi ✅ regenerated (609 shards)
- Corpus B cache: `/scratch/gilbreth/gupt1042/corpusB_new/cache` ✅ built and transferred (43 .npy files, 825,733 samples)
- Corpus C shards: `~/python/corpusC/` on Negishi ✅ 256 shards generated
- Corpus C cache: `/scratch/gilbreth/gupt1042/corpusC/cache` on Gilbreth ✅ built and transferred
- Layer III training: ✅ COMPLETE (60 epochs, `best.pt` saved)

---

## 4. Losses (`losses.py`)

`SetCriterion` implements Hungarian-matching bipartite loss:
- **Matching costs:** `cost_exist`, `cost_pos`, `cost_cp`, `cost_stab`
- **Losses:** exist (BCE), pos (Huber), charpoly (Huber), stab (CE), count (CE, odd-only), phys (ODE residual), thomas (Thomas index constraint, requires `has_pos_fb`)
- Loss weights: `w_exist=1, w_pos=5, w_cp=2, w_stab=1, w_count=0.5, w_phys=1, w_thomas=0.5`

**`batch_metrics()`** computes: `count_exact`, `regime_acc`, `stable_count_exact`, `fp_recall`, `stab_acc`, `pos_decades`, `brouwer_idx_acc`, `brouwer_consistent`, `thomas_violation`, `refined_converged`

`pos_decades`: mean absolute position error in log₁₀ units. **Lower = better.** Computed as `slot_error * LOG_SPAN / ln(10)` where `LOG_SPAN = -log(1e-6) ≈ 13.816`.

---

## 5. Training

### Layer I (`train.py`, `train_layer1.slurm`)

```bash
sbatch train_layer1.slurm
```
- Config: 60 epochs, batch_size=256, d_model=128, n_gnn_layers=4, n_attn_layers=1, n_dec_layers=4, n_heads=8, lr=3e-4, AMP=true
- **Best checkpoint:** epoch 39 (used as Layer I backbone — note: `train_layer2.py` loaded it reporting "epoch 57", likely a 1-indexed vs 0-indexed difference)
- **Checkpoint path:** `/scratch/gilbreth/gupt1042/runs/layer1/best.pt`
- Checkpoint format: `{"model": ..., "ema": ..., "opt": ..., "sched": ..., "epoch": ..., "best": ...}`
- Selection score: `count_exact + fp_recall`

### Layer II (`train_layer2.py`, `train_layer2.slurm`) ✅ COMPLETE

```bash
sbatch train_layer2.slurm
```
- Loads Layer I EMA weights from `best.pt`, freezes backbone (`requires_grad_(False)`)
- Trains only Φ_ψ + A_η + U_ω (~0.67M params out of 2.89M total)
- Config: 60 epochs, batch_size=128, d_model=128, m_dim=128, n_u_layers=2, lr=1e-4, AMP=true, freeze_backbone=true
- Corpus B cache: `/scratch/gilbreth/gupt1042/corpusB_new/cache` (825,733 train + 20,000 val samples)
- Output dir: `/scratch/gilbreth/gupt1042/runs/layer2/`
- Same checkpoint format as Layer I

**Training completed — 60 epochs, SLURM job 11929852 (Gilbreth, partition a10)**

**Final results (best.pt = epoch 35, selection score 1.6794):**

| Metric | Value |
|--------|-------|
| count_exact | 0.8763 |
| fp_recall | 0.8031 |
| selection score (`count_exact + fp_recall`) | **1.6794** |
| stab_acc | 0.9976 |
| brouwer_idx_acc | 0.9875 |
| brouwer_consistent | 0.9860 |
| thomas_violation | 0.0000 ✅ |
| val loss (epoch 35) | ~0.524 |

**Artifacts committed to GitHub (via Git LFS):** `artifacts/layer2/best.pt`, `artifacts/layer2/last.pt`, `artifacts/layer2/log.csv`

**Note on training dynamics:** Val loss reached minimum ~0.5205 around epoch 19, then rose to ~0.564 by epoch 60 as LR cosine-decayed to near zero (~2e-6) and training loss continued falling — mild overfitting late in schedule. The `selection_score` metric (maximized for `best.pt`) peaked at epoch 35, which is a better checkpoint than the val-loss minimum.

---

## 6. Bug Fixes Applied (branch `claude/eloquent-cerf-l0zqbo`)

### Fix 1 — Corpus B: Stratum mislabelling (CRITICAL)
**File:** `corpus_b_generator.py`, `sample_u_vals()`, the `n2` fallback else-branch  
**Problem:** When no opposite-regime candidates exist, the code fell back to a broad draw but labeled it stratum 1 (near-bifurcation) instead of stratum 2 (opposite-regime attempt).  
```python
# BEFORE (wrong):
strata.append(1); bif_types.append(BIF_NONE); bif_us.append(np.nan)
# AFTER (correct):
strata.append(2); bif_types.append(BIF_NONE); bif_us.append(np.nan)
```

### Fix 2 — Corpus B: Missing `_sizes` 6th element (structural)
**File:** `corpus_b_generator.py`, `finalize()`  
**Problem:** `_sizes` had only 5 elements (g, e, s, ep, fp) — missing the `cb` (port forcing) group. `build_cache_b` could not merge shards correctly.  
```python
out["_sizes"] = np.array([
    ids.size,
    out["e_src"].shape[0],
    cat["s_graph"].shape[0],
    out["ep_T"].shape[0] if "ep_T" in out else 0,
    out["fp_state"].shape[0],
    out["cb_port_gene"].shape[0],   # ← ADDED: group "cb"
], dtype=np.int64)
```

### Fix 3 — Corpus C: Neutral padding for absent edges (MODERATE)
**File:** `corpus_c_generator.py`, SCC edge-param array initialization  
**Problem:** `ep_T`, `ep_NH`, `ep_F` initialized to zeros for absent-edge positions. This creates `log(0)` in the GNN encoder for unused edge slots.  
```python
# BEFORE (wrong — degenerate log values):
ep_T  = np.zeros((N_MAX, N_MAX), dtype=np.float32)
# AFTER (correct — neutral padding):
ep_T  = np.ones((N_MAX, N_MAX), dtype=np.float32)   # H(x; F=1,T=1,n=1) = 0.5 (neutral)
```

### Fix 4 — Corpus C: Missing `_sizes` array (structural)
**File:** `corpus_c_generator.py`, `save()`  
**Problem:** `_sizes` array was absent entirely. `build_cache_c` requires it to merge shards efficiently.  
```python
_sizes = np.array([
    self._n_dags, self._n_sccs, self._n_iedges,
    self._n_branches, len(self.link_pb), self._n_bfps,
], dtype=np.int64)
```

### Fix 5 — data_b.py: Wrong shard glob pattern (commit `e0ce08e`)
**File:** `data_b.py`, `build_cache_b`  
**Problem:** Used Corpus A glob `"shard_[0-9][0-9][0-9][0-9].npz"` but Corpus B shards are named `shard_b_0000.npz`.  
```python
SHARD_GLOB_B = "shard_b_[0-9][0-9][0-9][0-9].npz"
```

### Fix 6 — data_b.py: IndexError in `_report_cache_b` (commit `3b6d0c2`)
**File:** `data_b.py`, `_report_cache_b`  
**Problem:** `cb_stratum` is per-port (1.24M entries) not per-sample (825K entries). Direct indexing by sample index caused size mismatch.  
```python
# AFTER (correct — index per-sample's first port):
stratum = np.asarray(a["cb_stratum"])[np.asarray(a["s_cb_ptr"])]
```

### Fix 7 — data_b.py: Missing `port_mask` in `collate_b` (commit `c68e73d`)
**File:** `data_b.py`, `collate_b`  
**Problem:** `model_b.py` expected `batch["port_mask"]` but `collate_b` never produced it. Caused `KeyError` at first forward pass.  
```python
"port_mask": (torch.arange(MAX_PORTS).unsqueeze(0) < np_.unsqueeze(1)),  # [B, MAX_PORTS]
```

### Fix 8 — model_b.py: GPU index crash in `FiLMInjection` (commit `5a3d05c`)
**File:** `model_b.py`, `FiLMInjection.forward()`  
**Problem:** `port_gene=-1` padding sentinel passed directly to GPU scatter/gather, causing `ScatterGatherKernel` assertion failure.  
```python
gene_k_safe = gene_k.clamp(min=0)   # inactive slots → index 0; zeroed by active mask
h_k  = h[torch.arange(B, device=h.device), gene_k_safe]
idx  = gene_k_safe.view(B, 1, 1).expand(B, 1, d)
delta.scatter_add_(1, idx, inc.unsqueeze(1))
```

### Fix 9 — data_c.py: Wrong shard glob + hardcoded `shards/` subdir (commits `07cd9c3`, `36ade8e`)
**File:** `data_c.py`, `build_cache_c`  
**Problem 1:** `build_cache_c` looked for shards under `corpus_dir/shards/` but generator writes directly to `corpus_dir`. Added fallback.  
**Problem 2:** Used Corpus A glob `"shard_[0-9][0-9][0-9][0-9].npz"` (imported from `data.py`), but Corpus C shards are named `corpus_c_shard_0000.npz`.  
```python
SHARD_GLOB_C = "corpus_c_shard_[0-9][0-9][0-9][0-9].npz"
shards = sorted(glob.glob(os.path.join(corpus_dir, "shards", SHARD_GLOB_C)))
if not shards:
    shards = sorted(glob.glob(os.path.join(corpus_dir, SHARD_GLOB_C)))
if not shards:
    raise FileNotFoundError(f"No shards found in {corpus_dir}/shards or {corpus_dir}")
```

### Fix 10 — data_c.py: Wrong sentinel pointer allocation (commit `2b5f8ee`)
**File:** `data_c.py`, `build_cache_c`  
**Problem:** `SENTINEL_PTRS_C` was a single dict mapping ptr_name → target group (what the pointer *points into*). But pointer arrays are sized by their *source* group (e.g., `branch_fp_ptr` has shape `n_branch+1`, not `n_bfp+1`). Used target size for allocation → shape mismatch crash.  
**Fix:** Split into two dicts — source group (for allocation) and target group (for running offset during merge):
```python
# SENTINEL_PTRS_C: ptr_name → SOURCE group (determines array length n_source+1)
SENTINEL_PTRS_C = {
    "dag_scc_ptr":     "dag",
    "dag_edge_ptr":    "dag",
    "scc_branch_ptr":  "scc",
    "branch_fp_ptr":   "branch",
    "branch_link_ptr": "branch",
}
# SENTINEL_TGTS_C: ptr_name → TARGET group (running offset during merge)
SENTINEL_TGTS_C = {
    "dag_scc_ptr":     "scc",
    "dag_edge_ptr":    "iedge",
    "scc_branch_ptr":  "branch",
    "branch_fp_ptr":   "bfp",
    "branch_link_ptr": "link",
}
```

### Fix 11 — data_c.py: Missing `xmax` in `collate_c` (commit `11182df`)
**File:** `data_c.py`, `collate_c`  
**Problem:** `losses.py SetCriterion._compute()` requires `xmax` (invariant-box bound); Corpus B loads it from shard data; Corpus C must compute it at collation time.  
**Fix:** Compute from P, K, F using the invariant-box formula:
```python
# xmax[b,i] = (1 + EPS_BOX) * P[b,i] / K[b,i] * prod_j max(1, F[b,j,i])
F_full = stack("F")[:, :nb, :nb].float()       # [B, nb, nb]
bound  = F_full.clamp_min(1.0).prod(dim=1)      # [B, nb]
out["xmax"] = ((1.0 + EPS_BOX) * out["P"].float() / out["K"].float() * bound).to(torch.float32)
```

### Fix 12 — data_c.py: Missing `has_pos_fb` in `collate_c` (commit `754f7a6`)
**File:** `data_c.py`, `collate_c`  
**Problem:** `losses.py` line 175 uses `has_pos_fb` for Thomas criterion weighting; not in Corpus C batch.  
**Fix:** Derive from `tc` (Thomas class): TC_NEG=1 has no positive feedback loop; TC_POS=2 and TC_BOTH=3 do.
```python
out["has_pos_fb"] = out["tc"] != 1   # True for TC_POS and TC_BOTH
```

### Fix 13 — train_layer3.py: Missing `torch.no_grad()` in val loop (commit `714d7ac`)
**File:** `train_layer3.py`, `evaluate_loader_c`  
**Problem:** Val pass built full autograd compute graphs for all 20K samples → memory pressure → training hung silently after epoch 1 val pass completed.  
**Fix:**
```python
with torch.no_grad(), autocast_ctx(cfg, device):
    out_cond, _out0 = model(batch, freeze_backbone=cfg.freeze_backbone)
```

### Fix 14 — losses.py: Wrong phase group size for Corpus C (commit `cf4c907`)
**File:** `losses.py`, `MetricAccumulator.GROUP_SIZES`  
**Problem:** `GROUP_SIZES["phase"] = 3` was hardcoded for Corpus B (phase values 0, 1, 2 → `minlength=3`). Corpus C uses `tc` as phase (values 1, 2, 3 → need `minlength=4`).  
```python
# BEFORE:
GROUP_SIZES = {"n": N_MAX + 1, "phase": 3, "consistent": 2, "has_pos_fb": 2}
# AFTER:
GROUP_SIZES = {"n": N_MAX + 1, "phase": 4, "consistent": 2, "has_pos_fb": 2}
```

### Fix 15 — transfer_corpusB.sh: Wrong meta file check (commit `3183b13`)
**File:** `transfer_corpusB.sh`  
**Problem:** Script checked for `meta_b.json` only, but `build_cache_b` writes `meta.json`.  
```bash
# AFTER:
if [ ! -f "$SRC/meta.json" ] && [ ! -f "$SRC/meta_b.json" ]; then
```

---

## 7. Git History (branch `claude/eloquent-cerf-l0zqbo`)

```
cf4c907  fix: GROUP_SIZES["phase"]=4 for Corpus C (tc values 1/2/3 need minlength=4)
714d7ac  fix: add torch.no_grad() to evaluate_loader_c (val hang after epoch 1)
754f7a6  fix: add has_pos_fb to collate_c output (tc != 1)
11182df  fix: compute xmax in collate_c from P/K/F invariant-box formula
2b5f8ee  fix: split SENTINEL_PTRS_C into source/target dicts (shape mismatch)
36ade8e  fix: correct SHARD_GLOB_C to corpus_c_shard_NNNN.npz naming
07cd9c3  fix: fallback glob for shards/ subdir in build_cache_c
9fdc88a  chore: add .gitignore (Python, cache dirs, slurm logs, OS/IDE artifacts)
f32dbb0  docs: update session context for Layer III (model_c, train_layer3, workflow)
509657f  feat: add Layer III model, training script, slurm job, and corpus transfer script
db1b21a  docs: fix HEAD ref and add final two commits to git history
b0b8d1e  docs: update session context with Layer II completion, LFS setup, and collaborator instructions
b9f7f1f  artifacts: add Layer II checkpoints and training log (epoch 35 best, score 1.6794)
61d3899  chore: add .gitattributes for Git LFS tracking (*.pt, *.npy, *.npz, *.csv)
5a3d05c  fix: clamp port_gene sentinel -1 before GPU index in FiLMInjection
c68e73d  Fix collate_b: add port_mask to batch
b5cd0f6  Add model.py (with embedding patch), model_b.py, train_layer1.slurm from main
0d297fd  Add train_layer2.py from main branch
3183b13  Fix transfer_corpusB.sh: cache writes meta.json not meta_b.json
3b6d0c2  Fix _report_cache_b: index cb_stratum per-sample via s_cb_ptr
e0ce08e  Fix SHARD_GLOB for Corpus B: shard_b_NNNN.npz naming convention
122b8f7  Add train_layer2.slurm and transfer_corpusB.sh for Negishi→Gilbreth workflow
27ff6e4  Fix corpus B/C bugs and add data loaders for B and C
```

---

## 8. Artifacts & GitHub LFS

`.gitattributes` (commit `61d3899`) tracks `*.pt`, `*.npy`, `*.npz`, `*.csv` via Git LFS.

**Layer II artifacts committed to the repo** under `artifacts/layer2/`:
- `best.pt` — best checkpoint (epoch 35, selection score 1.6794, ~27 MB)
- `last.pt` — final checkpoint (epoch 60, ~27 MB)
- `log.csv` — full per-iteration training log (~45 KB)

**Layer I checkpoint** is at `/scratch/gilbreth/gupt1042/runs/layer1/best.pt` (not yet committed to GitHub).

**Corpus B cache** (43 `.npy` files, ~hundreds of MB) is on Gilbreth at `/scratch/gilbreth/gupt1042/corpusB_new/cache` — too large for LFS free tier; rebuild from shards if needed (see §9).

---

## 9. Cluster Workflow Reference

### Complete Layer II workflow (already done — for reference)

```bash
# 1. On Negishi: regenerate Corpus B with fixes
python corpus_b_generator.py \
    --corpus_a_dir ~/python/corpusA \
    --out_dir ~/python/corpusB_new \
    --workers 8 --preset full
# Result: 609 shards, 825,733 samples, 38,969 graphs, 99.2% Brouwer-certified

# 2. On Negishi: build cache
python -c "from data_b import build_cache_b; build_cache_b('$HOME/python/corpusB_new', '$HOME/python/corpusB_new/cache')"

# 3. On Negishi: transfer to Gilbreth
bash transfer_corpusB.sh
# Validates .npy count locally vs remotely before finishing

# 4. On Gilbreth: pull latest code and submit
cd /scratch/gilbreth/gupt1042/GNN
git fetch origin && git reset --hard origin/claude/eloquent-cerf-l0zqbo
sbatch train_layer2.slurm
```

### Layer III (`train_layer3.py`, `train_layer3.slurm`) ✅ COMPLETE

```bash
sbatch train_layer3.slurm
```
- Loads Layer II EMA weights from `best.pt` (epoch 35), freezes backbone
- Trains only Φ_ψ + A_η + U_ω (~0.67M params out of 2.89M total) — same modules as Layer II
- Config: 60 epochs, batch_size=128, d_model=128, m_dim=128, n_u_layers=2, lr=5e-5, AMP=true, freeze_backbone=true
- Corpus C cache: `/scratch/gilbreth/gupt1042/corpusC/cache` (892,773 train + 20,000 val branches)
- Output dir: `/scratch/gilbreth/gupt1042/runs/layer3/`

**Training completed — 60 epochs on Gilbreth (partition a10)**

**Final results (best.pt ≈ epoch 59, val loss 0.3554):**

| Metric | Epoch 1 | Epoch 22 | Epoch 60 | Notes |
|--------|---------|----------|----------|-------|
| val loss | 0.5525 | 0.3779 | **0.3555** | Steady improvement |
| count_exact | 89.9% | 93.7% | **94.5%** | Correct FP count |
| regime_acc | 88.6% | 93.0% | **93.7%** | Mono vs multi classification |
| fp_recall | 78.0% | 81.2% | **82.1%** | Individual FP recall |
| pos_decades | 0.191 | 0.147 | **0.144** | Plateau after ~ep 20 |
| stab_acc | 99.85% | 99.85% | **99.86%** | Excellent (frozen) |
| thomas_violation | 0.000 | 0.000 | **0.000** | TC_NEG perfectly respected |
| brouwer_idx_acc | 99.40% | 99.46% | **99.46%** | |
| brouwer_consistent | 99.31% | 99.55% | **99.48%** | |
| refined_converged | 61.5% | 55.3% | **54.9%** | Newton convergence from pred position |
| ece | 0.034 | 0.027 | **0.026** | Well-calibrated |

**Key observations:**
- `thomas_violation = 0.000` throughout all 60 epochs — model perfectly respects TC_NEG constraints
- `pos_decades` plateaued around epoch 18-20 (0.147) and barely moved to 0.144 by epoch 60. Architectural ceiling: position head is bottlenecked by frozen backbone representations not trained for localization
- `refined_converged` drifted slightly downward (0.615 → 0.549) — mild late-training position drift; not alarming
- LR cosine-annealed from 5e-5 → ~1e-6 by epoch 60; epochs 59 and 60 nearly identical, confirming full convergence
- Main weakness: `fp_recall` 82.1% — 18% of true FPs unfound, primarily in TC_POS networks (79% certified vs 90% for TC_BOTH)

**Dataset breakdown (Corpus C, val set, 20K branches):**
- TC_NEG: 2,973 branches | certified 100.0% | mean 1.00 FP (no positive feedback → unique FP)
- TC_POS: 5,125 branches | certified 79.0% | mean 1.90 FP
- TC_BOTH: 11,902 branches | certified 90.2% | mean 1.40 FP

### Layer III workflow reference (already done)

```bash
# 1. On Negishi: generate Corpus C shards (256 shards × 500 DAGs)
python corpus_c_generator.py \
    --out_dir ~/python/corpusC \
    --n_shards 256 --dags_per_shard 500 \
    --n_workers 64 --k_min 2 --k_max 8 --seed 42

# 2. On Negishi: build Corpus C cache
python -c "from data_c import build_cache_c; build_cache_c('$HOME/python/corpusC', '$HOME/python/corpusC/cache')"

# 3. Transfer to Gilbreth
bash transfer_corpusC.sh

# 4. On Gilbreth: pull latest code and submit
cd /scratch/gilbreth/gupt1042/GNN
git fetch origin && git reset --hard origin/claude/eloquent-cerf-l0zqbo
sbatch train_layer3.slurm
```

---

## 10. Instructions for New Collaborators

### Cloning and getting all artifacts

```bash
# Install Git LFS (one time per machine)
git lfs install

# Clone
git clone https://github.com/pgvsec/GNN.git
cd GNN
git checkout claude/eloquent-cerf-l0zqbo

# Pull LFS files (checkpoints, logs)
git lfs pull
```

After this you will have:
- All source code
- `artifacts/layer2/best.pt` — Layer II trained weights (27 MB, via LFS)
- `artifacts/layer2/last.pt` — Layer II final weights (27 MB, via LFS)
- `artifacts/layer2/log.csv` — full training log

### Loading the Layer II model for inference

```python
import torch
from model import SccFixedPointNet
from model_b import CorpusBNet

# Load checkpoint
ck = torch.load("artifacts/layer2/best.pt", map_location="cpu", weights_only=False)

# Build model (must match training config)
from types import SimpleNamespace
cfg = SimpleNamespace(**ck["cfg"])
model = CorpusBNet(cfg)

# Load EMA weights (better than raw weights for inference)
model.load_state_dict(ck["ema"])   # ema stores shadow weights directly
model.eval()
```

### Corpus B cache (if you need to run training or eval)

The cache is too large for GitHub. Rebuild it:
```bash
# On a cluster with the Corpus B shards accessible:
python -c "
from data_b import build_cache_b
build_cache_b('/path/to/corpusB_new', '/path/to/corpusB_new/cache')
"
```
Or contact gupt1042 to rsync from `/scratch/gilbreth/gupt1042/corpusB_new/cache`.

### Running Layer I training from scratch

```bash
# On Gilbreth
cd /scratch/gilbreth/gupt1042/GNN
sbatch train_layer1.slurm
```

---

## 11. Pending Reconciliation Before Merging Branches

When merging `claude/eloquent-cerf-l0zqbo` → `main`, resolve these discrepancies:

| Item | `main` value | Dev branch value | Decision needed |
|------|-------------|------------------|-----------------|
| `SPLIT_TRAIN/VAL/TEST` in `data.py` | 0 / 1 / 2 | 1 / 2 / 3 | Pick one; regenerate caches |
| `MAX_PORTS` in `data_b.py` | 8 | 3 | Pick one; regenerate Corpus B |
| `data_b.py` implementation | 378 lines (main) | ~270 lines (dev) | Merge or pick dev (has all fixes) |

---

## 12. Architectural Decisions Made

| Decision | Rationale |
|----------|-----------|
| FiLM modulation at port genes only | Signal is local; non-port genes updated via GNN propagation |
| Corpus B absorbs u into P at gen-time | Lets `SetCriterion` and physics loss work unchanged |
| DeepSets (not attention) for aggregation | Provably permutation-invariant; ports are unordered |
| Pad `pos` to N_MAX before portrait flatten | Backbone returns `pos [B,K,n_actual]`; `portrait_raw_dim` must be fixed at 288 |
| EMA over full CorpusBNet (backbone + new) | Backbone frozen params are included in shadow; they just don't move |
| AMP with `GradScaler` (not pure bf16) | Stability on a10/a100 GPUs; consistent with Layer I |
| `selection_score = count_exact + fp_recall` | Same as Layer I: measures both count accuracy and point-level recall |
| `stratum` aliased to `phase` in eval | `MetricAccumulator` groups by `phase`; Corpus B uses `stratum` for the same concept |
| `n_u_layers=2` GNN layers in U_ω | Small budget: only ~0.67M new params; more layers risked overfitting with frozen backbone |
| Corpus C one-sample-per-branch | Each SCC's portrait is independent conditioned on parent FP selections |
| Corpus C CSR sentinel ptrs [n+1] | Standard CSR format; `branch_fp_ptr[b+1]-branch_fp_ptr[b]` = num FPs in branch b |
| Neutral padding (ep T/NH/F = 1) | Avoids log(0) in encoder for absent-edge positions; H(x; F=1, T=1, n=1) = 0.5 (neutral) |
| clamp port_gene to ≥0 before GPU ops | GPU scatter/gather kernels reject negative indices even when active=0 |
| CorpusCNet subclasses CorpusBNet (Option B) | Port tensors synthesized from branch_u inside forward(); data layer stays clean |
| xmax = P/K at forward time in CorpusCNet | Corpus C does not store xmax; P/K is the first-order steady-state approximation used by node_enc |
| has_pos_fb defaults False in CorpusCNet | Thomas loop topology not tracked in Corpus C; w_thomas=0 in train_layer3.py |
| Layer III warm-starts from Layer II best.pt | All parameter names identical between CorpusBNet and CorpusCNet |
| port_T/nh/F = 1.0/2.0/2.0 (neutral defaults) | iedge Hill params not pre-collated per branch; Φ_ψ still distinguishes via u magnitude |

---

## 13. Constants Quick Reference

```python
# from data.py
N_MAX = 10               # max genes in an SCC
NUM_STABILITY = 7        # stability classes (0-6)
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 1, 2, 3   # dev branch (main uses 0, 1, 2)
UNKNOWN_STABILITY_CODE = -1
UNKNOWN_STABILITY_CLASS = 6   # maps to index 6 in stab_logit

# from model.py
LOG_EPS = 1e-6
LOG_SPAN = -math.log(1e-6)   # ≈ 13.816  (~6 log-decades)
CHARPOLY_SCALE = 10.0

# from data_b.py (dev branch)
MAX_PORTS = 3            # main branch uses 8
GROUP_ORDER_B = ("g", "e", "s", "ep", "fp", "cb")
SHARD_GLOB_B  = "shard_b_[0-9][0-9][0-9][0-9].npz"

# from data_c.py
GROUP_ORDER_C = ("dag", "scc", "iedge", "branch", "link", "bfp")

# from model_b.py
DEFAULT_M_DIM = 128

# from model_c.py
MAX_PORTS_C = 8   # port slots synthesized from branch_u per Corpus C sample

# Network defaults (train_layer2.slurm / train_layer3.slurm / Config)
d_model    = 128
max_fp     = 16          # K slots
m_dim      = 128
n_u_layers = 2
```

---

## 14. How to Resume in a New Chat

1. Clone/pull the repo:
   ```bash
   git clone https://github.com/pgvsec/GNN
   cd GNN
   git checkout claude/eloquent-cerf-l0zqbo
   git lfs pull
   ```
2. Upload this file (`SESSION_CONTEXT_updated.md`).
3. The assistant should read the key source files before making any edits:
   - `model.py` (Layer I network + exposed embeddings)
   - `data.py` (constants and Corpus A loader)
   - `data_b.py` (Corpus B loader — dev branch version, has all fixes)
   - `data_c.py` (Corpus C loader — dev branch only)
   - `model_b.py` (Layer II/III modules)
   - `train_layer2.py` (Layer II training loop)
   - `losses.py` (criterion and metrics)
4. **Current status:** Layer I ✅ done. Layer II ✅ done (best.pt epoch 35, score 1.6794). Layer III ✅ done (best.pt ~epoch 59, val loss 0.3554, count_exact 94.5%, fp_recall 82.1%).

# GNN Project — Session Context for Handoff

**Last updated:** 2026-10-11  
**GitHub repo:** `pgvsec/GNN` (branch `main`, HEAD `3c500f8`)  
**Cluster:** Purdue Gilbreth (`gilbreth.rcac.purdue.edu`)  
**Scratch path:** `/scratch/gilbreth/gupt1042/`  
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

**PATCH made in this session:** `SccFixedPointNet.forward()` now exposes three intermediate tensors for use by Layer II/III:
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

---

## 3. Data Pipeline

### Corpus A (`data.py`) — Layer I training data
- Memory-mapped `.npy` cache under a single directory
- 5 key groups: `g_*` (graph), `e_*` (edge), `s_*` (sample), `ep_*` (edge-params), `fp_*` (fixed-points)
- Key constants in `data.py`: `N_MAX=10`, `NUM_STABILITY=7`, `SPLIT_TRAIN=0`, `SPLIT_VAL=1`, `SPLIT_TEST=2`, `UNKNOWN_STABILITY_CODE`, `UNKNOWN_STABILITY_CLASS`
- `make_loader()` → returns `(dataset, DataLoader)` with `collate_fn` that trims to max active nodes/fps in batch

### Corpus B (`data_b.py`) — Layer II/III training data (forced single-SCC)
- Same shard format as Corpus A, plus a 6th `cb_*` group for variable-length port forcing arrays
- Extra pointer: `s_cb_ptr` → into `cb_port_*` arrays
- **Corpus B absorbs forcing into P at generation time** (`apply_forcing` in `corpus_b_generator.py`): `P_forced[b_k] = P[b_k] * u_k`. So `batch["P"]` already contains `P_forced` and `SetCriterion` works unchanged.
- `MAX_PORTS = 8` cap on simultaneous forced genes
- `has_pos_fb` comes from `g_pos_fb` topology key (added by `corpus_b_generator.py`'s `finalize()`)
- `make_loader_b(cfg, split, train)` → `(dataset, DataLoader)` with `collate_b`

**Cache paths on cluster:**
- Corpus A: `/scratch/gilbreth/gupt1042/corpusA/cache`
- Corpus B shards: `/scratch/gilbreth/gupt1042/corpusB/` (built, confirmed exists)
- Corpus B cache: `/scratch/gilbreth/gupt1042/corpusB/cache` (build before submitting Layer II job)

**Build Corpus B cache (run once on cluster, CPU ok):**
```bash
python -c "from data_b import build_cache_b; build_cache_b('/scratch/gilbreth/gupt1042/corpusB', '/scratch/gilbreth/gupt1042/corpusB/cache')"
```

---

## 4. Losses (`losses.py`)

`SetCriterion` implements Hungarian-matching bipartite loss:
- **Matching costs:** `cost_exist`, `cost_pos`, `cost_cp`, `cost_stab`
- **Losses:** exist (BCE), pos (Huber), charpoly (Huber), stab (CE), count (CE, odd-only), phys (ODE residual), thomas (Thomas index constraint, requires `has_pos_fb`)
- Loss weights: `w_exist=1, w_pos=5, w_cp=2, w_stab=1, w_count=0.5, w_phys=1, w_thomas=0.5`

**`batch_metrics()`** computes: `count_exact`, `regime_acc`, `stable_count_exact`, `fp_recall`, `stab_acc`, `pos_decades`, `brouwer_idx_acc`, `brouwer_consistent`, `thomas_violation`, `refined_converged`

`pos_decades`: mean absolute position error in log₁₀ units. **Lower = better.** Computed from `LOG_SPAN = -log(1e-6) ≈ 13.816`.

---

## 5. Training

### Layer I (`train.py`, `train_layer1.slurm`)

```bash
sbatch train_layer1.slurm
```
- Config: 60 epochs, batch_size=256, d_model=128, n_gnn_layers=4, n_attn_layers=1, n_dec_layers=4, n_heads=8, lr=3e-4, AMP=true
- **Best checkpoint reached:** epoch 39, val loss 0.5867
- **Checkpoint path:** `/scratch/gilbreth/gupt1042/runs/layer1/best.pt`
- Checkpoint format: `{"model": ..., "ema": ..., "opt": ..., "sched": ..., "epoch": ..., "best": ...}`
- Selection score: `count_exact + fp_recall`

### Layer II/III (`train_layer2.py`, `train_layer2.slurm`)

```bash
sbatch train_layer2.slurm
```
- Loads Layer I EMA weights from `best.pt`, freezes backbone (`requires_grad_(False)`)
- Trains only Φ_ψ + A_η + U_ω (~0.67M params)
- Config: 60 epochs, batch_size=128, d_model=128, m_dim=128, n_u_layers=2, lr=1e-4, AMP=true
- Val metrics aliased: `batch["stratum"] → batch["phase"]` for `MetricAccumulator` group keys
- Output dir: `/scratch/gilbreth/gupt1042/runs/layer2/`
- Same checkpoint format as Layer I

**Status:** Script written and committed. **Not yet submitted.** Corpus B cache must be built first.

---

## 6. Git History (all commits on `main`)

```
3c500f8  Add Layer II/III infrastructure: Φ_ψ, A_η, U_ω on Corpus B
           • model.py: expose slot_emb, graph_emb, node_emb
           • data_b.py: new file (378 lines) — Corpus B DataLoader
           • model_b.py: new file (399 lines) — PhiPsi, AEta, UOmega, CorpusBNet
           • train_layer2.py: new file (511 lines) — Layer II/III training loop
           • train_layer2.slurm: new file (41 lines) — SLURM script

e208e87  Switch partition to a10 (idle nodes available)
30c5de6  Add SLURM account=ayg (required by Gilbreth)
58a98b6  Configure SLURM script for Gilbreth: gnn conda env, scratch paths
3e78794  Set partition to a100-40gb for Gilbreth GPU nodes
33b7ab2  Set Corpus A cache path to /home/gupt1042/python/corpusA/cache
403afbd  Add SLURM training script for Layer I on Gilbreth GPU cluster
ef9f0be  Update print statement from 'Hello' to 'Goodbye'
```

---

## 7. Key File Map

| File | Role | Status |
|------|------|--------|
| `model.py` | Layer I network (`SccFixedPointNet`) + GRN ODE functions | Modified (expose embeddings) |
| `data.py` | Corpus A DataLoader, constants (`N_MAX`, `NUM_STABILITY`, etc.) | Unchanged |
| `losses.py` | `SetCriterion`, `MetricAccumulator`, `CalibrationTable`, `batch_metrics` | Unchanged |
| `train.py` | Layer I training loop | Unchanged |
| `train_layer1.slurm` | SLURM script for Layer I | Done |
| `data_b.py` | Corpus B DataLoader (`MAX_PORTS=8`, port forcing, `has_pos_fb`) | New |
| `model_b.py` | `PhiPsi`, `AEta`, `FiLMInjection`, `UOmega`, `CorpusBNet` | New |
| `train_layer2.py` | Layer II/III training loop | New |
| `train_layer2.slurm` | SLURM script for Layer II/III | New |
| `corpus_b_generator.py` | Generates Corpus B shards (already run) | Unchanged |
| `corpus_c_generator.py` | Generates Corpus C shards (not yet run) | Unchanged |
| `solver.py` | ODE solver, topology helpers | Unchanged |
| `topology.py` | Graph topology generation | Unchanged |

---

## 8. Immediate Next Steps

1. **Build Corpus B cache** (one-time, run on Negishi/Gilbreth, CPU):
   ```bash
   cd ~/GNN
   python -c "from data_b import build_cache_b; build_cache_b('/scratch/gilbreth/gupt1042/corpusB', '/scratch/gilbreth/gupt1042/corpusB/cache')"
   ```

2. **Submit Layer II training:**
   ```bash
   sbatch train_layer2.slurm
   ```
   Monitor: `tail -f /scratch/gilbreth/gupt1042/runs/layer2/slurm_<jobid>.out`

3. **After Layer II converges**, plan Layer III (Corpus C):
   - Corpus C generator (`corpus_c_generator.py`) exists but has not been run yet
   - Layer III will use Corpus C which involves multi-SCC networks with inter-SCC edges
   - Will need `data_c.py`, `model_c.py`, `train_layer3.py` (analogous to the B-layer files)

---

## 9. Architectural Decisions Made

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

---

## 10. Constants Quick Reference

```python
# from data.py
N_MAX = 10               # max genes in an SCC
NUM_STABILITY = 7        # stability classes (0-6)
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
UNKNOWN_STABILITY_CODE = -1
UNKNOWN_STABILITY_CLASS = 6   # maps to index 6 in stab_logit

# from model.py
LOG_EPS = 1e-6
LOG_SPAN = -math.log(1e-6)   # ≈ 13.816  (~6 log-decades)
CHARPOLY_SCALE = 10.0

# from data_b.py
MAX_PORTS = 8

# from model_b.py  
DEFAULT_M_DIM = 128

# Network defaults (train_layer2.slurm / Config)
d_model = 128
max_fp  = 16            # K slots
m_dim   = 128
n_u_layers = 2
```

---

## 11. How to Resume in a New Chat

1. Clone/pull the repo: `git clone https://github.com/pgvsec/GNN` (or `git pull` if already cloned)
2. Upload this file.
3. The assistant should read the key source files before making any edits:
   - `model.py` (Layer I network + exposed embeddings)
   - `data.py` (constants and Corpus A loader)
   - `data_b.py` (Corpus B loader)
   - `model_b.py` (Layer II/III modules)
   - `train_layer2.py` (Layer II/III training loop)
   - `losses.py` (criterion and metrics)
4. All source is in `/home/claude/gnn/` in the Claude Code environment, or pull from `pgvsec/GNN` on GitHub.

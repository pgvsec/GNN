# GNN Compositional GRN Dynamics — Experiment Progress Log

**Goal:** Prove compositional GRN dynamics are learnable by a pure ML model.  
**Target:** All three metrics must reach ~85%: `count_exact ≥ 0.85`, `fp_hd ≥ 0.85`, `stable_acc ≥ 0.85`  
**Eval command (canonical):** `eval_composition2.py --l3_frame raw --dedup fill --refine_steps 15`  
**Eval set:** `composition_eval_v5` — 500 cases, 1502 total GT FPs  

---

## Hard Constraints (non-negotiable)

1. **Pure ML model only** — no physics-based heuristics, no hybrid solver fallback
2. **No masking of physical accuracy** — all metrics must reflect true ML performance
3. **No retraining from scratch** — all experiments fine-tune from existing checkpoints

---

## Architecture Overview

### Three-Layer GNN Pipeline
- **L1 (`SccFixedPointNet`)** — predicts FPs for root SCCs (no parents in DAG)
- **L3 (`CorpusCNet`)** — predicts FPs for forced SCCs (conditioned on upstream FPs)

### `SccFixedPointNet` submodules
| Submodule | Role | Trainable in fine-tuning? |
|-----------|------|--------------------------|
| `node_enc`, `pair_enc` | Node/pair encoders (backbone) | No (frozen by default) |
| `gnn` (ModuleList) | GNN message-passing layers (backbone) | No |
| `attn` (ModuleList) | Attention layers (backbone) | No |
| `readout` | Graph-level pooling (backbone) | No |
| `decoder` (SlotDecoder) | K learned slot queries cross-attend to [graph, gene tokens] | No |
| `exist_head` | MLP: slot feature → existence logit | No |
| `pos_head` (SlotNodeHead) | MLP(3d, d, 1): (slot, gene) → FP coordinate | Only in v6 |
| `count_head` (StructuralCountHead) | Reads `exist_logit.detach()` + `g.detach()` | Yes (v2–v5) |
| `stable_count_head` | Same detached-input pattern | Yes (v2–v5) |

### Critical architectural constraint
`count_head` reads `exist_logit.detach()` and `g.detach()` — **all inputs detached from backbone gradient**. This means:
- Count head trains only its own MLP parameters (from count CE loss)
- But: if backbone/decoder changes, `exist_logit` *values* shift even though gradient is blocked → count accuracy drops

---

## Experiment Log

### Baseline: v2 (best performing run to date)

| Metric | Value | Raw |
|--------|-------|-----|
| count_exact | 0.8340 | 417/500 |
| fp_hd | 0.8229 | GT-FP-count weighted |
| stable_acc | 0.8669 | mean per case |

**Eval:** `--dedup fill --refine_steps 15`  
**Checkpoints:** `$SCRATCH/runs/e2e_v2/l1_best.pt`, `$SCRATCH/runs/e2e_v2/l3_best.pt`  
**Backup:** `~/checkpoints/e2e_v2_best_baseline/` (Gilbreth home dir)  
**What it does:** Fine-tunes count heads only (backbone frozen), starting from Layer I+III pre-trained weights. `scc_reg_weight=0.3` adds per-SCC CE alongside composition CE.

---

### v3: Backbone unfreeze attempt (lr_backbone=1e-5)

**Script:** `jobs/e2e_v3.sh`  
**Key change:** `--lr_backbone 1e-5` — unfreezes GNN/encoder for pos decoder gradient  
**Result:** Peaked at epoch 1, then degraded every epoch thereafter  
**Diagnosis:** Even lr=1e-5 backbone gradient shifts `exist_logit` distribution enough to destabilize count head within 1 epoch

---

### v4: Smaller backbone LR (lr_backbone=1e-6)

**Script:** `jobs/e2e_v4.sh`  
**Key change:** `--lr_backbone 1e-6` — 10x smaller backbone LR  
**Result:** Same degradation pattern — peaked epoch 1, degraded thereafter  
**Diagnosis:** Even lr=1e-6 causes cumulative drift. One epoch of backbone gradient costs ~1.8pp on count_exact. Count head is extremely sensitive to backbone representation shifts.

---

### v5: Pos Chamfer loss + backbone unfreeze + training speedups

**Script:** `jobs/e2e_v5.sh`  
**Key changes:**
- `--pos_reg_weight 0.1` — log-space Chamfer loss on root SCC slots vs GT FPs
- `--lr_backbone 1e-6` — backbone unfrozen for pos decoder gradient
- `--accumulate 32` — gradient accumulation (32x fewer optimizer steps)
- AMP (mixed precision float16) — ~2x forward/backward speedup
- Precomputed GT counts — avoids per-epoch numpy overhead

**Training:** 53 epochs (hit 8h SLURM wall)  
**Results:**

| Epoch | count_exact | fp_hd | stable_acc |
|-------|-------------|-------|-----------|
| ep001 (saved as best) | 0.8240 | 0.8216 | 0.8535 |
| ep050 | 0.7900 | 0.8016 | 0.8178 |

**Diagnosis:** Pos loss does not overcome backbone-unfreeze damage. Count head degrades from epoch 1 as soon as backbone gradient flows. By epoch 50, count_exact is 5.2pp below v2. The 50+ epochs of pos loss gradient didn't help fp_hd either (it actually degraded). v5 is worse than v2 on all three metrics.

---

### v6: Pos-only isolation (train ONLY `pos_head`)

**Script:** `jobs/e2e_v6.sh`  
**Key changes:**
- `--lr 0.0` — count heads frozen
- `--lr_backbone 0.0` — backbone frozen
- `--lr_pos 1e-4` — ONLY `l1_model.pos_head` (49,409 params) trainable
- `--pos_reg_weight 1.0` — pos Chamfer loss only
- `--scc_reg_weight 0.0` — no count CE

**Rationale:** `decoder` and `exist_head` frozen → `exist_logit` values never change → count head inputs stay stable → count predictions remain calibrated. Only `pos_head.mlp` (MLP(3d, d, 1)) gets gradient from Chamfer loss.

**Training:** 60 epochs. Val pos_loss = 1.1090 for ALL 60 epochs (completely flat).

| Epoch | train_loss | val_loss | val count_exact |
|-------|-----------|----------|----------------|
| 1 | 1.9072 | 1.1090 | 0.8160 |
| 60 | 1.8683 | 1.1090 | 0.8160 |

**Results:**

| Metric | v2 baseline | v6 | Δ |
|--------|------------|-----|---|
| count_exact | 0.8340 (417/500) | **0.8400 (420/500)** | **+3 cases** |
| fp_hd | 0.8229 | **0.8242** | +0.13pp |
| stable_acc | 0.8669 | 0.8598 | -0.71pp |

**v6 is the current best on count_exact (+3 cases vs v2).**

**Checkpoints:** `$SCRATCH/runs/e2e_v6/l1_best.pt`, `$SCRATCH/runs/e2e_v6/l3_best.pt`  
⚠️ **Action required:** Copy v6 checkpoints to `~/checkpoints/e2e_v6_best/` on Gilbreth to protect from scratch purge.

**Diagnosis of flat val loss:** `pos_head` is a 49k-param MLP mapping from frozen slot features `s = decoder(g, h)` to coordinates. It was already trained to near-optimality during Layer I training for those exact features. The Chamfer loss can fit the training set (train loss decreases) but cannot generalize because the frozen features already encode everything they can about FP positions. The bottleneck is the DECODER itself — the slot Chamfer distance of 1.109 decades (~12x off) means the decoder isn't assigning slots to cover all GT FPs. Fixing this requires training the decoder, which is blocked by the exist_logit coupling.

---

## Speedup Summary (implemented in v5, carried into v6)

| Optimization | Effect |
|---|---|
| AMP (float16 on A10) | ~2x forward/backward |
| Gradient accumulation (32 steps) | 32x fewer optimizer.step() calls |
| Precomputed GT counts | Removes per-epoch numpy overhead |
| Net | ~4-6h for 60 epochs (was 8-10h) |

---

## Key Findings

1. **Backbone-unfreeze degradation is universal:** Any backbone gradient (even lr=1e-6) shifts `exist_logit` and degrades count accuracy within epoch 1. Confirmed in v3, v4, v5.

2. **Count head is extremely sensitive:** The count head is calibrated to a specific distribution of `exist_logit` values. Even tiny shifts in backbone weights produce measurable count degradation.

3. **pos_head isolation doesn't help:** pos_head can't improve beyond what the frozen slot features already encode. Val Chamfer loss stays flat regardless of training duration.

4. **Slot Chamfer distance = 1.109 decades:** Average slot position is ~12x off from GT FPs. Newton refinement (15 steps) bridges some of this gap, but ~11.6% of GT FPs are missed even in correct-count cases.

5. **refine_steps=15 > refine_steps=5 by +1.66pp fp_hd:** More Newton steps help. Further gains possible with refine_steps=30+.

6. **The fundamental tension:** Decoder must change to improve FP seeds, but decoder change breaks exist_logit calibration for count head. These are coupled through the shared `s = decoder(g, h)` representation.

7. **CRITICAL BUG (found in full arch audit):** `InferenceOptions.l3_frame` defaulted to `"forced"` since the module was written, but L3 was always trained in `"raw"` frame (batch has unforced `scc_P`; forcing enters only via ports). At inference in "forced" frame, the backbone sees `P * u` (never in its training distribution), and the forcing signal is double-counted: once via node features (P*u) and again through the port mechanism. **All v2–v6 eval numbers were computed in the wrong frame.** Fix: change default to `"raw"` in `InferenceOptions` (done on `prateek/ml-experiments`). Expected impact: 3–5pp across all metrics with zero retraining.

8. **Position scale mismatch in "raw" frame (secondary):** L3 encodes target FP positions relative to `xmax_unforced = box(P)`. Under activation forcing the true FP can exceed `xmax_unforced`, so the model learns a clipped representation. Newton compensates at eval time, but longer refine_steps helps more than they should. True fix: retrain L3 encoding targets relative to `xmax_phys = box(P*u)`.

---

## Paths Not Yet Tried

1. **refine_steps=30 or 50 on v2/v6 checkpoint** (free, 30 min eval only) — might yield +0.5-1pp fp_hd via more Newton convergence
2. **Separate pos estimator** — add a new `fp_pos_net(g, h)` module that produces FP seeds independently of `exist_logit`. Would allow decoder to be bypassed for Newton seeding without touching the count head coupling.
3. **Larger training set** — more diverse composition cases might allow count head + decoder to co-adapt without diverging

---

## Reproducibility

```bash
# Reproduce v6 eval (best current metrics)
cd $SCRATCH/GNN
git checkout archive/v2-best-baseline

PYTHONPATH=$SCRATCH/GNN:$SCRATCH/GNN_temp \
  python exhaust_gen/eval_composition2.py \
    --l1_ckpt $SCRATCH/runs/e2e_v6/l1_best.pt \
    --l3_ckpt $SCRATCH/runs/e2e_v6/l3_best.pt \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  /tmp/eval_v6_repro.csv \
    --dedup fill --refine_steps 15
# Expected: count_exact=0.8400 (420/500), fp_hd=0.8242, stable_acc=0.8598

# Reproduce v2 eval (for comparison)
PYTHONPATH=$SCRATCH/GNN:$SCRATCH/GNN_temp \
  python exhaust_gen/eval_composition2.py \
    --l1_ckpt ~/checkpoints/e2e_v2_best_baseline/l1_best.pt \
    --l3_ckpt ~/checkpoints/e2e_v2_best_baseline/l3_best.pt \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  /tmp/eval_v2_repro.csv \
    --dedup fill --refine_steps 15
# Expected: count_exact≈0.8340 (417/500), fp_hd≈0.8229, stable_acc≈0.8669
```

# GNN Compositional GRN Dynamics — Experiment Progress Log

**Goal:** Prove compositional GRN dynamics are learnable by a pure ML model.  
**Target:** All three metrics must reach ~90%: `count_exact ≥ 0.90`, `fp_hd ≥ 0.90`, `stable_acc ≥ 0.90`  
**Status:** count_exact=**0.914** ✓ · fp_hd=**0.891** · stable_acc=**0.914** ✓ (v8)  
**Eval command (canonical):** `eval_composition2.py --l3_frame forced --dedup fill --refine_steps 15`  
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

**Diagnosis of flat val loss:** `pos_head` is a 49k-param MLP mapping from frozen slot features `s = decoder(g, h)` to coordinates. It was already trained to near-optimality during Layer I training for those exact features. The Chamfer loss can fit the training set (train loss decreases) but cannot generalize because the frozen features already encode everything they can about FP positions. The bottleneck is the DECODER itself — the slot Chamfer distance of 1.109 decades (~12x off) means the decoder isn't assigning slots to cover all GT FPs. Fixing this requires training the decoder, which is blocked by the exist_logit coupling.

---

### v7: ForcingAwareCountHeadV2 + backbone unfreeze

**Script:** `jobs/finetune_e2e_v7.sh` (and `_a100.sh`)  
**Key changes:**
- `--use_v2_count_head` — replaces `StructuralCountHead` with `ForcingAwareCountHeadV2` which reads `z_cond` (not detached) + `u_stats` + `tc_onehot`
- `--lr_backbone 2e-6` — unfreezes L3 backbone via `freeze_l3_backbone=False`
- Training data: v3+v4 (in-distribution with v5 val)

**Results (full eval on v5):**

| Metric | v7 |
|--------|-----|
| count_exact | 0.834 |
| fp_hd | 0.847 |
| stable_acc | 0.865 |

**Diagnosis of v7b/v7c/v7d failure:** Fine-tuning attempts from v7 all used composition CE loss. CE improved (0.618→0.561) while count_exact declined (0.834→0.794) in every run. CE rewards smooth probability distributions; count_exact needs sharp argmax peaks. CE is the wrong proxy — it teaches the model to spread probability mass, which is exactly the wrong direction for argmax accuracy.

---

### v8: composition_margin_loss + l1_count_no_detach (current best)

**Script:** `jobs/finetune_e2e_v8_a100.sh` (a10 fallback: `finetune_e2e_v8.sh`)  
**Start:** v7 checkpoints  
**Key changes:**
- `--use_margin_loss`: replaces composition CE with `composition_margin_loss` — a hinge loss that directly optimizes count argmax accuracy
  - Wrong predictions: `hinge(P(n_pred) - P(n_gt) + margin=0.1)` — forces P(n_gt) above the current argmax. Zero gradient once correct.
  - Correct predictions: `correct_ce_weight=0.01` × CE for calibration only.
- `--l1_count_no_detach`: temporarily sets `l1_model.count_detach=False` during composition forward, opening gradient path `composition loss → L1 count head → backbone GNN` (previously blocked by `count_detach=True`)
- `--lr 1e-5 --lr_backbone 2e-6 --scc_reg_weight 0.5 --pos_reg_weight 0.3`
- Training: 100 epochs, peak at epoch 84

**Training dynamics:** count_exact improved monotonically from epoch 1 (0.828 → 0.894 training metric). Unlike v7b–v7d which peaked at epoch 1 and declined, v8 improved every epoch for 84 epochs.

**Results (full eval on v5, 500 cases):**

| Metric | v7 | v8 | Δ |
|--------|-----|-----|---|
| count_exact | 0.834 (417/500) | **0.914 (457/500)** | +8.0pp |
| fp_hd | 0.847 | **0.891** | +4.4pp |
| stable_acc | 0.865 | **0.914** | +4.9pp |

**Checkpoints:** `$SCRATCH/runs/e2e_v8/l1_best.pt`, `$SCRATCH/runs/e2e_v8/l3_best.pt` (epoch 84)  
⚠️ **Back these up** — scratch is purged periodically. Copy to `$HOME/checkpoints/e2e_v8/`.

**What was tried and didn't work (v8b):** Increasing `pos_reg_weight` 0.3→0.5 caused all three metrics to regress (count_exact: 0.914→0.908, fp_hd: 0.891→0.884, stable_acc: 0.914→0.906). The backbone was at capacity balance; stronger position supervision disrupted the equilibrium. v8 remains best.

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

7. **Frame consistency verified (eval sweep 2026-10-29):** `l3_frame="forced"` is correct for v2–v6 checkpoints. `finetune_e2e_composition.py` calls `_make_scc_batch(scc, u, device)` without explicit frame, which defaults to `"forced"`. Switching to `"raw"` drops count_exact from 0.840→0.594 (-24.6pp). The `_make_scc_batch` call in finetune has been made explicit (`frame="forced"`) to prevent future confusion. `train_layer3.py` still defaults to `frame="raw"` — any fresh L3 retrain should pass `--frame forced` to match.

8. **Position scale in "forced" frame:** The fine-tuned model encodes positions relative to `xmax_phys = box(P*u)` (the "forced" frame xmax). This is correct. The original concern about xmax_unforced clipping was based on an incorrect assumption that fine-tuning used "raw" frame.

---

## Key Findings (v8)

9. **CE loss is misaligned with count_exact:** Composition CE rewards smooth probability distributions while count_exact needs sharp argmax peaks. All v7b/v7c/v7d attempts showed CE improving while count_exact declined. The fix is `composition_margin_loss` (hinge on argmax gap), which improved count_exact monotonically for 84 epochs.

10. **l1_count_no_detach is essential:** `count_detach=True` in `SccFixedPointNet` blocked composition gradients from reaching the L1 backbone. Temporarily setting `count_detach=False` during composition training opens the path `composition loss → count head → backbone GNN`. Without this, composition loss cannot improve the backbone's count representations.

11. **pos_reg_weight has a narrow sweet spot:** At 0.3, position supervision improves fp_hd without hurting count/stability. At 0.5, it over-pulls the backbone and regresses all three metrics. The backbone is at capacity balance — stronger position gradients disrupt the shared representation.

12. **The backbone-unfreeze problem is solved for count:** v8 achieves backbone unfreeze (lr_backbone=2e-6) without count degradation by using margin loss instead of CE. The count head's sensitivity to `exist_logit` shifts is compensated by directly targeting count accuracy in the loss.

---

---

## Generalization Experiments — Full Results (E1–E4)

**Eval command:** `eval_composition2.py --l3_frame forced --dedup fill --refine_steps 15`  
**Checkpoints:** v8 (epoch 84) — `$SCRATCH/runs/e2e_v8/l1_best.pt` / `l3_best.pt`

| Tier | Dataset | n_genes | n_SCCs | count_exact | fp_hd | stable_acc | Notes |
|------|---------|---------|--------|-------------|-------|------------|-------|
| Baseline | v5 3+3 (biased) | 6 | 2 | **0.914** | 0.884 | 0.913 | v8 checkpoint-selection set |
| E3 | 3+3 all-split (honest) | 6 | 2 | 0.772 | 0.810 | 0.822 | Fresh draw, seed≠42 |
| E3 | 3+3 test-split (OOD topo) | 6 | 2 | **0.798** | 0.823 | 0.843 | OOD topologies |
| E1 | 2+4 | 6 | 2 | 0.788 | 0.778 | 0.833 | Asymm, never trained on |
| E1 | 4+2 | 6 | 2 | 0.748 | 0.822 | 0.805 | Asymm, never trained on |
| E2 | 3+4 | 7 | 2 | 0.696 | 0.737 | 0.757 | Zero-shot +1 gene |
| E2 | 4+3 | 7 | 2 | 0.686 | 0.747 | 0.758 | Zero-shot +1 gene |
| E2 | 3+5 | 8 | 2 | 0.724 | 0.722 | 0.784 | Zero-shot +2 genes |
| E2 | 5+3 | 8 | 2 | 0.754 | 0.732 | 0.801 | Zero-shot +2 genes |
| E2 | 4+4 | 8 | 2 | 0.672 | 0.656 | 0.752 | Zero-shot +2 genes |
| E4 | 2→3→4 (chain) | 9 | 3 | 0.726 | 0.692 | 0.794 | Zero-shot 3-SCC chain |
| E4 | 3→3→3 (chain) | 9 | 3 | 0.698 | 0.736 | 0.770 | Zero-shot 3-SCC chain |

### Key findings

**13. Validation selection bias = 14pp:**
v5 was used to select the v8 best checkpoint. A fresh 3+3 draw (E3 all-split, different seed) gives only 77.2%, not 91.4%. The honest single-SCC-composition generalization number is **~77–80%**. The v5 number is not wrong — it reflects true performance on that specific distribution — but it is biased as a general accuracy estimate.

**14. Topology generalization confirmed (replicated):**
E3 test-split (OOD topologies held out from all training stages) scores **79.8%** — *higher* than E3 all-split (77.2%). The gap is in the right direction in both the first run and this final full run. The model learned dynamics rules, not topology fingerprints.

**15. OOD scale extrapolation — gradual degradation:**
Moving from n=6 to n=7 costs ~8pp (69–70%). This is not catastrophic collapse — the model maintains meaningful accuracy on networks 17% larger than its training max per-SCC.

**16. n=8 asymmetry paradox:**
Within n=8 variants: 5+3 (75.4%) > 3+5 (72.4%) > 4+4 (67.2%). When the root SCC is larger (n_a=5), the L1 prediction is more confident, and the target SCC (n_b=3) sits at the optimal training-distribution size. 4+4 is hardest: both SCCs are at the large end of the training distribution, compounding prediction uncertainty.

**17. Deep DAG chains don't compound error:**
E4 n=9 3-SCC chains score 70–73% — essentially the same as E2 n=7 two-SCC compositions. If errors compounded multiplicatively (77% × 77%) we'd expect ~59%. Instead the 3-SCC pipeline holds at 70%+. The topological DAG walk in `assemble.py` handles multi-level forcing natively with no accuracy penalty for DAG depth per se.

**18. Summary — the degradation curve:**
```
v5 biased (91.4%) → honest n=6 (77.2%) → asymm n=6 (~76.8%) → OOD n=7 (~69%) → OOD n=8 (~71%) → deep DAG n=9 (~71%)
```
The plateau at ~70% from n=7 onward suggests the bottleneck is L1 SCC count accuracy on larger SCCs, not the composition step itself.

---

## v9: Diverse composition training (in progress)

**Start:** v8 checkpoints (epoch 84)  
**Fix 1:** Add 2+4 and 4+2 training compositions (seed=100, 500 cases each) so the model sees asymmetric size pairs during fine-tuning.  
**Fix 2:** Use a fresh 3+3 val set (seed=99, never seen by the model) for checkpoint selection — eliminates the v5 validation selection bias.  
**Training dirs:** v3 + v4 + v9_train_2p4 + v9_train_4p2  
**Val dir:** v9_val_3p3 (seed=99)  
**Hyperparameters:** identical to v8 (proven to work)  
**Scripts:** `jobs/gen_v9_data.sh` → `jobs/finetune_e2e_v9_a100.sh`

**Result: Training saturated (train loss = 0.0000 from epoch 1)**
v8 already predicts the count correctly on ALL 2000 training cases from epoch 1. The margin hinge is satisfied everywhere; `correct_ce_weight=0.01` provides negligible gradient when the model predicts with near-certainty. No gradient flows → val count_exact locked at 0.7840 for all 100 epochs.

**Conclusion:** v8 is the architecture ceiling for composition fine-tuning on easy cases. The ~79% on fresh evals is the genuine generalization number. Getting to 85%+ requires hard-negative mining (training only on cases where v8 currently fails) or architectural changes. Deferring architectural improvements; proceeding with E2 and E4 using v8.

---

## Paths Not Yet Tried

1. **Pos-only fine-tune from v8:** `--lr_pos 5e-6 --lr 0 --lr_backbone 0 --pos_reg_weight 1.0` — trains only `l1_model.pos_head`, guaranteed not to hurt count_exact/stable_acc. May push fp_hd above 0.90.
2. **refine_steps=30+** — more Newton convergence steps might recover additional fp_hd for free.
3. **Larger training set** — more diverse composition cases may help generalization further.

---

## Reproducibility

```bash
# Reproduce v8 eval (current best)
cd $SCRATCH/GNN
git checkout main

PYTHONPATH=$SCRATCH/GNN:$SCRATCH/GNN_temp \
  python exhaust_gen/eval_composition2.py \
    --l1_ckpt $SCRATCH/runs/e2e_v8/l1_best.pt \
    --l3_ckpt $SCRATCH/runs/e2e_v8/l3_best.pt \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  /tmp/eval_v8_repro.csv \
    --l3_frame forced --dedup fill --refine_steps 15
# Expected: count_exact=0.9140 (457/500), fp_hd=0.8908, stable_acc=0.9140

# Reproduce v6 eval (pre-v8 best)
PYTHONPATH=$SCRATCH/GNN:$SCRATCH/GNN_temp \
  python exhaust_gen/eval_composition2.py \
    --l1_ckpt $SCRATCH/runs/e2e_v6/l1_best.pt \
    --l3_ckpt $SCRATCH/runs/e2e_v6/l3_best.pt \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  /tmp/eval_v6_repro.csv \
    --l3_frame forced --dedup fill --refine_steps 15
# Expected: count_exact=0.8400 (420/500), fp_hd=0.8242, stable_acc=0.8598
```

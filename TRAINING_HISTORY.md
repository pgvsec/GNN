# GNN Training History & Current Status

**Last updated:** 2026-10-18  
**Current best checkpoint:** `layer3_v4/best.pt` (eval_v7/v8)  
**Best eval result:** v5 baseline — 75.00% count_exact, 86.50% fp_hd, 81.06% stable_acc

---

## 1. Project Overview

We are building a **pure ML model** (no hybrid numerical solver) that predicts compositional gene regulatory network (GRN) dynamics. Given a 6-gene GRN, the model must predict:

- How many fixed points (steady states) the network has (`count_exact`)
- Where those fixed points are in expression space (`fp_hd` — fraction of GT fixed points recovered within half a log-decade)
- Which fixed points are stable attractors (`stable_acc`)

### Architecture: Three-Layer Pipeline

| Layer | Class | Input | Output |
|---|---|---|---|
| L1 | `SccFixedPointNet` | Single SCC graph (≤6 genes) | Set of fixed points + stability |
| L2 | `CorpusBNet` | Forced/driven SCC | Conditioned fixed points |
| L3 | `CorpusCNet` | Full SCC-DAG composition | Composed 6-gene fixed points |

The L3 model assembles predictions from L1/L2 sub-models for each SCC in the condensation DAG of a 6-gene graph. The key challenge is **count accuracy**: the model must predict exactly how many fixed points the composed system has.

### Key Files

| File | Purpose |
|---|---|
| `model.py` | L1 network (`SccFixedPointNet`), SlotDecoder, count_head |
| `model_b.py` | L2 network (`CorpusBNet`) |
| `model_c.py` | L3 network (`CorpusCNet`) |
| `train.py` | L1 training loop |
| `train_layer2.py` | L2 training loop |
| `train_layer3.py` | L3 training loop |
| `losses.py` | `SetCriterion`: Hungarian matching, count loss |
| `exhaust_gen/generate.py` | Corpus A generator (SCC graphs, ground-truth FPs via numerical solver) |
| `exhaust_gen/gen_composition_eval.py` | Builds the 500-case composition eval dataset |
| `exhaust_gen/eval_composition2.py` | Evaluates L1+L3 on composition eval dataset |

---

## 2. Scope Reduction: n=1–10 → n=1–6

The original model targeted SCCs of up to 10 genes. This was narrowed to **n=1–6** for the following reasons:

- The composition eval task (predicting dynamics of 6-gene graphs made of two 3-gene SCCs) only requires n≤6 L1 predictions
- Exhaustive topology enumeration becomes computationally tractable at n≤6 (the `exhaust` preset)
- GPU memory and wall-clock time at n>6 were prohibitive per worker

The `exhaust_gen` module was built specifically for this: it enumerates all non-isomorphic SCC topologies for n=2..6, draws parameter samples, and solves for ground-truth fixed points using the same Gillespie/Newton ODE solver used at inference.

**Git commit:** `acab2e6` — "exhaust_gen: cap at n<=6, redirect n=7-10 compute to n=4-6"

---

## 3. Corpus Structure

### Corpus A (L1 training)
Single-SCC graphs, n=2..6, with ground-truth fixed points solved exhaustively.

**Cluster path:** `/scratch/gilbreth/gupt1042/corpus_exhaust_A/cache` (original)  
**Cluster path:** `/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache` (v2, see §5.1)

**Original `exhaust` preset:**
```
draws = [240, 200, 160, 60, 50, 0, 0, 0, 0]   # samples per topology, n=2..6
phase_fracs = (0.30, 0.50, 0.20)               # interior / switch / boundary
```
3.3M total samples, 591 shards.

### Corpus B (L2 training)
Forced/driven SCC graphs — one SCC receives external input signals.  
**Cluster path:** `/scratch/gilbreth/gupt1042/corpus_exhaust_B/cache`

### Corpus C (L3 training)
Full SCC-DAG composed graphs (the composition task).  
**Cluster path:** `/scratch/gilbreth/gupt1042/corpusC/cache`

### Composition Eval Dataset
500 test cases: two 3-gene SCCs drawn from Corpus A, joined with 2 directed cross-SCC edges. Ground-truth FPs solved numerically.  
Generated fresh each eval run from a fixed seed.

---

## 4. Evaluation Metrics

All evals run on 500 composition test cases.

| Metric | Definition |
|---|---|
| `count_exact` | Fraction of cases where `n_fp_pred == n_fp_gt` exactly |
| `fp_hd` | For each GT fixed point, whether ≥1 predicted FP is within half a log-decade (max over all components); mean over GT FPs |
| `stable_acc` | Stability label accuracy over predicted FPs matched to GT |

**Key observation:** `fp_hd` measures "did the model find a point near the right basin?" while `count_exact` measures "did the model know how many basins to predict?". These can diverge: a model might find the right locations but predict too many (high `fp_hd`, low `count_exact`).

---

## 5. Training Iterations

### Iteration 1 — v1 baseline (pre-exhaust)

**Checkpoints:** `exhaust_layer1/best.pt`, `exhaust_layer3/best.pt`  
**Corpus:** Original `corpus_exhaust_A` with `exhaust` preset  
**Eval:** Not formally recorded in this log (pre-eval infrastructure)

### Iteration 2 — v5 baseline

**Checkpoints:** `runs/layer1_v5/best.pt` (or similar), `runs/layer3/best.pt`  
**Corpus:** Original `corpus_exhaust_A` (30/50/20 phase split)  
**Model:** `model.py` with `count_head(g)` — count head reads from pre-decoder graph embedding `g`  
**Eval file:** `eval_v5.csv` (500 cases)

**Results:**
| count_exact | fp_hd | stable_acc |
|---|---|---|
| **75.00%** | **86.50%** | **81.06%** |

This is the **reference baseline**. All subsequent iterations are compared against these numbers.

**Known failure mode:** Systematic overcounting — many cases have `n_fp_pred > n_fp_gt`. The hypothesis was that the count head reads from `g` (pre-decoder graph embedding, computed before the SlotDecoder runs), so it cannot see what basins the decoder actually found.

---

### Iteration 3 — v3: Slot-Augmented Count Head + Corpus A v2

This iteration tried **two simultaneous improvements**:

#### Change 1: Richer Corpus A v2 (more boundary-phase samples)

**Motivation:** The "boundary phase" samples (near-bifurcation parameter regimes) are the hardest cases — networks near a bifurcation have more fixed points and the model needs to learn these landscapes. The original corpus had only 20% boundary-phase samples; doubling this should improve multi-basin accuracy.

**Implementation (`exhaust_gen/generate.py`):**
```python
"exhaust_v2": dict(
    topos=[10000, 15000, 12000, 0, 0, 0, 0],
    draws=[240, 200, 160, 60, 50, 0, 0, 0, 0],
    seeds=[96, 128, 160, 192, 224, 224, 224, 224, 224],
    n_max=6,
    phase_fracs=(0.20, 0.40, 0.40),   # was (0.30, 0.50, 0.20)
),
```
Boundary-phase fraction doubled from 20% → 40%. Verified in cache stats: phase 2 mean n_fp = 2.90 vs 2.08 for switch phase.  
**Cluster path:** `/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache`  
**Git commit:** `fe92ef5` — "Add exhaust_v2 preset with 40% boundary-phase fraction"

#### Change 2: Slot-Augmented Count Head

**Motivation:** The `count_head` reads from `g` (pre-decoder embedding). After the SlotDecoder runs, each slot `s[k]` contains basin-specific information. Feeding `s.mean(dim=1)` back to the count head lets it see what the decoder actually found before making its count decision.

**Implementation (`model.py`):**
```python
# Before (v5):
"count_logit": self.count_head(g),
"stable_count_logit": self.stable_count_head(g),

# After (v3):
g_count = g + s.mean(dim=1)   # residual: graph emb + mean slot emb
"count_logit": self.count_head(g_count),
"stable_count_logit": self.stable_count_head(g_count),
```
Same change applied in `model_b.py` for L2's `_run_backbone` and `forward` Step 5.  
**Git commit:** `7bc36e0` — "Feed slot embeddings back to count head"

#### Training commands (iteration 3):
```bash
# L1
PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp \
python train.py \
  --cache_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer1_v3 \
  --num_workers 8

# L2
python train_layer2.py \
  --cache_dir_b /scratch/gilbreth/gupt1042/corpus_exhaust_B/cache \
  --layer1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v3/best.pt \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer2_v3 \
  --num_workers 8

# L3
python train_layer3.py \
  --cache_dir_c /scratch/gilbreth/gupt1042/corpusC/cache \
  --layer2_ckpt /scratch/gilbreth/gupt1042/runs/layer2_v3/best.pt \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer3_v3 \
  --num_workers 8
```

#### Eval (eval_v6.csv):
```bash
# Step 1: generate eval dataset
PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp \
python exhaust_gen/gen_composition_eval.py \
  --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v3 \
  --n_cases 500 --n_cross 2 --seed 42

# Step 2: eval
python exhaust_gen/eval_composition2.py \
  --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v3/best.pt \
  --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v3/best.pt \
  --eval_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v3 \
  --out_csv /scratch/gilbreth/gupt1042/runs/eval_v6.csv
```

**Results (eval_v6.csv):**
| count_exact | fp_hd | stable_acc |
|---|---|---|
| 50.40% | **92.96%** | 74.94% |

**Analysis:** `fp_hd` improved substantially (+6.5pp), meaning the model found the right basin locations more often. But `count_exact` collapsed from 75% to 50% — the slot-augmented count head made the model predict more fixed points than exist. Inspection of rows showed many cases with `n_fp_pred >> n_fp_gt` (e.g. gt=5, pred=18; gt=1, pred=6). The mean slot embedding `s.mean(dim=1)` carries a "slots found something" signal that biases the count head upward.

---

### Iteration 4 — v4: Corpus A v2 Only (Slot Augmentation Reverted)

**Decision:** Revert the slot augmentation, keep Corpus A v2. This isolates whether the corpus improvement alone is sufficient.

**Implementation:** Reverted `model.py` and `model_b.py` to `count_head(g)`.  
**Git commit:** `b9f4d77` — "Revert slot-augmented count head; use plain graph embedding"

#### Training commands (iteration 4):
```bash
# L1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --cache_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer1_v4 \
  --num_workers 8

# L2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_layer2.py \
  --cache_dir_b /scratch/gilbreth/gupt1042/corpus_exhaust_B/cache \
  --layer1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v4/best.pt \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer2_v4 \
  --num_workers 8

# L3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_layer3.py \
  --cache_dir_c /scratch/gilbreth/gupt1042/corpusC/cache \
  --layer2_ckpt /scratch/gilbreth/gupt1042/runs/layer2_v4/best.pt \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer3_v4 \
  --num_workers 8
```

#### Eval (eval_v7.csv):
```bash
PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp \
python exhaust_gen/gen_composition_eval.py \
  --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v4 \
  --n_cases 500 --n_cross 2 --seed 42

python exhaust_gen/eval_composition2.py \
  --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v4/best.pt \
  --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v4/best.pt \
  --eval_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v4 \
  --out_csv /scratch/gilbreth/gupt1042/runs/eval_v7.csv
```

**Results (eval_v7.csv):**
| count_exact | fp_hd | stable_acc |
|---|---|---|
| 50.40% | 92.62% | 73.60% |

**Analysis:** Almost identical to eval_v6. Corpus A v2 alone also did not restore count_exact. This reveals that the drop from v5 → v6/v7 is not solely attributable to the slot augmentation — Corpus A v2 itself shifted the distribution in a way that hurt count accuracy. The richer boundary-phase training makes the model more sensitive to multi-basin landscapes but apparently also lowers its threshold for predicting additional fixed points.

---

### Iteration 5 — v8: Inference-Only Fixes (No Retraining)

**Checkpoints:** Same as v7 (`layer1_v4/best.pt`, `layer3_v4/best.pt`)  
**Changes (committed `a7098f5`):**

1. **`assemble.py` `_RESCUED_THRESH`: `1e-3 → 1e-6`** — Tighter Newton residual threshold for slot rescue. Genuine fixed points converge to residual ~1e-12 by step 15; spurious slots stay >> 1e-6 even after 20 Newton steps.

2. **`assemble.py` `refine_steps`: `5 → 20`** — More Newton iterations give genuine and spurious slots time to separate. Previously spurious slots at step 5 had residuals near 1e-3 and were incorrectly rescued.

**No model retraining — pure inference fix.**

**Results (eval_v8.csv):**
| count_exact | fp_hd | stable_acc |
|---|---|---|
| **68.00%** | 84.68% | — |

**Direction breakdown:**

| gt count | n_cases | exact | under | over |
|---|---|---|---|---|
| 1 | 225 | 89.9% | 0 | 28 |
| 3 | 142 | 51.4% | 56 | 11 |
| 5 | 98 | 32.7% | 33 | 0 |
| 7+ | ~35 | 0-7% | all under | 0 |

**Analysis:** The inference fix corrected the overcounting problem (43 over vs 117 under). The model now systematically undercounts for gt=3,5,7+ cases. This reveals the v7 model's count_head is conservatively biased toward k=1 predictions — it was trained with `w_count=2.0` and `eos_coef=0.3`, which did not strongly incentivize correct high-k predictions. The rescue mechanism previously masked this bias by adding extra slots post-decode.

The undercounting failure mode is now exposed and requires a training fix.

---

## 6. Summary Table

| Eval | Checkpoint | Key Change | count_exact | fp_hd | stable_acc |
|---|---|---|---|---|---|
| v5 | layer1/3 (original) | Baseline (original Corpus A, count_head(g)) | **75.00%** | 86.50% | **81.06%** |
| v6 | layer1_v3/layer3_v3 | Corpus A v2 (40% boundary) + slot-aug count head | 50.40% | **92.96%** | 74.94% |
| v7 | layer1_v4/layer3_v4 | Corpus A v2 only (slot aug reverted) | 50.40% | 92.62% | 73.60% |
| v8 | layer1_v4/layer3_v4 | Inference fix: rescue thresh 1e-3→1e-6, refine 5→20 | 68.00% | 84.68% | — |

**Key finding:** Corpus A v2 consistently improves `fp_hd` by ~+6pp but drops `count_exact` by ~25pp. The v8 inference fix partially recovers count_exact (68%) without retraining. The remaining 7pp gap to v5 (75%) is attributed to count_head undercalibration for high-k cases, requiring retraining with stronger `w_count` and `eos_coef`.

---

## 7. Root Cause Analysis: Count Undercalibration (v8 state)

After the v8 inference fix, the primary failure mode shifted from overcounting to **undercounting for high-k cases**. The root cause chain:

1. **`exhaust_v2` 40% boundary-phase** → more multi-basin samples, but also dilutes Brouwer-consistent samples relative to the count loss denominator
2. **`w_count=2.0`** too low → count loss signal dominated by existence loss; count_head learns conservative k=1 default
3. **`eos_coef=0.3`** insufficient penalty on false positives → at training time, model is not forced to be selective; at inference (tight threshold), this manifests as underprediction

**Fix (already committed to `train.py` in `a7098f5`):**
- `w_count: 2.0 → 3.0` — 50% stronger count supervision
- `eos_coef: 0.3 → 0.5` — stronger penalty on spurious slot activations

These changes require a full L1 + L3 retrain (v9). L2 retraining is optional unless L3 is still insufficient after v9.

---

## 7b. Root Cause Analysis: Overcounting (v6/v7 historical)

Inspection of eval_v6 and eval_v7 rows shows a clear pattern of **systematic overcounting**:

```
case 414: gt=5,  pred=18  (overcounting by 13)
case 192: gt=3,  pred=11  (overcounting by 8)
case 281: gt=3,  pred=8   (overcounting by 5)
case 166: gt=3,  pred=8   (overcounting by 5)
```

The overcounting is not a count head calibration issue — it persists even with `count_head(g)` (v7). This points to the **SlotDecoder filling too many slots with plausible-looking fixed points** even when fewer basins exist.

### Hypothesis

The `exist_logit` threshold (which determines which slots are "active" at inference) is too permissive. The decoder finds local Newton solutions in many slots, and the existence classifier does not suppress spurious ones aggressively enough.

Training signal: in `losses.py`, the `exist_head` is trained with a cross-entropy loss weighted by `eos_coef` for empty slots. If `eos_coef` is too low, the model is not penalized enough for activating empty slots.

### Current `eos_coef` default
```python
# train.py
eos_coef: float = 0.1   # weight on EOS/empty-slot classification loss
```

---

## 8. v9 Retraining Plan (Next Step)

### Changes
- `w_count=3.0` (train.py default already updated)
- `eos_coef=0.5` (train.py default already updated)
- Same Corpus A v2 (exhaust_v2 preset)
- Same model architecture (no structural changes)

### Commands

```bash
# Retrain L1 (v9)
PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --cache_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
    --out_dir   /scratch/gilbreth/gupt1042/runs/layer1_v9 \
    --num_workers 8

# Retrain L3 with existing L2 checkpoint (L2 architecture unchanged)
PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_layer3.py \
    --layer2_ckpt /scratch/gilbreth/gupt1042/runs/layer2_v4/best.pt \
    --cache_dir_c /scratch/gilbreth/gupt1042/corpusC/cache \
    --out_dir     /scratch/gilbreth/gupt1042/runs/layer3_v9 \
    --num_workers 8
```

> **Note on L2:** L2 (`CorpusBNet`) is a frozen-backbone wrapper — it shares weights with L1. A new L1 checkpoint has a different backbone, so strictly L2 should also be retrained with the new L1. However since L3 (`CorpusCNet`) loads from a L2 checkpoint, and L3 is the inference-time model, try the following sequence:
> 1. Retrain L2 with new L1: `python train_layer2.py --layer1_ckpt .../layer1_v9/best.pt --cache_dir_b .../corpus_exhaust_B/cache --out_dir .../layer2_v9`
> 2. Retrain L3 with new L2: `python train_layer3.py --layer2_ckpt .../layer2_v9/best.pt ...`

### Eval (v9)

```bash
python exhaust_gen/eval_composition2.py \
    --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v9/best.pt \
    --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v9/best.pt \
    --eval_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v4 \
    --out_csv  /scratch/gilbreth/gupt1042/runs/eval_v9.csv
```

### Target
- `count_exact ≥ 75%` (match or beat v5 baseline)
- `fp_hd ≥ 86.5%` (match v5; current 84.68% is slightly below)
- `stable_acc ≥ 81.06%` (match v5)

---

## 8b. ML Improvement Changes (v10 — in progress)

**Hypothesis:** Three simultaneous changes target the overcounting root cause from three angles, going beyond the v9 plan.

### 8b.1 Loss Changes (losses.py)

#### Existence Soft-Count Consistency Loss (`w_exist_cnt=0.5`)
**New loss term:** `l_exist_cnt = MSE(sigmoid(exist_logit).sum(1), k_true.float())` on Brouwer-consistent samples.

Unlike the slot-by-slot existence BCE, this is a *global* constraint: it simultaneously penalises over-activating empty slots AND corrects the average activation when the model is counting too high. The BCE alone can be satisfied by a model that gives every slot p_exist ≈ 0.5 (moderate activation, moderate BCE loss), but the soft-count loss forces `Σ p_exist → k_true`.

#### Focal Count CE (`count_focal_gamma=2.0`)
**Modified count loss:** `l_count = ((1-pt)^2 * ce_k + ce_s) * cons1 / n_cons`

On Corpus A v2, many Brouwer-consistent samples have k=1 (interior-phase dominant). The model already gets those right, leaving hard multi-basin cases underweighted in the gradient. Focal reweighting down-weights easy examples, concentrating the count gradient on the hard cases where the model is actually wrong.

#### Stronger existence penalty (`eos_coef=0.5`, already in v9)
Penalises spurious slot activations more symmetrically with true-positive activations.

### 8b.2 Architecture Change (model.py)

#### Deeper Count Head (`count_head_layers=3`)
Increased MLP depth from 2 → 3 layers for `count_head` and `stable_count_head`. Fixed-point count is determined by the graph's bifurcation structure, which is a complex nonlinear function requiring more capacity.

### 8b.3 Config Changes (train.py / train_layer2.py / train_layer3.py)

| Parameter | Was | Now | Reason |
|---|---|---|---|
| `eos_coef` | 0.1 | 0.5 | Stronger empty-slot penalty |
| `w_count` | 0.5 | 3.0 | Compensate for diluted Brouwer-consistent fraction |
| `w_exist_cnt` | — | 0.5 | New: soft-count consistency loss |
| `count_focal_gamma` | — | 2.0 | New: focal reweighting of count CE |
| `count_head_layers` | 2 | 3 | New: deeper count head |

### 8b.4 Inference Change (model.py)

`decode_slots` now accepts `exist_thresh` (default 0.0 = off). After retraining, this threshold can be tuned on a held-out validation set to further reduce overcounting without retraining.

### 8b.5 Training Commands

```bash
cd /scratch/gilbreth/gupt1042/GNN_temp
git pull origin claude/zealous-wright-eum1ur

# L1 (original corpus_exhaust_A — 30/50/20 split — for cleaner count signal)
sbatch exhaust_gen/train_layer1_exhaust.slurm   # → exhaust_layer1/best.pt

# L2 (after L1 finishes)
sbatch exhaust_gen/train_layer2_exhaust.slurm   # → exhaust_layer2/best.pt

# L3 (after L2 finishes)
sbatch exhaust_gen/train_layer3_exhaust.slurm   # → exhaust_layer3/best.pt
```

### 8b.6 Open Questions After Retrain
- Does `exist_thresh=0.3` further improve `count_exact` post-training?
  (tune on val set: try 0.1, 0.2, 0.3 in `eval_composition2.py`)
- Is Corpus A v2 still needed, or does the deeper count head + focal loss recover fp_hd on the original corpus?
- Can `count_focal_gamma` be tuned per corpus (higher gamma when more easy-k=1 cases)?


```bash
cd /scratch/gilbreth/gupt1042/GNN_temp
git pull origin claude/zealous-wright-eum1ur

# L1 (uses original corpus_exhaust_A — 30/50/20 split — for cleaner count signal)
sbatch exhaust_gen/train_layer1_exhaust.slurm   # → exhaust_layer1/best.pt

# L2 (after L1 finishes)
sbatch exhaust_gen/train_layer2_exhaust.slurm   # → exhaust_layer2/best.pt

# L3 (after L2 finishes)
sbatch exhaust_gen/train_layer3_exhaust.slurm   # → exhaust_layer3/best.pt
```

Or with Corpus A v2 (more boundary-phase, keeps fp_hd gain):
```bash
# Override CACHE_DIR in the L1 script or run manually:
python train.py \
  --cache_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer1_v5 \
  --num_workers 8

# Corresponding eval:
python exhaust_gen/gen_composition_eval.py \
  --corpus_cache /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v5 \
  --n_cases 500 --n_cross 2 --seed 42

python exhaust_gen/eval_composition2.py \
  --l1_ckpt /scratch/gilbreth/gupt1042/runs/layer1_v5/best.pt \
  --l3_ckpt /scratch/gilbreth/gupt1042/runs/layer3_v5/best.pt \
  --eval_dir /scratch/gilbreth/gupt1042/runs/composition_eval_v5 \
  --out_csv /scratch/gilbreth/gupt1042/runs/eval_v8.csv
```

### 8.6 Open Questions After v5 Retrain
- Does `exist_thresh=0.3` further improve `count_exact` post-training?  
  (tune on val set: try 0.1, 0.2, 0.3 in `eval_composition2.py`)
- Is Corpus A v2 still needed, or does the deeper count head + focal loss recover fp_hd on the original corpus?
- Can `count_focal_gamma` be tuned per corpus (higher gamma when more easy-k=1 cases)?

---

## 9. Infrastructure Notes

### Running on Gilbreth (Purdue)

All training runs from `/scratch/gilbreth/gupt1042/GNN_temp` (the git working tree).

```bash
# Always set PYTHONPATH before any command
export PYTHONPATH=/scratch/gilbreth/gupt1042/GNN_temp

# OOM prevention for 16+ workers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Git pull before each run to get latest model changes
git pull origin claude/zealous-wright-eum1ur
```

### Cluster Paths

| Path | Contents |
|---|---|
| `/scratch/gilbreth/gupt1042/GNN_temp` | Git working tree (pull before training) |
| `/scratch/gilbreth/gupt1042/corpus_exhaust_A/cache` | Original Corpus A (30/50/20 phase split) |
| `/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache` | Corpus A v2 (20/40/40 phase split) |
| `/scratch/gilbreth/gupt1042/corpus_exhaust_B/cache` | Corpus B (forced SCC) |
| `/scratch/gilbreth/gupt1042/corpusC/cache` | Corpus C (L3 training) |
| `/scratch/gilbreth/gupt1042/runs/layer1_v4/best.pt` | Current best L1 checkpoint |
| `/scratch/gilbreth/gupt1042/runs/layer2_v4/best.pt` | Current best L2 checkpoint |
| `/scratch/gilbreth/gupt1042/runs/layer3_v4/best.pt` | Current best L3 checkpoint |

### Eval Pipeline (2 steps)

```bash
# Step 1: generate 500 test cases (fast, ~1 min)
python exhaust_gen/gen_composition_eval.py \
  --corpus_cache <corpus_A_cache> \
  --out_dir <eval_dir> \
  --n_cases 500 --n_cross 2 --seed 42

# Step 2: run model eval (requires GPU, ~5 min)
python exhaust_gen/eval_composition2.py \
  --l1_ckpt <layer1_best.pt> \
  --l3_ckpt <layer3_best.pt> \
  --eval_dir <eval_dir> \
  --out_csv <out.csv>
```

Note: `eval_composition.py` (without the `2`) tries to find qualifying topologies inside the corpus cache directly and currently returns 0 results — use `eval_composition2.py` instead.

---

## 10. Hard Constraints (Non-Negotiable)

1. **Pure ML only** — no hybrid ML + numerical solver. The numerical solver is used only for generating ground-truth training data, never at inference.
2. **Physical accuracy** — do not mask physical accuracy with architectural tricks. Changes must not smooth over or hide non-linear dynamics.
3. **n≤6 scope** — the model targets SCCs of at most 6 genes and composed 6-gene graphs.
4. **Three-layer architecture** — L1/L2/L3 pipeline is fixed. Improvements are within-layer only (training data, loss weights, model hyperparameters).

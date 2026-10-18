# GNN Training History & Current Status

**Last updated:** 2026-10-18  
**Current best checkpoint:** `layer3_v4/best.pt` (eval_v7)  
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

## 6. Summary Table

| Eval | Checkpoint | Key Change | count_exact | fp_hd | stable_acc |
|---|---|---|---|---|---|
| v5 | layer1/3 (original) | Baseline (original Corpus A, count_head(g)) | **75.00%** | 86.50% | **81.06%** |
| v6 | layer1_v3/layer3_v3 | Corpus A v2 (40% boundary) + slot-aug count head | 50.40% | **92.96%** | 74.94% |
| v7 | layer1_v4/layer3_v4 | Corpus A v2 only (slot aug reverted) | 50.40% | 92.62% | 73.60% |

**Key finding:** Corpus A v2 consistently improves `fp_hd` by ~+6pp but drops `count_exact` by ~25pp. There is a tension between "find the right basin locations" and "know exactly how many basins exist".

---

## 7. Root Cause Analysis: Overcounting

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

## 8. Open Hypotheses & Next Steps

The following approaches have NOT yet been tried and are the natural next experiments:

### 8.1 Stronger Existence Penalty (High priority)
Increase `--eos_coef` during L1 training to make the model more conservative about declaring slots occupied.

```bash
python train.py \
  --cache_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache \
  --out_dir /scratch/gilbreth/gupt1042/runs/layer1_v5 \
  --eos_coef 0.3 \      # try 0.3 or 0.5 (default is 0.1)
  --num_workers 8
```

### 8.2 Count Prior Bias at Inference
At inference time the model decodes k via `argmax` over odd counts of `count_logit`. Adding a small prior penalty on higher k values (without retraining) could reduce overcounting.

This would be a change in `model_c.py` or `demo_assemble.py` in the `decode_slots` function.

### 8.3 Separate Existence Threshold Tuning
Instead of argmax on count_logit, threshold `exist_logit` at a higher value. Currently the top-k exist logits are used where k = predicted count. If instead a fixed threshold on `sigmoid(exist_logit) > τ` determines occupancy, τ can be tuned post-training on a held-out validation set.

### 8.4 Revert to Original Corpus A
If the goal is to match or beat v5 on count_exact, the most conservative approach is to go back to the original 30/50/20 phase split corpus and only change the model architecture. The Corpus A v2 fp_hd gain may not be worth the count_exact penalty.

### 8.5 Curriculum: Train on Easy Cases First
Phase 1: train only on n_fp=1 cases (easy count prediction). Phase 2: gradually introduce multi-basin cases. This could teach the model count discipline before being exposed to hard multi-attractor landscapes.

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

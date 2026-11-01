# GNN — Compositional GRN Dynamics Predictor

A three-layer GNN pipeline that predicts fixed-point dynamics of large Gene Regulatory Networks (GRNs) compositionally, via SCC decomposition. Target metrics: **count_exact ≥ 0.90 · fp_hd ≥ 0.90 · stable_acc ≥ 0.90**.

Current best (v8, `dedup=fill`, `refine_steps=15`): **0.914 / 0.891 / 0.914**

---

## Repository layout

```
GNN/
├── model.py                    # L1: SccFixedPointNet (root SCC predictor)
├── model_b.py                  # L2: CorpusBNet (inter-SCC signal)
├── model_c.py                  # L3: CorpusCNet (forced SCC predictor)
├── data.py                     # L1 dataset + N_MAX, NUM_STABILITY constants
├── data_b.py                   # L2 dataset
├── data_c.py                   # L3 dataset
├── solver.py                   # Newton–Raphson fixed-point refinement
├── losses.py                   # Hungarian-matched set loss, Chamfer, etc.
├── topology.py                 # Tarjan SCC decomposition, DAG utilities
├── assemble.py                 # GRNPredictor — full inference pipeline
├── train.py                    # L1 training script
├── train_layer2.py             # L2 training script
├── train_layer3.py             # L3 training script
├── finetune_e2e_composition.py # End-to-end composition fine-tuning
├── exhaust_gen/
│   ├── generate.py             # Corpus A generation (exhaustive)
│   ├── gen_corpus_d.py         # Corpus D generation (composition train set)
│   ├── gen_composition_eval.py # Composition eval set generation
│   ├── eval_composition2.py    # Canonical evaluation script
│   └── topology.py             # DAG utilities for generation
├── jobs/
│   ├── finetune_e2e_v8_a100.sh # v8 training job (A100)
│   └── finetune_e2e_v8.sh      # v8 training job (A10 fallback)
├── tests/                      # Unit tests (pytest)
└── PROGRESS.md                 # Experiment log (v2 → v8)
```

---

## Three-layer architecture

```
Input GRN → Tarjan SCC decomposition → condensation DAG
    ↓
  L1 (SccFixedPointNet): root SCCs → slot positions + exist logits + count
    ↓ Hill function (analytic)
  L3 (CorpusCNet): forced SCCs → conditioned FP predictions
    ↓
  assemble.py: enumerate branch paths → Newton refine → full FP set
```

**Key constraint**: L3's count head reads `exist_logit.detach()` from L1's decoder. v8 uses `--l1_count_no_detach` during composition fine-tuning to open the gradient path `composition loss → L1 count head → backbone`, which was blocked in v2–v7. Pure ML — no physics heuristics.

---

## Training

All jobs run on Gilbreth SLURM (`partition=a10`, `account=ayg`).

```bash
# Stage 1 — train L1 on Corpus A
python train.py --cache_dir <corpus_a_cache> ...

# Stage 2 — train L3 on Corpus C (or Corpus D for composition)
python train_layer3.py --cache_dir_c <corpus_c_cache> ...

# Stage 3 — composition fine-tuning (current: v8)
sbatch jobs/finetune_e2e_v8_a100.sh   # A100 partition
# or
sbatch jobs/finetune_e2e_v8.sh        # A10 fallback
```

See `PROGRESS.md` for checkpoint paths and hyperparameters for each experiment.

---

## Evaluation

```bash
# Canonical eval (v8 checkpoints)
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP python exhaust_gen/eval_composition2.py \
    --l1_ckpt $SCRATCH/runs/e2e_v8/l1_best.pt \
    --l3_ckpt $SCRATCH/runs/e2e_v8/l3_best.pt \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  eval_v8.csv \
    --l3_frame forced --dedup fill --refine_steps 15
```

`--l3_frame forced` matches the fine-tuning frame used by `finetune_e2e_composition.py`. `--dedup fill` trusts the ML count head exactly. `--refine_steps 15` gives +1.66pp fp_hd vs. 5 steps.

---

## Experiment summary

| Version | count_exact | fp_hd  | stable_acc | Notes                                              |
|---------|------------|--------|------------|----------------------------------------------------|
| v2      | 0.834      | 0.823  | 0.867      | Best baseline; checkpoints archived                |
| v6      | 0.840      | 0.824  | 0.860      | pos-only isolation; pos_head fine-tuned            |
| v7      | 0.834      | 0.847  | 0.865      | ForcingAwareCountHeadV2; backbone lr=2e-6          |
| **v8**  | **0.914**  | **0.891** | **0.914** | **composition_margin_loss + l1_count_no_detach** |

v8 checkpoints: `$SCRATCH/runs/e2e_v8/l1_best.pt` and `l3_best.pt` (epoch 84). **Back these up** — scratch is purged periodically.

Full details and next steps in `PROGRESS.md`.

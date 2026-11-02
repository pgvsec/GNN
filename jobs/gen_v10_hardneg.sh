#!/bin/bash
#SBATCH --job-name=gen_v10_hardneg
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/v10_hardneg/gen_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/v10_hardneg/gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# ── v10 hard-negative mining pipeline ────────────────────────────────────────
#
# Step 1: generate 5 000 fresh 3+3 compositions (seeds 201-205, 1 000 each)
# Step 2: run v8 eval on the combined 5 000 cases
# Step 3: filter to cases where count_exact == 0  → hard_neg_3p3 dataset
#
# The resulting hard_neg_3p3/ directory is then used as one of the training
# dirs in finetune_e2e_v10_a100.sh alongside the existing asymmetric datasets.
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

GEN_SCRIPT=$REPO_GNN/exhaust_gen/gen_composition_eval.py
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
EVAL_SCRIPT=$REPO_GNN/exhaust_gen/eval_composition2.py
FILTER_SCRIPT=$REPO_GNN/exhaust_gen/filter_hard_negatives.py

L1_CKPT=$SCRATCH/runs/e2e_v8/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v8/l3_best.pt

OUT_BASE=$SCRATCH/runs/v10_hardneg
mkdir -p $OUT_BASE

# ── Step 1: generate 5 000 fresh 3+3 cases (5 × 1 000) ─────────────────────
echo "=== Step 1: generating 5 000 fresh 3+3 cases ==="
MERGED_DIR=$OUT_BASE/fresh_3p3_5k
mkdir -p $MERGED_DIR

for SEED in 201 202 203 204 205; do
    PART_DIR=$OUT_BASE/fresh_3p3_seed${SEED}
    echo "  Generating seed=${SEED} → $PART_DIR"
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN_SCRIPT \
        --corpus_cache $CORPUS_CACHE \
        --n_cases 1000 \
        --seed $SEED \
        --out_dir $PART_DIR
    if [ $? -ne 0 ]; then
        echo "ERROR: generation failed for seed $SEED" && exit 1
    fi
done

# Merge 5 parts into one directory using merge_datasets.py (if available)
# Fallback: use the first 5k-case directory directly if a merger is not needed.
# eval_composition2.py reads from a single directory; merge them first.
MERGE_SCRIPT=$REPO_GNN/exhaust_gen/merge_datasets.py
if [ -f "$MERGE_SCRIPT" ]; then
    echo "  Merging 5 parts into $MERGED_DIR"
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $MERGE_SCRIPT \
        --in_dirs $(for SEED in 201 202 203 204 205; do echo $OUT_BASE/fresh_3p3_seed${SEED}; done) \
        --out_dir $MERGED_DIR
    if [ $? -ne 0 ]; then
        echo "ERROR: merge failed" && exit 1
    fi
else
    echo "  No merge_datasets.py found — concatenating via numpy inline"
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

base = Path("/scratch/gilbreth/gupt1042/runs/v10_hardneg")
seeds = [201, 202, 203, 204, 205]
parts = [base / f"fresh_3p3_seed{s}" for s in seeds]
out   = base / "fresh_3p3_5k"
out.mkdir(parents=True, exist_ok=True)

scalar_files = ["adj.npy", "par.npy", "K.npy", "T.npy", "nh.npy", "F.npy", "n_fp.npy"]
for fname in scalar_files:
    arrays = [np.load(p / fname) for p in parts if (p / fname).exists()]
    if arrays:
        merged = np.concatenate(arrays, axis=0)
        np.save(out / fname, merged)
        print(f"  {fname}: {merged.shape}")

# Rebuild fp_ptr and concatenate fp_state/fp_stab
state_parts, stab_parts, n_fp_parts = [], [], []
for p in parts:
    if (p / "fp_state.npy").exists():
        state_parts.append(np.load(p / "fp_state.npy"))
    if (p / "fp_stab.npy").exists():
        stab_parts.append(np.load(p / "fp_stab.npy"))
    if (p / "n_fp.npy").exists():
        n_fp_parts.append(np.load(p / "n_fp.npy"))

if state_parts:
    fp_state = np.concatenate(state_parts)
    np.save(out / "fp_state.npy", fp_state)
    print(f"  fp_state.npy: {fp_state.shape}")
if stab_parts:
    fp_stab = np.concatenate(stab_parts)
    np.save(out / "fp_stab.npy", fp_stab)
    print(f"  fp_stab.npy: {fp_stab.shape}")
if n_fp_parts:
    n_fp = np.concatenate(n_fp_parts)
    ptr  = np.concatenate([[0], np.cumsum(n_fp)[:-1]])
    np.save(out / "n_fp.npy",   n_fp)
    np.save(out / "fp_ptr.npy", ptr)
    print(f"  n_fp.npy: {n_fp.shape}, fp_ptr.npy: {ptr.shape}")

print("Merge done.")
PYEOF
    if [ $? -ne 0 ]; then
        echo "ERROR: inline merge failed" && exit 1
    fi
fi

echo "Merged dataset at $MERGED_DIR"

# ── Step 2: run v8 eval on the 5 000 fresh cases ────────────────────────────
echo ""
echo "=== Step 2: evaluating v8 on 5 000 fresh 3+3 cases ==="
EVAL_CSV=$OUT_BASE/eval_fresh_3p3_5k.csv
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL_SCRIPT \
    --l1_ckpt  $L1_CKPT \
    --l3_ckpt  $L3_CKPT \
    --eval_dir $MERGED_DIR \
    --out_csv  $EVAL_CSV \
    --l3_frame forced \
    --dedup fill \
    --refine_steps 15
if [ $? -ne 0 ]; then
    echo "ERROR: eval failed" && exit 1
fi

# ── Step 3: filter to hard negatives ────────────────────────────────────────
echo ""
echo "=== Step 3: filtering to hard negatives (count_exact == 0) ==="
HARDNEG_DIR=$OUT_BASE/hard_neg_3p3
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $FILTER_SCRIPT \
    --eval_csv    $EVAL_CSV \
    --dataset_dir $MERGED_DIR \
    --out_dir     $HARDNEG_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: filter failed" && exit 1
fi

echo ""
echo "==================================================================="
echo " Hard-negative dataset ready at: $HARDNEG_DIR"
echo " Use as --train_dirs argument in finetune_e2e_v10_a100.sh"
echo "==================================================================="

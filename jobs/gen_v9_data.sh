#!/bin/bash
#SBATCH --job-name=gen_v9_data
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v9/gen_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v9/gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Generate training and validation data for v9.
#
# Problem v8 revealed: the model was fine-tuned and checkpoint-selected
# entirely on 3+3 compositions (v3/v4 train, v5 val).  Eval on fresh
# 3+3 draws (E3-all-split) gives ~77%, not 91%, showing ~14pp of the v5
# score was validation-selection bias.  Eval on 2+4 and 4+2 (E1) gives
# 74-79%, showing the model never learned asymmetric compositions.
#
# Fix for v9:
#   1. Add 2+4 and 4+2 composition examples to the training set.
#      The backbone already handles 2-gene and 4-gene SCCs; the fine-tuning
#      just needs to see the asymmetric composition step.
#   2. Use a fresh 3+3 val set (seed=99, never used for checkpoint selection)
#      so v9 best-epoch selection is unbiased.
#
# Seeds:
#   seed=42  → eval sets (E1/E3 experiments, never use for training)
#   seed=99  → v9 fresh validation (no training data drawn from it)
#   seed=100 → v9 training data (2+4 and 4+2)

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $SCRATCH/runs/e2e_v9

GEN="$REPO_GNN/exhaust_gen/gen_composition_eval_general.py"

# --- Training: 2+4 compositions (seed=100) --------------------------------
echo "=== v9_train_2p4 (train, n_a=2, n_b=4, seed=100) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $SCRATCH/runs/v9_train_2p4 \
    --n_a 2 --n_b 4 \
    --n_cases    500 \
    --n_cross_ab   2 \
    --seed       100

# --- Training: 4+2 compositions (seed=100) --------------------------------
echo "=== v9_train_4p2 (train, n_a=4, n_b=2, seed=100) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $SCRATCH/runs/v9_train_4p2 \
    --n_a 4 --n_b 2 \
    --n_cases    500 \
    --n_cross_ab   2 \
    --seed       100

# --- Validation: fresh 3+3 (seed=99, NEVER used as training data) ---------
echo "=== v9_val_3p3 (val, n_a=3, n_b=3, seed=99) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $SCRATCH/runs/v9_val_3p3 \
    --n_a 3 --n_b 3 \
    --n_cases    500 \
    --n_cross_ab   2 \
    --seed        99

echo "=== v9 data generation complete ==="
ls -lh $SCRATCH/runs/v9_train_2p4/adj.npy \
       $SCRATCH/runs/v9_train_4p2/adj.npy \
       $SCRATCH/runs/v9_val_3p3/adj.npy 2>/dev/null

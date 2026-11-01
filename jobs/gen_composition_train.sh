#!/bin/bash
#SBATCH --job-name=gen_train
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/gen_train_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/gen_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Generate 8 x 500-case composition training batches (seeds 1001-1008).
# These are in the same format as composition_eval_v3/v4, ready to be passed
# as --train_dirs.  After this completes, run jobs/finetune_e2e_v7b.sh.
#
# Total new training data: 4000 cases.  Combined with v3+v4 (1000 cases):
# 5000 training cases vs the current 1000 — 5x increase.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

for SEED in 1001 1002 1003 1004 1005 1006 1007 1008; do
    OUT=$SCRATCH/runs/composition_train_s${SEED}
    if [ -f "$OUT/n_fp.npy" ]; then
        echo "=== seed $SEED already exists, skipping ==="
        continue
    fi
    echo "=== Generating seed $SEED ==="
    mkdir -p $OUT
    PYTHONPATH=$REPO_GNN_TEMP:$REPO_GNN $PYTHON exhaust_gen/gen_composition_eval.py \
        --corpus_cache $CORPUS_CACHE \
        --out_dir      $OUT \
        --n_cases      500 \
        --n_cross      2 \
        --seed         $SEED
    echo "=== Done seed $SEED ==="
done

echo ""
echo "All generation complete. Training dirs ready:"
for SEED in 1001 1002 1003 1004 1005 1006 1007 1008; do
    echo "  $SCRATCH/runs/composition_train_s${SEED}"
done

#!/bin/bash
#SBATCH --job-name=e2e_v7d_a100
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v7d/train_a100_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v7d/train_a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a100
#SBATCH --account=ayg

# v7d: use ONLY v3+v4 training data (same distribution as val set v5).
#
# v7b/v7c diagnosis: adding s1001-s1008 (harder distribution) caused val to
# decline every epoch. In v7c, train count_exact = 0.70 while val = 0.82 at
# epoch 1 — train data is harder than val. Training on it un-learns the
# v3/v4/v5 composition statistics, dragging val down while train improves.
#
# Fix: remove s1001-s1008. Use only v3+v4 (1000 cases, same distribution as
# v5). The model should improve monotonically since train and val share the
# same distribution.
#
# With batching optimisation (v7c code), 1000 cases runs at ~1-2 min/epoch.
# 200 epochs ≈ 3-4hr — fits in 8hr window.
#
# LR: very gentle to avoid disturbing v7's already-good representations.
#   lr=5e-6  (count head — now only 1x over backbone, near-identical rates)
#   lr_backbone=5e-6 (backbone — same as before)
#
# Starting from v7 checkpoints (best available: 0.834 / 0.847 / 0.865).

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v7d
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v7/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v7/l3_best.pt

# Only v3+v4 — same distribution as composition_eval_v5 (val set)
TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 \
            $SCRATCH/runs/composition_eval_v4"

VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON finetune_e2e_composition.py \
    --l1_ckpt     $L1_CKPT \
    --l3_ckpt     $L3_CKPT \
    --train_dirs  $TRAIN_DIRS \
    --val_dir     $VAL_DIR \
    --out_dir     $OUT_DIR \
    --epochs      200 \
    --lr          5e-6 \
    --lr_backbone 5e-6 \
    --lr_decoder  0.0 \
    --lr_pos      0.0 \
    --use_v2_count_head \
    --scc_reg_weight 0.3 \
    --pos_reg_weight 0.3 \
    --p_predicted 0.5 \
    --accumulate  32 \
    --wd          1e-4

echo "=== v7d a100 training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v7d.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

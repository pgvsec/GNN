#!/bin/bash
#SBATCH --job-name=e2e_v9_a10
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v9/train_a10_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v9/train_a10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# v9: diverse composition training to close the generalization gap.
#
# Problem with v8:
#   v8 was trained and checkpoint-selected entirely on 3+3 compositions
#   (train: v3+v4, val: v5).  This created two problems:
#     (a) Validation selection bias: v5 gave 91.4%, but fresh 3+3 draws
#         (E3-all-split) give only 77.2% — ~14pp gap from overfitting to v5.
#     (b) Size-split mismatch: 2+4 and 4+2 evals give 74-79% because the
#         model never trained on asymmetric compositions.
#
# Fix:
#   1. Augment training set with 2+4 and 4+2 compositions (seed=100).
#      Backbone handles all sizes; fine-tuning just learns the composition
#      step for asymmetric pairs.
#   2. Use a FRESH 3+3 val set (seed=99) for checkpoint selection — this
#      means the saved best checkpoint is not overfit to v5's distribution.
#
# Target: count_exact ≥ 0.85 on E1 (2+4, 4+2) and E3 (test-split, all-split).
#
# Prerequisite: run gen_v9_data.sh first to populate:
#   $SCRATCH/runs/v9_train_2p4/
#   $SCRATCH/runs/v9_train_4p2/
#   $SCRATCH/runs/v9_val_3p3/
#
# Start: v8 checkpoints (epoch 84: count_exact=0.914 on v5).
# All other hyperparameters identical to v8 — they are proven to work.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v9
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v8/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v8/l3_best.pt

# Original 3+3 training data (v3+v4) + new asymmetric compositions
TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 \
            $SCRATCH/runs/composition_eval_v4 \
            $SCRATCH/runs/v9_train_2p4 \
            $SCRATCH/runs/v9_train_4p2"

# Fresh 3+3 val (seed=99) — never used for checkpoint selection before
VAL_DIR=$SCRATCH/runs/v9_val_3p3

cd $REPO_GNN

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON finetune_e2e_composition.py \
    --l1_ckpt     $L1_CKPT \
    --l3_ckpt     $L3_CKPT \
    --train_dirs  $TRAIN_DIRS \
    --val_dir     $VAL_DIR \
    --out_dir     $OUT_DIR \
    --epochs      100 \
    --lr          1e-5 \
    --lr_backbone 2e-6 \
    --lr_decoder  0.0 \
    --lr_pos      0.0 \
    --use_v2_count_head \
    --use_margin_loss \
    --margin      0.1 \
    --correct_ce_weight 0.01 \
    --l1_count_no_detach \
    --scc_reg_weight 0.5 \
    --pos_reg_weight 0.3 \
    --p_predicted 0.5 \
    --accumulate  32

#!/bin/bash
#SBATCH --job-name=e2e_v7b_a100
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v7b/train_a100_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v7b/train_a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a100
#SBATCH --account=ayg

# v7b on a100: fits full 40-epoch run in the 4hr standby window.
# a100 is ~4x faster than a10 → ~5 min/epoch → 40 epochs ≈ 3.3hr.
#
# 40 epochs × 5000 training cases = 3.3x more total training data
# than v7 (60 epochs × 1000 cases), which is the key improvement.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v7b
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v7/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v7/l3_best.pt

TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 \
            $SCRATCH/runs/composition_eval_v4 \
            $SCRATCH/runs/composition_train_s1001 \
            $SCRATCH/runs/composition_train_s1002 \
            $SCRATCH/runs/composition_train_s1003 \
            $SCRATCH/runs/composition_train_s1004 \
            $SCRATCH/runs/composition_train_s1005 \
            $SCRATCH/runs/composition_train_s1006 \
            $SCRATCH/runs/composition_train_s1007 \
            $SCRATCH/runs/composition_train_s1008"

VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON finetune_e2e_composition.py \
    --l1_ckpt     $L1_CKPT \
    --l3_ckpt     $L3_CKPT \
    --train_dirs  $TRAIN_DIRS \
    --val_dir     $VAL_DIR \
    --out_dir     $OUT_DIR \
    --epochs      40 \
    --lr          1e-4 \
    --lr_backbone 5e-6 \
    --lr_decoder  0.0 \
    --lr_pos      0.0 \
    --use_v2_count_head \
    --scc_reg_weight 0.5 \
    --pos_reg_weight 0.3 \
    --p_predicted 0.5 \
    --accumulate  32 \
    --wd          1e-4

echo "=== v7b a100 training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v7b.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

#!/bin/bash
#SBATCH --job-name=e2e_v8_a100
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v8/train_a100_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v8/train_a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a100
#SBATCH --account=ayg

# v8: margin loss replaces composition CE.
#
# ROOT CAUSE of v7b/v7c/v7d failure:
#   All three showed val CE improving (0.618→0.561) while val count_exact
#   DECLINED (0.834→0.794).  CE is misaligned with count_exact: CE rewards
#   smooth probability mass, but count_exact rewards sharp argmax peaks.
#   Fine-tuning with CE teaches the model to spread probability smoothly —
#   exactly the wrong direction for argmax accuracy.
#
# Fix: composition_margin_loss (new in v8)
#   - Wrong predictions: hinge(P(n_pred) - P(n_gt) + margin=0.1)
#     Forces the correct count above the current argmax.  Gradient is ZERO
#     once P(n_gt) > P(n_pred) + margin — never pushes a correct prediction wrong.
#   - Correct predictions: tiny CE (weight=0.01) for calibration only.
#   This directly optimises count_exact instead of CE.
#
# Additional fix: --l1_count_no_detach
#   In v7, l1_model.count_detach=True blocks composition gradient from reaching
#   the L1 backbone (count head reads g.detach(), exist_logit.detach()).
#   Setting count_detach=False opens this path:
#     composition margin → L1 count head → g → backbone GNN
#   Combined with pos_reg_weight=0.3 to protect slot positions.
#
# Training data: v3+v4 only (same distribution as v5 val, avoids domain shift).
# LR: very gentle.  100 epochs ≈ 2-3hr on a100.
# Start: v7 (best: 0.834 / 0.847 / 0.865).

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v8
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v7/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v7/l3_best.pt

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
    --accumulate  32 \
    --wd          1e-4

echo "=== v8 a100 training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v8.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

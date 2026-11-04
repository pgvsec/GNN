#!/bin/bash
#SBATCH --job-name=e2e_v10_a100
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v10/train_a100_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v10/train_a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a100
#SBATCH --account=ayg

# ── v10: existence-count consistency loss + hard-negative mining ─────────────
#
# Changes vs v8:
#   1. Existence-count consistency loss (--exist_consistency_weight 1.0):
#        loss += smooth_l1( sum(sigmoid(10 * exist_logit)) - gt_k )
#      Forces L1's K=16 existence probabilities toward bimodal distribution
#      matching GT FP count.  Directly addresses the root cause of OOD failure:
#      soft/uncertain exist_logits for novel compositions.
#      Gradient flows: consistency_loss → exist_logit → exist_head → backbone.
#
#   2. Hard-negative mining:
#        --train_dirs: adds hard_neg_3p3 (cases where v8 predicted wrong count
#        on fresh 3+3 data).  Trains exclusively on the failure distribution.
#
#   3. Cross-edge diversity via asymmetric compositions:
#        --train_dirs: retains v9 2+4 and 4+2 datasets alongside v3/v4/hardneg.
#        These vary n_cross_ab exposure during training.
#
# PREREQUISITE: run gen_v10_hardneg.sh first to produce hard_neg_3p3/.
#
# Eval command is printed at the end.
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

# Start from v8 checkpoints (best composition-trained weights so far)
L1_CKPT=$SCRATCH/runs/e2e_v8/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v8/l3_best.pt

OUT_DIR=$SCRATCH/runs/e2e_v10
mkdir -p $OUT_DIR

# Training data:
#   v3 + v4    — original 3+3 (same as v8, 500+500 cases)
#   v9_2p4     — 2+4 asymmetric compositions
#   v9_4p2     — 4+2 asymmetric compositions
#   hard_neg   — 3+3 cases where v8 failed (mined by gen_v10_hardneg.sh)
TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 \
            $SCRATCH/runs/composition_eval_v4 \
            $SCRATCH/runs/generalization/e1_2p4 \
            $SCRATCH/runs/generalization/e1_4p2 \
            $SCRATCH/runs/v10_hardneg/hard_neg_3p3"

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
    --use_margin_loss \
    --margin      0.1 \
    --correct_ce_weight 0.01 \
    --l1_count_no_detach \
    --exist_consistency_weight 1.0 \
    --exist_consistency_alpha  10.0 \
    --scc_reg_weight 0.5 \
    --pos_reg_weight 0.3 \
    --p_predicted 0.5 \
    --accumulate  32 \
    --wd          1e-4

echo "=== v10 a100 training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate on canonical v5 set:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v10_v5.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"
echo ""
echo "To evaluate on fresh 3+3 (honest OOD baseline):"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/v10_hardneg/fresh_3p3_5k \\"
echo "    --out_csv  $OUT_DIR/eval_v10_fresh3p3.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

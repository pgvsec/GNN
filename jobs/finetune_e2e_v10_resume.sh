#!/bin/bash
#SBATCH --job-name=e2e_v10_resume
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v10/resume_a100_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v10/resume_a100_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a100
#SBATCH --account=ayg

# ── v10 resume: continues training from l1_last.pt / l3_last.pt ─────────────
#
# After the first job finishes (~30 epochs in 3 hrs), run this script to
# continue from where it left off.  The training script prints the exact
# values to fill in below at the end of the first run:
#
#   To resume: --l1_ckpt .../l1_last.pt --l3_ckpt .../l3_last.pt
#              --start_epoch 30 --best_val 0.8860
#
# Set the three RESUME_* variables below from that printout, then sbatch.
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

OUT_DIR=$SCRATCH/runs/e2e_v10

# ── Fill these in from the "To resume:" line printed by the first job ────────
# First run stopped at epoch 29; best checkpoint is epoch 25 (val CE=0.8920).
# There is no ep029 checkpoint, so resume from l1_best.pt (= ep025).
RESUME_START_EPOCH=25          # epoch of the checkpoint we're loading
RESUME_BEST_VAL=0.8920         # best val count_exact seen so far
# ────────────────────────────────────────────────────────────────────────────

L1_CKPT=$OUT_DIR/l1_best.pt
L3_CKPT=$OUT_DIR/l3_best.pt

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
    --epochs      75 \
    --start_epoch $RESUME_START_EPOCH \
    --best_val    $RESUME_BEST_VAL \
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

echo "=== v10 resume complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate on canonical v5 set:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v10_v5.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

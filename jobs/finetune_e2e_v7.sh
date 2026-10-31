#!/bin/bash
#SBATCH --job-name=e2e_v7
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v7/train_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v7/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# v7: end-to-end backbone training with ForcingAwareCountHeadV2.
#
# Key change vs v6 / decoder_v1:
#   --lr_backbone 1e-6    : unfreezes the full backbone (GNN encoder, A_eta,
#                           U_omega, SlotDecoder, exist_head).  With V2 head,
#                           composition CE gradient now flows:
#                             count CE -> V2 head -> z_cond (live) -> U_omega -> GNN
#                           This is the first time the backbone receives direct
#                           gradient from composition-level count accuracy.
#
#   --use_v2_count_head   : ForcingAwareCountHeadV2 — reads z_cond (not detached),
#                           no exist_logit dependency.  Backbone changes cannot
#                           destabilise count calibration.
#
#   --pos_reg_weight 0.3  : Chamfer loss on slot positions for BOTH root SCCs (L1)
#                           and forced SCCs (L3, new in v7).  With backbone unfrozen,
#                           this gradient reaches the shared SlotDecoder from
#                           both sides.
#
# Trainable params:
#   - l1 count_head, stable_count_head     (lr=1e-4)
#   - l3 forcing_count_head (V2, fresh)    (lr=1e-4)
#   - full backbone: GNN, A_eta, U_omega,  (lr=1e-6)
#     SlotDecoder, exist_head (shared)
#
# Starting point: v6 checkpoints (count_exact=0.840, fp_hd=0.822, stable_acc=0.860)

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v7
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v6/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v6/l3_best.pt

TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 $SCRATCH/runs/composition_eval_v4"
VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON finetune_e2e_composition.py \
    --l1_ckpt     $L1_CKPT \
    --l3_ckpt     $L3_CKPT \
    --train_dirs  $TRAIN_DIRS \
    --val_dir     $VAL_DIR \
    --out_dir     $OUT_DIR \
    --epochs      60 \
    --lr          1e-4 \
    --lr_backbone 1e-6 \
    --lr_decoder  0.0 \
    --lr_pos      0.0 \
    --use_v2_count_head \
    --scc_reg_weight 0.3 \
    --pos_reg_weight 0.3 \
    --p_predicted 0.5 \
    --accumulate  32 \
    --wd          1e-4

echo "=== v7 training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v7.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

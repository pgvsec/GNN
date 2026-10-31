#!/bin/bash
#SBATCH --job-name=decoder_v1
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_decoder_v1/train_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_decoder_v1/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# 5-epoch diagnostic: decoder unfreeze with ForcingAwareCountHeadV2.
#
# Goal: validate that decoder training improves fp_hd before committing to
# a full 60-epoch run.  Expected signal: val fp_hd should visibly improve
# vs v6 baseline (0.822) within 5 epochs if the decoder is the bottleneck.
#
# Architecture change:
#   --use_v2_count_head   : ForcingAwareCountHeadV2 (no exist_logit input)
#                           forcing_count_head is reinitialised from scratch
#   --lr_decoder 1e-5     : unfreezes l1_model.decoder + l1_model.exist_head
#                           (same backbone shared with l3_model during training)
#
# Trainable params:
#   - l1 count_head, stable_count_head         (lr=1e-4)
#   - l3 forcing_count_head (V2, fresh)        (lr=1e-4)
#   - l1/l3 SlotDecoder + exist_head           (lr=1e-5)
#   - GNN encoder/backbone: FROZEN
#
# Starting point: v6 checkpoints (count_exact=0.840, fp_hd=0.822, stable_acc=0.860)

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_decoder_v1
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
    --epochs      5 \
    --lr          1e-4 \
    --lr_decoder  1e-5 \
    --lr_backbone 0.0 \
    --lr_pos      0.0 \
    --use_v2_count_head \
    --scc_reg_weight 0.3 \
    --pos_reg_weight 0.5 \
    --p_predicted 0.5 \
    --accumulate  32 \
    --wd          1e-4

echo "=== 5-epoch diagnostic complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate (run separately):"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_decoder_v1.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

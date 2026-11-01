#!/bin/bash
#SBATCH --job-name=e2e_v7b
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v7b/train_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v7b/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# v7b: all three levers pulled simultaneously.
#
# vs v7:
#   1. Training data: 5000 cases (v3+v4+s1001-s1008) vs 1000 cases.
#      5x more data → backbone gradient signal is far richer, reduces overfitting.
#   2. scc_reg_weight 1.0 (was 0.3): much stronger per-SCC direct supervision.
#      Forces each SCC count head to be right independently, not just the product.
#   3. lr_backbone 5e-6 (was 1e-6): 5x faster backbone adaptation.
#      At 1e-6 the backbone barely moved in 60 epochs. 5e-6 should give meaningful
#      representation learning within the same wall-time.
#
# Starting from v7 checkpoints (count_exact=0.834, fp_hd=0.847, stable_acc=0.865).
# The V2 count head from v7 is already partially converged — no cold-start penalty.
#
# PREREQUISITE: jobs/gen_composition_train.sh must have completed successfully,
# creating composition_train_s1001 through composition_train_s1008.

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
    --epochs      60 \
    --max_train   1500 \
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

echo "=== v7b training complete ==="
echo "Checkpoints: $OUT_DIR/l1_best.pt  $OUT_DIR/l3_best.pt"
echo ""
echo "To evaluate:"
echo "PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \\"
echo "    --l1_ckpt $OUT_DIR/l1_best.pt \\"
echo "    --l3_ckpt $OUT_DIR/l3_best.pt \\"
echo "    --eval_dir $SCRATCH/runs/composition_eval_v5 \\"
echo "    --out_csv  $OUT_DIR/eval_v7b.csv \\"
echo "    --l3_frame forced --dedup fill --refine_steps 15"

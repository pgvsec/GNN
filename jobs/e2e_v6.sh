#!/bin/bash
#SBATCH --job-name=e2e_v6
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v6/e2e_v6_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v6/e2e_v6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=165G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/e2e_v6
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

# Start from v2 checkpoints (best baseline: 84.2%/82.29%/86.69% with rs=15)
L1_CKPT=$SCRATCH/runs/e2e_v2/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v2/l3_best.pt

if ls $SCRATCH/runs/composition_train_s1001 &>/dev/null; then
    TRAIN_DIRS="$SCRATCH/runs/composition_train_s1001 $SCRATCH/runs/composition_train_s1002 $SCRATCH/runs/composition_train_s1003 $SCRATCH/runs/composition_train_s1004"
else
    TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 $SCRATCH/runs/composition_eval_v4"
fi

VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

# v6: pos-only isolation mode
#   --lr 0           : count heads frozen (no CE gradient, no count param update)
#   --lr_backbone 0  : backbone GNN frozen (exist_logit stays stable)
#   --lr_pos 1e-4    : only l1_model.pos_head trained with pos Chamfer loss
#   --pos_reg_weight 1.0  : pos loss is the only gradient signal
#   --scc_reg_weight 0.0  : no per-SCC CE (count heads frozen anyway)
#   --p_predicted 0.0     : teacher forcing (cleaner pos supervision)
#
# Why this works: pos_head receives (s, h) from frozen decoder+backbone.
# Its output pos is supervised directly by GT FP coordinates via Chamfer loss.
# count_head reads exist_logit.detach() — exist_logit comes from frozen exist_head(s).
# Since decoder is frozen, s is unchanged, so exist_logit is unchanged.
# count_head inputs never shift → count predictions stay calibrated.
PYTHONPATH=$REPO_GNN_TEMP:$REPO_GNN $PYTHON finetune_e2e_composition.py \
    --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT \
    --train_dirs $TRAIN_DIRS --val_dir $VAL_DIR \
    --out_dir $OUT_DIR \
    --epochs 60 \
    --lr 0.0 \
    --lr_backbone 0.0 \
    --lr_pos 1e-4 \
    --scc_reg_weight 0.0 \
    --pos_reg_weight 1.0 \
    --p_predicted 0.0 \
    --seed 42

# Evaluate best checkpoint
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
    --l1_ckpt $OUT_DIR/l1_best.pt --l3_ckpt $OUT_DIR/l3_best.pt \
    --eval_dir $VAL_DIR \
    --out_csv $SCRATCH/runs/eval_e2e_v6_fill.csv \
    --dedup fill --refine_steps 15

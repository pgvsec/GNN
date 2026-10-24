#!/bin/bash
#SBATCH --job-name=e2e_v5
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v5/e2e_v5_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v5/e2e_v5_%j.err
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
OUT_DIR=$SCRATCH/runs/e2e_v5
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

# Start from v2 checkpoints (best performing baseline: 84.2%/82.29%/86.69% with rs=15)
L1_CKPT=$SCRATCH/runs/e2e_v2/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v2/l3_best.pt

if ls $SCRATCH/runs/composition_train_s1001 &>/dev/null; then
    TRAIN_DIRS="$SCRATCH/runs/composition_train_s1001 $SCRATCH/runs/composition_train_s1002 $SCRATCH/runs/composition_train_s1003 $SCRATCH/runs/composition_train_s1004"
else
    TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 $SCRATCH/runs/composition_eval_v4"
fi

VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

# Key changes vs v4:
#   --pos_reg_weight 0.1  : direct Chamfer position loss on root SCC slots vs GT FPs
#   --lr_backbone 1e-6    : unfreeze backbone so pos decoder receives gradient
#   --p_predicted 0.0     : teacher forcing only (no scheduled sampling noise)
#   --scc_reg_weight 0.3  : per-SCC CE regularisation (same as v2/v3/v4)
PYTHONPATH=$REPO_GNN_TEMP:$REPO_GNN $PYTHON finetune_e2e_composition.py --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT --train_dirs $TRAIN_DIRS --val_dir $VAL_DIR --out_dir $OUT_DIR --epochs 80 --lr 5e-5 --lr_backbone 1e-6 --scc_reg_weight 0.3 --p_predicted 0.0 --pos_reg_weight 0.1 --seed 42

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py --l1_ckpt $OUT_DIR/l1_best.pt --l3_ckpt $OUT_DIR/l3_best.pt --eval_dir $VAL_DIR --out_csv $SCRATCH/runs/eval_e2e_v5_fill.csv --dedup fill --refine_steps 15

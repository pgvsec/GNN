#!/bin/bash
#SBATCH --job-name=e2e_v3
#SBATCH --output=e2e_v3_%j.out
#SBATCH --error=e2e_v3_%j.err
#SBATCH -A ayg
#SBATCH -p a10
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# e2e_v3: teacher forcing fix (p_predicted=0.5)
#
# Resumes from e2e_v2 checkpoints (83.4%/80.63%/86.33% on composition_eval_v5).
# Introduces scheduled sampling: 50% of forced-SCC training calls use L1's
# predicted parent FPs instead of GT, closing the train-test gap that caps fp_hd.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp

OUT_DIR=$SCRATCH/runs/e2e_v3
mkdir -p $OUT_DIR

# Checkpoints: start from e2e_v2 best
L1_CKPT=$SCRATCH/runs/e2e_v2/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v2/l3_best.pt

# Training data: use composition_train_* if available, else eval_v3+v4
if ls $SCRATCH/runs/composition_train_s1001 &>/dev/null; then
    TRAIN_DIRS=""
    for d in $SCRATCH/runs/composition_train_s100{1,2,3,4}; do
        [ -d "$d" ] && TRAIN_DIRS="$TRAIN_DIRS $d"
    done
else
    TRAIN_DIRS="$SCRATCH/runs/composition_eval_v3 $SCRATCH/runs/composition_eval_v4"
fi

VAL_DIR=$SCRATCH/runs/composition_eval_v5

echo "=== e2e_v3 training ==="
echo "L1 ckpt : $L1_CKPT"
echo "L3 ckpt : $L3_CKPT"
echo "Train   : $TRAIN_DIRS"
echo "Val     : $VAL_DIR"
echo "Out     : $OUT_DIR"
echo "========================================"

source /home/gupt1042/.conda/envs/2025.06-py313/etc/profile.d/conda.sh
conda activate 2025.06-py313

cd $REPO_GNN

PYTHONPATH=$REPO_GNN_TEMP:$REPO_GNN python finetune_e2e_composition.py \
    --l1_ckpt   $L1_CKPT \
    --l3_ckpt   $L3_CKPT \
    --train_dirs $TRAIN_DIRS \
    --val_dir    $VAL_DIR \
    --out_dir    $OUT_DIR \
    --epochs     60 \
    --lr         5e-5 \
    --lr_backbone 0.0 \
    --scc_reg_weight 0.3 \
    --p_predicted 0.5 \
    --seed       42

echo "=== Training done. Running eval ==="

# Eval on composition_eval_v5 with dedup=fill
EVAL_OUT=$SCRATCH/runs/eval_e2e_v3_fill.csv

PYTHONPATH=$REPO_GNN_TEMP:$REPO_GNN python exhaust_gen/eval_composition2.py \
    --l1_ckpt  $OUT_DIR/l1_best.pt \
    --l3_ckpt  $OUT_DIR/l3_best.pt \
    --eval_dir $VAL_DIR \
    --out_csv  $EVAL_OUT \
    --dedup fill

echo "=== Results saved to $EVAL_OUT ==="

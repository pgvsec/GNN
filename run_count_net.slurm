#!/bin/bash
#SBATCH --job-name=grn_count_net
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/comp_fix/count_net_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/comp_fix/count_net_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg
#
# Step 2 of the composition count fix (after run_composition_fix.slurm):
#   1. post-processing ablation of _infer_scc on the existing models (pure ML, no fallback)
#   2. CountNet: Layer I copy trained end to end for the count on Corpus A + composition
#      branches (train_count_net.py)
#   3. held-out evaluation with CountNet
# Uses the composition_train_s100* sets written by run_composition_fix.slurm.

set -euo pipefail

RUNS=${RUNS:-/scratch/gilbreth/gupt1042/runs}
L1_CKPT=${L1_CKPT:-$RUNS/layer1_v10/best.pt}
L3_CKPT=${L3_CKPT:-$RUNS/layer3_v12/best.pt}
CORPUS_A_CACHE=${CORPUS_A_CACHE:-/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache}
TEST_DIR=${TEST_DIR:-$RUNS/composition_eval_v5}
TRAIN_DIRS=${TRAIN_DIRS:-"$RUNS/composition_eval $RUNS/composition_eval_v3 $RUNS/composition_eval_v4 \
$RUNS/composition_train_s1001 $RUNS/composition_train_s1002 $RUNS/composition_train_s1003 $RUNS/composition_train_s1004"}
OUT=${OUT:-$RUNS/comp_fix}
EPOCHS=${EPOCHS:-20}

# conda's activate/deactivate hooks read unset variables, so relax -u around them.
set +u
module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT"
export OMP_NUM_THREADS=8

L1F="forced_model=l1,solver_fallback=0"
HEAD="rescue=0,dedup=none,parity_drop=0"

# 1. Post-processing ablation (existing models, L1 on every SCC, no solver fallback)
#    legacy : rescue + linear dedup + parity drop (current behaviour)
#    log    : dedup in log space instead of 2% of xmax
#    norescue / noparity : one step removed at a time
#    head   : count exactly as the count head says
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --eval_dir "$TEST_DIR" --out_dir "$OUT/ablate_post" \
    --variants legacy:$L1F log:$L1F,dedup=log norescue:$L1F,rescue=0 \
               noparity:$L1F,parity_drop=0 head:$L1F,$HEAD

# 2. CountNet
python train_count_net.py --l1_ckpt "$L1_CKPT" --corpus_a_cache "$CORPUS_A_CACHE" \
    --train_dirs $TRAIN_DIRS --val_frac 0.15 --test_dir "$TEST_DIR" \
    --out_dir "$OUT/count_net_v1" --epochs "$EPOCHS"

# 3. Held-out evaluation with CountNet
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --count_net "$OUT/count_net_v1/best.pt" --eval_dir "$TEST_DIR" --out_dir "$OUT/final_count_net" \
    --variants cn_legacy:$L1F,count_net=1 cn_log:$L1F,dedup=log,count_net=1 \
               cn_head:$L1F,$HEAD,count_net=1

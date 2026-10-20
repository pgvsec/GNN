#!/bin/bash
#SBATCH --job-name=grn_verify
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/comp_fix/verify_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/comp_fix/verify_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg
#
# Leak-free verification of the CountNet result.
# CountNet v1 scored 96% teacher-forced on composition_eval_v5 but 78.7% on its own validation
# split, which points to v5 cases inside its training dirs. This job:
#   1. measures the overlap of v5 with every dir CountNet v1 trained on
#   2. builds a strict test set from Corpus A *test-split* topologies only (no model saw them)
#   3. builds training sets from *train-split* topologies only
#   4. retrains CountNet (v2) on those sets alone
#   5. evaluates original L1 and CountNet v2 on the strict set and on v5

set -euo pipefail

RUNS=${RUNS:-/scratch/gilbreth/gupt1042/runs}
L1_CKPT=${L1_CKPT:-$RUNS/layer1_v10/best.pt}
L3_CKPT=${L3_CKPT:-$RUNS/layer3_v12/best.pt}
CORPUS_A_CACHE=${CORPUS_A_CACHE:-/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache}
V5=${V5:-$RUNS/composition_eval_v5}
STRICT=${STRICT:-$RUNS/composition_test_strict}
OUT=${OUT:-$RUNS/comp_fix}
EPOCHS=${EPOCHS:-20}

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

# 1. Overlap of v5 with the CountNet v1 training dirs
python check_overlap.py --test "$V5" --train \
    $RUNS/composition_eval $RUNS/composition_eval_v3 $RUNS/composition_eval_v4 \
    $RUNS/composition_train_s1001 $RUNS/composition_train_s1002 \
    $RUNS/composition_train_s1003 $RUNS/composition_train_s1004

# 2. Strict test set: unseen topologies, new seed
if [ ! -f "$STRICT/n_fp.npy" ]; then
    PYTHONPATH=. python exhaust_gen/gen_composition_eval.py --corpus_cache "$CORPUS_A_CACHE" \
        --out_dir "$STRICT" --n_cases 1000 --n_cross 2 --seed 9001 --split test
fi

# 3. Training sets from train-split topologies only
TRAIN=""
for i in 1 2 3 4; do
    d="$RUNS/composition_train_strict_s$((2000 + i))"
    if [ ! -f "$d/n_fp.npy" ]; then
        PYTHONPATH=. python exhaust_gen/gen_composition_eval.py --corpus_cache "$CORPUS_A_CACHE" \
            --out_dir "$d" --n_cases 2000 --n_cross 2 --seed $((2000 + i)) --split train
    fi
    TRAIN="$TRAIN $d"
done
python check_overlap.py --test "$STRICT" --train $TRAIN
python check_overlap.py --test "$V5" --train $TRAIN

# 4. CountNet v2 on the strict training sets only
python train_count_net.py --l1_ckpt "$L1_CKPT" --corpus_a_cache "$CORPUS_A_CACHE" \
    --train_dirs $TRAIN --val_frac 0.15 --test_dir "$STRICT" \
    --out_dir "$OUT/count_net_v2" --epochs "$EPOCHS"

# 5. Evaluation (pure ML: no solver fallback)
L1F="forced_model=l1,solver_fallback=0"
HEAD="rescue=0,dedup=none,parity_drop=0"
for name in strict:$STRICT v5:$V5; do
    tag=${name%%:*}
    dir=${name#*:}
    python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
        --count_net "$OUT/count_net_v2/best.pt" --eval_dir "$dir" --out_dir "$OUT/verify_$tag" \
        --variants legacy:$L1F log:$L1F,dedup=log head:$L1F,$HEAD \
                   cn_log:$L1F,dedup=log,count_net=1 cn_head:$L1F,$HEAD,count_net=1
done

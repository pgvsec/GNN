#!/bin/bash
#SBATCH --job-name=grn_l1_forced
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/comp_fix/l1_forced_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/comp_fix/l1_forced_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg
#
# Pure-ML improvement of the per-SCC predictor (the learned part of the decomposition):
#   0. diagnostic: branch accuracy vs how far P*u lies outside the Corpus A range
#   1. forced-SCC samples (Corpus A train-split SCCs under composition-like input), full labels
#   2. fine-tune Layer I on Corpus A + forced samples with its full loss; select on a tuning set
#   3. network-only evaluation on the strict test set (count from the count head, no rescue),
#      original vs fine-tuned Layer I; the hybrid log row is kept for reference only

set -euo pipefail

RUNS=${RUNS:-/scratch/gilbreth/gupt1042/runs}
L1_CKPT=${L1_CKPT:-$RUNS/layer1_v10/best.pt}
L3_CKPT=${L3_CKPT:-$RUNS/layer3_v12/best.pt}
CORPUS_A_CACHE=${CORPUS_A_CACHE:-/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache}
TUNE=${TUNE:-$RUNS/composition_train_strict_s2001}
STRICT=${STRICT:-$RUNS/composition_test_strict}
FORCED_DIR=${FORCED_DIR:-/scratch/gilbreth/gupt1042/forced_scc}
N_FORCED=${N_FORCED:-150000}          # draws per file, 2 files
OUT=${OUT:-$RUNS/comp_fix}
NEW_L1=$RUNS/layer1_forced_v1

set +u
module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT" "$FORCED_DIR"
export OMP_NUM_THREADS=8

# 0. Diagnostic on the original Layer I
python diag_forcing_range.py --l1_ckpt "$L1_CKPT" --eval_dir "$STRICT"

# 1. Forced-SCC samples
for s in 1 2; do
    f="$FORCED_DIR/forced_s$s.npz"
    [ -f "$f" ] || python gen_forced_scc.py --corpus_a_cache "$CORPUS_A_CACHE" --out "$f" \
        --n_samples "$N_FORCED" --seed "$s"
done

# 2. Fine-tune Layer I (full loss)
python finetune_l1_forced.py --l1_ckpt "$L1_CKPT" --corpus_a_cache "$CORPUS_A_CACHE" \
    --forced "$FORCED_DIR/forced_s1.npz" "$FORCED_DIR/forced_s2.npz" \
    --tune_dir "$TUNE" --out_dir "$NEW_L1"

# 3. Evaluation on the strict test set
python diag_forcing_range.py --l1_ckpt "$L1_CKPT" "$NEW_L1/best.pt" --eval_dir "$STRICT"
L1F="forced_model=l1,solver_fallback=0"
PURE="$L1F,rescue=0,dedup=none,parity_drop=0"
for tag in orig:$L1_CKPT forced:$NEW_L1/best.pt; do
    name=${tag%%:*}
    ckpt=${tag#*:}
    python composition_eval.py --l1_ckpt "$ckpt" --l3_ckpt "$L3_CKPT" \
        --eval_dir "$STRICT" --out_dir "$OUT/l1_forced_eval_$name" \
        --variants pure_nopolish:$PURE,refine=0 pure:$PURE pure_fill:$L1F,dedup=fill \
                   hybrid_log:$L1F,dedup=log
done

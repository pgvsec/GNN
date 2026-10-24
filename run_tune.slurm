#!/bin/bash
#SBATCH --job-name=grn_tune
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/comp_fix/tune_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/comp_fix/tune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg
#
# Tune the slot refinement of the log-dedup pipeline (Newton steps per slot x dedup tolerance)
# on a tuning set that no model trained on, select by the worst of the three metrics, and
# score only the selected setting (plus the untuned baseline) on the strict test set.

set -euo pipefail

RUNS=${RUNS:-/scratch/gilbreth/gupt1042/runs}
L1_CKPT=${L1_CKPT:-$RUNS/layer1_v10/best.pt}
L3_CKPT=${L3_CKPT:-$RUNS/layer3_v12/best.pt}
TUNE=${TUNE:-$RUNS/composition_train_strict_s2001}
STRICT=${STRICT:-$RUNS/composition_test_strict}
OUT=${OUT:-$RUNS/comp_fix}

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

BASE="forced_model=l1,solver_fallback=0,dedup=log"
VARIANTS=""
for r in 5 20 50; do
    for t in 0.01 0.05 0.2; do
        VARIANTS="$VARIANTS r${r}_t${t}:$BASE,refine=$r,dedup_tol=$t"
    done
done

# 1. Grid on the tuning set (first 1000 cases)
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --eval_dir "$TUNE" --max_cases 1000 --out_dir "$OUT/tune" --variants $VARIANTS

# 2. Select by the worst of count_exact / fp_hd / stable_acc
BEST=$(python - "$OUT/tune/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
score = {k: min(v["all"]["count_exact"], v["all"]["fp_hd"], v["all"]["stable_acc"]) for k, v in s.items()}
for k in sorted(score, key=score.get, reverse=True):
    a = s[k]["all"]
    print(f"{k:>14}  min {score[k]:.4f}  CE {a['count_exact']:.4f}  hd {a['fp_hd']:.4f}  st {a['stable_acc']:.4f}", file=sys.stderr)
print(max(score, key=score.get))
EOF
)
echo "selected: $BEST"
r=${BEST#r}; r=${r%%_*}
t=${BEST#*_t}

# 3. Strict test: untuned log baseline and the selected setting
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --eval_dir "$STRICT" --out_dir "$OUT/tuned_strict" \
    --variants log_base:$BASE "tuned_${BEST}:$BASE,refine=$r,dedup_tol=$t"

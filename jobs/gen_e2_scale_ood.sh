#!/bin/bash
#SBATCH --job-name=gen_e2_scale_ood
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/generalization/e2_scale_ood_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/generalization/e2_scale_ood_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# E2: Out-of-distribution scale extrapolation (n_total > 6)
#
# Training max is n=6.  These compositions produce n=7 or n=8 gene systems,
# never seen during ANY stage of training (L1, L3, or composition fine-tuning).
#
# Scientific question: Does the learned compositional operator generalize beyond
# the training support?  The GRN dynamics become increasingly nonlinear at larger
# n, so any drop in accuracy here is informative.
#
# n=7 variants:
#   3+4 — canonical 3-gene root, 4-gene target
#   4+3 — 4-gene root, canonical 3-gene target
#
# n=8 variants:
#   4+4 — equal-size large composition
#   3+5 — small root, max-size target
#   5+3 — max-size root, small target

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
OUT_BASE=$SCRATCH/runs/generalization
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_BASE

GEN="$REPO_GNN/exhaust_gen/gen_composition_eval_general.py"

for NA_NB in "3 4" "4 3" "4 4" "3 5" "5 3"; do
    read NA NB <<< "$NA_NB"
    TAG="e2_${NA}p${NB}"
    OUT_DIR="$OUT_BASE/$TAG"
    echo "=== Generating $TAG (n_a=$NA, n_b=$NB, total=$((NA+NB))) → $OUT_DIR ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
        --corpus_cache $CORPUS_CACHE \
        --out_dir      $OUT_DIR \
        --n_a  $NA \
        --n_b  $NB \
        --n_cases    500 \
        --n_cross_ab   2 \
        --seed        42
done

echo "=== E2 generation complete ==="
ls -lh $OUT_BASE/e2_*/adj.npy 2>/dev/null

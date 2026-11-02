#!/bin/bash
#SBATCH --job-name=gen_e1_size_var
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/generalization/e1_size_var_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/generalization/e1_size_var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# E1: In-distribution size variation (n_total=6, split != canonical 3+3)
#
# Hypothesis: The model learned a compositional operator, not a size-specific
# memorization.  If it generalizes within n=6, these splits (each never seen
# in the composition eval training set) should achieve ~90% accuracy.
#
# Variants:
#   2+4 — tiny root SCC forced into large target
#   4+2 — large root SCC forced into tiny target
#
# Note: 1+5 and 5+1 are excluded — Corpus A covers n=2..6 only; there are
# no n=1 single-gene SCC samples to draw from.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
OUT_BASE=$SCRATCH/runs/generalization
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_BASE

GEN="$REPO_GNN/exhaust_gen/gen_composition_eval_general.py"

for NA_NB in "2 4" "4 2"; do
    read NA NB <<< "$NA_NB"
    TAG="e1_${NA}p${NB}"
    OUT_DIR="$OUT_BASE/$TAG"
    echo "=== Generating $TAG (n_a=$NA, n_b=$NB) → $OUT_DIR ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
        --corpus_cache $CORPUS_CACHE \
        --out_dir      $OUT_DIR \
        --n_a  $NA \
        --n_b  $NB \
        --n_cases    500 \
        --n_cross_ab   2 \
        --seed        42
done

echo "=== E1 generation complete ==="
ls -lh $OUT_BASE/e1_*/adj.npy 2>/dev/null

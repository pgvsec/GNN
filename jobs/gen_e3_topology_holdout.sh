#!/bin/bash
#SBATCH --job-name=gen_e3_topo_holdout
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/generalization/e3_topo_holdout_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/generalization/e3_topo_holdout_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# E3: Topology holdout — same size (3+3=6 genes), OOD graph topologies
#
# Corpus A is split into train/val/test by graph TOPOLOGY (adjacency pattern),
# not by kinetic parameters.  The L1 and L3 models were trained only on
# train+val topologies; the test topologies were never seen during training.
#
# This experiment composes two SCCs drawn ONLY from test topologies (--split test).
# The combined system has:
#   (a) the same n=6 gene count as the canonical eval
#   (b) SCC subgraphs with entirely novel edge patterns
#
# If the model generalizes to OOD topologies, it has learned dynamics rules,
# not topology fingerprints.
#
# Comparison baseline: re-generate the canonical 3+3 with --split all (no
# topology restriction) to confirm the generator reproduces the v5 eval quality.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
OUT_BASE=$SCRATCH/runs/generalization
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_BASE

GEN="$REPO_GNN/exhaust_gen/gen_composition_eval_general.py"

# Test-topology holdout
echo "=== Generating e3_3p3_test_split → $OUT_BASE/e3_3p3_test_split ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $OUT_BASE/e3_3p3_test_split \
    --n_a  3 --n_b 3 \
    --split test \
    --n_cases    500 \
    --n_cross_ab   2 \
    --seed        42

# Sanity-check baseline: canonical size, no split restriction
echo "=== Generating e3_3p3_all_split (sanity check) → $OUT_BASE/e3_3p3_all_split ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $OUT_BASE/e3_3p3_all_split \
    --n_a  3 --n_b 3 \
    --split all \
    --n_cases    500 \
    --n_cross_ab   2 \
    --seed        42

echo "=== E3 generation complete ==="
ls -lh $OUT_BASE/e3_*/adj.npy 2>/dev/null

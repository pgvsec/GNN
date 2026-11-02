#!/bin/bash
#SBATCH --job-name=gen_e4_deep_dag
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/generalization/e4_deep_dag_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/generalization/e4_deep_dag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# E4: Deep DAG — three-SCC linear chains  A → B → C
#
# DAG depth = 2 (two levels of forcing) vs. depth = 1 in all previous evals.
#
# assemble.py already handles multi-SCC chains natively: run_inference() walks
# the condensation DAG in topological order and calls L3 on each forced SCC
# given all of its parents' FP combinations.  No architecture changes needed.
#
# Chain topologies:
#   3→3→3 : 9-gene system, symmetric chain
#   2→3→4 : 9-gene system, asymmetric (growing) chain
#
# This tests whether a single forward pass through depth-1 training generalises
# to depth-2 forced chains.  The composition operator must chain correctly:
#   root A → forced B | A → forced C | B
# Each conditioning step sees a fresh FP combination as its upstream signal.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
CORPUS_CACHE=$SCRATCH/corpus_exhaust_A/cache
OUT_BASE=$SCRATCH/runs/generalization
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_BASE

GEN="$REPO_GNN/exhaust_gen/gen_composition_eval_general.py"

# Symmetric 3→3→3 chain
echo "=== Generating e4_3ch3ch3 (3→3→3, n=9) → $OUT_BASE/e4_3ch3ch3 ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $OUT_BASE/e4_3ch3ch3 \
    --n_a  3 --n_b 3 --n_c 3 \
    --n_cases    500 \
    --n_cross_ab   2 \
    --n_cross_bc   2 \
    --seed        42

# Asymmetric 2→3→4 chain
echo "=== Generating e4_2ch3ch4 (2→3→4, n=9) → $OUT_BASE/e4_2ch3ch4 ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $GEN \
    --corpus_cache $CORPUS_CACHE \
    --out_dir      $OUT_BASE/e4_2ch3ch4 \
    --n_a  2 --n_b 3 --n_c 4 \
    --n_cases    500 \
    --n_cross_ab   2 \
    --n_cross_bc   2 \
    --seed        42

echo "=== E4 generation complete ==="
ls -lh $OUT_BASE/e4_*/adj.npy 2>/dev/null

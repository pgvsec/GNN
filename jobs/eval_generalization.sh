#!/bin/bash
#SBATCH --job-name=eval_generalization
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/generalization/eval_all_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/generalization/eval_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Evaluate v8 checkpoints on all generalization experiment datasets.
#
# Prerequisite: run gen_e1 / gen_e2 / gen_e3 / gen_e4 jobs first to
# populate $OUT_BASE/e{1,2,3,4}_* directories.
#
# Outputs one CSV per dataset + a summary table printed to stdout.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
L1_CKPT=$SCRATCH/runs/e2e_v8/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v8/l3_best.pt
OUT_BASE=$SCRATCH/runs/generalization
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

EVAL="$REPO_GNN/exhaust_gen/eval_composition2.py"
COMMON_FLAGS="--l3_frame forced --dedup fill --refine_steps 15"

# ── Canonical baseline (v5 eval set, reproduced for comparison) ──────────────
echo "=== Canonical 3+3 v5 eval (baseline) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL \
    --l1_ckpt  $L1_CKPT \
    --l3_ckpt  $L3_CKPT \
    --eval_dir $SCRATCH/runs/composition_eval_v5 \
    --out_csv  $OUT_BASE/results_v5_repro.csv \
    $COMMON_FLAGS
echo ""

# ── E1: In-distribution size variation ───────────────────────────────────────
for TAG in e1_2p4 e1_4p2 e1_1p5 e1_5p1; do
    DIR="$OUT_BASE/$TAG"
    if [ ! -d "$DIR" ]; then
        echo "SKIP $TAG — directory not found (run gen_e1_size_variation.sh first)"
        continue
    fi
    echo "=== $TAG ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL \
        --l1_ckpt  $L1_CKPT \
        --l3_ckpt  $L3_CKPT \
        --eval_dir $DIR \
        --out_csv  $OUT_BASE/results_${TAG}.csv \
        $COMMON_FLAGS
    echo ""
done

# ── E2: OOD scale extrapolation ───────────────────────────────────────────────
for TAG in e2_3p4 e2_4p3 e2_4p4 e2_3p5 e2_5p3; do
    DIR="$OUT_BASE/$TAG"
    if [ ! -d "$DIR" ]; then
        echo "SKIP $TAG — directory not found (run gen_e2_scale_ood.sh first)"
        continue
    fi
    echo "=== $TAG ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL \
        --l1_ckpt  $L1_CKPT \
        --l3_ckpt  $L3_CKPT \
        --eval_dir $DIR \
        --out_csv  $OUT_BASE/results_${TAG}.csv \
        $COMMON_FLAGS
    echo ""
done

# ── E3: Topology holdout ─────────────────────────────────────────────────────
for TAG in e3_3p3_test_split e3_3p3_all_split; do
    DIR="$OUT_BASE/$TAG"
    if [ ! -d "$DIR" ]; then
        echo "SKIP $TAG — directory not found (run gen_e3_topology_holdout.sh first)"
        continue
    fi
    echo "=== $TAG ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL \
        --l1_ckpt  $L1_CKPT \
        --l3_ckpt  $L3_CKPT \
        --eval_dir $DIR \
        --out_csv  $OUT_BASE/results_${TAG}.csv \
        $COMMON_FLAGS
    echo ""
done

# ── E4: Deep DAG (3-SCC chains) ──────────────────────────────────────────────
for TAG in e4_3ch3ch3 e4_2ch3ch4; do
    DIR="$OUT_BASE/$TAG"
    if [ ! -d "$DIR" ]; then
        echo "SKIP $TAG — directory not found (run gen_e4_deep_dag.sh first)"
        continue
    fi
    echo "=== $TAG ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON $EVAL \
        --l1_ckpt  $L1_CKPT \
        --l3_ckpt  $L3_CKPT \
        --eval_dir $DIR \
        --out_csv  $OUT_BASE/results_${TAG}.csv \
        $COMMON_FLAGS
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "==================================================================="
echo " SUMMARY — count_exact / fp_hd / stable_acc"
printf "  %-26s  %s  %s  %s\n" "dataset" "count_exact" "fp_hd" "stable_acc"
echo "==================================================================="
PYTHON_AGGS=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
for CSV in $(ls $OUT_BASE/results_*.csv 2>/dev/null | sort); do
    TAG=$(basename $CSV .csv | sed 's/results_//')
    RESULT=$($PYTHON_AGGS -c "
import csv, sys
try:
    with open('$CSV') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print('(empty)')
        sys.exit(0)
    n = len(rows)
    # try both possible column names
    def get(r, *keys):
        for k in keys:
            if k in r: return float(r[k])
        return 0.0
    ce = sum(get(r, 'count_correct', 'count_exact') for r in rows) / n
    fh = sum(get(r, 'fp_hd') for r in rows) / n
    sa = sum(get(r, 'stable_acc') for r in rows) / n
    print(f'{ce:.4f}  {fh:.4f}  {sa:.4f}  (n={n})')
except Exception as e:
    print(f'error: {e}')
" 2>&1)
    printf "  %-26s  %s\n" "$TAG" "$RESULT"
done
echo "==================================================================="

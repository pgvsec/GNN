#!/bin/bash
#SBATCH --job-name=eval_gen_v10
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v10/eval_gen_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v10/eval_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# ── Full generalization table: v10 checkpoints across all composition types ──
#
# Reproduces the v8 baseline table with v10 l1_best.pt / l3_best.pt.
# Each row corresponds to one dataset:
#
#   Dataset                  Path
#   ─────────────────────────────────────────────────────────────────────────
#   v5 3+3 (biased)         composition_eval_v5           (checkpoint-select set)
#   3+3 honest (all-split)  v10_hardneg/fresh_3p3_5k      (fresh draw, seed≠42)
#   2+4  asymm              generalization/e1_2p4         (never trained, 2-SCC)
#   4+2  asymm              generalization/e1_4p2         (never trained, 2-SCC)
#   3+4  zero-shot +1       generalization/e1_3p4         (3-gene root, 4-gene forced)
#   4+3  zero-shot +1       generalization/e1_4p3         (4-gene root, 3-gene forced)
#   3+5  zero-shot +2       generalization/e1_3p5         (3-gene root, 5-gene forced)
#   5+3  zero-shot +2       generalization/e1_5p3         (5-gene root, 3-gene forced)
#   4+4  zero-shot +2       generalization/e1_4p4         (4-gene root, 4-gene forced)
#   2→3→4 chain             generalization/chain_2p3p4    (3-SCC chain)
#   3→3→3 chain             generalization/chain_3p3p3    (3-SCC chain)
#
# If a path doesn't exist on the cluster the eval step is skipped with a warning.
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

L1_CKPT=$SCRATCH/runs/e2e_v10/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v10/l3_best.pt
OUT_DIR=$SCRATCH/runs/e2e_v10/eval_gen

mkdir -p $OUT_DIR
cd $REPO_GNN

# Helper: run eval if the directory exists, otherwise print a skip message.
run_eval() {
    local label="$1"
    local eval_dir="$2"
    local out_csv="$3"
    if [ ! -d "$eval_dir" ]; then
        echo "SKIP [$label]: $eval_dir not found"
        return
    fi
    echo "=== $label ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
        --l1_ckpt $L1_CKPT \
        --l3_ckpt $L3_CKPT \
        --eval_dir $eval_dir \
        --out_csv  $out_csv \
        --l3_frame forced --dedup fill --refine_steps 15
    echo ""
}

run_eval "v5 3+3 (biased)" \
    $SCRATCH/runs/composition_eval_v5 \
    $OUT_DIR/v5_canonical.csv

# run_eval "3+3 honest (5k fresh)" \
#     $SCRATCH/runs/v10_hardneg/fresh_3p3_5k \
#     $OUT_DIR/fresh_3p3_5k.csv

run_eval "2+4 asymm" \
    $SCRATCH/runs/generalization/e1_2p4 \
    $OUT_DIR/e1_2p4.csv

run_eval "4+2 asymm" \
    $SCRATCH/runs/generalization/e1_4p2 \
    $OUT_DIR/e1_4p2.csv

run_eval "3+4 zero-shot +1" \
    $SCRATCH/runs/generalization/e2_3p4 \
    $OUT_DIR/e2_3p4.csv

run_eval "4+3 zero-shot +1" \
    $SCRATCH/runs/generalization/e2_4p3 \
    $OUT_DIR/e2_4p3.csv

run_eval "3+5 zero-shot +2" \
    $SCRATCH/runs/generalization/e2_3p5 \
    $OUT_DIR/e2_3p5.csv

run_eval "5+3 zero-shot +2" \
    $SCRATCH/runs/generalization/e2_5p3 \
    $OUT_DIR/e2_5p3.csv

run_eval "4+4 zero-shot +2" \
    $SCRATCH/runs/generalization/e2_4p4 \
    $OUT_DIR/e2_4p4.csv

run_eval "2→3→4 chain (3-SCC)" \
    $SCRATCH/runs/generalization/e4_2ch3ch4 \
    $OUT_DIR/e4_2ch3ch4.csv

run_eval "3→3→3 chain (3-SCC)" \
    $SCRATCH/runs/generalization/e4_3ch3ch3 \
    $OUT_DIR/e4_3ch3ch3.csv

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "=== v10 Generalization Summary (refine_steps=15) ==="
echo ""

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON - <<PYEOF
import csv, os
from pathlib import Path

OUT_DIR = "$OUT_DIR"

rows = [
    ("v5 3+3 (biased)",       "v5_canonical.csv",   6, 2, "checkpoint-select set"),
    ("3+3 honest (5k fresh)", "fresh_3p3_5k.csv",   6, 2, "fresh draw seed≠42"),
    ("2+4 asymm",             "e1_2p4.csv",          6, 2, "never trained on"),
    ("4+2 asymm",             "e1_4p2.csv",          6, 2, "never trained on"),
    ("3+4 zero-shot +1",      "e2_3p4.csv",          7, 2, "zero-shot +1 gene"),
    ("4+3 zero-shot +1",      "e2_4p3.csv",          7, 2, "zero-shot +1 gene"),
    ("3+5 zero-shot +2",      "e2_3p5.csv",          8, 2, "zero-shot +2 genes"),
    ("5+3 zero-shot +2",      "e2_5p3.csv",          8, 2, "zero-shot +2 genes"),
    ("4+4 zero-shot +2",      "e2_4p4.csv",          8, 2, "zero-shot +2 genes"),
    ("2→3→4 chain",           "e4_2ch3ch4.csv",      9, 3, "zero-shot 3-SCC chain"),
    ("3→3→3 chain",           "e4_3ch3ch3.csv",      9, 3, "zero-shot 3-SCC chain"),
]

# v8 baseline for comparison
v8 = {
    "v5_canonical.csv":  (0.914, 0.884, 0.913),
    "fresh_3p3_5k.csv":  (0.772, 0.810, 0.822),
    "e1_2p4.csv":        (0.788, 0.778, 0.833),
    "e1_4p2.csv":        (0.748, 0.822, 0.805),
    "e2_3p4.csv":        (0.696, 0.737, 0.757),
    "e2_4p3.csv":        (0.686, 0.747, 0.758),
    "e2_3p5.csv":        (0.724, 0.722, 0.784),
    "e2_5p3.csv":        (0.754, 0.732, 0.801),
    "e2_4p4.csv":        (0.672, 0.656, 0.752),
    "e4_2ch3ch4.csv":    (0.726, 0.692, 0.794),
    "e4_3ch3ch3.csv":    (0.698, 0.736, 0.770),
}

hdr = f"{'Dataset':<26} {'n_g':>4} {'SCCs':>5}  {'count_exact':>11}  {'fp_hd':>7}  {'stable_acc':>10}  {'Notes'}"
print(hdr)
print("-" * len(hdr))

for label, fname, ng, nsccs, notes in rows:
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        b = v8.get(fname, (None, None, None))
        bstr = f"  [v8: {b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f}]" if b[0] else ""
        print(f"  {label:<24} {ng:>4} {nsccs:>5}   SKIPPED{bstr}")
        continue
    data = list(csv.DictReader(open(path)))
    if not data:
        print(f"  {label:<24} {ng:>4} {nsccs:>5}   EMPTY")
        continue
    n = len(data)
    ce = sum(float(r["count_exact"]) for r in data) / n
    hd_vals = [float(r["fp_hd"]) for r in data if r.get("fp_hd","") not in ("", "nan")]
    sa_vals = [float(r["stable_acc"]) for r in data if r.get("stable_acc","") not in ("", "nan")]
    hd = sum(hd_vals) / len(hd_vals) if hd_vals else float("nan")
    sa = sum(sa_vals) / len(sa_vals) if sa_vals else float("nan")
    b  = v8.get(fname, (None, None, None))
    if b[0] is not None:
        delta = f"  Δ={ce-b[0]:+.3f}/{hd-b[1]:+.3f}/{sa-b[2]:+.3f}"
    else:
        delta = ""
    print(f"  {label:<24} {ng:>4} {nsccs:>5}   {ce:.3f}  {hd:.3f}  {sa:.3f}  {delta}")

print()
print("Format: count_exact / fp_hd / stable_acc  (refine_steps=15)")
print("Δ = v10 − v8 baseline")
PYEOF

echo ""
echo "CSVs in: $OUT_DIR"

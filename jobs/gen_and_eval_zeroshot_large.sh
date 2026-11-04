#!/bin/bash
#SBATCH --job-name=gen_zs_large
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v10/gen_zs_large_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v10/gen_zs_large_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Generate and evaluate larger zero-shot compositions: 4+5, 5+4, 5+5, 5+6, 6+6
#
# IMPORTANT: The corpus cache must contain graphs of the requested sizes.
# If a size is missing (e.g., no 6-gene graphs in corpus A), the script
# will print an error and skip that dataset — not fail the whole job.
# ─────────────────────────────────────────────────────────────────────────────

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

CORPUS=$SCRATCH/corpus_exhaust_A/cache
GEN_DIR=$SCRATCH/runs/generalization
L1_CKPT=$SCRATCH/runs/e2e_v10/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v10/l3_best.pt
OUT_DIR=$SCRATCH/runs/e2e_v10/eval_gen

mkdir -p $GEN_DIR $OUT_DIR
cd $REPO_GNN

# ── Generate ──────────────────────────────────────────────────────────────────
gen_dataset() {
    local label="$1"
    local na="$2"
    local nb="$3"
    local tag="$4"
    local out=$GEN_DIR/$tag
    if [ -d "$out" ] && [ -f "$out/adj.npy" ] && [ "$(python3 -c "import numpy as np; print(len(np.load('$out/adj.npy')))" 2>/dev/null)" != "0" ]; then
        echo "SKIP generate [$label]: $out already exists and is non-empty"
        return
    fi
    # remove any empty/failed previous attempt
    rm -rf "$out"
    echo "=== Generating $label ($tag) ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/gen_composition_eval_N.py \
        --corpus_cache $CORPUS \
        --out_dir      $out \
        --na $na --nb $nb \
        --n_cases 500 \
        --n_cross 2 \
        --seed 42
    echo ""
}

gen_dataset "4+5" 4 5 "e3_4p5"
gen_dataset "5+4" 5 4 "e3_5p4"
gen_dataset "5+5" 5 5 "e3_5p5"
gen_dataset "5+6" 5 6 "e3_5p6"
gen_dataset "6+6" 6 6 "e3_6p6"

# ── Evaluate ──────────────────────────────────────────────────────────────────
eval_dataset() {
    local label="$1"
    local tag="$2"
    local eval_dir=$GEN_DIR/$tag
    local out_csv=$OUT_DIR/${tag}.csv
    if [ ! -d "$eval_dir" ] || [ ! -f "$eval_dir/adj.npy" ]; then
        echo "SKIP eval [$label]: $eval_dir missing (generation likely failed)"
        return
    fi
    echo "=== Evaluating $label ==="
    PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
        --l1_ckpt $L1_CKPT \
        --l3_ckpt $L3_CKPT \
        --eval_dir $eval_dir \
        --out_csv  $out_csv \
        --l3_frame forced --dedup fill --refine_steps 15
    echo ""
}

eval_dataset "4+5" "e3_4p5"
eval_dataset "5+4" "e3_5p4"
eval_dataset "5+5" "e3_5p5"
eval_dataset "5+6" "e3_5p6"
eval_dataset "6+6" "e3_6p6"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Large Zero-Shot Results (v10, refine_steps=15) ==="
echo ""

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON - <<PYEOF
import csv, os

OUT_DIR = "$OUT_DIR"

rows = [
    ("4+5 zero-shot +3",  "e3_4p5.csv",   9, 2),
    ("5+4 zero-shot +3",  "e3_5p4.csv",   9, 2),
    ("5+5 zero-shot +4",  "e3_5p5.csv",  10, 2),
    ("5+6 zero-shot +5",  "e3_5p6.csv",  11, 2),
    ("6+6 zero-shot +6",  "e3_6p6.csv",  12, 2),
]

# Reference: v10 on trained/seen sizes for comparison
ref = {
    "3+3 honest":  (0.808, 0.929, 0.844),
    "4+4":         (0.674, 0.847, 0.765),
}

print(f"  {'Dataset':<22} {'n_g':>4} {'SCCs':>5}  {'count_exact':>11}  {'fp_hd':>7}  {'stable_acc':>10}")
print("  " + "-"*68)
for label, fname, ng, nsccs in rows:
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"  {label:<22} {ng:>4} {nsccs:>5}   MISSING")
        continue
    data = list(csv.DictReader(open(path)))
    if not data:
        print(f"  {label:<22} {ng:>4} {nsccs:>5}   EMPTY")
        continue
    n = len(data)
    ce = sum(float(r["count_exact"]) for r in data) / n
    hd_vals = [float(r["fp_hd"]) for r in data if r.get("fp_hd","") not in ("","nan")]
    sa_vals = [float(r["stable_acc"]) for r in data if r.get("stable_acc","") not in ("","nan")]
    hd = sum(hd_vals)/len(hd_vals) if hd_vals else float("nan")
    sa = sum(sa_vals)/len(sa_vals) if sa_vals else float("nan")
    print(f"  {label:<22} {ng:>4} {nsccs:>5}   {ce:.3f}  {hd:.3f}  {sa:.3f}")

print()
print("  Reference (v10):")
for k, (ce, hd, sa) in ref.items():
    print(f"    {k}: count_exact={ce:.3f}  fp_hd={hd:.3f}  stable_acc={sa:.3f}")
print()
print("  Format: count_exact / fp_hd / stable_acc  (refine_steps=15)")
PYEOF

echo ""
echo "CSVs in: $OUT_DIR"

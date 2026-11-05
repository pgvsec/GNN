#!/bin/bash
#SBATCH --job-name=gap_gene_grn
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/real_grn/gap_gene_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/real_grn/gap_gene_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# ─────────────────────────────────────────────────────────────────────────────
#  Real GRN experiments: Drosophila gap gene network
#
#  Two topologies tested:
#    Model A — Reinitz/Manu 2009: single 4-gene SCC {Hb, Kr, Gt, Kni}
#              GNN runs L1 on the full 4-gene SCC (single-SCC prediction)
#    Model B — Papatsenko-Levine 2011: two 2-gene SCCs {Hb,Kni}→{Kr,Gt}
#              GNN runs the full composition pipeline (L1 on SCC_A, L3 on SCC_B)
#
#  Three AP positions tested (37% EL / 45% EL / 55% EL) → 3 cases per model.
#
#  Pipeline:
#    1. gen_gap_gene_dataset.py  — solve Hill-function ODE → ground truth
#       (ground truth is verified against literature: Manu 2009 / Papatsenko 2011)
#    2. eval_gap_gene.py         — run GNN prediction → compare to ground truth
#
#  Output:
#    $DATA_DIR/gap_gene_model_a/   ground-truth dataset + summary JSON  (Model A)
#    $DATA_DIR/gap_gene_model_b/   ground-truth dataset + summary JSON  (Model B)
#    $OUT_DIR/gap_gene_results.json  per-case prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

L1_CKPT=$SCRATCH/runs/e2e_v10/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v10/l3_best.pt
DATA_DIR=$SCRATCH/runs/real_grn
OUT_DIR=$SCRATCH/runs/real_grn

mkdir -p "$DATA_DIR"

export PYTHONPATH=$REPO_GNN:$SCRATCH/GNN_temp
cd $REPO_GNN

# ── Step 1: Generate ground-truth datasets ───────────────────────────────────
echo ""
echo "=== Step 1: Generating ground-truth fixed-point datasets ==="
echo ""
echo "  This runs the Hill-function ODE solver on both gap gene topologies."
echo "  Output: 3 cases per model (AP positions: 37%, 45%, 55% EL)."
echo "  Brouwer certification confirms completeness of the fixed-point sets."
echo ""

$PYTHON real_grn/gen_gap_gene_dataset.py \
    --model both \
    --out_base "$DATA_DIR"

# ── Step 2: Literature verification ──────────────────────────────────────────
echo ""
echo "=== Step 2: Ground-truth verification vs. literature ==="
echo ""
echo "  Expected (from Manu 2009 / Papatsenko-Levine 2011):"
echo "  Model A, 37% EL: 2–4 coexisting attractors (literature: 3 at Manu params)"
echo "  Model B, 37% EL: each switch bistable (up to 2×2 = 4 product attractors)"
echo "  → Check the counts above. Parameter deviations from Reinitz fitted circuits"
echo "    are expected (different ODE form), but qualitative structure should match."
echo ""

# Display ground-truth summaries
for model in a b; do
    summary="$DATA_DIR/gap_gene_model_${model}/ground_truth_summary.json"
    if [ -f "$summary" ]; then
        echo "--- Model $(echo $model | tr 'a-z' 'A-Z') ground truth ---"
        $PYTHON -c "
import json
d = json.load(open('$summary'))
print(f\"  Model: {d['model']}  |  Gene order: {d['gene_order']}\")
for case in d['cases']:
    n_s = case['n_stable']
    n_fp = case['n_fp']
    print(f\"  {case['ap_position']}: {n_fp} FPs  ({n_s} stable)\")
    for fp in case['fixed_points']:
        print(f\"    [{fp['stability']}]  x={fp['state']}\")
"
        echo ""
    fi
done

# ── Step 3: GNN prediction (pure ML, refine_steps=0) ─────────────────────────
echo ""
echo "=== Step 3: GNN prediction (pure ML, refine_steps=0) ==="
echo ""
echo "  count_exact is a PURE ML metric (argmax of count head)."
echo "  fp_hd at refine_steps=0 is also pure ML (no Newton refinement)."
echo ""

$PYTHON real_grn/eval_gap_gene.py \
    --l1_ckpt "$L1_CKPT" \
    --l3_ckpt "$L3_CKPT" \
    --data_base "$DATA_DIR" \
    --model both \
    --refine_steps 0 \
    --out_json "$OUT_DIR/gap_gene_results_refine0.json"

# ── Step 4: GNN prediction (with Newton refinement, refine_steps=15) ─────────
echo ""
echo "=== Step 4: GNN prediction (ML seeds + Newton refinement, refine_steps=15) ==="
echo ""
echo "  fp_hd at refine_steps=15: ML seeds + Newton refinement."
echo "  count_exact is still pure ML."
echo ""

$PYTHON real_grn/eval_gap_gene.py \
    --l1_ckpt "$L1_CKPT" \
    --l3_ckpt "$L3_CKPT" \
    --data_base "$DATA_DIR" \
    --model both \
    --refine_steps 15 \
    --out_json "$OUT_DIR/gap_gene_results_refine15.json"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo ""

OUT_DIR_PY="$OUT_DIR"
$PYTHON - <<PYEOF
import json, os, math

for tag, fname in [("Pure ML (refine=0)", "gap_gene_results_refine0.json"),
                   ("ML+Newton (refine=15)", "gap_gene_results_refine15.json")]:
    path = os.path.join("$OUT_DIR_PY", fname)
    if not os.path.exists(path):
        print(f"  {tag}: MISSING")
        continue
    rows = json.load(open(path))
    print(f"\n  {tag}:")
    print(f"  {'Model':<10} {'AP pos':<10} {'GT FPs':>6} {'Pred':>5} {'count_exact':>11} {'fp_hd':>7} {'stable_acc':>10}")
    print("  " + "-"*60)
    for r in rows:
        hd = f"{r['fp_hd']:.3f}" if not math.isnan(r['fp_hd']) else "  nan"
        sa = f"{r['stable_acc']:.3f}" if not math.isnan(r['stable_acc']) else "  nan"
        print(f"  Model {r['model']:<4} {r['ap_position']:<10} {r['n_fp_gt']:>6} {r['n_fp_pred']:>5} "
              f"{r['count_exact']:>11}   {hd:>7}  {sa:>10}")
PYEOF

echo ""
echo "Results JSON in: $OUT_DIR"
echo ""
echo "IMPORTANT notes:"
echo "  - Model A tests SINGLE-SCC prediction (L1 only, 4-gene SCC, never seen in training)"
echo "  - Model B tests COMPOSITION prediction (L1 on SCC_A=2 genes, L3 on SCC_B=2 genes)"
echo "  - Both use v10 checkpoints trained on 3+3 / 2+4 / 4+2 compositions"
echo "  - 2+2 is a subset of 2+4 training → Model B is in-distribution for gene count"
echo "  - 4-gene single SCC (Model A) was seen in Corpus A training for L1"

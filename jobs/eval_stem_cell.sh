#!/bin/bash
#SBATCH --job-name=stem_cell_grn
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/real_grn/stem_cell_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/real_grn/stem_cell_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# ─────────────────────────────────────────────────────────────────────────────
#  Real GRN experiments: Pluripotency → Hematopoiesis → Lymphocyte activation
#
#  Three-SCC depth-3 chain (10 genes):
#    SCC₁ {OCT4, SOX2, NANOG}     — pluripotency triad, mutual activation (3 genes)
#    SCC₂ {GATA1, GATA2, KLF1,    — hematopoietic TF network, GATA/PU.1 toggle (5 genes)
#          SPI1, TAL1}
#    SCC₃ {BATF, IRF4}             — lymphocyte activation loop (2 genes)
#
#  Inter-SCC edges:
#    SCC₁ → SCC₂: NANOG⊣GATA1, OCT4⊣GATA1  (pluripotency represses commitment)
#    SCC₂ → SCC₃: SPI1→IRF4                  (PU.1 activates lymphocyte programme)
#
#  Three signaling contexts: ESC / progenitor / myeloid
#
#  DAG depth = 3 — genuinely OOD (Corpus C trained on max depth 2: L1+single L3).
#  GNN routing: L1 → SCC₁, L3 → SCC₂ (forced by SCC₁), L3 → SCC₃ (forced by SCC₁+SCC₂).
#
#  Pipeline:
#    Step 1: gen_stem_cell_dataset.py — solve 10-gene ODE → ground truth
#    Step 2: display ground truth summary
#    Step 3: eval refine=0  (pure ML)
#    Step 4: eval refine=15 (ML + Newton)
#
#  Sources:
#    SCC₁: Chickarmane & Peterson 2008 (OCT4/SOX2/NANOG mutual activation)
#    SCC₂: Huang et al. 2007 (GATA1/PU.1 bistable toggle) + TRRUST v2 / DoRothEA A
#    SCC₃: Glasmacher et al. 2012 (BATF/IRF4 AICE loop, DoRothEA A)
#    SCC₁→SCC₂: Babaie 2007 (OCT4⊣GATA1), Chambers 2007 (NANOG⊣GATA1)
#    SCC₂→SCC₃: Glasmacher 2010 (SPI1→IRF4, DoRothEA A)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO=$SCRATCH/GNN
L1_CKPT=$SCRATCH/runs/e2e_v10/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v10/l3_best.pt
DATA_BASE=$SCRATCH/runs/real_grn
OUT_DIR=$DATA_BASE/stem_cell

export PYTHONPATH=$REPO:$SCRATCH/GNN_temp

echo "============================================================"
echo "  Stem cell differentiation network — real GRN experiment"
echo "  $(date)"
echo "  REPO:     $REPO"
echo "  L1 ckpt:  $L1_CKPT"
echo "  L3 ckpt:  $L3_CKPT"
echo "  Data:     $DATA_BASE"
echo "============================================================"

cd $REPO

# ── Step 1: Generate ground truth ───────────────────────────────────────────
echo ""
echo "── Step 1: Ground truth (Hill-function ODE solver) ──────────────────"
python real_grn/gen_stem_cell_dataset.py \
    --out_base "$DATA_BASE"
echo "  Ground truth saved → $OUT_DIR"

# ── Step 2: Display ground truth summary ────────────────────────────────────
echo ""
echo "── Step 2: Ground truth summary ─────────────────────────────────────"
python -c "
import json, sys
from pathlib import Path
data = json.load(open('$OUT_DIR/ground_truth_summary.json'))
print(f\"  Network: {data['network']}\")
print(f\"  Gene order: {data['gene_order']}\")
print(f\"  SCC1={data['scc1']}  SCC2={data['scc2']}  SCC3={data['scc3']}\")
print()
for case in data['cases']:
    ns = case['n_stable']
    print(f\"  {case['context']:12s} : {case['n_fp']} FPs, {ns} stable\")
    for fp in case['fixed_points']:
        s = fp['stability']
        on = fp['on_genes']
        print(f\"    [{s}]: ON={on}\")
"

# ── Step 3: GNN eval — pure ML (refine=0) ───────────────────────────────────
echo ""
echo "── Step 3: GNN predictions — pure ML (refine_steps=0) ───────────────"
python real_grn/eval_stem_cell.py \
    --l1_ckpt "$L1_CKPT" \
    --l3_ckpt "$L3_CKPT" \
    --data_base "$DATA_BASE" \
    --refine_steps 0 \
    --out_json "$OUT_DIR/results_refine0.json"

# ── Step 4: GNN eval — ML + Newton (refine=15) ──────────────────────────────
echo ""
echo "── Step 4: GNN predictions — ML + Newton (refine_steps=15) ──────────"
python real_grn/eval_stem_cell.py \
    --l1_ckpt "$L1_CKPT" \
    --l3_ckpt "$L3_CKPT" \
    --data_base "$DATA_BASE" \
    --refine_steps 15 \
    --out_json "$OUT_DIR/results_refine15.json"

echo ""
echo "============================================================"
echo "  DONE  $(date)"
echo "  Results: $OUT_DIR/results_refine0.json"
echo "           $OUT_DIR/results_refine15.json"
echo "============================================================"

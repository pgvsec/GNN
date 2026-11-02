#!/bin/bash
#SBATCH --job-name=diag_v10
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/e2e_v10/diag_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/e2e_v10/diag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:20:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Diagnostic: why does v10 training produce Train Loss=0.0000, Train CE=0.0000?
# Root hypothesis: valid_items is empty in run_epoch because either
#   (a) all dags are None (decompose_grn throws for all cases), or
#   (b) all scc_gt_counts are None (extract_scc_gt_counts throws for all cases), or
#   (c) scc_gt_counts = {} (empty dict) for all cases.
#
# This script runs each diagnostic step and prints detailed output so we
# can identify exactly where the pipeline breaks.

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $SCRATCH/runs/e2e_v10

echo "====== v10 Training Diagnostic ======"
echo "Date: $(date)"
echo ""

PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON - <<'PYEOF'
import sys, traceback, numpy as np
sys.path.insert(0, "/scratch/gilbreth/gupt1042/GNN")
import importlib

# ── Step 1: import check ─────────────────────────────────────────────────────
print("=== Step 1: Import checks ===")
for mod in ["finetune_e2e_composition", "assemble", "model"]:
    try:
        importlib.import_module(mod)
        print(f"  {mod}: OK")
    except Exception as e:
        print(f"  {mod}: FAILED — {e}")
        traceback.print_exc()
print()

from finetune_e2e_composition import (
    gather_cases, precompute_dags, precompute_gt_counts,
    extract_scc_gt_counts, load_corpus, run_epoch,
    _compute_chunk_loss
)
from pathlib import Path

# ── Step 2: load each training dir and print shapes ──────────────────────────
print("=== Step 2: Load each training dir ===")
SCRATCH = "/scratch/gilbreth/gupt1042"
TRAIN_DIRS = [
    f"{SCRATCH}/runs/composition_eval_v3",
    f"{SCRATCH}/runs/composition_eval_v4",
    f"{SCRATCH}/runs/generalization/e1_2p4",
    f"{SCRATCH}/runs/generalization/e1_4p2",
    f"{SCRATCH}/runs/v10_hardneg/hard_neg_3p3",
]

for d in TRAIN_DIRS:
    try:
        corp = load_corpus(Path(d))
        n = len(corp["adj"])
        adj_shape = corp["adj"].shape
        fp_shape  = corp["fp_state"].shape
        print(f"  {d.split('/')[-1]}: {n} cases, adj={adj_shape}, fp_state={fp_shape}")
    except Exception as e:
        print(f"  {d.split('/')[-1]}: LOAD FAILED — {e}")
print()

# ── Step 3: gather_cases ─────────────────────────────────────────────────────
print("=== Step 3: gather_cases ===")
try:
    train_cases = gather_cases(TRAIN_DIRS)
    print(f"  Total cases: {len(train_cases)}")
    if train_cases:
        c0 = train_cases[0]
        print(f"  case[0] keys: {list(c0.keys())}")
        print(f"  case[0] shapes: adj={c0['adj'].shape}, P={c0['P'].shape}, fp_state={c0['fp_state'].shape}")
except Exception as e:
    print(f"  FAILED — {e}")
    traceback.print_exc()
print()

# ── Step 4: precompute_dags ──────────────────────────────────────────────────
print("=== Step 4: precompute_dags ===")
try:
    dags = precompute_dags(train_cases)
    n_none = sum(1 for d in dags if d is None)
    n_ok   = len(dags) - n_none
    print(f"  Total: {len(dags)}, OK: {n_ok}, None: {n_none}")
    # Show first few non-None dags
    for i, (c, d) in enumerate(zip(train_cases[:10], dags[:10])):
        if d is None:
            # Re-run to see the exception
            from assemble import decompose_grn
            try:
                decompose_grn(c["adj"], c["P"], c["K"], c["T"], c["nh"], c["F"])
            except Exception as e:
                print(f"  case[{i}] DAG failed: {e}")
        else:
            root_ids = [s.scc_id for s in d.sccs if len(s.parents) == 0]
            print(f"  case[{i}] DAG OK: {len(d.sccs)} SCCs, roots={root_ids}")
except Exception as e:
    print(f"  FAILED — {e}")
    traceback.print_exc()
print()

# ── Step 5: precompute_gt_counts ─────────────────────────────────────────────
print("=== Step 5: precompute_gt_counts ===")
try:
    gtc_cache = precompute_gt_counts(train_cases, dags)
    n_none  = sum(1 for g in gtc_cache if g is None)
    n_empty = sum(1 for g in gtc_cache if g is not None and len(g) == 0)
    n_ok    = len(gtc_cache) - n_none - n_empty
    print(f"  Total: {len(gtc_cache)}, OK: {n_ok}, None: {n_none}, Empty-dict: {n_empty}")
    # Show first few
    for i, (c, d, g) in enumerate(zip(train_cases[:5], dags[:5], gtc_cache[:5])):
        if g is None:
            # Re-run to see the exception
            if d is not None:
                try:
                    extract_scc_gt_counts(d, c["fp_state"])
                except Exception as e:
                    print(f"  case[{i}] gtc failed: {e}")
                    traceback.print_exc()
        else:
            print(f"  case[{i}] gtc keys={list(g.keys())}, vals={[(k,v[0]) for k,v in g.items()]}")
except Exception as e:
    print(f"  FAILED — {e}")
    traceback.print_exc()
print()

# ── Step 6: simulate valid_items build ──────────────────────────────────────
print("=== Step 6: Simulate valid_items ===")
valid_count = 0
error_count = 0
empty_gtc   = 0
for i in range(len(train_cases)):
    dag = dags[i]
    if dag is None:
        error_count += 1
        continue
    gtc = gtc_cache[i]
    if gtc is None:
        error_count += 1
        continue
    if len(gtc) == 0:
        empty_gtc += 1
        continue
    valid_count += 1
print(f"  valid_items: {valid_count}")
print(f"  errors (dag/gtc=None): {error_count}")
print(f"  empty gtc dicts: {empty_gtc}")
print()

# ── Step 7: simulate _compute_chunk_loss on first valid chunk ────────────────
print("=== Step 7: Simulate _compute_chunk_loss on first 5 valid cases ===")
import torch
valid_items = [
    (train_cases[i], dags[i], gtc_cache[i])
    for i in range(len(train_cases))
    if dags[i] is not None and gtc_cache[i] is not None and len(gtc_cache[i]) > 0
]

if not valid_items:
    print("  No valid items — cannot simulate forward pass")
else:
    # Load model
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    from finetune_e2e_composition import load_l1, load_l3
    L1_CKPT = f"{SCRATCH}/runs/e2e_v8/l1_best.pt"
    L3_CKPT = f"{SCRATCH}/runs/e2e_v8/l3_best.pt"

    try:
        l1_model, l1_cfg, l1_cfg_dict = load_l1(L1_CKPT, device)
        l3_model, l3_cfg, l3_cfg_dict = load_l3(L3_CKPT, l1_model, device)
        l1_model = l1_model.to(device).train()
        l3_model = l3_model.to(device).train()
        print("  Models loaded OK")
    except Exception as e:
        print(f"  Model load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    chunk = valid_items[:5]
    try:
        loss, pred_counts_list = _compute_chunk_loss(
            chunk, l1_model, l3_model, device,
            scc_reg_weight=0.5, max_n=128, p_predicted=0.5,
            pos_reg_weight=0.3, freeze_l3_backbone=False,
            use_margin_loss=True, margin=0.1, correct_ce_weight=0.01,
            l1_count_no_detach=True,
            exist_consistency_weight=1.0, exist_consistency_alpha=10.0)

        print(f"  loss={loss.item():.6f}, requires_grad={loss.requires_grad}")
        for i, pc in enumerate(pred_counts_list):
            gt_n = chunk[i][0]["n_fp_gt"]
            print(f"  case[{i}]: pred_counts={pc}, gt_n={gt_n}")
    except Exception as e:
        print(f"  _compute_chunk_loss FAILED: {e}")
        traceback.print_exc()

print()
print("=== Diagnostic complete ===")
PYEOF

echo ""
echo "Done."

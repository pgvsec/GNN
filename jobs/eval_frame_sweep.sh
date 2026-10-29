#!/bin/bash
#SBATCH --job-name=eval_frame
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/eval_frame_sweep/eval_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/eval_frame_sweep/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Eval sweep: compare l3_frame=raw vs l3_frame=forced, also sweeps refine_steps=15 and 30.
# All four runs use the same v6 checkpoints — no retraining.
#
# Result (2026-10-29): forced frame is correct for v2-v6 checkpoints.
# finetune_e2e_composition.py calls _make_scc_batch without explicit frame,
# which defaults to "forced". Switching to "raw" drops count_exact 0.840->0.594.
# Summary: forced_rs15=0.840/0.822/0.860, forced_rs30=0.840/0.826/0.860
#          raw_rs15=0.594/0.750/0.681,    raw_rs30=0.594/0.758/0.678

SCRATCH=/scratch/gilbreth/gupt1042
REPO_GNN=$SCRATCH/GNN
REPO_GNN_TEMP=$SCRATCH/GNN_temp
OUT_DIR=$SCRATCH/runs/eval_frame_sweep
PYTHON=/home/gupt1042/.conda/envs/2025.06-py313/gnn/bin/python
export PYTHONUNBUFFERED=1

mkdir -p $OUT_DIR

L1_CKPT=$SCRATCH/runs/e2e_v6/l1_best.pt
L3_CKPT=$SCRATCH/runs/e2e_v6/l3_best.pt
VAL_DIR=$SCRATCH/runs/composition_eval_v5

cd $REPO_GNN

echo "=== 1/4: raw frame, rs=15 (WRONG for v2-v6 checkpoints; drops count_exact to 0.594) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
    --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT \
    --eval_dir $VAL_DIR \
    --out_csv $OUT_DIR/raw_rs15.csv \
    --l3_frame raw --dedup fill --refine_steps 15

echo "=== 2/4: raw frame, rs=30 ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
    --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT \
    --eval_dir $VAL_DIR \
    --out_csv $OUT_DIR/raw_rs30.csv \
    --l3_frame raw --dedup fill --refine_steps 30

echo "=== 3/4: forced frame, rs=15 (correct for v2-v6 — matches finetune training frame) ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
    --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT \
    --eval_dir $VAL_DIR \
    --out_csv $OUT_DIR/forced_rs15.csv \
    --l3_frame forced --dedup fill --refine_steps 15

echo "=== 4/4: forced frame, rs=30 ==="
PYTHONPATH=$REPO_GNN:$REPO_GNN_TEMP $PYTHON exhaust_gen/eval_composition2.py \
    --l1_ckpt $L1_CKPT --l3_ckpt $L3_CKPT \
    --eval_dir $VAL_DIR \
    --out_csv $OUT_DIR/forced_rs30.csv \
    --l3_frame forced --dedup fill --refine_steps 30

echo "=== Summary ==="
python3 -c "
import csv, os, glob

configs = [
    ('raw_rs15',    'raw',    15),
    ('raw_rs30',    'raw',    30),
    ('forced_rs15', 'forced', 15),
    ('forced_rs30', 'forced', 30),
]
print(f'{'Config':<20} {'count_exact':>12} {'fp_hd':>8} {'stable_acc':>11}')
print('-' * 55)
for name, frame, rs in configs:
    path = os.path.join('$OUT_DIR', name + '.csv')
    if not os.path.exists(path):
        print(f'{name:<20}  NOT FOUND')
        continue
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f'{name:<20}  EMPTY')
        continue
    n = len(rows)
    ce = sum(float(r['count_exact']) for r in rows) / n
    hd = sum(float(r['fp_hd']) for r in rows if r['fp_hd'] != 'nan') / n
    sa = sum(float(r['stable_acc']) for r in rows if r['stable_acc'] != 'nan') / n
    print(f'{name:<20}  {ce:>12.4f}  {hd:>8.4f}  {sa:>11.4f}')
"

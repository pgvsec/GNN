#!/bin/bash
#SBATCH --job-name=grn_layer1
#SBATCH --output=runs/layer1/slurm_%j.out
#SBATCH --error=runs/layer1/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gilbreth

# ---------------------------------------------------------------------------
# Usage:
#   sbatch train_layer1.slurm
#
# Before first run, build the cache (CPU, run once on Negishi or Gilbreth):
#   python train.py --mode cache \
#       --corpus_dir /path/to/corpusA \
#       --cache_dir /path/to/corpusA/cache
#
# Override any Config field on the command line (add to the python call below):
#   --epochs 60 --batch_size 256 --d_model 128 ...
# ---------------------------------------------------------------------------

# -- Edit these two paths --
CACHE_DIR="$HOME/corpusA/cache"
OUT_DIR="$HOME/runs/layer1"

mkdir -p "$OUT_DIR"

cd "$SLURM_SUBMIT_DIR"

# Load modules — adjust to what's available on your allocation:
#   module avail 2>&1 | grep -i torch
#   module avail 2>&1 | grep -i conda
module load anaconda/2024.02-py311 2>/dev/null || true
module load cuda/12.1             2>/dev/null || true

# If you use a conda environment with PyTorch:
# conda activate my_torch_env

python train.py \
    --mode train \
    --cache_dir "$CACHE_DIR" \
    --out_dir   "$OUT_DIR"  \
    --epochs         60     \
    --batch_size    256     \
    --d_model       128     \
    --n_gnn_layers    4     \
    --n_attn_layers   1     \
    --n_dec_layers    4     \
    --n_heads         8     \
    --num_workers     8     \
    --lr           3e-4     \
    --amp          true

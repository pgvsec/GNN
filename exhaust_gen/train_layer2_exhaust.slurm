#!/bin/bash
#SBATCH --job-name=grn_l2_exhaust
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/exhaust_layer2/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/exhaust_layer2/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Prerequisites:
#   1. exhaust Layer I best.pt must exist (from train_layer1_exhaust.slurm)
#   2. Corpus B (exhaust) cache must exist (from run_corpus_b_exhaust_gpu.slurm
#      + python train_layer2.py --mode cache or data_b.build_cache_b())

LAYER1_CKPT="/scratch/gilbreth/gupt1042/runs/exhaust_layer1/best.pt"
CACHE_B_DIR="/scratch/gilbreth/gupt1042/corpus_exhaust_B/cache"
OUT_DIR="/scratch/gilbreth/gupt1042/runs/exhaust_layer2"

# Build Corpus B cache if not present (run once, ~5 min CPU)
if [ ! -d "${CACHE_B_DIR}" ]; then
    echo "Cache not found — building Corpus B cache..."
    python -c "
from data_b import build_cache_b
build_cache_b('/scratch/gilbreth/gupt1042/corpus_exhaust_B', '${CACHE_B_DIR}')
"
    echo "Cache built."
fi

REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_layer2.py \
    --mode          train           \
    --layer1_ckpt   "$LAYER1_CKPT" \
    --cache_dir_b   "$CACHE_B_DIR" \
    --out_dir       "$OUT_DIR"     \
    --epochs            60         \
    --batch_size        128        \
    --d_model           128        \
    --n_gnn_layers        4        \
    --n_attn_layers       1        \
    --n_dec_layers        4        \
    --n_heads             8        \
    --m_dim             128        \
    --n_u_layers          2        \
    --num_workers         8        \
    --lr              1e-4         \
    --freeze_backbone true         \
    --amp             true

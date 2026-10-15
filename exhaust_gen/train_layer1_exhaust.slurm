#!/bin/bash
#SBATCH --job-name=grn_l1_exhaust
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/exhaust_layer1/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/exhaust_layer1/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Corpus A (exhaust, n=2..6) cache path — built by run_corpus_a_exhaust_gpu.slurm
CACHE_DIR="/scratch/gilbreth/gupt1042/corpus_exhaust_A/cache"
OUT_DIR="/scratch/gilbreth/gupt1042/runs/exhaust_layer1"

# Build Corpus A cache if not present (run once, ~5 min CPU)
if [ ! -d "${CACHE_DIR}" ]; then
    echo "Cache not found — building Corpus A cache..."
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    python "${REPO_ROOT}/train.py" --mode cache \
        --corpus_dir /scratch/gilbreth/gupt1042/corpus_exhaust_A \
        --cache_dir  "${CACHE_DIR}"
    echo "Cache built."
fi

mkdir -p "$OUT_DIR"
cd "$(dirname "$0")/.."

module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
    --mode      train          \
    --cache_dir "$CACHE_DIR"   \
    --out_dir   "$OUT_DIR"     \
    --epochs         60        \
    --batch_size    256        \
    --d_model       128        \
    --n_gnn_layers    4        \
    --n_attn_layers   1        \
    --n_dec_layers    4        \
    --n_heads         8        \
    --num_workers     8        \
    --lr           3e-4        \
    --amp          true

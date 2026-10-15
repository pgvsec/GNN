#!/bin/bash
#SBATCH --job-name=grn_l3_exhaust
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/exhaust_layer3/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/exhaust_layer3/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# Prerequisites:
#   1. exhaust Layer II best.pt must exist (from train_layer2_exhaust.slurm)
#   2. Corpus C (exhaust) cache must exist (from exhaust_gen/corpus_c_generator.py
#      + python data_c.py --mode cache or data_c.build_cache_c())

LAYER2_CKPT="/scratch/gilbreth/gupt1042/runs/exhaust_layer2/best.pt"
CACHE_C_DIR="/scratch/gilbreth/gupt1042/corpus_exhaust_C/cache"
OUT_DIR="/scratch/gilbreth/gupt1042/runs/exhaust_layer3"

# Build Corpus C cache if not present (run once, ~5 min CPU)
if [ ! -d "${CACHE_C_DIR}" ]; then
    echo "Cache not found — building Corpus C cache..."
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    python -c "
from data_c import build_cache_c
build_cache_c('/scratch/gilbreth/gupt1042/corpus_exhaust_C',
              '${CACHE_C_DIR}')
"
    echo "Cache built."
fi

mkdir -p "$OUT_DIR"
cd "$(dirname "$0")/.."

module load anaconda/2025.06-py313
source activate gnn

python train_layer3.py \
    --mode             train           \
    --layer2_ckpt      "$LAYER2_CKPT" \
    --cache_dir_c      "$CACHE_C_DIR" \
    --out_dir          "$OUT_DIR"     \
    --epochs               60         \
    --batch_size           128        \
    --d_model              128        \
    --n_gnn_layers           4        \
    --n_attn_layers          1        \
    --n_dec_layers           4        \
    --n_heads                8        \
    --m_dim                128        \
    --n_u_layers             2        \
    --max_ports_c            8        \
    --num_workers            8        \
    --lr                 5e-5         \
    --freeze_backbone    true         \
    --only_solved_val    true         \
    --amp                true

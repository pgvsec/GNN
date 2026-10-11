#!/bin/bash
#SBATCH --job-name=grn_layer2
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/layer2/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/layer2/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg

# ---------------------------------------------------------------------------
# Usage:
#   sbatch train_layer2.slurm
#
# Prerequisites (run once before submitting):
#   1. Regenerate Corpus B on Negishi with the fixed generator:
#        python corpus_b_generator.py \
#            --corpus_a_dir ~/python/corpusA \
#            --out_dir ~/python/corpusB_new \
#            --workers 8 --preset full
#
#   2. Build the Corpus B cache on Negishi (CPU ok):
#        python -c "from data_b import build_cache_b; \
#            build_cache_b('$HOME/python/corpusB_new', '$HOME/python/corpusB_new/cache')"
#
#   3. Transfer the cache to Gilbreth (run transfer_corpusB.sh from Negishi):
#        bash transfer_corpusB.sh
#
#   4. Confirm Layer I checkpoint exists:
#        ls /scratch/gilbreth/gupt1042/runs/layer1/best.pt
# ---------------------------------------------------------------------------

LAYER1_CKPT="/scratch/gilbreth/gupt1042/runs/layer1/best.pt"
CACHE_B_DIR="/scratch/gilbreth/gupt1042/corpusB_new/cache"
OUT_DIR="/scratch/gilbreth/gupt1042/runs/layer2"

mkdir -p "$OUT_DIR"
cd "$SLURM_SUBMIT_DIR"

module load anaconda/2025.06-py313
source activate gnn

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

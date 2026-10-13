#!/bin/bash
#SBATCH --job-name=grn_layer3
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/layer3/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/layer3/slurm_%j.err
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
#   sbatch train_layer3.slurm
#
# Prerequisites (run before submitting):
#   1. Generate Corpus C on Negishi (v2 schema required):
#        sbatch run_corpus_c.slurm
#        (256 shards × 500 DAGs × 64 workers → ~500-640K branches, ~6-8 min)
#
#   2. Build the Corpus C cache on Negishi:
#        python -c "from data_c import build_cache_c; \
#            build_cache_c('$HOME/python/corpusC', '$HOME/python/corpusC/cache')"
#
#   3. Transfer the cache to Gilbreth:
#        bash transfer_corpusC.sh
#
#   4. Confirm Layer II checkpoint exists:
#        ls /scratch/gilbreth/gupt1042/runs/layer2/best.pt
#
# The script warms-starts from the Layer II best.pt, loading Φ_ψ + A_η + U_ω
# weights in addition to the frozen backbone (E_θ + D_φ).
# ---------------------------------------------------------------------------

LAYER2_CKPT="/scratch/gilbreth/gupt1042/runs/layer2/best.pt"
CACHE_C_DIR="/scratch/gilbreth/gupt1042/corpusC/cache"
OUT_DIR="/scratch/gilbreth/gupt1042/runs/layer3"

mkdir -p "$OUT_DIR"
cd "$SLURM_SUBMIT_DIR"

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

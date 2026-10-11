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

LAYER1_CKPT="/scratch/gilbreth/gupt1042/runs/layer1/best.pt"
CACHE_B_DIR="/scratch/gilbreth/gupt1042/corpusB/cache"
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

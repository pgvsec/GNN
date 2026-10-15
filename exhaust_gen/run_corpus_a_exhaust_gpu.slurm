#!/bin/bash
#SBATCH --job-name=GRN_A_exhaust
#SBATCH --output=corpus_a_exhaust_%j.out
#SBATCH --error=corpus_a_exhaust_%j.err
#SBATCH -A ayg
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e

# ----------------- Environment Setup -----------------
module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn

export TMPDIR="/scratch/gilbreth/$(whoami)/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Prevent BLAS thread conflicts across workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

# REPO_ROOT is the GNN checkout; this script lives in exhaust_gen/ inside it
REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
EXHAUST_DIR="${REPO_ROOT}/exhaust_gen"

cd "${REPO_ROOT}"

echo "SLURM_JOB_ID  : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "CPUs          : ${SLURM_CPUS_PER_TASK}"
echo "REPO_ROOT     : ${REPO_ROOT}"
echo "Start time    : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ----------------- Execute ---------------------------

OUT_DIR="/scratch/gilbreth/${USER}/corpus_exhaust_A"

PYTHONPATH="${EXHAUST_DIR}:${REPO_ROOT}" \
python "${EXHAUST_DIR}/generate.py" \
    --preset          exhaust  \
    --out_dir         "${OUT_DIR}" \
    --device          cuda     \
    --workers         32       \
    --graphs_per_shard 128     \
    --seed            0

# ----------------- Cleanup ---------------------------
rm -rf "${TMPDIR}"

echo "Done. End time: $(date)"
echo "Output in: ${OUT_DIR}"

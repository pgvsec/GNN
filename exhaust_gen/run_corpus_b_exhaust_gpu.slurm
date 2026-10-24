#!/bin/bash
#SBATCH --job-name=GRN_B_exhaust
#SBATCH --output=corpus_b_exhaust_%j.out
#SBATCH --error=corpus_b_exhaust_%j.err
#SBATCH -A ayg
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# ----------------- Preflight Checks ------------------

CORPUS_A_DIR="/scratch/gilbreth/$(whoami)/corpus_exhaust_A"
CORPUS_B_DIR="/scratch/gilbreth/$(whoami)/corpus_exhaust_B"

if [ ! -f "${CORPUS_A_DIR}/topologies.npz" ]; then
    echo "ERROR: ${CORPUS_A_DIR}/topologies.npz not found." >&2
    echo "       Run the Corpus A job first." >&2
    exit 1
fi

if [ ! -d "${CORPUS_A_DIR}/cache" ]; then
    echo "Corpus A cache not found — building it now..."
    PYTHONPATH="${EXHAUST_DIR}:${REPO_ROOT}" \
    python train.py --mode cache \
        --corpus_dir "${CORPUS_A_DIR}" \
        --cache_dir  "${CORPUS_A_DIR}/cache"
    echo "Cache built."
fi

echo "Preflight OK."

# ----------------- Execute ---------------------------

# 16 workers × ~1.2 GB each fits within 22 GB VRAM.
# elements_per_batch=6M caps tensor spikes during n=5/6 bifurcation sweeps.
PYTHONPATH="${EXHAUST_DIR}:${REPO_ROOT}" \
python "${EXHAUST_DIR}/corpus_b_generator.py" \
    --corpus_a_dir       "${CORPUS_A_DIR}" \
    --out_dir            "${CORPUS_B_DIR}" \
    --preset             exhaust           \
    --device             cuda              \
    --workers            16                \
    --elements_per_batch 6000000           \
    --graphs_per_shard   128               \
    --rounds             5                 \
    --verify             true              \
    --seed               0

# ----------------- Cleanup ---------------------------
rm -rf "${TMPDIR}"

echo "Done. End time: $(date)"
echo "Output in: ${CORPUS_B_DIR}"

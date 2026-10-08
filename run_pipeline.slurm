#!/bin/bash
#SBATCH --job-name=GRN_A_corpus
#SBATCH --output=corpus_%j.out
#SBATCH --error=corpus_%j.err
#SBATCH -A ayg
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Stop script execution immediately if any command fails
set -e

# ----------------- Environment Setup -----------------
module --force purge
module load anaconda

# Initialize conda for bash subshells and activate your environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grn_env

export TMPDIR="/scratch/negishi/${USER}/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Prevent worker thread conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# ----------------- Execute Pipeline ------------------

# 1. Dataset generation utilizing available CPU worker processes
python generate.py --preset full --out_dir corpusA --device cpu --workers 64

# 2. Build flat memory-mapped cache arrays
python train.py --mode cache --corpus_dir corpusA --cache_dir corpusA/cache

# 3. Generate summary reports and visual panels
python analyze_dataset.py --cache_dir corpusA/cache --out_dir corpusA/analysis

# ----------------- Cleanup ---------------------------
rm -rf "${TMPDIR}"
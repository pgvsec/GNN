#!/bin/bash
#SBATCH --job-name=GRN_B_corpus
#SBATCH --output=corpus_b_%j.out
#SBATCH --error=corpus_b_%j.err
#SBATCH -A ayg
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -e

# ----------------- Environment Setup -----------------
module --force purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grn_env

export TMPDIR="/scratch/negishi/${USER}/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Prevent BLAS thread conflicts across workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Pin hash seed for reproducibility across spawned workers
export PYTHONHASHSEED=0

cd "$SLURM_SUBMIT_DIR"

# ----------------- Preflight Checks ------------------

CORPUS_A_DIR="corpusA"
CORPUS_B_DIR="corpusB"

if [ ! -f "${CORPUS_A_DIR}/topologies.npz" ]; then
    echo "ERROR: ${CORPUS_A_DIR}/topologies.npz not found." >&2
    echo "       Run the Corpus A job first and verify it completed." >&2
    exit 1
fi

if [ ! -d "${CORPUS_A_DIR}/cache" ]; then
    echo "ERROR: ${CORPUS_A_DIR}/cache/ not found." >&2
    echo "       Run: python train.py --mode cache --corpus_dir ${CORPUS_A_DIR} --cache_dir ${CORPUS_A_DIR}/cache" >&2
    exit 1
fi

# Verify the cache has the arrays Corpus B actually needs at worker-init time
for arr in s_graph.npy s_n_fp.npy s_cons.npy fp_state.npy s_fp_ptr.npy; do
    if [ ! -f "${CORPUS_A_DIR}/cache/${arr}" ]; then
        echo "ERROR: Missing cache array ${arr} under ${CORPUS_A_DIR}/cache/." >&2
        echo "       Re-run the cache step for Corpus A." >&2
        exit 1
    fi
done

echo "Preflight OK — Corpus A cache verified."
echo "SLURM_JOB_ID : ${SLURM_JOB_ID}"
echo "SLURM_NODEID : ${SLURM_NODEID}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Start time    : $(date)"

# ----------------- Execute Pipeline ------------------

# 1. Generate Corpus B
#
#    --workers 32  : 32 processes, each given --threads 2 (= 64 logical cores total).
#                    Each worker runs bifurcation sweeps *serially* before the batched
#                    solve, so fewer workers with more threads each is faster than 64
#                    single-threaded workers fighting over the CPU.
#    --threads 2   : torch linear-algebra threads per worker (Newton solve + eigenvalues).
#    --graphs_per_shard 64 : unchanged from Corpus A; keeps shard I/O granular.
#    --preset full : full draw budget (30/28/… draws per graph, n=2..10).
#    --verify true : independent correctness checks on the first batch of every shard.
#    --rounds 5    : solver rounds before giving up on a draw (same as Corpus A).
#
python corpus_b_generator.py \
    --corpus_a_dir "${CORPUS_A_DIR}" \
    --out_dir      "${CORPUS_B_DIR}" \
    --preset       full             \
    --device       cpu              \
    --workers      32               \
    --threads      2                \
    --graphs_per_shard 64           \
    --rounds       5                \
    --verify       true             \
    --seed         42

# 2. Visual sanity checks
python analyze_corpus_b.py \
    --corpus_b_dir "${CORPUS_B_DIR}" \
    --corpus_a_dir "${CORPUS_A_DIR}" \
    --out_dir      "${CORPUS_B_DIR}/analysis"

# ----------------- Cleanup ---------------------------
rm -rf "${TMPDIR}"

echo "Done. End time: $(date)"

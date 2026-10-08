#!/bin/bash
#SBATCH --job-name=GRN_C_corpus
#SBATCH --output=corpus_c_%j.out
#SBATCH --error=corpus_c_%j.err
#SBATCH -A ayg
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00

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
CORPUS_C_DIR="corpusC"

# Corpus C needs Corpus A topologies (topology.py reads N_MIN, N_MAX, etc.)
if [ ! -f "${CORPUS_A_DIR}/topologies.npz" ]; then
    echo "ERROR: ${CORPUS_A_DIR}/topologies.npz not found." >&2
    echo "       Run the Corpus A job first and verify it completed." >&2
    exit 1
fi

# Corpus C does NOT require the Corpus A cache or Corpus B outputs —
# it calls topology.sample_graphs and generate.sample_params directly.
# We do a quick sanity check that Corpus B exists (implies A ran cleanly).
if [ ! -d "${CORPUS_B_DIR}" ]; then
    echo "WARNING: ${CORPUS_B_DIR}/ not found — are you sure Corpus B has run?" >&2
    echo "         Continuing anyway; Corpus C is independent of Corpus B at runtime." >&2
fi

echo "Preflight OK — proceeding with Corpus C generation."
echo "SLURM_JOB_ID : ${SLURM_JOB_ID}"
echo "SLURM_NODEID : ${SLURM_NODEID}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Start time    : $(date)"

# ----------------- Execute Pipeline ------------------

# Generate Corpus C
#
#   --n_shards 64         : 64 shards × 500 DAGs/shard = 32 000 DAG samples total.
#                           Each shard is a self-contained .npz written atomically;
#                           failed shards can be re-run individually with --shard_start.
#   --dags_per_shard 500  : balanced against shard I/O overhead vs. shard file count.
#   --n_workers 32        : 32 parallel shard processes.
#                           Each worker runs one shard serially (no inner parallelism),
#                           so 32 workers × 2 natural threads ≈ 64 cores used.
#   --k_min 2             : minimum SCCs per DAG (chain of 2).
#   --k_max 8             : maximum SCCs per DAG (matches MAX_DAG_SCCS in generator).
#   --edge_prob 0.35      : per-candidate-edge draw probability for DAG topology
#                           (decays with layer distance; see sample_dag_topology).
#   --device cpu          : no GPU needed; Newton solve is batch-vectorised over
#                           the SCC size (N=2..10), not the DAG size.
#   --seed 42             : base RNG seed; shard i uses seed + i * 100_003.
#
python corpus_c_generator.py \
    --out_dir        "${CORPUS_C_DIR}" \
    --n_shards       64               \
    --dags_per_shard 500              \
    --n_workers      32               \
    --k_min          2                \
    --k_max          8                \
    --edge_prob      0.35             \
    --device         cpu              \
    --seed           42               \
    --log_level      INFO

# ----------------- Cleanup ---------------------------
rm -rf "${TMPDIR}"

echo "Done. End time: $(date)"

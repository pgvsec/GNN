#!/bin/bash
#SBATCH --job-name=grn_comp_fix
#SBATCH --output=/scratch/gilbreth/gupt1042/runs/comp_fix/slurm_%j.out
#SBATCH --error=/scratch/gilbreth/gupt1042/runs/comp_fix/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=a10
#SBATCH --account=ayg
#
# Composition count fix: ablation of the inference fixes, more composition training data,
# count-head fine-tuning on exact per-branch labels, and the final held-out evaluation.
# See COMPOSITION_FIXES.md. Every path can be overridden from the environment, e.g.
#   L3_CKPT=/path/to/other.pt sbatch run_composition_fix.slurm
# Make the output directory before submitting:  mkdir -p /scratch/gilbreth/gupt1042/runs/comp_fix

set -euo pipefail

RUNS=${RUNS:-/scratch/gilbreth/gupt1042/runs}
L1_CKPT=${L1_CKPT:-$RUNS/layer1_v10/best.pt}
L3_CKPT=${L3_CKPT:-$RUNS/layer3_v12/best.pt}
CORPUS_A_CACHE=${CORPUS_A_CACHE:-/scratch/gilbreth/gupt1042/corpus_exhaust_A_v2/cache}
TEST_DIR=${TEST_DIR:-$RUNS/composition_eval_v5}
TRAIN_DIRS=${TRAIN_DIRS:-"$RUNS/composition_eval $RUNS/composition_eval_v3 $RUNS/composition_eval_v4"}
N_EXTRA_SETS=${N_EXTRA_SETS:-4}          # extra composition training sets ...
N_EXTRA_CASES=${N_EXTRA_CASES:-2000}     # ... of this many cases each
OUT=${OUT:-$RUNS/comp_fix}

module --force purge
module load external
module load anaconda/2025.06-py313
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/gupt1042/.conda/envs/2025.06-py313/gnn

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT"
export OMP_NUM_THREADS=8

# 1. Inference ablation with the existing checkpoints on the held-out set.
#    old      : no Thomas class for L3 (what the assembler did before)
#    tc       : Thomas class passed, L3 sees P*u (the historical frame)
#    tc_raw   : Thomas class passed, L3 sees unforced P (the frame L3 was trained in)
#    l1_forced: L1 on forced SCCs as autonomous SCCs with P*u
#    *_nofb   : same without the exact-solver fallback (pure ML)
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --eval_dir "$TEST_DIR" --out_dir "$OUT/ablate_base" \
    --variants old:pass_tc=0 tc:pass_tc=1 tc_raw:l3_frame=raw l1_forced:forced_model=l1 \
               tc_nofb:solver_fallback=0 tc_raw_nofb:l3_frame=raw,solver_fallback=0 \
               l1_forced_nofb:forced_model=l1,solver_fallback=0

# 2. More composition training data (new seeds, so no case overlaps the test set).
EXTRA=""
for i in $(seq 1 "$N_EXTRA_SETS"); do
    d="$RUNS/composition_train_s$((1000 + i))"
    if [ ! -f "$d/n_fp.npy" ]; then
        PYTHONPATH=. python exhaust_gen/gen_composition_eval.py \
            --corpus_cache "$CORPUS_A_CACHE" --out_dir "$d" \
            --n_cases "$N_EXTRA_CASES" --n_cross 2 --seed $((1000 + i))
    fi
    EXTRA="$EXTRA $d"
done

# 3. Count-head fine-tuning for each inference variant. Selection uses a split of the
#    training cases; the test set is scored once at the end with the full pipeline.
for variant in "forced:--l3_frame forced" "raw:--l3_frame raw" "l1f:--forced_model l1"; do
    name=${variant%%:*}
    flags=${variant#*:}
    python finetune_count_heads.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
        --train_dirs $TRAIN_DIRS $EXTRA --val_frac 0.15 --test_dir "$TEST_DIR" \
        --out_dir "$OUT/ft_$name" $flags --eval_assemble
done

# 4. Summary: pure-ML numbers of every fine-tuned variant on the held-out set.
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --heads "$OUT/ft_forced/count_heads.pt" --eval_dir "$TEST_DIR" --out_dir "$OUT/final_forced" \
    --variants ft_forced_nofb:solver_fallback=0
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --heads "$OUT/ft_raw/count_heads.pt" --eval_dir "$TEST_DIR" --out_dir "$OUT/final_raw" \
    --variants ft_raw_nofb:l3_frame=raw,solver_fallback=0
python composition_eval.py --l1_ckpt "$L1_CKPT" --l3_ckpt "$L3_CKPT" \
    --heads "$OUT/ft_l1f/count_heads.pt" --eval_dir "$TEST_DIR" --out_dir "$OUT/final_l1f" \
    --variants ft_l1f_nofb:forced_model=l1,solver_fallback=0

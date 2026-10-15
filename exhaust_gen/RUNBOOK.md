# Exhaust Corpus Generation Runbook

Generates Corpus A, B, C for n=2..6 with exhaustive dynamical sampling.
All code lives in `exhaust_gen/` — original repo files are untouched.

---

## 1. Clone the repo (gilbreth)

```bash
ssh gilbreth.rcac.purdue.edu
cd /scratch/gilbreth/$USER
git clone https://<YOUR_PAT>@github.com/pgvsec/GNN.git GNN_temp
cd GNN_temp
git checkout claude/zealous-wright-eum1ur
```

> **GitHub auth:** HTTPS password login is disabled. Use a Personal Access Token
> (Settings → Developer settings → Tokens → classic, `repo` scope) in place of
> `<YOUR_PAT>`, or use SSH: `git clone git@github.com:pgvsec/GNN.git GNN_temp`

---

## 2. Sanity check (optional, ~30 sec)

Confirms the topology builder stops at n=6 with the right counts.

```bash
cd /scratch/gilbreth/$USER/GNN_temp
PYTHONPATH=exhaust_gen:. python -c "
import exhaust_gen.topology as topology
gs = topology.build_graphs([10000,15000,12000,0,0,0,0], seed=42, n_max=6)
from collections import Counter
c = Counter(g.n for g in gs)
for n in sorted(c): print(f'  n={n}: {c[n]:6d}')
"
```

Expected output:
```
  n=2:     21
  n=3:   1934
  n=4:   8813
  n=5:  15000
  n=6:  12000
```

---

## 3. Corpus A (GPU, ~2-3 hrs)

```bash
cd /scratch/gilbreth/$USER/GNN_temp/exhaust_gen
sbatch run_corpus_a_exhaust_gpu.slurm
```

Monitor:
```bash
squeue -u $USER
tail -f corpus_a_exhaust_<JOBID>.out
```

Output: `/scratch/gilbreth/$USER/corpus_exhaust_A/`

---

## 4. Corpus C (CPU on login node, ~10 min)

Run in parallel with Corpus A (independent — does not need A to finish first).

```bash
cd /scratch/gilbreth/$USER/GNN_temp
PYTHONPATH=exhaust_gen:. python exhaust_gen/corpus_c_generator.py \
    --out_dir /scratch/gilbreth/$USER/corpus_exhaust_C \
    --n_shards 96 \
    --dags_per_shard 500 \
    --n_workers 24
```

Run this in a tmux session so it survives terminal disconnect:
```bash
tmux new -s corpus_c
# paste the command above, then Ctrl+B D to detach
```

Output: `/scratch/gilbreth/$USER/corpus_exhaust_C/`

---

## 5. Corpus B (GPU, ~3-5 hrs) — run after Corpus A finishes

```bash
cd /scratch/gilbreth/$USER/GNN_temp/exhaust_gen
sbatch run_corpus_b_exhaust_gpu.slurm
```

The script auto-builds the Corpus A cache (`train.py --mode cache`) if it
doesn't exist yet, then runs Corpus B.

Output: `/scratch/gilbreth/$USER/corpus_exhaust_B/`

---

## Notes

- All three sbatch scripts use `-p gpu`, 1 GPU, 16 workers, `--device cuda`.
- If the GPU partition name differs on your cluster, check with `sinfo | grep gpu`
  and update the `#SBATCH -p` line in the `.slurm` files.
- The negishi CPU alternative (10+ hrs) uses `run_pipeline.slurm` in the repo
  root with `--preset full` — the exhaust scripts above are the fast path.

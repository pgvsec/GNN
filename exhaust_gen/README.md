# exhaust_gen — High-Parameter-Density Generator

Modified versions of the three corpus generator files tuned for **dynamical
exhaustiveness**: each topology sees ~2× more independent parameter draws than
the baseline `"full"` preset (~40 per topology for large n, up from ~20), so
the 4–5 dynamical regimes a topology can exhibit are all reliably sampled.
Topology breadth is kept at the same level as `"full"` (n=2,3 exhaustive
structural enumeration; n=4–10 random sampling with WL-hash dedup).

**Original files in the repo root are unchanged.** All other modules
(model.py, solver.py, data.py, etc.) are imported from the parent directory.

---

## What changed

### generate.py — new `"exhaust"` preset

| n  | Topologies (same as `full`) | Draws per topo (`full` → `exhaust`) |
|----|-----------------------------|--------------------------------------|
| 2  | exhaustive (21)             | 120 → **240**                        |
| 3  | exhaustive (1934)           | 100 → **200**                        |
| 4  | 6 000                       | 80  → **160**                        |
| 5  | 10 000                      | 30  → **60**                         |
| 6  | 7 000                       | 25  → **50**                         |
| 7  | 5 000                       | 20  → **40**                         |
| 8  | 4 000                       | 16  → **32**                         |
| 9  | 3 000                       | 12  → **24**                         |
| 10 | 2 400                       | 10  → **20**                         |

Solver seed budget per draw is identical to `"full"`.

### topology.py

Adds `enumerate_exhaustive_chunked` (used for n≤3, same as the original
`enumerate_exhaustive`; ready for n=4 if ever needed — raises for n≥5).
`build_graphs`/`build_and_save` accept an `exhaust_n` parameter (default 3,
matching original behavior; the `"exhaust"` preset does not override it).

### corpus_c_generator.py

Pre-builds exhaustive topology pools for SCC sizes n ≤ `EXHAUST_SCC_N=4` at
shard startup. SCC topology selection for small n draws uniformly from the
pool (all 21 / 1934 / ~all-n=4 topologies get equal representation).

---

## Running the three generators

The generators have the following dependency:

```
Corpus A  (generate.py)         ─── writes corpusA/topologies.npz + shards
    └── Corpus B (corpus_b_generator.py)  ─── reads corpusA/topologies.npz
Corpus C  (corpus_c_generator.py)          ─── fully independent
```

**Corpus C can run in parallel with Corpus A.** Corpus B must wait for A.

### Recommended: three terminals from the repo root

```bash
# Terminal 1 — Corpus A  (topology build + parameter sampling)
cd /path/to/GNN
PYTHONPATH=. python exhaust_gen/generate.py \
    --preset exhaust \
    --out_dir corpusA_exhaust \
    --workers 8 \
    --device cpu          # or --device cuda

# Terminal 2 — Corpus C  (start in parallel with A)
cd /path/to/GNN
PYTHONPATH=. python exhaust_gen/corpus_c_generator.py \
    --out_dir corpusC_exhaust \
    --n_shards 32 \
    --workers 8 \
    --device cpu

# Terminal 3 — Corpus B  (start AFTER Corpus A finishes)
cd /path/to/GNN
PYTHONPATH=. python corpus_b_generator.py \
    --topo_path corpusA_exhaust/topologies.npz \
    --out_dir corpusB_exhaust \
    --workers 8 \
    --device cpu
```

### Or with `nohup` / `screen` to keep jobs running

```bash
# From the repo root
PYTHONPATH=. nohup python exhaust_gen/generate.py \
    --preset exhaust --out_dir corpusA_exhaust --workers 8 \
    > logs/corpusA.log 2>&1 &

PYTHONPATH=. nohup python exhaust_gen/corpus_c_generator.py \
    --out_dir corpusC_exhaust --n_shards 32 --workers 8 \
    > logs/corpusC.log 2>&1 &

# Wait for Corpus A, then:
PYTHONPATH=. nohup python corpus_b_generator.py \
    --topo_path corpusA_exhaust/topologies.npz \
    --out_dir corpusB_exhaust --workers 8 \
    > logs/corpusB.log 2>&1 &
```

### Quick sanity check before a long run

```bash
cd /path/to/GNN
PYTHONPATH=. python exhaust_gen/generate.py \
    --preset quick --out_dir /tmp/test_exhaust_gen --workers 2
```

---

## Expected runtime (CPU, 8 workers)

| Corpus | Preset    | Approx. time |
|--------|-----------|--------------|
| A      | `exhaust` | 4–12 h       |
| B      | —         | 2–6 h        |
| C      | 32 shards | 2–8 h        |

A and C run concurrently, so wall-clock is roughly max(A, C) + B.

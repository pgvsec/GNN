# exhaust_gen — Exhaustive Topology Generator

Modified versions of the three corpus generator files that exhaustively sample
all physically possible non-isomorphic topologies for n ≤ 6.

**Original files are unchanged.** This directory contains standalone modified
copies. All other modules (model.py, solver.py, assemble.py, data.py, etc.)
are imported from the parent directory via `sys.path` or the Python path.

## What changed

### topology.py

- **New**: `enumerate_exhaustive_chunked(n, seen, chunk_size=2_000_000)`
  Iterates all 3^(n×n) ternary sign matrices in memory-safe chunks, filtering
  strongly connected graphs and deduplicating by WL-hash canonical key.
  Delegates to `enumerate_exhaustive` for n ≤ 3 and raises `ValueError` for
  n ≥ 5 (3^25 = 847 billion candidates — infeasible).

- **Modified**: `build_graphs(topos_per_n, seed, exhaust_n=4)`
  Uses `enumerate_exhaustive_chunked` for n ≤ exhaust_n, random sampling
  (unchanged) for n > exhaust_n.

- **Modified**: `build_and_save(topos_per_n, seed, path, exhaust_n=4)`
  Passes `exhaust_n` through to `build_graphs`.

### generate.py

- **New preset** `"exhaust"` in `PRESETS`:
  - n = 2, 3, 4: exhaustive enumeration (`exhaust_n=4`)
  - n = 5: 1 000 000 quota (near-complete for this size)
  - n = 6: 500 000 quota
  - n = 7–10: same as `"full"` preset
  - Same draws/seeds schedule as `"full"`

- **Modified**: `generate()` reads `preset.get("exhaust_n", 3)` and passes it
  to `topology.build_and_save` (backward-compatible: all existing presets
  default to the original `exhaust_n=3` behavior).

### corpus_c_generator.py

- **New constant**: `EXHAUST_SCC_N = 4`

- **Modified**: `run_shard_c()` pre-builds exhaustive topology pools for
  n = 2 … EXHAUST_SCC_N at startup using `enumerate_exhaustive_chunked`.
  When an SCC of size n ≤ EXHAUST_SCC_N is needed, one graph is sampled
  uniformly at random from the pool (repeats allowed, which is fine because
  different DAGs get independent parameter draws). For n > EXHAUST_SCC_N the
  original `sample_graphs` path is unchanged.

## Feasibility notes

| n  | Candidates   | Approx. distinct SC sign digraphs | Method         |
|----|-------------|-----------------------------------|----------------|
| 2  | 81          | 21                                | exhaustive     |
| 3  | 19 683      | 1 934                             | exhaustive     |
| 4  | 43 046 721  | ~1–8 million (TBD)               | exhaustive (chunked, ~10–30 min) |
| 5  | 847 billion | unknown; huge                     | high-quota sampling |
| 6  | 150 trillion| unknown; enormous                 | high-quota sampling |

For n = 4 the full enumeration is the computational bottleneck. Expect it to
take 10–30 minutes on a modern CPU. The resulting topology file will be large.

## Usage

Run from the parent directory so that sibling modules are importable:

```bash
cd /path/to/GNN

# Corpus A with exhaustive n=2..4 coverage
python exhaust_gen/generate.py --preset exhaust --out_dir corpusA_exhaust --workers 4

# Corpus C with exhaustive SCC topologies for n<=4
python exhaust_gen/corpus_c_generator.py --out_dir corpusC_exhaust --n_shards 16 --workers 4
```

Or add `exhaust_gen/` to `PYTHONPATH` and run directly:

```bash
PYTHONPATH=. python exhaust_gen/generate.py --preset exhaust --out_dir corpusA_exhaust
```

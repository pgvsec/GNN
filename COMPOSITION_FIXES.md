# Composition count fixes

Status after the 2026-10-19 session: count_exact 66.6%, fp_hd 63.4% on composition_eval_v5,
against targets of 75% / 86.5% / 81.06% (stable_acc). fp_hd is 94.2% on cases whose count
is right, so the count is the bottleneck. This change fixes four defects on that path and
replaces the count-head fine-tuning. None of it has been run on the real checkpoints yet;
`run_composition_fix.slurm` does that in one job.

## Defects found

| # | Where | Defect | Effect on count |
|---|-------|--------|-----------------|
| 1 | `assemble.py` | The SCC batch never carried `tc`, so `ForcingAwareCountHead` read `tc = 1` (TC_NEG, "multistability impossible") for **every** forced SCC. `finetune_count_heads.py` did pass `tc`, so the fine-tuned head was trained on one input and evaluated on another. | Pushes forced branches toward k = 1: the dominant −2 error. Also explains why fine-tuning reported 82.6% val but 66.6% in the real pipeline on the same set. |
| 2 | `assemble.py`, `model_b.py`, `model_c.py` | Layer III depended on how far a batch was padded. `UOmega` builds its portrait from the backbone `pos` of every gene slot, and padded slots hold `sigmoid(...)` values rather than zeros. `_forcing_stats` took a `max` over genes after zeroing padded entries, so a zero beat the negative values of weakly forced genes. Training collates trim each batch to its largest graph; the assembler padded every SCC to `N_MAX = 10`. | At inference L3 was conditioned on inputs it never saw in training. In a random-weight test model the count-head inputs of the same SCC differed by up to 5.6 (abs) between the two paddings; the size of the effect on the trained checkpoint is unknown until the ablation runs. |
| 3 | `finetune_count_heads.py` | Every branch of a forced SCC was trained toward the **modal** count over its parent states, only the first parent's forcing was used, and the reported "val CE" multiplied per-SCC counts instead of using the sum formula. | A child that is bistable under one parent state and monostable under two was labelled k = 1 everywhere. That trains in the under-prediction. |
| 4 | `losses.py` (L3 training) | The ODE-residual loss used the unforced `P` while the GT positions belong to `P * u`. | The physics term fought the position term on every forced branch. It does not affect the current checkpoint at inference; it matters on the next L3 retrain. |

Also: `_model_is_degenerate` called `topk(k)` with k from L3's 33-way count head, which can exceed
the slot count (crash). And the exact-solver fallback in `_infer_scc` is a non-ML path. It is now
counted (`n_solver_fallback` in every eval) and can be turned off (`solver_fallback=0`).

## Open question: L3 input frame

L2 was trained on Corpus B with `P = P * u` (forcing absorbed into P). L3 was trained on
Corpus C with the **unforced** `P` and forcing only through the ports (`data_c.load_branch`).
The assembler fed `P * u` *and* the ports. Which frame suits the current L3 checkpoint is
an empirical question, so it is an option (`InferenceOptions.l3_frame`) and the ablation
reports both. Newton refinement always uses the physical `P * u`.

A third option needs no L3 at all: under constant forcing a forced SCC *is* an autonomous
SCC with production `P * u`, so L1 (trained on 111k+ Corpus A samples) can predict it
directly (`forced_model=l1`). This is an exact reparametrisation of the input, not a heuristic.

## What changed

- `assemble.py`: `InferenceOptions` (`l3_frame`, `forced_model`, `pass_tc`, `solver_fallback`),
  `thomas_class()`. `_make_scc_batch` trims to the SCC size and carries `tc`, `has_pos_fb`,
  `P_phys`, `xmax_phys`. Refinement, residuals and dedup use the physical system.
  `INFER_STATS` counts L1/L3 calls and solver fallbacks.
- `model_b.py`, `model_c.py`: padded genes are masked out of the portrait and the forcing max.
- `composition_labels.py` (new): exact per-branch labels from composition GT, plus
  `enumerate_count()` (the sum formula on any DAG) as a self-check. The check reproduces the GT
  count on 299/300 synthetic cases; the one miss is a GT gap. Branches with even counts or
  contradictory contexts are flagged (`consistent=False`). About 1% of synthetic GT cases are
  visibly incomplete; on those, count_exact penalises a correct model.
- `composition_eval.py` (new): evaluates several inference variants in one run. It reports
  count_exact, fp_hd and stable_acc (using `eval_composition2.py`'s own metric functions) on all
  cases and on clean-GT cases, under/over counts, root-vs-forced error attribution, and solver
  fallbacks.
- `finetune_count_heads.py` (rewritten): exact per-branch labels. Batches come from the same
  builder and frame as inference. The head inputs are cached once through forward hooks, and the
  heads are trained on the cache with real minibatches. A sweep over loss weighting
  (none/sqrt/linear/balanced) × learning rate picks the best configuration on a validation split
  of the *training* dirs. The test dir is scored once at the end, so there is no selection on
  test. Optional Corpus C branches (`--corpus_c_cache`); `--eval_assemble` reruns the full
  pipeline before and after.
- `data_c.py`, `losses.py`, `train_layer3.py`: `P_phys`/`xmax_phys` in the collate and
  `apply_frame()`. Physics losses use `P_phys`. `train_layer3.py --frame raw|forced` selects
  the input frame for a retrain.
- `tests/test_composition_fixes.py`: regression tests (padding invariance, batch contents,
  Corpus C path == assembly path, label self-check). The padding tests fail on the old model code.

## How to run

```bash
mkdir -p /scratch/gilbreth/gupt1042/runs/comp_fix
sbatch run_composition_fix.slurm
```

Steps (all paths overridable from the environment):
1. Ablation of the existing checkpoints on composition_eval_v5 (`comp_fix/ablate_base/summary.json`).
2. Four new composition training sets of 2000 cases (seeds 1001–1004), about 0.15 s per case.
3. Count-head fine-tuning for the `forced`, `raw` and `l1` variants (`comp_fix/ft_*`).
4. Pure-ML (no solver fallback) held-out numbers of each fine-tuned variant (`comp_fix/final_*`).

Single runs:
```bash
python composition_eval.py --l1_ckpt L1 --l3_ckpt L3 --eval_dir DIR --out_dir OUT \
    --variants tc:pass_tc=1 tc_raw:l3_frame=raw l1_forced:forced_model=l1
python finetune_count_heads.py --l1_ckpt L1 --l3_ckpt L3 --train_dirs D1 D2 ... \
    --test_dir composition_eval_v5 --out_dir OUT --l3_frame raw --eval_assemble
python composition_eval.py ... --heads OUT/count_heads.pt --variants ft:l3_frame=raw
```

## Reading the results

- `blame` splits count errors into root SCC (L1), forced SCC (L3) and both. The `tc` variants should
  mostly shrink the forced share. If `root` dominates afterwards, the next lever is L1, starting
  with `finetune_l1_on_corpusA.py`.
- `clean CE` is count_exact on cases whose GT passes the consistency check. The gap to plain
  count_exact is error that no model can remove.
- fp_hd should follow count_exact (94.2% when the count is right).
- Report the `*_nofb` numbers: they are the pure-ML ones.

## If still below target

1. Retrain L3 (`train_layer3.py`) with the physics fix and the frame that wins the ablation,
   with `w_count_kscale > 0`. The padding fix also removes a train-time artifact (mixed-size
   batches fed garbage positions into the portrait).
2. Unfreeze L1's decoder and exist head on Corpus A with k-weighted count loss
   (`finetune_l1_on_corpusA.py` currently trains only the count head).
3. Generate more composition training data. It is cheap, and it matches the test distribution.

## Round 2: results of run_composition_fix.slurm (composition_eval_v5, 500 cases)

| variant | count_exact | fp_hd | stable_acc |
|---|---|---|---|
| old (no tc) | 64.8% | 62.8% | 71.2% |
| tc | 65.4% | 64.2% | 71.6% |
| tc_raw | 64.8% | 64.8% | 71.6% |
| l1_forced (no solver fallback) | **66.6%** | **68.1%** | **72.8%** |
| fine-tuned heads, best variant | 66.6% | 68.1% | 72.8% |

What the numbers say:
- The inference fixes give +1.8 pp. Running L1 on forced SCCs (with P * u) beats L3.
- Head fine-tuning on frozen features is saturated. For `l1_forced` the sweep kept the original
  heads; per-branch accuracy stays near 78% for k = 3 and 54% for k = 5.
- With counts taken from the head under true forcing, 74.7% of cases are exactly right. The
  pipeline gets 67.4% on the same clean cases. fp_hd is ~94% when the count is right, so the ~7 pp
  are lost in `_infer_scc` post-processing: linear dedup at 2% of xmax merges distinct
  low-expression points, and the parity drop then removes another slot (3 -> 2 -> 1).

Changes:
- `InferenceOptions.rescue / dedup (linear|log|none) / parity_drop`. The pure-ML head-only count
  is `rescue=0,dedup=none,parity_drop=0`. `INFER_STATS` counts post-processing count changes
  (`post up/down` column).
- `count_net.py`, `train_count_net.py`: CountNet is a Layer I copy with every weight trained for
  the count (`SccFixedPointNet.count_detach = False`). It trains on Corpus A plus all composition
  branches as autonomous SCCs with P * u, and is selected on the teacher-forced composition score.
  At inference it only replaces `count_logit`; slots, positions and stability stay with Layer I.
- `run_count_net.slurm`: post-processing ablation -> CountNet training -> held-out eval.
- composition_eval reports `count_right` (fp_hd and stable_acc on count-correct cases). This
  tells whether stable_acc needs its own fix once the count is right.

## Round 3: post-processing and CountNet results (composition_eval_v5)

| variant (L1 on every SCC, no solver fallback) | count_exact | fp_hd | stable_acc |
|---|---|---|---|
| legacy post-processing | 66.6% | 68.1% | 72.8% |
| **log-space dedup** | **81.2%** | **88.1%** | **83.4%** |
| head-only count (pure ML) | 72.8% | 79.3% | 78.9% |
| CountNet v1 + log dedup | 87.4% | 87.6% | 87.8% |
| CountNet v1 + head-only | 93.0% | 87.0% | 92.3% |

- The 2%-of-xmax dedup was the largest single defect (159 SCC calls lost points). Log-space
  dedup changes no weights and clears all three targets on v5. The variant was chosen after
  seeing v5, so it still needs a fresh test set.
- **CountNet v1 is not yet trustworthy.** Its teacher-forced score is 78.7% on its validation
  split but 96.0% on v5, while L1's is 74.1% vs 74.8% and CountNet's Corpus A accuracy did not
  move. That pattern indicates v5 cases (or duplicates of them) in its training dirs.
- `run_verify.slurm`: `check_overlap.py` measures the overlap. A strict test set is built from
  Corpus A test-split topologies (`gen_composition_eval.py --split test`). CountNet v2 trains
  only on train-split compositions. Everything is re-scored on the strict set and on v5.
- Note: the log and legacy variants keep the residual "rescue" step (Newton polish). The
  head-only rows are the strictly pure-ML count.

`dedup=fill` (added after round 3): the count is exactly the count head's k, and the k slots are
filled with distinct refined points. A slot that converges onto an earlier pick is replaced by the
next-ranked one. Head-only lost fp_hd this way (90.3% on count-correct cases vs 95.3% with log
dedup) without any change to the count. `cn_fill` is the candidate for >85% on all three metrics.

## Round 4: leak-free verification (run_verify.slurm)

- **Leak confirmed:** composition_eval_v3, _v4 and _v5 are the same 500 cases. Every "v5 test"
  number from a model trained on v3/v4 (count_ft_v2, CountNet v1) was scored on training data.
- **CountNet does not generalise:** trained only on train-split compositions (v2), it never beat
  the original L1, and validation fell from epoch 1. Dropped.
- **Strict test set** (1000 cases, Corpus A test-split topologies, `composition_test_strict`):

| variant (L1 on every SCC, no solver fallback) | count_exact | fp_hd | stable_acc |
|---|---|---|---|
| legacy | 69.4% | 66.6% | 73.4% |
| log dedup | **81.7%** | **85.4%** | **84.3%** |
| head-only count (pure ML) | 74.1% | 74.8% | 79.3% |

On count-correct cases fp_hd is 95.0% and stable_acc 96.5%: the count is the only remaining gap.
The log row keeps the Newton-verified rescue (model proposes slots, Newton confirms them).

`run_tune.slurm`: tunes Newton steps per slot (`refine`) and the log dedup tolerance
(`dedup_tol`) on composition_train_strict_s2001, selects by the worst metric, and scores only the
selected setting on the strict set.

## Round 5: pure-ML improvement of the per-SCC predictor (run_l1_forced.slurm)

The decomposition is exact (block-triangular Jacobian over the acyclic SCC graph), so composite
accuracy is set by how well Layer I predicts one SCC under constant input. A forced SCC is an
autonomous SCC with production P * u, and composition inputs push P * u up to ~2 decades outside
the Corpus A range P in [0.01, 100].

- `gen_forced_scc.py`: Corpus A train-split SCCs (n = 2..6) under composition-like input, solved
  with the Corpus A solver. Full labels in data.load_sample format; residuals are verified ~1e-7.
- `finetune_l1_forced.py`: Layer I fine-tuned on Corpus A + forced samples with its full
  SetCriterion loss. Selection uses the teacher-forced composition score on a tuning set.
- `diag_forcing_range.py`: branch accuracy binned by decades outside the Corpus A P range.
- Headline numbers are network-only: count from the count head, no rescue (`pure`, and
  `pure_nopolish` with no Newton step at all). `hybrid_log` is reported for reference only.

**Loader bug (fixed):** building L3 around the L1 model and then loading the L3 checkpoint
overwrote the L1 weights with L3's stored `backbone.*` copy (composition_eval.load_models,
assemble.load_l3_model, and the original eval_composition2.py). Every earlier pipeline number
used L3's backbone for root SCCs, not the `--l1_ckpt` given. L3 now gets its own backbone copy,
and the eval prints how far the two differ. Rows in run_l1_forced.slurm are the first to use
the given Layer I.

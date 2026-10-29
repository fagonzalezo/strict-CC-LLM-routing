# Dataset Construction — RouterEval-CC and EmbedLLM-CC

The paper's two Cold–Cold datasets, **RouterEval-CC** and **EmbedLLM-CC**, are called
`router-eval` and `embedllm` in code and directory names. This document describes how each is
derived from its upstream release. The commands to run the pipeline are in the
[README](../README.md#tier-b--full-rerun-from-raw-data).


**Scope**: how `data/<dataset>/processed/llm_perf_dataset.pkl` is built from the upstream releases  
**Datasets**: `router-eval` (from `linggm/RouterEval`) · `embedllm` (from `RZ412/EmbedLLM`)  
**Build code**: `src/data_pipeline/router_eval/`, `src/data_pipeline/embedllm/`  
**Consumed by**: `src/eval/data.py::load_splits` via `DatasetSpec` (`src/eval/spec.py`)

---

## 1. Overview

Both pipelines turn a public LLM-evaluation release into the *same* artifact: a single pickle
holding query embeddings, LLM embeddings, and a binary correctness matrix, partitioned cold-cold so
that neither the queries nor the LLMs in the test split were seen during fitting. Because the output
schema is identical, every router in `src/routers/` runs on either dataset unchanged — the only
dataset-specific objects are the cost proxy, the benchmark grouping, and the file path, all three
isolated in `DatasetSpec`.

The two pipelines also share most of their *design*: the same LLM-fingerprinting recipe, the same
seed-42 partitioning discipline, the same assembly and verification steps. §2 describes what they
share, §3 lists the handful of real differences, §4 tracks item counts from the original release to
the processed dataset, and §5–§6 give each pipeline's particulars.

---

## 2. The common approach

### 2.1 Stage 1 — Acquire the upstream release

Both datasets start from a HuggingFace dataset repo, fetched by a `download_data.py` that skips
anything already on disk and is safe to re-run. Neither pipeline re-runs the underlying *evaluation*
— the correctness labels are taken as published.

### 2.2 Stage 2 — Behavioural LLM fingerprinting

This is the step that makes routing over *unseen* LLMs possible. An LLM is represented not by
metadata but by **what it writes**:

1. Fix a probe set of **64 questions**.
2. Pull that model's generated output for each probe question.
3. Encode each output with **`Qwen/Qwen3-Embedding-4B`**, truncated to **32 dims** via Matryoshka
   representation learning (`truncate_dim=32`), L2-normalised, fp16.
4. **Concatenate** the 64 × 32 values in probe order → one **2048-d** row per model.

This recipe is implemented once, in `src/fingerprints/` (`build_fingerprints`), not per dataset —
see [`method.md`](method.md) for what consumes its
output. Building a fingerprint is two stages, always in that order:

- **`src/data_pipeline/fingerprints/collect_answers.py --dataset {router-eval,embedllm}`**
  gathers each model's raw answers to the probe set and persists them to
  `data/fingerprints/answers/<dataset>/<model_key>.json` (gitignored, regenerable). RouterEval's
  models can't be re-run, so this downloads their existing lighteval generations; EmbedLLM's
  answers are generations the authors ran against the probe set, committed in
  `data/embedllm/raw/model_outputs/` — see §6.2.
- **`build_fingerprints.py --dataset {router-eval,embedllm}`** reads those answers back and
  encodes them, never touching a network or a generation model. It writes the canonical
  `data/fingerprints/<dataset>/fingerprints.pkl`, and for RouterEval also a legacy-schema remap
  at `gsm8k_model_embeddings.pkl` so `build_llm_perf_dataset.py` needs no changes.

Missing probe answers leave that probe's 32-d slot as **explicit zeros** — a model with `N`
missing answers gets status `missing_probes:N`, a model whose answers were entirely unreachable
gets `missing`, and anything else is recorded verbatim as `error: <message>`. See §7 for how
each dataset fares.

**Both datasets are now probed with the exact same 64 questions** —
`data/fingerprints/probes/gsm8k_seed42_n64.csv` (tracked in git; §2.2's step 1 above, frozen).
GSM8K is the only probe source usable for *both*: it is the only RouterEval task whose
lighteval zips hold generated text rather than per-choice log-likelihoods (every other old-
leaderboard task — arc, hellaswag, mmlu, truthfulqa, winogrande — stores log-probability arrays),
and RouterEval's 5,000-model pool cannot be re-run to produce text for a different question set.
So the probe set had to be RouterEval's GSM8K questions, frozen once, with EmbedLLM's models
re-generated against them.

Because the recipe, the encoder, the dimensionality, *and the probe set itself* are identical
across datasets, LLM-side geometry is genuinely comparable between them.

### 2.3 Stage 3 — Cold-cold partitioning, seed 42

Both datasets are split three ways along **both** axes, with `numpy.random.default_rng(42)`:

- **Query splits** are pairwise disjoint sets of questions.
- **LLM splits** are pairwise disjoint sets of models.

So the test block pairs unseen queries with unseen LLMs. Fitting uses `train`, hyperparameter
selection uses `val`, and `test` is touched only for reporting. Who *owns* each split differs
between the datasets (§3).

### 2.4 Stage 4 — Assembly into the shared schema

| Key | Contents |
|---|---|
| `dataset["queries"][split]["embeddings"]` | `(Nq, d_q)` float32 |
| `dataset["queries"][split]["task_ids"]` | `(Nq,)` object — benchmark label per query |
| `dataset["queries"][split]["original_indices"]` | `(Nq,)` int64 — row in the source array |
| `dataset["llms"][split]["embeddings"]` | `(Nl, 2048)` float32 — behavioural fingerprint |
| `dataset["llms"][split]["names"]` | `(Nl,)` object — model identifiers |
| `dataset["llms"][split][<index field>]` | `(Nl,)` int — column in the full score matrix |
| `dataset["scores"][q_split][l_split]` | `(Nq, Nl)` uint8 — binary correctness, **all 9 blocks** |
| `dataset["metadata"]` | dict — sources, split sizes, seed, git commit, timestamps |

All nine (query-split × LLM-split) blocks are materialised, not just the diagonal, so cross-split
protocols (e.g. train queries against test LLMs) need no re-derivation.

### 2.5 Stage 5 — Verification and provenance

Both builds assert before writing: query-split and LLM-split leakage checks, index-alignment checks
between independently-sourced arrays, and shape/coverage audits. `metadata` records the source file
paths, the partition seed, and the git commit, and each build emits a markdown report next to the
pickle.

---

## 3. Where the two pipelines differ

| | **router-eval** | **embedllm** |
|---|---|---|
| Upstream repo | `linggm/RouterEval` | `RZ412/EmbedLLM` |
| Query embeddings | RoBERTa, 768-d — **downloaded**, not computed here | the EmbedLLM paper's `question_embeddings.pth`, 768-d, unit-norm — **downloaded** |
| Probe questions for fingerprints | the shared 64-question set, `data/fingerprints/probes/gsm8k_seed42_n64.csv` — **identical on both sides** (§2.2) | same |
| How each model answers the probes | already on record: sliced from that model's own lighteval GSM8K zip | read from the shipped `raw/model_outputs/<model_id>.json` (§6.2) — these already answer the shared probe set, verified by probe-position alignment; a GPU regeneration path exists but isn't the default |
| Extra LLM embedding | — | `embeddings_mf`, the paper's trained matrix-factorisation model embedding |
| **Query splits** | **inherited** from RouterEval's per-benchmark `split_index` | **generated here**: fractions 0.60 / 0.10 / rest, seed 42 |
| **LLM splits** | **generated here**: 0.8 / 0.1 / rest, seed 42 | **generated here**: 0.65 / 0.15 / rest, seed 42 |
| Model eligibility | name must contain a parseable size token | model output JSON must exist on disk |
| Binarisation | `score >= 0.95 → 1` | none — the CSV `label` column is already binary |
| Cost-proxy source | `size_strings` stored in `model_partition.pkl` (a build artifact) | re-parse HF model names + a 15-entry manual lookup |
| Benchmark label per query | 6 router keys (MMLU subtasks keep their own IDs) | 80+ category strings straight from the label CSVs |

---

## 4. Item counts — original → processed

### 4.1 router-eval

| Item | Original release | Kept | Processed splits (train / val / test) |
|---|---:|---:|---|
| LLMs | 5,000 | 3,749 (75.0%) — name carries a size token | **2,999 / 374 / 376** |
| LLMs dropped | — | 1,251 (25.0%) — no parseable size | — |
| Benchmark tasks | 62 | 62 (all) — grouped into 6 router keys | — |
| Questions | 28,659 | 28,659 (none dropped) | **22,906 / 2,859 / 2,894** |
| Query embedding dim | 768 (RoBERTa) | 768 | 768 |
| LLM embedding dim | — (built here) | 2,048 | 2,048 |

The 62 leaderboard tasks map onto the 6 router keys as 5 stand-alone benchmarks plus 57
`harness_hendrycksTest_*` subtasks collapsed into `mmlu`. No question is discarded: the query splits
are RouterEval's own, and they cover the full question set.

### 4.2 embedllm

| Item | Original release | Kept | Processed splits (train / val / test) |
|---|---:|---:|---|
| LLMs listed (upstream `model_order.csv`; our roster `model_list.csv`) | 107 | 107 valid — output JSON present for all | **70 / 16 / 21** |
| LLMs dropped | — | 0 | — |
| Queries (`question_order.csv`) | 35,673 | 35,673 (none dropped) | **21,404 / 3,567 / 10,702** |
| Label rows (train+val+test CSV) | ~3.8 M | scattered into a dense 35,673 × 107 matrix | coverage ≈100%, no imputation |
| Probe queries for fingerprints | — (built here) | 64 (the shared GSM8K set, `gsm8k_seed42_n64`) | not part of any routing split |
| Query embedding dim | 768 (paper's) | 768 | 768 |
| LLM embedding dim | — (built here) | 2,048 (+ `embeddings_mf`) | 2,048 |
| Query categories | 80+ | 80+ | 80+ |

> The fingerprint array has **107** rows, one per listed model, none all-zero (§6.2). All **107**
> are partitioned, giving 70 + 16 + 21. `task_id` (the dense 0..106 re-index `model_list.csv`
> and `AnswerStore` use) and `model_id` (the original 0..111 numbering with gaps, used by
> `raw/model_outputs/<model_id>.json` filenames) are different id spaces — see §6.2.
>
> Every committed `experiments/embedllm/*` result uses this 107-model roster.

### 4.3 router-eval — per-task question counts

| Router key | Leaderboard task(s) | Questions | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| `arc` | `harness_arc_challenge_25` | 1,172 | 937 | 117 | 118 |
| `gsm8k` | `harness_gsm8k_5` | 1,319 | 1,055 | 132 | 132 |
| `hellaswag` | `harness_hellaswag_10` | 10,042 | 8,033 | 1,004 | 1,005 |
| `harness_truthfulqa_mc_0` | `harness_truthfulqa_mc_0` | 817 | 653 | 82 | 82 |
| `winogrande` | `harness_winogrande_5` | 1,267 | 1,013 | 127 | 127 |
| `mmlu` | 57 × `harness_hendrycksTest_*` | 14,042 | 11,215 | 1,397 | 1,430 |
| **Total** | **62 tasks** | **28,659** | **22,906** | **2,859** | **2,894** |

> "Router key" is the build-time grouping key in `build_llm_perf_dataset.py::TASK_MAP`. The
> `task_id` actually stored per query is the leaderboard task name — for MMLU, the individual
> `harness_hendrycksTest_*` subtask — and `ROUTER_EVAL_GROUPS` in `src/eval/spec.py` re-derives the
> six reporting groups (`arc`, `gsm8k`, `hellaswag`, `winogrande`, `truthfulqa`, `mmlu`) from those
> IDs at evaluation time.

### 4.4 Score blocks

Both datasets materialise all nine blocks, `uint8` in {0, 1}.

**router-eval** — mean correctness ≈ 0.56

| | LLM train (2,999) | LLM val (374) | LLM test (376) |
|---|---|---|---|
| **Q train** (22,906) | 0.5632 | 0.5576 | 0.5565 |
| **Q val** (2,859) | 0.5640 | 0.5591 | 0.5575 |
| **Q test** (2,894) | 0.5696 | 0.5635 | 0.5632 |

**embedllm** — mean correctness ≈ 0.41 (measured on an earlier 102-model version of the roster;
the per-block means of the current 70 / 16 / 21 split are printed by
`build_perf_dataset.py` when the bundle is built).

The two datasets sit at markedly different base rates (≈0.56 vs ≈0.41). Three things differ at once
— the model pools, the question mixes (router-eval's leaderboard mean is 0.5643 across all 62 tasks,
with winogrande at 0.75 and hellaswag at 0.60), and the label definition (§3: router-eval binarises
at 0.95, EmbedLLM takes the published label) — so the gap should not be attributed to any one of
them without a dedicated comparison.

---

## 5. router-eval particulars

Upstream: **`linggm/RouterEval`**. Pipeline, in execution order:

### 5.1 `download_data.py`

Fetches three zips and extracts only the members needed:

| Archive | Extracted member | Contents |
|---|---|---|
| `leaderboard_score.zip` | `leaderboard_old.pkl` | 5,000 models × 62 tasks, `{model: [...], data: {task: {correctness}}}` |
| `leaderboard_embed.zip` | `leaderboard_old_embed/RoBERTa_result.pkl` | 768-d query embeddings per task |
| `router_dataset.zip` | the six `<task>_router_dataset.pkl` files used | one file per benchmark, with `split_index`, `embedding`, `prompt` |

### 5.2 `partition_models.py` — the LLM pool

The leaderboard's 5,000 entries are HuggingFace model IDs. A regex keeps only those whose **name
carries a parseable size token** (`7B`, `13b`, `1.5B`, `248M`, `4x7B`, …) — 3,749 of 5,000. The kept
subset is shuffled with seed 42 and cut 80 / 10 / 10 into 2,999 / 374 / 376.

The output `model_partition.pkl` stores `subset.{indices, names, size_strings}` (indices being
positions in the original 5,000-model array) and `splits.{train, val, test}`. The size filter is not
cosmetic: `size_strings` is exactly what `src/cost_proxy.py` later converts to parameter counts, so
**the cost axis of every cost-aware experiment is fixed at this stage** and coverage is 100% by
construction.

### 5.3 Behavioural fingerprints (§2.2's shared pipeline)

For each of the 3,749 models: `RouterEvalLightevalSource` (`src/fingerprints/sources/
router_eval.py`) downloads its per-model lighteval GSM8K zip from the same HF repo, reads the
`harness|gsm8k|5` predictions, and slices them at the frozen probe set's `gsm8k_index`
positions — 64 questions sampled once with `numpy.random.default_rng(42)` and frozen in
`data/fingerprints/probes/gsm8k_seed42_n64.csv`. `collect_answers.py --dataset
router-eval` runs this and persists the raw answers; `build_fingerprints.py --dataset
router-eval` applies the §2.2 recipe over them and writes both the canonical
`data/fingerprints/router-eval/fingerprints.pkl` and the legacy-schema
`gsm8k_model_embeddings.pkl` (a `(3749, 2048)` array plus per-model status) that
`build_llm_perf_dataset.py` reads.

**All 3,749 models returned `ok`** in the run behind the paper (per-model status is written to
`data/router-eval/processed/gsm8k_encode_stats.json`: 3,749 `ok`, 0 `missing`, 0 `error`), so this
dataset contains no zero rows despite the zero-init mechanism. A re-run extracts the same 64 probe
texts per model — checked for 5 models: the frozen probe prompts matched each zip's own
`full_prompt` at the recorded index, and the extracted answers matched a from-scratch
re-extraction byte for byte. The encoded fingerprints match up to the floating-point
nondeterminism of the fp16 encoder (batching, library versions), so they are close to, not
bit-identical with, the ones behind the committed results.

### 5.4 `build_llm_perf_dataset.py` — assembly

1. **Resolve the MMLU concatenation order.** The 57 `harness_hendrycksTest_*` tasks must be
   concatenated in the same order RouterEval used when it produced `mmlu_router_dataset.pkl`, or
   every MMLU index would be wrong. Rather than assume, `resolve_mmlu_order` *tests* candidate
   orderings — alphabetical, then score-file insertion order — by concatenating the RoBERTa
   embeddings under each and checking a 200-row sample against the router_dataset's own pre-split
   embeddings. **Alphabetical matched**; the build aborts if neither does.
2. **The index-alignment oracle.** For every task and split, the build slices the RoBERTa embeddings
   with the router_dataset indices and asserts them `allclose` (atol 1e-4) against the
   router_dataset's stored `<split>_embed`. A mismatch raises and stops the build. This assertion is
   the reason query rows and score rows can be trusted to be co-indexed — the property that would
   silently invalidate every experiment if it broke.
3. **Query splits** come from each benchmark's `split_index`; tasks are concatenated in sorted-key
   order (`arc, gsm8k, harness_truthfulqa_mc_0, hellaswag, mmlu, winogrande`). MMLU global indices
   are decomposed back into `(subtask name, local index)` so each MMLU query keeps its own
   `task_id`.
4. **Binarisation** at `score >= 0.95`. A pre-binarisation audit found only
   `harness_truthfulqa_mc_0` carried continuous scores (3.8 M unique values); the other five tasks
   were already {0, 1}.
5. **LLM columns.** The score matrices are `(Nq, 5000)`; each split selects its columns by
   `indices_in_5000`, and a reverse map `idx5k_to_3749` picks the matching fingerprint rows.
6. **Leakage assertions** on query `original_indices` per task and on LLM positions, then the nine
   score blocks are written, plus `processed/llm_perf_dataset_report.md`.

---

## 6. embedllm particulars

Upstream: **`RZ412/EmbedLLM`** supplies the queries, correctness labels and query embeddings.
Two inputs are **this repository's own data**, committed rather than downloaded:

- `data/embedllm/raw/model_list.csv` — the 107-model roster: `task_id` (dense 0..106), upstream
  `model_id`, display name, HuggingFace id, and whether the model was run quantized.
- `data/embedllm/raw/model_outputs/<model_id>.json` — each model's greedy generations on the 64
  shared GSM8K probes, in `probe_index` order. Upstream EmbedLLM releases no generations.

### 6.1 `download_data.py`

Fetches only what the pipeline reads: `question_order.csv` (row → `prompt_id` for all 35,673
queries), `train.csv` / `val.csv` / `test.csv` (the ~3.8 M `(prompt_id, model_id, label,
category)` rows), `question_embeddings.pth` (the EmbedLLM paper's 768-d query embeddings), and
`optimal_mf_model_embeddings.pth` (its trained MF model embedding, attached to the bundle but used
by no router).

### 6.2 Behavioural fingerprints, then `encode_outputs.py` Phases B–D (GPU for encoding only)

Fingerprinting runs in §2.2's shared pipeline. The committed `model_outputs/<model_id>.json`
files answer the shared GSM8K probe set (`gsm8k_seed42_n64`) in `probe_index` order — verified by
probe-position alignment (a token unique to probe *i*'s question appears in the
matching answer 94.4% of the time, vs 2.5–4.7% at wrong offsets, across all 107 files).

- **`collect_answers.py --dataset embedllm`** defaults to `--source local`
  (`EmbedLLMLocalOutputs` in `src/fingerprints/sources/embedllm.py`), which reads each model's
  answers straight from the shipped `raw/model_outputs/<model_id>.json`. **Tier A — no GPU, no
  network, all 107 models in seconds.** The file is keyed by `model_id`, not the `task_id` that
  `AnswerStore` and `model_list.csv` otherwise use — the two ids diverge for 78 of 107 rows, so a
  `task_id → model_id` map from `model_list.csv` is mandatory. Each answer gets the same
  `_truncate_at_stop` + `.strip()` normalization described below for the generate path, so the two
  sources can never diverge in how they clean generated text.

  `--source generate` regenerates instead, with `EmbedLLMGenerator`: the generation script
  of the original EmbedLLM fingerprinting code, ported with four deliberate changes: no
  `truncation=True, max_length=512` (a 5-shot GSM8K prompt runs ~700-1000 tokens and HF's
  right-truncation would delete the question being asked); prompts come from the probe set's
  `prompt` column rather than a dataset-specific CSV; decoding is greedy with
  `max_new_tokens=256` to match lighteval's own `harness|gsm8k|5` generation config; and each
  decoded answer is truncated at the next `"\nQuestion:"` or blank-line boundary
  (`_truncate_at_stop`) — checked directly against the cached RouterEval predictions, a real
  lighteval answer stops at its own `#### <n>` line and never rambles into a fabricated next
  turn. GPU; one model per invocation is the intended entry point — `--model-idx N` for a job
  array — since the roster spans 7B-72B parameters and nothing keeps more than one model
  resident. It is a port of the script that produced the committed `model_outputs/`, kept for
  provenance and for the day the probe set changes; it is not the default.

  **Roster: all 107 `model_list.csv` rows, unfiltered.** `model_list.csv` is already the 107
  valid models — no separate exclusion list is applied.
- **`build_fingerprints.py --dataset embedllm`** encodes those answers per §2.2 over all 107
  `model_list.csv` rows (in CSV order) and writes `data/fingerprints/embedllm/fingerprints.pkl`.
  **This step still needs `Qwen/Qwen3-Embedding-4B` and is Tier B.** Against the shipped
  `--source local` answers this produces **105 `ok` + 2 `missing_probes` + 0 `missing`** (see
  §7.1) — every model now gets a row that reflects a real answer set.
- **`encode_outputs.py`** then only runs what used to be **Phases B–D**, loading the fingerprint
  matrix from that artifact instead of computing it:
  - **Phase B — query embeddings.** Unchanged. The script can encode prompts itself, but the
    production run passed `--hf-query-embeddings data/embedllm/raw/question_embeddings.pth`, so the 35,673 × 768 unit-normalised embeddings are the **paper's
    own**, recorded as `query_embed_source="hf_precomputed"`. Row order follows
    `question_order.csv`; the file carries no IDs, so `query_ids` stays `None`.
  - **Phase C — splits.** Unchanged. Models: the **valid** ones only (`stats != "missing"`) — with
    `--source local` that is now all 107, split at fractions 0.65 / 0.15 / rest with seed 42 →
    **70 / 16 / 21**. Queries: all 35,673 row indices, split at 0.60 / 0.10 / rest →
    **21,404 / 3,567 / 10,702**. (The script's own docstring says "65/15/20" and "60/10/30";
    those are the nominal fractions, and the realised counts are the ones above.) See §4.2 for
    which committed `experiments/embedllm/*` results are current against this 107-model roster.
  - **Phase D** — atomic save to `processed/embedllm_fingerprints.pkl`.

### 6.3 `build_perf_dataset.py` — assembly (CPU)

1. Load the fingerprint pickle (embeddings, query embeddings, splits, stats).
2. Map `question_order.csv` rows → `prompt_id`, and `prompt_id` → row.
3. Read the three label CSVs for `(prompt_id, model_id, label, category)` only — the `prompt` field
   contains embedded newlines, so pandas' Python engine is required.
4. Scatter the labels into a dense `S` of shape **35,673 × 107** via vectorised index lookup.
   Coverage is ≈100%: every (query, valid model) pair has a label, so no imputation is needed.
5. Derive each query's `task_id` from its benchmark `category` string — 80+ distinct values (57 MMLU
   subtopics, 15 GPQA variants, asdiv/gsm8k/mathqa, medmcqa, logiqa, piqa, social_iqa,
   truthfulqa_mc1).
6. Emit the nine blocks, attaching to each LLM split both the Qwen fingerprint (`embeddings`) and
   the paper's MF embedding (`embeddings_mf`), indexed by `model_id`.

---

## 7. Caveats

### 7.1 embedllm — fingerprint completeness (affects experiments)

This table describes `collect_answers.py --dataset embedllm --source local` (the default) against
the shipped `raw/model_outputs/`, answered against the shared probe set (§2.2, §6.2):

| Status | Count | Effect |
|---|---:|---|
| `ok` — all 64 probe outputs present | 105 | clean fingerprint |
| `missing_probes:N` — 1..63 outputs absent | 2 (`task_id` 52: 53 missing, 100: 2 missing) | fingerprint kept, **zero-padded** in the absent 32-d slots |
| `missing` — no output JSON at all | 0 | — |

`_normalize_answer` (§6.2) strips before truncating specifically to avoid a third case: a decoded
answer whose *entire* content is thrown away because it opens with a blank line. `task_id 75`
(`microsoft/Orca-2-13b`) is the concrete instance — all 64 of its committed answers open with
`"\n\n"`, which a naive truncate-then-strip would treat as an immediate stop and empty every one
of them; strip-first correctly counts it as `ok`.

The 2 partially-padded models **are** among the 107 valid models and therefore appear in
train / val / test. Their fingerprints are systematically less informative than the 64 clean ones.
`src/fingerprints/encode.py` writes explicit zeros for a missing probe answer rather than
embedding a placeholder string. Worth checking before attributing an EmbedLLM
result to model geometry.

### 7.2 router-eval — what is deliberately not used

RouterEval ships a candidate-pool design of its own: `easy` / `hard` regimes × `all_strong`,
`all_weak`, `strong_to_weak` strategies, at k ∈ {3, 5, 10, 100, 1000}. **None of it is used here.**
This project takes only `split_index` and `embedding` from `router_dataset`, and builds its own
open, cold-model candidate pool from the 5,000-model leaderboard. A reader familiar with RouterEval
should not assume those pools are in play.

### 7.3 Other caveats

- **router-eval model coverage.** 1,251 of 5,000 models (25%) are dropped for lacking a parseable
  size token. The kept pool is therefore biased toward models that advertise their size in the name.
- **Fingerprints are GSM8K-only, for both datasets now.** One task's generations stand in for
  behaviour across all six router-eval benchmarks (and, for EmbedLLM, across the 80+ categories
  its own labels span) — a deliberate consequence of using one probe set for both datasets (§2.2).
- **Cost is a size proxy.** Both `src/cost_proxy.py` and `src/embedllm_cost_proxy.py` return
  parameter count in billions, not $/token, and the vector is derived at evaluation time — it is not
  stored in the pickle. `bigcode__octocoder` (`task_id` 77) has no parseable size token and was
  added to `_MANUAL_SIZES_B` (15.5B, StarCoder-based). It lands in the test split, and the eval
  harness asserts full cost-vector coverage on val and test.
- **embedllm `query_ids` is `None`.** Recover absolute prompt IDs by joining
  `queries[split]["original_indices"]` against `raw/question_order.csv`.
- **Number provenance.** The figures in this document were read back from the build artifacts and
  the build scripts' own reports.

---

## 8. Reproduction

See the README's [tier-B section](../README.md#tier-b--full-rerun-from-raw-data) for the ordered
commands. Every stage is resumable or idempotent; `collect_answers.py` skips any model already on
disk unless `--force`, and `build_fingerprints.py` checkpoints and accepts `--resume`.

---

## 9. Downstream

- **Fingerprint building.** `src/fingerprints/` (the shared recipe) and
  `src/data_pipeline/fingerprints/` (the `collect_answers.py` / `build_fingerprints.py` pipeline
  scripts) are consumed by `build_llm_perf_dataset.py` (router-eval, via the legacy-schema
  `gsm8k_model_embeddings.pkl` remap) and `encode_outputs.py` (embedllm, directly). Neither of
  those two assembly scripts changed its own output schema — only where the `["embeddings"]`
  values come from.
- **Re-embedding.** `src/data_pipeline/*/encode_embeddings_mlpdot.py` consumes the built `llm_perf_dataset.pkl` and emit drop-in replacement bundles with only
  `["embeddings"]` swapped — see [`method.md`](method.md)
  for the MLPDot case.
- **Loading.** `src/eval/spec.py` defines `ROUTER_EVAL_SPEC` and `EMBEDLLM_SPEC`;
  `src.eval.load_splits(spec)` returns the split arrays plus the cost vector.
- **Auto-generated build reports.** `data/router-eval/processed/llm_perf_dataset_report.md` and
  `data/router-eval/processed/model_partition_report.md` (gitignored, written by the build).

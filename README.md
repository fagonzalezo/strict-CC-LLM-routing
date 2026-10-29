# Strict Cold-Start Open-Pool LLM Routing

Code, data and result artifacts for the paper

> **Strict Cold-Start Open-Pool LLM Routing with Behavioral Fingerprints and Dual-Kernel Memory**
> Jose G. Moreno and Fabio A. González

The paper studies **Cold–Cold routing**: picking which LLM should answer a query when both the
query and every candidate LLM are unseen at training time. A new LLM is onboarded once, from its
ungraded answers to a fixed set of 64 probe questions (its *behavioral fingerprint*). The method,
**BF-DKR** (Behavioral Fingerprint Dual-Kernel Router), scores a query–LLM pair with a
kernel-density posterior over a learned query×LLM space. It routes to models absent from its
training memory without retraining.

The repository contains:

- the two Cold–Cold datasets, **RouterEval-CC** and **EmbedLLM-CC**: the authors' new data plus
  scripts that rebuild everything else from the public upstream releases;
- BF-DKR and the six baselines (kNN, MLPDot, DropoutNet, CLCRec, AGNN, FSGNN);
- every experiment in the paper, with its committed result JSON;
- one generator per paper table and figure.

**Two ways to reproduce:**

- **Tier A** re-derives every number in the paper from the committed results in seconds, on a
  laptop.
- **Tier B** reruns the experiments from the raw upstream data and needs a GPU.

---

## Contents

```
src/                     all code
  routers/               BF-DKR (dmkdmr.py), MLPDot, kNN, DropoutNet, CLCRec, static policies
  baselines/coldrec/     vendored ColdRec (AGNN, FSGNN), MIT
  fingerprints/          the behavioral-fingerprint recipe (probe set, answers, encoding)
  eval/                  dataset-independent evaluation harness (splits, metrics, selection)
  data_pipeline/         upstream data -> dataset bundles (router_eval/, embedllm/, fingerprints/, metadata/)
  experiments/           one directory per experiment: runners and aggregators
  paper/                 one generator per paper table/figure + reproduce_all.sh
  tests/                 layout, tier-A and data-integrity tests
experiments/             results only: <dataset>/<experiment>/{README.md, results/*.json}
data/                    the authors' new data (below); upstream downloads land here too (gitignored)
docs/                    dataset_construction.md (how the datasets are built), method.md (paper -> code)
```

**Naming.** The paper and the code use different names for the same things:

| Paper | Code and directories |
|---|---|
| BF-DKR | `DMKDMRRouter` (`src/routers/dmkdmr.py`); results under `dmkdmr_encoded/`; directories `bfdkr-*` |
| RouterEval-CC, EmbedLLM-CC | `router-eval`, `embedllm` (`router_eval` in Python package paths) |
| σ_q, σ_m | `sigma_x`, `sigma_y` |

`cost_auc` (the area under the per-query budget-cap curve) is the primary metric, and
AIQ = `cost_auc` − floor `cost_auc`. Cost is parameter count in billions.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pytest   # tier A
pip install -r requirements.txt          # tier B (add a CUDA build of torch for GPU runs)
```

Run every command from the repository root. Scripts find the root on their own; set
`BFDKR_ROOT` to override.

---

## Tier A — re-derive the paper from committed results

```bash
bash src/paper/reproduce_all.sh     # tables -> paper/tables/*.txt, Fig. 3 -> paper/figures/
python -m pytest src/tests -q       # layout, generators, data integrity
```

Each generator prints a `read: <path>` line for every artifact it uses, so any number in the
paper can be traced to a JSON file. `gen_tab_main.py` exits non-zero if
AIQ = `cost_auc` − floor fails for any row.

| Paper item | Generator (`src/paper/`) | Reads (`experiments/…/results/`) |
|---|---|---|
| Table II — dataset statistics | `gen_tab_data.py` (tier B: needs the bundles) | `data/*/processed/llm_perf_dataset.pkl` |
| Table III — all routers, both datasets; §VI-A numbers | `gen_tab_main.py` | `*/all-methods-comparison`, `router-eval/bfdkr-ablation/full`, `router-eval/bfdkr-sweep` |
| Fig. 3 — RouterEval-CC budget-cap curves | `gen_fig_routereval_cost_curve.py` | `router-eval/bfdkr-sweep` |
| Table IV, Table S4 — component ablation | `gen_tab_ablation.py` | `*/bfdkr-ablation` |
| Table V, Table S3 — static policies | `gen_tab_static.py` | `*/static-policy-baseline` |
| Table VI — fingerprint vs. metadata | `gen_tab_metadata.py` | `*/metadata-ablation`, `*/static-policy-baseline` |
| Table VII — validation-based selection audit | `gen_tab_selection.py` | `*/knn-baseline`, `*/mlpdot-baseline` |
| Tables S1, S2 — per-group results | `gen_tab_pergroup.py` | `router-eval/bfdkr-ablation`, `embedllm/bfdkr-sweep` |

Every experiment directory has a README. It covers what the experiment tests, the commands that
regenerate it, and which result files feed the paper.

| Experiment | RouterEval-CC | EmbedLLM-CC |
|---|---|---|
| kNN baseline | [`knn-baseline`](experiments/router-eval/knn-baseline) | [`knn-baseline`](experiments/embedllm/knn-baseline) |
| MLPDot baseline (also selects BF-DKR's encoder) | [`mlpdot-baseline`](experiments/router-eval/mlpdot-baseline) | [`mlpdot-baseline`](experiments/embedllm/mlpdot-baseline) |
| BF-DKR seed × objective × k sweep | [`bfdkr-sweep`](experiments/router-eval/bfdkr-sweep) | [`bfdkr-sweep`](experiments/embedllm/bfdkr-sweep) |
| BF-DKR component ablation | [`bfdkr-ablation`](experiments/router-eval/bfdkr-ablation) | [`bfdkr-ablation`](experiments/embedllm/bfdkr-ablation) |
| DropoutNet, CLCRec | [`dropoutnet-clcrec-baselines`](experiments/router-eval/dropoutnet-clcrec-baselines) | [`dropoutnet-clcrec-baselines`](experiments/embedllm/dropoutnet-clcrec-baselines) |
| AGNN, FSGNN | [`coldrec-baselines`](experiments/coldrec-baselines) (both datasets) | ← |
| Table III aggregation | [`all-methods-comparison`](experiments/router-eval/all-methods-comparison) | [`all-methods-comparison`](experiments/embedllm/all-methods-comparison) |
| Static policies | [`static-policy-baseline`](experiments/router-eval/static-policy-baseline) | [`static-policy-baseline`](experiments/embedllm/static-policy-baseline) |
| Fingerprint vs. metadata | [`metadata-ablation`](experiments/router-eval/metadata-ablation) | [`metadata-ablation`](experiments/embedllm/metadata-ablation) |

---

## The datasets

Both datasets re-split a public routing benchmark **Cold–Cold**: queries *and* LLMs are pairwise
disjoint across train/validation/test, with seed 42. Each LLM also gets a 2048-d behavioral
fingerprint and a parameter-count cost.

| | Queries train / val / test | LLMs train / val / test | Upstream (license) |
|---|---|---|---|
| RouterEval-CC | 22,906 / 2,859 / 2,894 | 2,999 / 374 / 376 | [`linggm/RouterEval`](https://huggingface.co/datasets/linggm/RouterEval) (MIT) |
| EmbedLLM-CC | 21,404 / 3,567 / 10,702 | 70 / 16 / 21 | [`RZ412/EmbedLLM`](https://huggingface.co/datasets/RZ412/EmbedLLM) (Apache-2.0) |

**New data in this repository** (`data/`, committed):

| File | What it is |
|---|---|
| `data/fingerprints/probes/gsm8k_seed42_n64.csv` (+ `.meta.json` with sha256) | The 64 GSM8K probe questions, with their 5-shot prompts, used to fingerprint every LLM on both datasets |
| `data/embedllm/raw/model_outputs/<model_id>.json` (107 files) | Greedy generations of each EmbedLLM-CC model on the 64 probes, run by the authors (`collect_answers.py --source generate` is a port of the generation script); upstream EmbedLLM releases no generations |
| `data/embedllm/raw/model_list.csv` | The 107-model EmbedLLM-CC roster: dense `task_id`, upstream `model_id`, HF id, quantization |
| `src/embedllm_cost_proxy.py::_MANUAL_SIZES_B` | Manual parameter counts for the 15 EmbedLLM models whose name has no size token |

**Not shipped, but rebuilt by tier B:**

- the upstream correctness labels and query embeddings;
- RouterEval-CC's probe answers: each model's own GSM8K generations from the Open LLM
  Leaderboard (v1) details, as redistributed by RouterEval, downloaded per model;
- the fingerprint matrices, the LLM splits, and the assembled dataset bundles.

All steps are deterministic given seed 42, except fingerprint encoding (see
[Reproducibility notes](#reproducibility-notes)).
[`docs/dataset_construction.md`](docs/dataset_construction.md) gives the full upstream →
derived accounting.

---

## Tier B — full rerun from raw data

The paper's runs used one GPU per job. Every stage below is a plain command, so on a cluster each
loop becomes a job array. Stages are resumable or idempotent.

> Running a stage **overwrites** the committed JSON under `experiments/…/results/`. Compare your
> rerun with the reference using `git diff`, then regenerate the tables with
> `bash src/paper/reproduce_all.sh`.

### 1. Build RouterEval-CC

```bash
python src/data_pipeline/router_eval/download_data.py      # network: ~0.8 GB of zips
python src/data_pipeline/router_eval/partition_models.py    # CPU, seconds: 3,749 sized models -> 2,999/374/376
python src/data_pipeline/fingerprints/collect_answers.py --dataset router-eval \
    --model-partition data/router-eval/processed/model_partition.pkl \
    --leaderboard data/router-eval/raw/leaderboard_score/leaderboard_old.pkl
    # network: one GSM8K details zip per model (3,749 x ~3.4 MB, ~12.7 GB)
python src/data_pipeline/fingerprints/build_fingerprints.py --dataset router-eval \
    --model-partition data/router-eval/processed/model_partition.pkl \
    --leaderboard data/router-eval/raw/leaderboard_score/leaderboard_old.pkl
    # GPU, hours: Qwen3-Embedding-4B over 3,749 x 64 answers; --resume to continue
python src/data_pipeline/router_eval/build_llm_perf_dataset.py   # CPU, minutes
```

### 2. Build EmbedLLM-CC

```bash
python src/data_pipeline/embedllm/download_data.py          # network
python src/data_pipeline/fingerprints/collect_answers.py --dataset embedllm \
    --model-list data/embedllm/raw/model_list.csv           # CPU, seconds: reads the committed model_outputs/
python src/data_pipeline/fingerprints/build_fingerprints.py --dataset embedllm \
    --model-list data/embedllm/raw/model_list.csv           # GPU, minutes
python src/data_pipeline/embedllm/encode_outputs.py \
    --hf-query-embeddings data/embedllm/raw/question_embeddings.pth   # splits 70/16/21 and 21,404/3,567/10,702
python src/data_pipeline/embedllm/build_perf_dataset.py     # CPU, ~1 min
```

`python src/paper/gen_tab_data.py` now prints Table II.

### 3. Run the experiments

Run the steps in this order for `D=router-eval P=router_eval`, then for `D=embedllm P=embedllm`.
The EmbedLLM-CC ablation aggregator also reads the RouterEval-CC ablation.

```bash
# (a) baselines on the raw representation; MLPDot also fixes BF-DKR's encoder architecture
python src/experiments/$P/knn_baseline/run_knn_pooled_eval.py
python src/experiments/$P/mlpdot_baseline/run_mlp_dot_pooled_eval.py

# (b) BF-DKR's representation: frozen MLPDot encoders, one bundle per encoder seed (GPU)
for s in 0 1 2; do python src/data_pipeline/$P/encode_embeddings_mlpdot.py --seed $s; done

# (c) BF-DKR sweep: 3 seeds x {bce, pq_auc} x k in {128, 512, 2048, 4096}
for i in $(seq 0 23); do python src/experiments/$P/bfdkr_sweep/run_dmkdmr_encoded_eval.py --config-idx $i; done
python src/experiments/$P/bfdkr_sweep/build_comparison.py

# (d) component ablation (fixed k = 4096)
for s in 0 1 2; do python src/experiments/$P/bfdkr_ablation/run_ablation_encoded_eval.py --seed-idx $s; done
python src/experiments/$P/bfdkr_ablation/build_ablation_comparison.py

# (e) DropoutNet and CLCRec (9-config HP search, then 3 seeds)
python src/experiments/$P/dropoutnet_clcrec_baselines/run_dropoutnet_eval.py
python src/experiments/$P/dropoutnet_clcrec_baselines/run_clcrec_eval.py
python src/experiments/$P/dropoutnet_clcrec_baselines/build_comparison.py
```

Then, once per repository:

```bash
# (f) AGNN and FSGNN: HP search, finals and summaries -- exact loops in experiments/coldrec-baselines/README.md

# (g) Table III aggregation (self-checking); gates (h) and (i)
python src/experiments/router_eval/all_methods_comparison/build_seed_summary.py
python src/experiments/embedllm/all_methods_comparison/build_seed_summary.py
```

and again per dataset:

```bash
# (h) static policies: BF-DKR reference refit (saves P_test.npy), then S1/S2/S2b/S3
for s in 0 1 2; do python src/experiments/$P/static_policy_baseline/run_bfdkr_reference.py --seed-idx $s; done
python src/experiments/$P/static_policy_baseline/run_static_policies.py

# (i) fingerprint vs. metadata: variant bundles, encoding, 4 arms x 3 seeds
#     -- exact loops in experiments/$D/metadata-ablation/README.md
```

Finally, run `bash src/paper/reproduce_all.sh`.

Most runners accept `--smoke`, a tiny end-to-end run for checking an installation, and
`--device cpu|cuda`. `--help` on any script is authoritative. Run the scripts as files, not with
`python -m`.

---

## Reproducibility notes

- **Bit-exactness.** Fingerprints are encoded with Qwen3-Embedding-4B in fp16, so a rerun
  reproduces the probe *texts* exactly but the fingerprint *values* only up to floating-point
  nondeterminism (batching, GPU, library versions). Downstream numbers from a full tier-B rerun
  should therefore match the committed results within seed noise, not to the last digit.
  - On a fixed bundle, RouterEval-CC's BF-DKR refits are exact (drift 0.0000).
  - EmbedLLM-CC's Brent bandwidth search is not bit-reproducible: drift is at most 0.0024 cost
    AUC per seed. That is why its fingerprint reference reads 0.4984 in Tables VI and S3
    against 0.4987 in Table III.
- **Selection.** BF-DKR, kNN, MLPDot, DropoutNet and CLCRec are selected on validation `cost_auc`.
  AGNN and FSGNN are selected on pooled *test* `pq_auc`, which favours them; the paper says so.
- **Seeds.** All Cold–Cold splits use seed 42. Encoder and model seeds are 0, 1, 2.
- **Uncertainty.** Paired-bootstrap intervals exist only where both routers persisted per-query
  test predictions: BF-DKR vs MLPDot, the ablation arms, the static-policy arms, and the metadata
  arms. Other cross-method gaps are point estimates.
- **Legacy experiment IDs.** Code comments and some JSON keys (e.g. `anchor_exp26`) use the
  original study's experiment numbers:

  | ID | Directory |
  |---|---|
  | `exp_31` | `router-eval/knn-baseline` |
  | `exp_32` | `router-eval/mlpdot-baseline` |
  | `exp_37` | `router-eval/bfdkr-sweep` |
  | `exp_38` | `router-eval/bfdkr-ablation` |
  | `exp_39` | `router-eval/dropoutnet-clcrec-baselines` |
  | `exp_22` | `embedllm/knn-baseline` |
  | `exp_23` | `embedllm/mlpdot-baseline` |
  | `exp_25` | `embedllm/bfdkr-sweep` |

  Other numbers refer to earlier exploratory runs that are not part of this release. The
  pre-registered anchors (`anchor_exp*`) that Table VII compares against are stored in the
  baselines' `selection_audit.json`.

---

## License and citation

The code is released under the [MIT license](LICENSE). The vendored ColdRec code keeps its own
MIT license ([`src/baselines/coldrec/LICENSE`](src/baselines/coldrec/LICENSE)).

The derived datasets build on:
- RouterEval (MIT);
- EmbedLLM (Apache-2.0);
- GSM8K (MIT), the source of the probe questions;
- the Open LLM Leaderboard (v1) details, the source of RouterEval-CC's probe answers. These are
  downloaded from RouterEval, not redistributed here.

The committed EmbedLLM-CC probe answers are generations of the open-weight models listed in
`model_list.csv`.

If you use this code or the datasets, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{moreno2026strictcc,
  title   = {Strict Cold-Start Open-Pool {LLM} Routing with Behavioral Fingerprints and Dual-Kernel Memory},
  author  = {Moreno, Jose G. and Gonz{\'a}lez, Fabio A.},
  journal = {Under review},
  year    = {2026}
}
```

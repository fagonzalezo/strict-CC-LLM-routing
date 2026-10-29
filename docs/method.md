# BF-DKR in code

The paper (Section IV) is the reference for the method. This note maps it onto the code and
records the configuration that produced the committed results.

## Names

| Paper | Code |
|---|---|
| BF-DKR (Behavioral Fingerprint Dual-Kernel Router) | `DMKDMRRouter` in `src/routers/dmkdmr.py`; result dirs `dmkdmr_encoded/` |
| Representation encoder + temporary dot-product head; the **MLPDot** baseline | `MLPDotRouter` in `src/routers/mlp_dot.py` |
| Behavioral fingerprint $y_m$ (64 probes × 32-d, 2048-d) | `src/fingerprints/`, built by `src/data_pipeline/fingerprints/` |
| $\sigma_q$, $\sigma_m$ (query / LLM bandwidths) | `sigma_x`, `sigma_y` (also `σ_q` / `σ_l` in some outputs) |
| kNN baseline; "Encoded kNN" ablation arm | `src/routers/knn_baseline.py`; arm `knn_enc` |
| DropoutNet, CLCRec | `src/routers/dropoutnet.py`, `src/routers/clcrec.py` (reimplementations) |
| AGNN, FSGNN | `src/baselines/coldrec/` (vendored ColdRec) driven by `src/experiments/coldrec_baselines/run_coldrec.py` |
| Static policies S1, S2, S2b, S3 | `src/routers/static_policy.py` |

Every router follows one contract: `fit(...)`, then `predict(Q, L)` returns `P` of shape
`(n_queries, n_llms)` in `[0, 1]`, the estimated probability that each LLM answers each query
correctly. Cold-model routing is structural: the memory holds the *training* LLMs, and the
candidates are whatever LLM embeddings `predict` receives.

## Pipeline

1. **Representation initialization (paper §IV-B).**
   `src/data_pipeline/{router_eval,embedllm}/encode_embeddings_mlpdot.py --seed S` fits MLPDot at
   the architecture selected by `experiments/<dataset>/mlpdot-baseline` (read from its
   `results/selection_audit.json`), then discards the dot-product head. Its frozen towers $f_q$ and
   $f_m$ re-embed every query and LLM. The output
   `data/<dataset>/processed/mlpdot_encoded_seed{S}/llm_perf_dataset_mlpdot_encoded.pkl` is a
   drop-in copy of the dataset bundle with only the `["embeddings"]` arrays swapped. Runners point
   a `DatasetSpec` at it with `dataclasses.replace(get_spec(...), dataset_path=...)`.
2. **Dual-kernel memory posterior (paper §IV-C–E).** `DMKDMRRouter.fit` stores the encoded
   training queries, training LLMs and the sparse correctness matrix as its memory. It anchors
   both bandwidths at the median pairwise distance of 1,000-point subsamples, then selects the
   multipliers by nested bounded Brent search on the validation block. The objective is
   `pq_auc` or `bce`; the multiplier range is `(0.0625, 4.0)`; `tol = 0.05` in log units.
   `predict` evaluates the factorized posterior (paper Eq. 11) over the `k` nearest training queries.
3. **Selection (paper §V-C).** The sweep runner fits one router per (encoder seed, objective, k).
   `build_comparison.py` then selects per seed by validation `cost_auc`, with validation Brier as
   tie-break.

## Configuration behind the committed results

| | RouterEval-CC | EmbedLLM-CC |
|---|---|---|
| Encoder hidden width / output dim / layers | 256 / 128 / 2 | 128 / 32 / 2 |
| Encoder training | AdamW, lr 1e-3, wd 1e-4, dropout 0.1, ≤200 epochs, batch 256, checkpoint on val BCE | same |
| Encoder seeds | 0, 1, 2 (one encoded bundle each; BF-DKR's seed is tied to it) | same |
| `k` grid (sweep) | 128, 512, 2048, 4096 | same |
| Brent objective (headline) | `pq_auc` | `pq_auc` |
| Fixed-k protocol (ablation, per-group tables) | k = 4096 | k = 4096 (ablation), 2048 (per-group) |

## Ablation arms (`src/experiments/*/bfdkr_ablation/ablated_dmkdmr.py`)

Each knockout is a limiting case of the same estimator, so a delta is attributable to one
component:

| Arm | Limit | Paper row |
|---|---|---|
| `full` | BF-DKR as above, fixed k = 4096 | Full BF-DKR |
| `qk_off` | $\sigma_q \to \infty$: the k neighbours are averaged uniformly | No query kernel |
| `lk_off` | $\sigma_m \to 0^+$: each candidate borrows from its nearest memory LLM | Hard nearest LLM |
| `bw_off` | no Brent search; both bandwidths at their median anchors | No bandwidth search |
| `knn_enc` | plain kNN on the same encoded representation (external reference) | Encoded kNN |
| `full_ext` | `full` with the multiplier range widened to `(1/1024, 4.0)` (RouterEval-CC only; a robustness check, not in the paper) | — |

The $\sigma_m \to 0^+$ limit uses a kernel stabilized per candidate column. The column factor
cancels in the posterior, so for any finite bandwidth it equals the base kernel exactly.

## Determinism

On RouterEval-CC, refitting BF-DKR from the same bundle and seed reproduces the committed numbers
exactly (the static-policy and metadata-ablation reproduction gates record drift 0.0000). On
EmbedLLM-CC, Brent's search is not bit-reproducible across runs. The drift is at most 0.0024
cost_auc per seed, far below the seed standard deviation. This is why EmbedLLM-CC's `fp` reference
reads 0.4984 in Table VI and Table S3 against 0.4987 in Table III.

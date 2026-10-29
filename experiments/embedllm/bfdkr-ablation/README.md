# BF-DKR component ablation — EmbedLLM-CC

Knocks out BF-DKR's components one at a time, each as an exact limiting case of the same
estimator (arms in [docs/method.md](../../../docs/method.md)):
`full`, `qk_off` (no query kernel), `lk_off` (hard nearest LLM), `bw_off` (no bandwidth search),
`knn_enc` (plain kNN on the same encoded representation).

**Paper:** Table S4 and the §VI-C cross-dataset paragraph.

## Protocol
Fixed-k protocol: k = 4096 and the `pq_auc` objective for every arm, so memory starvation cannot
confound a knockout. Encoder seeds 0, 1, 2. Each arm is compared with `full` on the same seed by
a paired bootstrap over test queries: 1,000 resamples from one shared resample-index matrix,
with a 95% percentile interval.

## Run
Needs the MLPDot-encoded bundles; `build_ablation_comparison.py` also reads `bfdkr-sweep`
for a reproduction gate and `router-eval/bfdkr-ablation` for a cross-dataset readout.
```bash
for s in 0 1 2; do
  python src/experiments/embedllm/bfdkr_ablation/run_ablation_encoded_eval.py --seed-idx $s
done
python src/experiments/embedllm/bfdkr_ablation/build_ablation_comparison.py
```
Run the runner as a script, not with `python -m`: it imports its sibling `ablated_dmkdmr.py` by
bare name.

## Results
- `results/<arm>/seed{S}/`: `metrics.json`, `cost_curve.json`, `config_audit.json`, …
- `results/attribution_table.json`: 3-seed mean ± s.d. per arm (the table's `cost_auc` and ECE columns)
- `results/h1_query_kernel.json`, `h2_llm_kernel.json`, `h3_brent.json`, `h5_knn.json`: per-seed paired deltas and intervals against `full` (the Δ column)

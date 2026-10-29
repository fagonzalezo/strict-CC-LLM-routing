# kNN baseline — EmbedLLM-CC

Query-and-LLM k-nearest-neighbour collaborative filtering on the raw (unencoded) query
embeddings and fingerprints. A query's score for a candidate LLM averages the correctness of
its `k_q` nearest training queries on the candidate's `k_l` nearest training LLMs. This is a
memory-only router with no learned parameters.

**Paper:** Table III (kNN row; one deterministic run, so no seed axis) and Table VII.

## Protocol
Grid: `k_q` ∈ 1..4,096 (13 values) × `k_l` ∈ 1..70 (8 values) = 104 cells. Selection: highest
validation `cost_auc`, ties broken by lowest validation Brier. The test block is touched only
for reporting.

## Run (CPU)
```bash
python src/experiments/embedllm/knn_baseline/run_knn_pooled_eval.py
```

## Results
- `results/selected_metrics.json`: per-group test metrics of the selected cell (`group == "pooled"` is the Table III row)
- `results/selection_audit.json`: the full grid on validation, the selected cell, and
  `h2_val_test_agreement` (Spearman ρ of validation vs test `cost_auc` → Table VII)
- `results/sweep_metrics.json`, `cost_curve.json`, `reliability.json`, `cost_coverage.json`

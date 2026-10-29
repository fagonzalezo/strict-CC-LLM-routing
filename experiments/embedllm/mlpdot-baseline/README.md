# MLPDot baseline — EmbedLLM-CC

The two-tower encoder of paper §IV-B, with its dot-product head kept as the readout:
`sigmoid(α·z_qᵀz_m + β)`. Trained by per-pair binary cross-entropy on the train block, with the
checkpoint selected on validation BCE.

It plays two roles:
1. It is the **MLPDot** row of Table III and a router in Tables VI–VII.
2. Its **selected architecture** is the one BF-DKR's representation encoder uses:
   `src/data_pipeline/embedllm/encode_embeddings_mlpdot.py` reads it from
   `results/selection_audit.json`. This experiment must therefore run before the encoded bundles
   are built.

## Protocol
Grid: `target_dim` ∈ {32, 64, 128} × `hidden_dim` ∈ {128, 256} × `n_layers` ∈ {2, 3} ×
seed ∈ {0, 1, 2} = 36 fits. AdamW (lr 1e-3, wd 1e-4), dropout 0.1, ≤200 epochs. Architecture
selection: highest validation `cost_auc`, ties broken by lowest validation Brier.

## Run (GPU recommended; `--device cpu` works)
```bash
python src/experiments/embedllm/mlpdot_baseline/run_mlp_dot_pooled_eval.py
```

## Results
- `results/selection_audit.json`: all 36 fits with validation and test scores, the selected
  architecture, the pre-registered anchor, and `h2_val_test_agreement` (→ Table VII)
- `results/selected_metrics.json`, `sweep_metrics.json`, `cost_curve.json`, `reliability.json`,
  `cost_coverage.json`

Table III's 3-seed MLPDot row is aggregated in `all-methods-comparison`.

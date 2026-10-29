# Static-policy comparison — EmbedLLM-CC

Does per-query routing beat one query-independent preference over LLMs? Every arm is evaluated
through the same per-query budget-cap curve.

| Arm | Policy |
|---|---|
| S1 | always the largest affordable model (ties broken over the whole tied-largest size stratum) |
| S2 | ridge regression (α = 10) from raw fingerprint to mean train-query accuracy, fitted on training LLMs only |
| S2b | BF-DKR's own test predictions averaged over test queries |
| S3 | ranking by **test-set** mean accuracy: an anchor, flagged `uses_test_labels: true`, never a baseline |

**Paper:** Table S3 (Supplement §S2).

## Run
Stage 1 refits BF-DKR at its per-seed validation-selected `k`. It checks a reproduction gate
against `all-methods-comparison/results/seed_summary.json` and saves its test prediction matrix
(`P_test.npy`, gitignored), which S2b and `metadata-ablation` reuse. Stage 2 scores the static
arms and runs the paired bootstrap against BF-DKR.
```bash
for s in 0 1 2; do
  python src/experiments/embedllm/static_policy_baseline/run_bfdkr_reference.py --seed-idx $s
done
python src/experiments/embedllm/static_policy_baseline/run_static_policies.py
```

## Results
- `results/comparison_metrics.json`: pooled test `cost_auc`/AIQ per arm (3-seed mean ± s.d. for S2b and BF-DKR)
- `results/gap_bootstrap.json`: per-seed paired gaps, BF-DKR minus each arm
- `results/bfdkr_ref/seed{S}/reproduction_gate.json`: refit-vs-Table III drift (≤ 0.0024; EmbedLLM-CC's Brent search is not bit-reproducible)
- `results/s1_*.json`, `s2_*.json`, `s2b_*.json`, `s3_*.json`, `h*.json`: per-arm details

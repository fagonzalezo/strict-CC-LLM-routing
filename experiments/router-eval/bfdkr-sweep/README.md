# BF-DKR sweep — RouterEval-CC

BF-DKR (`DMKDMRRouter`, see [docs/method.md](../../../docs/method.md)) fitted on the frozen
MLPDot-encoded representation. The sweep covers encoder seed × Brent objective × neighbour
count `k`. This experiment holds the headline BF-DKR numbers.

**Paper:** Table III (the BF-DKR row, via `all-methods-comparison`), Fig. 3 and the §VI-A single-seed
numbers (seed 1, `pq_auc`, k = 2048).

## Protocol
24 configurations: seed ∈ {0, 1, 2} × objective ∈ {`bce`, `pq_auc`} × k ∈ {128, 512, 2048, 4096}.
Each fit selects (σ_q, σ_m) by nested Brent search on the validation block. Reported
configuration: per seed, the `k` with the highest validation `cost_auc` (ties broken by Brier),
under the `pq_auc` objective.

## Run
Needs `data/router-eval/processed/mlpdot_encoded_seed{0,1,2}/` (README, tier B).
```bash
for i in $(seq 0 23); do
  python src/experiments/router_eval/bfdkr_sweep/run_dmkdmr_encoded_eval.py --config-idx $i
done
python src/experiments/router_eval/bfdkr_sweep/build_comparison.py
```
`--smoke` runs a tiny subset and `--device cpu` forces CPU.

## Results
- `results/dmkdmr_encoded/seed{S}/{bce,pq_auc}/k{K}/`: `metrics.json` (per group),
  `cost_curve.json`, `reliability.json`, `gap_bootstrap.json` (paired bootstrap against the
  same-seed MLPDot), `config_audit.json` (selected σ's, Brent trace)
- `results/mlpdot_bce_reference_seed{S}.json`: the same-seed MLPDot reference
- `results/best_config.json`, `seed_spread.json`, `comparison_metrics.json`, `h3_objective_tradeoff.json`

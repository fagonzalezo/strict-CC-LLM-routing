# All-methods comparison — RouterEval-CC

Pure aggregation: no model is fit here. `build_seed_summary.py` collects every router's
per-seed pooled test metrics into one file, under one protocol per router:
- kNN, MLPDot, BF-DKR (`pq_auc` and `bce` objectives), DropoutNet, CLCRec, AGNN, FSGNN
- BF-DKR's ablation arms
- a reconciliation of BF-DKR's three `k` protocols (per-seed validation-selected; fixed k = 2048;
  fixed k = 4096)

The script runs more than ten consistency checks against the source experiments and exits
non-zero if one fails.

**Paper:** Table III, via `src/paper/gen_tab_main.py`. Its `seed_summary.json` is also the
reproduction gate for `static-policy-baseline` and `metadata-ablation`.

## Run (CPU, seconds)
Reads only committed results of `knn-baseline`, `mlpdot-baseline`, `bfdkr-sweep`,
`bfdkr-ablation`, `dropoutnet-clcrec-baselines` and `coldrec-baselines`.
```bash
python src/experiments/router_eval/all_methods_comparison/build_seed_summary.py
```

## Results
- `results/seed_summary.json`: per router, `per_seed` rows, `mean`/`std` (population s.d.) of
  `cost_auc`, AIQ, `pq_auc` and ECE; floor and oracle `cost_auc`; the protocol reconciliation

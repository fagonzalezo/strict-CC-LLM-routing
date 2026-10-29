# AGNN and FSGNN baselines — both datasets

Two graph-based cold-start recommenders, run with the ColdRec implementations vendored in
`src/baselines/coldrec/`. A local patch adds a both-sides-cold mode (`cold_object='both'`); see
that directory's README. Both are adapted to Cold–Cold by deriving both sides' content from the
inputs BF-DKR receives: query embeddings and fingerprints.

- **AGNN**: generates a cold node's embedding with a VAE over a content-similarity graph. Runs
  without its MF-pretrained backbone (Xavier initialization).
- **FSGNN**: adaptive feature and link completion before a LightGCN backbone. On RouterEval-CC it
  runs with personalized-PageRank overrides (5 iterations, top-5) to avoid out-of-memory failures.

Both train with a BPR ranking loss on **correct (query, LLM) pairs only**: incorrect answers are
treated as unobserved. On RouterEval-CC both train on a random 5,000,000 of the ≈38.7M training
positives.

**Paper:** Table III (AGNN and FSGNN rows), §VI-B, and the hyperparameters in Supplement §S4.

## Protocol
1. **HP search, seed 0**: 24 AGNN configurations (embedding size × lr × kNN-graph degree ×
   dropout) and 16 FSGNN configurations (embedding size × lr × λ_fc × μ_sc), defined in
   `run_coldrec.py`.
2. **Winner**: highest **pooled test** `pq_auc`. This is optimistic for these two baselines, and
   the paper says so.

   | | AGNN | FSGNN |
   |---|---|---|
   | RouterEval-CC | config 18 | config 2 |
   | EmbedLLM-CC | config 21 | config 0 |

3. **Finals**: refit each winner at seeds 0, 1, 2.

## Run (GPU)
```bash
# HP search (example: AGNN on RouterEval-CC; FSGNN uses --config-idx 0..15)
for c in $(seq 0 23); do
  python src/experiments/coldrec_baselines/run_coldrec.py --model AGNN --dataset router-eval --config-idx $c --seed 0
done
# Finals at the winning configurations
for s in 0 1 2; do
  python src/experiments/coldrec_baselines/run_coldrec.py --model AGNN  --dataset router-eval --config-idx 18 --seed $s
  python src/experiments/coldrec_baselines/run_coldrec.py --model FSGNN --dataset router-eval --config-idx 2  --seed $s
  python src/experiments/coldrec_baselines/run_coldrec.py --model AGNN  --dataset embedllm    --config-idx 21 --seed $s
  python src/experiments/coldrec_baselines/run_coldrec.py --model FSGNN --dataset embedllm    --config-idx 0  --seed $s
done
python src/experiments/coldrec_baselines/build_router_eval_finals_summary.py
python src/experiments/coldrec_baselines/build_embedllm_finals_summary.py
```

## Results
- `results/{router-eval,embedllm}/<MODEL>/config<i>/seed<s>/metrics.json`: the winning
  configurations' three seeds. Also a few non-winning RouterEval-CC FSGNN sweep cells
  (configs 9–15), which back the observation that every FSGNN configuration gives the same
  top-1 accuracy.
- `results/router_eval_finals_summary.json`, `embedllm_finals_summary.json`: 3-seed mean ±
  s.d. at the winner (the Table III rows)

The full HP-search grids are not committed. Nine of the sixteen RouterEval-CC FSGNN sweep
configurations lost their metric files to a storage-synchronization incident. The winning
configuration's final runs were re-executed, so no reported number depends on them.

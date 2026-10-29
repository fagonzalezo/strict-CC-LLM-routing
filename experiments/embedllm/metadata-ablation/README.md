# LLM-side representation ablation — EmbedLLM-CC

Replaces the behavioral fingerprint with cheap model metadata at the dataset-bundle level. No
router code changes; the encoder and every router are refit on each input.

| Arm | LLM-side input |
|---|---|
| `fp` | behavioral fingerprint, 2048-d (reference) |
| `meta_struct` | log10 parameter count, MoE flag, top-n organization/family one-hots (23-d; MoE flag dropped as zero-variance) |
| `meta_name` | Qwen3-Embedding-4B embedding of the `org__model` id, truncated to 2048-d |
| `concat` | `fp` ⊕ `meta_struct` |

**Paper:** Table VI (the EmbedLLM-CC block).

## Protocol
For each arm and encoder seed, the MLPDot encoder is refit on that arm's bundle. BF-DKR, MLPDot
and encoded kNN are then refit, with `k` re-selected on validation **per arm**. The `fp` arm reuses
`static-policy-baseline`'s BF-DKR predictions. Gaps to `fp` use the same paired bootstrap as the
other studies.

## Run
```bash
D=embedllm; P=embedllm
# 1. variant bundles (meta_struct, concat: CPU; meta_name: GPU, Qwen3-Embedding-4B)
for v in meta_struct concat meta_name; do
  python src/data_pipeline/metadata/build_metadata_bundle.py --dataset $D --variant $v
done
# 2. MLPDot-encode each variant, seeds 0-2
for v in meta_struct meta_name concat; do for s in 0 1 2; do
  V=data/$D/processed/metadata_variants/$v
  python src/data_pipeline/$P/encode_embeddings_mlpdot.py --source $V/llm_perf_dataset.pkl \
      --out-root $V --reports-dir $V/reports --seed $s
done; done
# 3. 4 arms x 3 seeds, then diagnostics and aggregation
for i in $(seq 0 11); do
  python src/experiments/$P/metadata_ablation/run_metadata_ablation.py --config-idx $i
done
python src/experiments/$P/metadata_ablation/check_degeneracy.py
python src/experiments/$P/metadata_ablation/build_comparison.py
```
`check_degeneracy.py` and `build_comparison.py` reload the dataset bundle and the per-arm test
prediction matrices (`P_test.npy`, gitignored) that the runner writes, so they run only after a
tier-B rerun.

## Results
- `results/comparison_metrics.json`: arm × router, 3-seed mean ± s.d. of pooled test `cost_auc`
- `results/gap_bootstrap.json`: BF-DKR per-seed paired gaps, `fp` minus each arm
- `results/seed{S}/<arm>/`: per-fit metrics; `other_routers_metrics.json` holds MLPDot and encoded kNN
- `results/representation_degeneracy.json`: distinct LLM vectors per arm and column-permutation robustness
- `results/metadata_coverage.json`: one-hot vocabulary coverage of test LLMs under Cold–Cold splits

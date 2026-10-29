# DropoutNet and CLCRec baselines — EmbedLLM-CC

Two two-tower cold-start recommenders, reimplemented in MLPDot's tower family and adapted to
Cold–Cold: both towers take only the inputs BF-DKR receives (query embedding, fingerprint), and
fitting uses train-block interactions only.

- **DropoutNet** (`src/routers/dropoutnet.py`, Volkovs et al., NeurIPS 2017). Stage A pretrains
  logistic matrix factorization on the train block (replacing the original weighted MF). Stage B
  trains MLP towers over [pretrained embedding ‖ content] with input dropout on the pretrained
  branch. Every test entity takes that branch zeroed, so at test time the model is content-only.
- **CLCRec** (`src/routers/clcrec.py`, Wei et al., ACM MM 2021). Content towers trained jointly
  with a BCE scoring head, a self-discrimination term aligning each training LLM's content
  embedding with a free embedding, and a per-query softmax over training LLMs.

**Paper:** Table III (DropoutNet and CLCRec rows), §VI-B, and the hyperparameters in Supplement §S4.

## Protocol
Each runner has two phases:
1. **HP search** at seed 0 over 9 configurations:
   - DropoutNet: MF dimension × input-dropout rate.
   - CLCRec: λ_RE × τ.
   The winner is the configuration with the highest validation `cost_auc` (ties broken by Brier).
2. **Refit** the winner at seeds 0, 1, 2.

Training: 200 epochs, lr 1e-3, wd 1e-4, batch 256.

## Run (GPU recommended)
```bash
python src/experiments/embedllm/dropoutnet_clcrec_baselines/run_dropoutnet_eval.py
python src/experiments/embedllm/dropoutnet_clcrec_baselines/run_clcrec_eval.py
python src/experiments/embedllm/dropoutnet_clcrec_baselines/build_comparison.py
```
`--probe --probe-config-idx i` times a single configuration; `--smoke` runs a tiny subset.

## Results
- `results/{dropoutnet,clcrec}/hp_search/<config>/`: per-configuration metrics; `selection_audit.json`,
  `sweep_metrics.json` and `seed_spread.json` next to `hp_search/`
- `results/{dropoutnet,clcrec}/final/seed{S}/`: `selected_metrics.json`, `config_audit.json` (incl. `best_epoch`), cost curves
- `results/comparison_metrics.json`: pooled comparison with kNN, MLPDot and BF-DKR

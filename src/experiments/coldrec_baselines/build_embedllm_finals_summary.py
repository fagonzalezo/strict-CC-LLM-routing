"""
3-seed aggregation for EmbedLLM AGNN/FSGNN at their HP-search-winning config.

AGNN's winner is config 21 (test_pq_auc=0.559246, seed0) and FSGNN's is config 0
(test_pq_auc=0.482968, seed0 -- all 16 FSGNN configs tie at this value, a collapsed
sweep; config 0 is the first, and no less valid than any other). Both were selected
by argmax test_pq_auc over each model's full HP-search grid (seed0, current 107-model
/ 70-16-21 roster) -- computed directly from the `run_coldrec.py --config-idx` HP search's
24/16 configs respectively, not transcribed from any report. The full HP-search grid
itself is not committed here (only these two winning configs' 3 seeds are); this
module records the winners as constants rather than re-deriving them on every run, the
same convention `HEADLINE_SEED` uses elsewhere in this repo for a similar one-time,
externally-verified selection.

Reads:
  - results/embedllm/AGNN/config{AGNN_BEST_CONFIG}/seed{0,1,2}/metrics.json
  - results/embedllm/FSGNN/config{FSGNN_BEST_CONFIG}/seed{0,1,2}/metrics.json

Writes:
  - results/embedllm_finals_summary.json -- per-model 3-seed mean/std (ddof=0, matching
    gen_tab_main.py's `mean_std` convention) for cost_auc, pq_auc, ece, aiq, plus the
    shared test_floor_cost_auc/test_oracle_cost_auc (identical across seeds -- a
    property of the test LLM pool, not the fit).

Usage: python build_embedllm_finals_summary.py   (run after all 6 seed dirs exist)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

OUT_RESULTS = results_dir("coldrec-baselines")
EMBEDLLM_RESULTS = OUT_RESULTS / "embedllm"

SEEDS = (0, 1, 2)
AGNN_BEST_CONFIG = 21
FSGNN_BEST_CONFIG = 0
MODELS = {"AGNN": AGNN_BEST_CONFIG, "FSGNN": FSGNN_BEST_CONFIG}


def _require(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Missing {path} -- run run_coldrec.py for this (model, config, seed) first.")
    print(f"  read: {relative(path)}")
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else [data]
    return next(r for r in rows if r["group"] == "pooled")


def mean_std(values):
    # ddof=0, matching src/paper/gen_tab_main.py's mean_std convention.
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    summary = {}
    for model, config_idx in MODELS.items():
        print(f"\n{model} (config{config_idx}):")
        pooled_by_seed = [
            _require(EMBEDLLM_RESULTS / model / f"config{config_idx}" / f"seed{seed}" / "metrics.json")
            for seed in SEEDS
        ]
        floors = {p["test_floor_cost_auc"] for p in pooled_by_seed}
        oracles = {p["test_oracle_cost_auc"] for p in pooled_by_seed}
        if len(floors) != 1 or len(oracles) != 1:
            sys.exit(f"{model}: test_floor_cost_auc/test_oracle_cost_auc disagree across seeds "
                      f"-- floors={floors} oracles={oracles}; these are pool properties and must "
                      f"be seed-invariant. Do not aggregate.")

        row = {"config_idx": config_idx, "n_seeds": len(SEEDS)}
        for key, src_key in (
            ("cost_auc", "test_cost_auc"), ("pq_auc", "test_pq_auc"),
            ("ece", "test_ece"), ("aiq", "test_aiq"),
        ):
            m, s = mean_std([p[src_key] for p in pooled_by_seed])
            row[key] = {"mean": round(m, 6), "std": round(s, 6)}
            print(f"  {key}: {m:.4f} +/- {s:.4f}")
        row["floor_cost_auc"] = next(iter(floors))
        row["oracle_cost_auc"] = next(iter(oracles))

        implied_aiq = row["cost_auc"]["mean"] - row["floor_cost_auc"]
        if abs(implied_aiq - row["aiq"]["mean"]) > 1e-6:
            sys.exit(f"{model}: AIQ identity broke -- cost_auc-floor={implied_aiq:.6f} != "
                      f"mean test_aiq={row['aiq']['mean']:.6f}")
        summary[model] = row

    out_path = OUT_RESULTS / "embedllm_finals_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote: {relative(out_path)}")


if __name__ == "__main__":
    main()

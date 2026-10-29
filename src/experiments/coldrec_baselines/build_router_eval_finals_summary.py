"""
3-seed aggregation for RouterEval AGNN/FSGNN at their HP-search-winning config.

Sibling to `build_embedllm_finals_summary.py` -- same schema, different dataset, own
constants (this repo deliberately duplicates per-dataset for this family; see e.g.
the two `build_*_finals_summary.py` scripts).

AGNN's winner is **config 18** (emb_size=64, lr=5e-4, agnn_knn_k=5, agnn_dropout=0.3;
test_pq_auc=0.660883, seed0) -- selected by a *fresh* full 24-config HP-search grid run
2026-09-13/14 after the original results tree (and its old transcribed winner, config12)
was lost to a storage-synchronization incident. Config12 does not reproduce; config18
clears the non-collapsed cost_auc plateau on both cost_auc and pq_auc and is corroborated
by config22's near-tied cost_auc (within 0.0002) -- see
`experiments/coldrec-baselines/README.md`. FSGNN's winner is **config 2** (emb_size=32, lr=1e-3, fsgnn_lambda_fc=0.5,
fsgnn_mu_sc=0.1; test_pq_auc=0.6069, seed0), selected the same way (argmax test_pq_auc
over the full 16-config grid) and unchanged from the original selection -- the full-grid
rerun confirmed rather than overturned it.

Both winners were selected by argmax test_pq_auc over each model's full HP-search grid
(seed0), computed directly from the `run_coldrec.py --config-idx` HP search's 24/16
configs respectively, not transcribed from any report. The full HP-search grid itself is
not committed here (only these two winning configs' 3 seeds are); this module records the
winners as constants rather than re-deriving them on every run, the same convention
`build_embedllm_finals_summary.py` uses.

Reads:
  - results/router-eval/AGNN/config{AGNN_BEST_CONFIG}/seed{0,1,2}/metrics.json
  - results/router-eval/FSGNN/config{FSGNN_BEST_CONFIG}/seed{0,1,2}/metrics.json

Writes:
  - results/router_eval_finals_summary.json -- per-model 3-seed mean/std (ddof=0, matching
    gen_tab_main.py's `mean_std` convention) for cost_auc, pq_auc, ece, aiq, plus the
    shared test_floor_cost_auc/test_oracle_cost_auc (identical across seeds -- a
    property of the test LLM pool, not the fit).

Usage: python build_router_eval_finals_summary.py   (run after all 6 seed dirs exist)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

OUT_RESULTS = results_dir("coldrec-baselines")
ROUTER_EVAL_RESULTS = OUT_RESULTS / "router-eval"

SEEDS = (0, 1, 2)
AGNN_BEST_CONFIG = 18
FSGNN_BEST_CONFIG = 2
MODELS = {"AGNN": AGNN_BEST_CONFIG, "FSGNN": FSGNN_BEST_CONFIG}

# Independent anchor: AGNN config18/seed0 was already transcribed into
# coldrec-baselines/reports/summary.md section 5.4 before this script existed --
# this is the one number here NOT derived from the same six files it checks.
SUMMARY_MD_ANCHOR = {"model": "AGNN", "config_idx": 18, "seed": 0,
                      "test_pq_auc": 0.6609, "test_cost_auc": 0.6674, "tol": 1e-3}

EXPECTED_FLOOR = 0.538449
EXPECTED_ORACLE = 0.947298


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
    pooled_cache = {}
    for model, config_idx in MODELS.items():
        print(f"\n{model} (config{config_idx}):")
        pooled_by_seed = [
            _require(ROUTER_EVAL_RESULTS / model / f"config{config_idx}" / f"seed{seed}" / "metrics.json")
            for seed in SEEDS
        ]
        pooled_cache[model] = pooled_by_seed
        floors = {p["test_floor_cost_auc"] for p in pooled_by_seed}
        oracles = {p["test_oracle_cost_auc"] for p in pooled_by_seed}
        if len(floors) != 1 or len(oracles) != 1:
            sys.exit(f"{model}: test_floor_cost_auc/test_oracle_cost_auc disagree across seeds "
                      f"-- floors={floors} oracles={oracles}; these are pool properties and must "
                      f"be seed-invariant. Do not aggregate.")
        floor, oracle = next(iter(floors)), next(iter(oracles))
        if abs(floor - EXPECTED_FLOOR) > 1e-6 or abs(oracle - EXPECTED_ORACLE) > 1e-6:
            sys.exit(f"{model}: floor/oracle {floor}/{oracle} != shared RouterEval constants "
                      f"{EXPECTED_FLOOR}/{EXPECTED_ORACLE}")

        row = {"config_idx": config_idx, "n_seeds": len(SEEDS)}
        for key, src_key in (
            ("cost_auc", "test_cost_auc"), ("pq_auc", "test_pq_auc"),
            ("ece", "test_ece"), ("aiq", "test_aiq"),
        ):
            m, s = mean_std([p[src_key] for p in pooled_by_seed])
            row[key] = {"mean": round(m, 6), "std": round(s, 6)}
            print(f"  {key}: {m:.4f} +/- {s:.4f}")
        row["floor_cost_auc"] = floor
        row["oracle_cost_auc"] = oracle

        implied_aiq = row["cost_auc"]["mean"] - row["floor_cost_auc"]
        if abs(implied_aiq - row["aiq"]["mean"]) > 1e-6:
            sys.exit(f"{model}: AIQ identity broke -- cost_auc-floor={implied_aiq:.6f} != "
                      f"mean test_aiq={row['aiq']['mean']:.6f}")
        summary[model] = row

    # Guard: the one independent (non-self-referential) check available for this data --
    # AGNN config18/seed0 was already transcribed into summary.md section 5.4.
    a = SUMMARY_MD_ANCHOR
    anchor_row = pooled_cache[a["model"]][a["seed"]]
    if (abs(anchor_row["test_pq_auc"] - a["test_pq_auc"]) > a["tol"]
            or abs(anchor_row["test_cost_auc"] - a["test_cost_auc"]) > a["tol"]):
        sys.exit(f"summary.md section 5.4 anchor check failed: "
                  f"{a['model']} config{a['config_idx']}/seed{a['seed']} "
                  f"pq_auc={anchor_row['test_pq_auc']:.6f} cost_auc={anchor_row['test_cost_auc']:.6f} "
                  f"vs transcribed pq_auc={a['test_pq_auc']} cost_auc={a['test_cost_auc']}")
    print(f"\nsummary.md section 5.4 anchor check OK: {a['model']} config{a['config_idx']}/"
          f"seed{a['seed']} matches (pq_auc={anchor_row['test_pq_auc']:.6f}, "
          f"cost_auc={anchor_row['test_cost_auc']:.6f}).")

    out_path = OUT_RESULTS / "router_eval_finals_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote: {relative(out_path)}")


if __name__ == "__main__":
    main()

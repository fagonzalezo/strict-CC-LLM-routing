"""
Aggregation for DropoutNet-style and CLCRec-style cold-start routers vs.
kNN, MLPDot, and DMKDMR-encoded (EmbedLLM).

Adapted from
`src/experiments/router_eval/dropoutnet_clcrec_baselines/build_comparison.py` with the
DMKDMR-raw column dropped: EmbedLLM never had a DMKDMR-on-raw-embeddings experiment
released (it is not part of the paper). This script depends on nothing missing and is fully
runnable.

Reads:
  - results/{dropoutnet,clcrec}/final/seed1/selected_metrics.json -- headline
    (seed 1, matching router-eval's convention for cross-experiment comparability).
  - results/{dropoutnet,clcrec}/final/seed1/{four_cell.json, re_per_model.json}.
  - embedllm/knn-baseline, embedllm/mlpdot-baseline's selected_metrics.json.
  - embedllm/bfdkr-sweep's comparison_metrics.json, row router=="DMKDMR-encoded (bce)".

All comparisons are POINT-ESTIMATE deltas read directly from each experiment's
already-computed pooled test metrics -- same convention as the router-eval condition
(no per-pair bootstrap CI; none of kNN/MLPDot/DMKDMR-encoded persisted a raw per-query
P_test for EmbedLLM either).

H1 (four-cell monotonicity, DropoutNet only): identical check to the router-eval
    condition.
H2 (new router vs. 3 baselines): point-estimate delta_cost_auc / delta_pq_auc /
    delta_ece per (new router, baseline) pair, 6 pairs total.
H3 (CLCRec vs. DropoutNet): point-estimate delta, same fields.
H4 (re_loss quartile vs. accuracy, CLCRec only, descriptive): from re_per_model.json,
    top-25%-by-re_loss vs. bottom-25%-by-re_loss mean train_accuracy gap. Quartiles
    are ~17-18 models each here (70 train LLMs vs. router-eval's ~750) -- noisier,
    flagged in the report rather than hidden.

Usage: python build_comparison.py   (run after both run scripts have completed)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import experiment_dir  # noqa: E402

EMBEDLLM = experiment_dir("embedllm")
EXP_DIR = experiment_dir("embedllm", "dropoutnet-clcrec-baselines")
OUT_RESULTS = EXP_DIR / "results"
OUT_REPORTS = EXP_DIR / "reports"

HEADLINE_SEED = 1

TEST_KEYS_A = [
    "test_pq_auc", "test_top1_acc", "test_top1_regret", "test_mrr",
    "test_recall_at_1", "test_recall_at_3", "test_recall_at_5", "test_recall_at_10",
    "test_ndcg_at_1", "test_ndcg_at_3", "test_ndcg_at_5", "test_ndcg_at_10",
    "test_llm_spearman", "test_llm_kendall",
]
TEST_KEYS_B = ["test_brier", "test_ece"]
TEST_KEYS_C = [
    "test_cost_auc", "test_floor_cost_auc", "test_oracle_cost_auc",
    "test_aiq", "test_aiq_deploy", "test_apgr", "test_cost_at_target_quality",
]
LOWER_IS_BETTER = {"test_top1_regret", "test_brier", "test_ece", "test_cost_at_target_quality"}


def _require(path: Path, hint: str) -> dict:
    if not path.exists():
        sys.exit(f"Missing {path} -- {hint}")
    return json.loads(path.read_text())


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def best_col(vals, lower_is_better=False):
    valid = {k: v for k, v in vals.items() if v is not None and v == v}
    if not valid:
        return None
    if len(set(valid.values())) == 1:
        return "shared"
    return min(valid, key=valid.get) if lower_is_better else max(valid, key=valid.get)


def render_metric_table(lines, rows_by_router, keys, title, routers, labels):
    lines.append(f"### {title}\n")
    header = "| Metric | " + " | ".join(labels[r] for r in routers) + " | Best |"
    sep = "|---|" + "---:|" * len(routers) + "---|"
    lines.append(header)
    lines.append(sep)
    for k in keys:
        vals = {r: rows_by_router[r].get(k) for r in routers}
        lower = k in LOWER_IS_BETTER
        b = best_col(vals, lower_is_better=lower)
        cells = []
        for r in routers:
            v = vals[r]
            s = fmt(v) if v is not None and v == v else "-"
            if b == r:
                s = f"**{s}**"
            cells.append(s)
        label = k.replace("test_", "")
        best_label = "(shared)" if b == "shared" else ("-" if b is None else labels[b])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {best_label} |")
    lines.append("")


SOURCE_PATHS = {
    "knn": EMBEDLLM / "knn-baseline" / "results" / "selected_metrics.json",
    "mlpdot": EMBEDLLM / "mlpdot-baseline" / "results" / "selected_metrics.json",
    "dropoutnet": OUT_RESULTS / "dropoutnet" / "final" / f"seed{HEADLINE_SEED}" / "selected_metrics.json",
    "clcrec": OUT_RESULTS / "clcrec" / "final" / f"seed{HEADLINE_SEED}" / "selected_metrics.json",
}
SOURCE_EXPS = {
    "knn": "embedllm/knn-baseline", "mlpdot": "embedllm/mlpdot-baseline",
    "dmkdmr_encoded": "embedllm/bfdkr-sweep (bce, val-selected)",
    "dropoutnet": f"embedllm/dropoutnet-clcrec-baselines/dropoutnet/final/seed{HEADLINE_SEED}",
    "clcrec": f"embedllm/dropoutnet-clcrec-baselines/clcrec/final/seed{HEADLINE_SEED}",
}
ROUTER_LABELS = {
    "knn": "kNN", "mlpdot": "MLPDot", "dmkdmr_encoded": "DMKDMR-encoded (bce)",
    "dropoutnet": "DropoutNet", "clcrec": "CLCRec",
}
NEW_ROUTERS = ["dropoutnet", "clcrec"]
ROUTERS = ["knn", "mlpdot", "dmkdmr_encoded", "dropoutnet", "clcrec"]
BASELINES = ["knn", "mlpdot", "dmkdmr_encoded"]


def _dmkdmr_encoded_row() -> dict:
    merged = _require(EMBEDLLM / "bfdkr-sweep" / "results" / "comparison_metrics.json",
                       "embedllm/bfdkr-sweep must be complete.")
    return next(r for r in merged
                if r["router"] == "DMKDMR-encoded (bce)" and r["group"] == "pooled")


def _pooled_row(router: str) -> dict:
    if router == "dmkdmr_encoded":
        return _dmkdmr_encoded_row()
    data = _require(SOURCE_PATHS[router], f"run the upstream experiment for {router} first.")
    return next(r for r in data if r["group"] == "pooled")


def _h1_four_cell():
    path = OUT_RESULTS / "dropoutnet" / "final" / f"seed{HEADLINE_SEED}" / "four_cell.json"
    cell = _require(path, "run run_dropoutnet_eval.py first.")

    def ok(a, b, key, lower_better):
        va, vb = cell[a][key], cell[b][key]
        return va <= vb if lower_better else va >= vb

    checks = {
        "bce_warm_le_cold_query": ok("warm", "cold_query", "bce", True),
        "bce_warm_le_cold_model": ok("warm", "cold_model", "bce", True),
        "bce_cold_query_le_cold_cold": ok("cold_query", "cold_cold", "bce", True),
        "bce_cold_model_le_cold_cold": ok("cold_model", "cold_cold", "bce", True),
        "pqauc_warm_ge_cold_query": ok("warm", "cold_query", "pq_auc", False),
        "pqauc_warm_ge_cold_model": ok("warm", "cold_model", "pq_auc", False),
        "pqauc_cold_query_ge_cold_cold": ok("cold_query", "cold_cold", "pq_auc", False),
        "pqauc_cold_model_ge_cold_cold": ok("cold_model", "cold_cold", "pq_auc", False),
    }
    all_seeds = {}
    for seed in (0, 1, 2):
        p = OUT_RESULTS / "dropoutnet" / "final" / f"seed{seed}" / "four_cell.json"
        if p.exists():
            all_seeds[f"seed{seed}"] = json.loads(p.read_text())

    return {
        "headline_seed": HEADLINE_SEED, "four_cell": cell, "checks": checks,
        "H1_confirmed": all(checks.values()),
        "all_seeds": all_seeds,
    }


def _point_delta(new_row: dict, base_row: dict, baseline: str) -> dict:
    return {
        "baseline": baseline,
        "delta_cost_auc": round(new_row["test_cost_auc"] - base_row["test_cost_auc"], 6),
        "delta_pq_auc": round(new_row["test_pq_auc"] - base_row["test_pq_auc"], 6),
        "delta_ece": round(new_row["test_ece"] - base_row["test_ece"], 6),
        "new_cost_auc": new_row["test_cost_auc"], "baseline_cost_auc": base_row["test_cost_auc"],
        "new_pq_auc": new_row["test_pq_auc"], "baseline_pq_auc": base_row["test_pq_auc"],
    }


def _h2_vs_baselines(baselines):
    baseline_rows = {b: _pooled_row(b) for b in baselines}
    results = {}
    for router in NEW_ROUTERS:
        new_row = _pooled_row(router)
        results[router] = {}
        for b in baselines:
            gap = _point_delta(new_row, baseline_rows[b], b)
            results[router][b] = gap
            print(f"  {router} vs {b}: delta_cost_auc={gap['delta_cost_auc']:+.6f}  "
                  f"delta_pq_auc={gap['delta_pq_auc']:+.6f}  delta_ece={gap['delta_ece']:+.6f}")
    return results


def _h3_clcrec_vs_dropoutnet():
    dn_row = _pooled_row("dropoutnet")
    cl_row = _pooled_row("clcrec")
    gap = _point_delta(cl_row, dn_row, "dropoutnet")
    gap["direction"] = "CLCRec - DropoutNet"
    if gap["delta_cost_auc"] > 0:
        verdict = "CLCRec ahead on cost_auc (point estimate)"
    elif gap["delta_cost_auc"] < 0:
        verdict = "DropoutNet ahead on cost_auc (point estimate)"
    else:
        verdict = "tie on cost_auc"
    gap["H3_verdict"] = verdict
    return gap


def _seed_spread_summary():
    out = {}
    for router in NEW_ROUTERS:
        p = OUT_RESULTS / router / "seed_spread.json"
        if p.exists():
            out[router] = json.loads(p.read_text())
    return out


def _h4_re_loss_vs_accuracy():
    path = OUT_RESULTS / "clcrec" / "final" / f"seed{HEADLINE_SEED}" / "re_per_model.json"
    data = _require(path, "run run_clcrec_eval.py first.")
    names = list(data.keys())
    re_loss = np.array([data[n]["re_loss"] for n in names])
    acc = np.array([data[n]["train_accuracy"] for n in names])

    order = np.argsort(re_loss)
    n = len(names)
    q = max(1, n // 4)
    lowest_idx = order[:q]     # best discrimination (lowest re_loss)
    highest_idx = order[-q:]   # worst discrimination (highest re_loss)

    acc_low_re = float(acc[lowest_idx].mean())
    acc_high_re = float(acc[highest_idx].mean())
    gap = acc_high_re - acc_low_re

    return {
        "n_models": n, "quartile_size": int(q),
        "mean_train_accuracy_lowest_re_loss_quartile": round(acc_low_re, 6),
        "mean_train_accuracy_highest_re_loss_quartile": round(acc_high_re, 6),
        "gap_high_minus_low": round(gap, 6),
        "H4_confirmed": bool(gap < 0),
        "note": f"quartile_size={q} models per group (70 train LLMs total) -- much "
                "noisier than router-eval's ~750-model quartiles; treat as "
                "suggestive, not confirmatory. gap < 0 means worst-discriminating "
                "models (highest re_loss) route worse on their own train set.",
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("Loading EmbedLLM dropoutnet-clcrec-baselines results ...")

    print("\nH1: DropoutNet four-cell monotonicity ...")
    h1 = _h1_four_cell()
    print(f"  checks: {sum(h1['checks'].values())}/{len(h1['checks'])} passed  "
          f"-> H1_confirmed={h1['H1_confirmed']}")
    (OUT_RESULTS / "h1_four_cell.json").write_text(json.dumps(h1, indent=2))

    print(f"\nH2: DropoutNet/CLCRec vs. {'/'.join(ROUTER_LABELS[b] for b in BASELINES)} "
          "(point-estimate deltas, no bootstrap) ...")
    h2 = _h2_vs_baselines(BASELINES)
    (OUT_RESULTS / "h2_vs_baselines.json").write_text(json.dumps(h2, indent=2))

    print("\nH3: CLCRec vs. DropoutNet (point-estimate delta) ...")
    h3 = _h3_clcrec_vs_dropoutnet()
    print(f"  delta_cost_auc={h3['delta_cost_auc']:+.6f}  delta_pq_auc={h3['delta_pq_auc']:+.6f}  "
          f"-> {h3['H3_verdict']}")
    (OUT_RESULTS / "h3_clcrec_vs_dropoutnet.json").write_text(json.dumps(h3, indent=2))

    print("\nH4: re_loss_per_model_ quartile vs. train accuracy (descriptive) ...")
    h4 = _h4_re_loss_vs_accuracy()
    print(f"  gap(high_re - low_re)={h4['gap_high_minus_low']:+.4f}  "
          f"-> H4_confirmed={h4['H4_confirmed']}")
    (OUT_RESULTS / "h4_re_loss_vs_accuracy.json").write_text(json.dumps(h4, indent=2))

    print("\nSeed spread (3 final seeds per router) ...")
    seed_spread = _seed_spread_summary()
    for router, spread in seed_spread.items():
        ca = spread.get("test_cost_auc", {})
        print(f"  {router}: test_cost_auc mean={ca.get('mean')} std={ca.get('std')}")
    (OUT_RESULTS / "seed_spread_summary.json").write_text(json.dumps(seed_spread, indent=2))

    # --- 5-way comparison table ---
    sources = {}
    for router in ("knn", "mlpdot"):
        data = _require(SOURCE_PATHS[router], f"run the upstream experiment for {router} first.")
        sources[router] = {r["group"]: r for r in data}
    dmkdmr_enc_merged = _require(EMBEDLLM / "bfdkr-sweep" / "results" / "comparison_metrics.json",
                                  "embedllm/bfdkr-sweep must be complete.")
    sources["dmkdmr_encoded"] = {r["group"]: r for r in dmkdmr_enc_merged
                                  if r["router"] == "DMKDMR-encoded (bce)"}
    for router in NEW_ROUTERS:
        data = _require(SOURCE_PATHS[router], f"run run_{router}_eval.py first.")
        sources[router] = {r["group"]: r for r in data}

    merged = []
    for router in ROUTERS:
        for group, row in sources[router].items():
            entry = {"router": ROUTER_LABELS[router], "group": group, "source_exp": SOURCE_EXPS[router]}
            for k, v in row.items():
                if k.startswith("test_") or k in ("Nq_test", "n_groups"):
                    entry[k] = v
            merged.append(entry)
    (OUT_RESULTS / "comparison_metrics.json").write_text(json.dumps(merged, indent=2))

    # --- reports/comparison_table.md ---
    baseline_label_str = "/".join(ROUTER_LABELS[b] for b in BASELINES)
    lines = []
    lines.append(f"# EmbedLLM: DropoutNet-style and CLCRec-style cold-start routers "
                  f"vs. {baseline_label_str}\n")
    lines.append(
        "Source: `experiments/embedllm/{knn-baseline,mlpdot-baseline,bfdkr-sweep}/results/**` "
        "+ `dropoutnet-clcrec-baselines/results/{dropoutnet,clcrec}/final/seed1/*.json`, "
        f"merged by `build_comparison.py`. No DMKDMR-raw row (never ported for EmbedLLM). "
        f"H2/H3 are point-estimate deltas over each experiment's own already-computed "
        f"pooled test metrics -- no per-pair bootstrap CI (none of {baseline_label_str} "
        f"persisted a raw P_test for EmbedLLM either).\n"
    )

    lines.append("## H1: DropoutNet four-cell monotonicity (S_train)\n")
    lines.append(f"Checks passed: {sum(h1['checks'].values())}/{len(h1['checks'])} "
                  f"-> **H1 {'CONFIRMED' if h1['H1_confirmed'] else 'NOT CONFIRMED'}**\n")
    lines.append("| Cell | bce | pq_auc | top1_acc |")
    lines.append("|---|---:|---:|---:|")
    for cell, m in h1["four_cell"].items():
        lines.append(f"| {cell} | {m['bce']:.4f} | {m['pq_auc']:.4f} | {m['top1_acc']:.4f} |")
    lines.append("")

    lines.append("## H2: DropoutNet/CLCRec vs. baselines (point-estimate deltas)\n")
    lines.append("| New router | Baseline | Δcost_auc | Δpq_auc | Δece |")
    lines.append("|---|---|---:|---:|---:|")
    for router in NEW_ROUTERS:
        for b, gap in h2[router].items():
            lines.append(
                f"| {ROUTER_LABELS[router]} | {ROUTER_LABELS[b]} | "
                f"{gap['delta_cost_auc']:+.6f} | {gap['delta_pq_auc']:+.6f} | "
                f"{gap['delta_ece']:+.6f} |")
    lines.append("")

    lines.append("## H3: CLCRec vs. DropoutNet (point-estimate delta)\n")
    lines.append(f"Δcost_auc={h3['delta_cost_auc']:+.6f}  Δpq_auc={h3['delta_pq_auc']:+.6f}  "
                  f"Δece={h3['delta_ece']:+.6f}\n")
    lines.append(f"**{h3['H3_verdict']}**\n")

    lines.append("## H4: re_loss_per_model_ quartile vs. train accuracy (descriptive)\n")
    lines.append(f"Lowest-re_loss quartile mean train_accuracy: "
                  f"{h4['mean_train_accuracy_lowest_re_loss_quartile']:.4f}  \n"
                  f"Highest-re_loss quartile mean train_accuracy: "
                  f"{h4['mean_train_accuracy_highest_re_loss_quartile']:.4f}  \n"
                  f"Gap (high - low): {h4['gap_high_minus_low']:+.4f}  "
                  f"-> **H4 {'CONFIRMED' if h4['H4_confirmed'] else 'NOT CONFIRMED'}** "
                  f"(quartile_size={h4['quartile_size']} models -- noisy, see note)\n")

    lines.append("## Seed spread (3 final seeds per router, headline config)\n")
    lines.append("| Router | Metric | seed0 | seed1 | seed2 | mean | std |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for router in NEW_ROUTERS:
        spread = seed_spread.get(router, {})
        for metric in ("test_cost_auc", "test_pq_auc", "test_brier", "test_ece"):
            d = spread.get(metric)
            if d is None:
                continue
            lines.append(
                f"| {ROUTER_LABELS[router]} | {metric.replace('test_', '')} | "
                f"{d['seed0']:.6f} | {d['seed1']:.6f} | {d['seed2']:.6f} | "
                f"{d['mean']:.6f} | {d['std']:.6f} |")
    lines.append("")

    lines.append(f"## {len(ROUTERS)}-way comparison (pooled, all test queries)\n")
    pooled_by_router = {r: sources[r]["pooled"] for r in ROUTERS}
    render_metric_table(lines, pooled_by_router, TEST_KEYS_A, "A. Ranking", ROUTERS, ROUTER_LABELS)
    render_metric_table(lines, pooled_by_router, TEST_KEYS_B, "B. Calibration", ROUTERS, ROUTER_LABELS)
    render_metric_table(lines, pooled_by_router, TEST_KEYS_C, "C. Cost-aware (post-hoc)", ROUTERS, ROUTER_LABELS)

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\nWrote:")
    for p in ("h1_four_cell.json", "h2_vs_baselines.json", "h3_clcrec_vs_dropoutnet.json",
              "h4_re_loss_vs_accuracy.json", "seed_spread_summary.json", "comparison_metrics.json"):
        print(f"  {OUT_RESULTS / p}")
    print(f"  {OUT_REPORTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

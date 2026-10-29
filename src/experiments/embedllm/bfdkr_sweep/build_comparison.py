"""
Aggregation for DMKDMR-on-MLPDot-encoded-embeddings vs. MLPDot and kNN (EmbedLLM). Ports
router-eval exp_37's build_comparison.py.

There is no per-arm sigma grid to select over here -- DMKDMR's own nested Brent search already
tunes sigma_x/sigma_y per fit. The tested grid is (seed, objective, k); this script selects the
best k per (seed, objective) by val cost_auc, picks the overall-best config across all 24 fits
and builds H3.

EmbedLLM has no DMKDMR-raw or gradient-trained KDM-head baseline in this repo, so neither H1
(encoded vs. raw) nor H2 (vs. kdm_frozen) is applicable here and both are omitted. The
comparison is 4-way: kNN, MLPDot, and the two DMKDMR-encoded headline arms (bce, pq_auc).

Reads:
  - results/dmkdmr_encoded/seed{S}/{objective}/k{K}/metrics.json  -- 24 fits' full A/B/C suite
  - results/dmkdmr_encoded/seed{S}/{objective}/k{K}/gap_bootstrap.json -- in-task (vs MLPDot) bootstrap
  - results/mlpdot_bce_reference_seed{S}.json                      -- per-seed shared MLPDot reference
  - knn-baseline/mlpdot-baseline's selected_metrics.json           -- kNN/MLPDot baselines

H3 (objective effect, bce vs pq_auc): matched-grid paired sign test over the 12 (seed, k) points
    shared by both arms -- same sign-test methodology as router-eval exp_37's H3, with (seed, k)
    as the matching key.
Reference vs MLPDot: the one comparison WITH a real paired bootstrap CI, computed in-task by
    run_dmkdmr_encoded_eval.py against that seed's precomputed MLPDot prediction.

Usage: python build_comparison.py   (run after all 24 dmkdmr_encoded array tasks have completed)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import experiment_dir  # noqa: E402

from src.eval.selection import select_by_cost_auc

EMBEDLLM = experiment_dir("embedllm")
EXP_DIR = experiment_dir("embedllm", "bfdkr-sweep")
OUT_RESULTS = EXP_DIR / "results"
OUT_REPORTS = EXP_DIR / "reports"

CONDITION = "dmkdmr_encoded"
SEEDS = [0, 1, 2]
OBJECTIVES = ["bce", "pq_auc"]
K_CANDIDATES = [128, 512, 2048, 4096]
HEADLINE_SEED = 1  # matches MLPDot's own exp_23-selected seed

TEST_KEYS_A = [  # ranking
    "test_pq_auc", "test_top1_acc", "test_top1_regret", "test_mrr",
    "test_recall_at_1", "test_recall_at_3", "test_recall_at_5", "test_recall_at_10",
    "test_ndcg_at_1", "test_ndcg_at_3", "test_ndcg_at_5", "test_ndcg_at_10",
    "test_llm_spearman", "test_llm_kendall",
]
TEST_KEYS_B = ["test_brier", "test_ece"]  # calibration
TEST_KEYS_C = [  # cost-aware
    "test_cost_auc", "test_floor_cost_auc", "test_oracle_cost_auc",
    "test_aiq", "test_aiq_deploy", "test_apgr", "test_cost_at_target_quality",
]
LOWER_IS_BETTER = {"test_top1_regret", "test_brier", "test_ece", "test_cost_at_target_quality"}


SOURCE_PATHS = {
    "knn": EMBEDLLM / "knn-baseline" / "results" / "selected_metrics.json",
    "mlpdot": EMBEDLLM / "mlpdot-baseline" / "results" / "selected_metrics.json",
}
SOURCE_EXPS = {
    "knn": "embedllm/knn-baseline", "mlpdot": "embedllm/mlpdot-baseline",
    "dmkdmr_enc_bce": f"dmkdmr_encoded/seed{HEADLINE_SEED}/bce (val-selected k)",
    "dmkdmr_enc_pqauc": f"dmkdmr_encoded/seed{HEADLINE_SEED}/pq_auc (val-selected k)",
}
ROUTER_LABELS = {
    "knn": "kNN", "mlpdot": "MLPDot",
    "dmkdmr_enc_bce": "DMKDMR-encoded (bce)", "dmkdmr_enc_pqauc": "DMKDMR-encoded (pq_auc)",
}
ROUTERS = ["knn", "mlpdot", "dmkdmr_enc_bce", "dmkdmr_enc_pqauc"]


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


def render_metric_table(lines, rows_by_router, keys, title, routers=ROUTERS, labels=ROUTER_LABELS):
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


# ---------------------------------------------------------------------------
# Load the 24 exp_25 fits
# ---------------------------------------------------------------------------

def _cdir(seed: int, objective: str, k: int) -> Path:
    return OUT_RESULTS / CONDITION / f"seed{seed}" / objective / f"k{k}"


def _load_all() -> list[dict]:
    """Pooled metrics + gap_bootstrap + config_audit for all 24 fits."""
    rows = []
    for seed in SEEDS:
        for objective in OBJECTIVES:
            for k in K_CANDIDATES:
                cdir = _cdir(seed, objective, k)
                metrics = _require(
                    cdir / "metrics.json",
                    f"run run_dmkdmr_encoded_eval.py for seed={seed}, objective={objective}, "
                    f"k={k} first.")
                pooled = next(r for r in metrics if r["group"] == "pooled")
                gap = _require(cdir / "gap_bootstrap.json", f"missing gap_bootstrap for {cdir}")
                row = dict(pooled)
                row["seed"], row["objective"], row["k"] = seed, objective, k
                row["gap_bootstrap_vs_mlpdot"] = gap
                rows.append(row)
    return rows


def _load_full_metrics(seed: int, objective: str, k: int) -> list[dict]:
    return _require(_cdir(seed, objective, k) / "metrics.json", "missing metrics.json")


# ---------------------------------------------------------------------------
# Selection: best k per (seed, objective); overall-best config across all 24
# ---------------------------------------------------------------------------

def _select_best_k(rows, seed, objective):
    cand = [r for r in rows if r["seed"] == seed and r["objective"] == objective]
    cand.sort(key=lambda r: r["k"])
    idx = select_by_cost_auc([r["val_cost_auc"] for r in cand], [r["val_brier"] for r in cand])
    return cand[idx]


def _select_overall_best(rows):
    idx = select_by_cost_auc([r["val_cost_auc"] for r in rows], [r["val_brier"] for r in rows])
    return rows[idx]


# ---------------------------------------------------------------------------
# H3: matched-grid (seed, k) paired comparison, bce vs pq_auc
# ---------------------------------------------------------------------------

def _sign_test_p(wins: int, n: int) -> float:
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(wins, n - wins) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail) if wins != n - wins else 1.0


def _h3_matched_grid(rows):
    by_key_bce = {(r["seed"], r["k"]): r for r in rows if r["objective"] == "bce"}
    by_key_pq = {(r["seed"], r["k"]): r for r in rows if r["objective"] == "pq_auc"}
    metrics = ["test_ece", "test_brier", "test_cost_auc", "test_pq_auc", "test_top1_acc"]
    lower_better = {"test_ece", "test_brier"}

    per_point, diffs = [], {m: [] for m in metrics}
    for key in sorted(by_key_bce):
        b, p = by_key_bce[key], by_key_pq[key]
        point = {"seed": key[0], "k": key[1]}
        for m in metrics:
            d = b[m] - p[m]  # bce - pq_auc
            diffs[m].append(d)
            point[f"{m}_bce_minus_pqauc"] = round(d, 6)
        per_point.append(point)

    summary = {}
    for m in metrics:
        vals = np.array(diffs[m])
        lower = m in lower_better
        wins = int((vals < 0).sum()) if lower else int((vals > 0).sum())
        n = len(vals)
        summary[m] = {
            "n": n, "bce_wins": wins, "mean_diff_bce_minus_pqauc": round(float(vals.mean()), 6),
            "std_diff": round(float(vals.std()), 6),
            "sign_test_p": round(_sign_test_p(wins, n), 4),
            "lower_is_better": lower,
        }
    return {"per_point": per_point, "summary": summary}


def main():
    print("Loading exp_25's 24 fits (3 seeds x 2 objectives x 4 k) ...")
    rows = _load_all()

    # --- Selection ---
    print("\nSelecting best k per (seed, objective), and overall-best config ...")
    headline_bce = _select_best_k(rows, HEADLINE_SEED, "bce")
    headline_pqauc = _select_best_k(rows, HEADLINE_SEED, "pq_auc")
    overall_best = _select_overall_best(rows)
    print(f"  Headline (seed={HEADLINE_SEED}, bce):    k={headline_bce['k']}  "
          f"test_cost_auc={headline_bce['test_cost_auc']:.5f}  test_ece={headline_bce['test_ece']:.5f}")
    print(f"  Headline (seed={HEADLINE_SEED}, pq_auc): k={headline_pqauc['k']}  "
          f"test_cost_auc={headline_pqauc['test_cost_auc']:.5f}  test_ece={headline_pqauc['test_ece']:.5f}")
    print(f"  Overall best (across all 24, by val cost_auc): seed={overall_best['seed']} "
          f"objective={overall_best['objective']} k={overall_best['k']}  "
          f"test_cost_auc={overall_best['test_cost_auc']:.5f}")

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "best_config.json").write_text(json.dumps({
        "seed": overall_best["seed"], "objective": overall_best["objective"], "k": overall_best["k"],
        "val_cost_auc": overall_best["val_cost_auc"], "test_cost_auc": overall_best["test_cost_auc"],
        "test_pq_auc": overall_best["test_pq_auc"], "test_ece": overall_best["test_ece"],
        "sigma_x": None, "sigma_y": None,  # filled from config_audit.json below
    }, indent=2))
    # Enrich best_config.json with sigma_x/sigma_y from that config's audit file
    audit = _require(_cdir(overall_best["seed"], overall_best["objective"], overall_best["k"])
                      / "config_audit.json", "missing config_audit for overall-best config")
    best_config = json.loads((OUT_RESULTS / "best_config.json").read_text())
    best_config["sigma_x"] = audit["config"]["sigma_x"]
    best_config["sigma_y"] = audit["config"]["sigma_y"]
    best_config["sigma_x_med"] = audit["config"]["sigma_x_med"]
    best_config["sigma_y_med"] = audit["config"]["sigma_y_med"]
    (OUT_RESULTS / "best_config.json").write_text(json.dumps(best_config, indent=2))
    print(f"  Wrote {OUT_RESULTS / 'best_config.json'} (sigma_x={best_config['sigma_x']}, "
          f"sigma_y={best_config['sigma_y']})")

    # --- H3 ---
    print("\nH3: matched-grid (seed, k) paired comparison, bce vs pq_auc (12 points) ...")
    h3 = _h3_matched_grid(rows)
    ece_s, pq_s = h3["summary"]["test_ece"], h3["summary"]["test_pq_auc"]
    print(f"  ECE: bce wins {ece_s['bce_wins']}/{ece_s['n']}  "
          f"mean(bce-pq_auc)={ece_s['mean_diff_bce_minus_pqauc']:+.5f}  sign_test_p={ece_s['sign_test_p']}")
    print(f"  pq_auc: bce wins {pq_s['bce_wins']}/{pq_s['n']}  "
          f"mean(bce-pq_auc)={pq_s['mean_diff_bce_minus_pqauc']:+.5f}  sign_test_p={pq_s['sign_test_p']}")
    h3_confirmed = ece_s["bce_wins"] >= 9 and ece_s["sign_test_p"] < 0.10
    print(f"  -> H3_bce_better_calibrated={h3_confirmed}")

    (OUT_RESULTS / "h3_objective_tradeoff.json").write_text(json.dumps(
        {**h3, "H3_bce_better_calibrated": h3_confirmed}, indent=2))

    # --- 4-way comparison table ---
    sources = {}
    for router in ("knn", "mlpdot"):
        data = _require(SOURCE_PATHS[router], f"run the upstream experiment for {router} first.")
        sources[router] = {r["group"]: r for r in data}

    bce_full = _load_full_metrics(HEADLINE_SEED, "bce", headline_bce["k"])
    pqauc_full = _load_full_metrics(HEADLINE_SEED, "pq_auc", headline_pqauc["k"])
    sources["dmkdmr_enc_bce"] = {r["group"]: r for r in bce_full}
    sources["dmkdmr_enc_pqauc"] = {r["group"]: r for r in pqauc_full}

    merged = []
    for router in ROUTERS:
        for group, row in sources[router].items():
            entry = {"router": ROUTER_LABELS[router], "group": group, "source_exp": SOURCE_EXPS[router]}
            for k, v in row.items():
                if k.startswith("test_") or k in ("Nq_test", "n_groups"):
                    entry[k] = v
            merged.append(entry)
    (OUT_RESULTS / "comparison_metrics.json").write_text(json.dumps(merged, indent=2))

    # --- Seed spread (dmkdmr_encoded bce arm, best-k selected per seed) ---
    seed_rows = [_select_best_k(rows, s, "bce") for s in SEEDS]
    seed_spread = {
        "cost_auc": {f"seed{r['seed']}": r["test_cost_auc"] for r in seed_rows},
        "pq_auc": {f"seed{r['seed']}": r["test_pq_auc"] for r in seed_rows},
        "ece": {f"seed{r['seed']}": r["test_ece"] for r in seed_rows},
    }
    for metric in seed_spread:
        vals = list(seed_spread[metric].values())
        seed_spread[metric]["mean"] = round(float(np.mean(vals)), 6)
        seed_spread[metric]["std"] = round(float(np.std(vals)), 6)
    (OUT_RESULTS / "seed_spread.json").write_text(json.dumps(seed_spread, indent=2))

    # --- reports/comparison_table.md ---
    lines = []
    lines.append("# EmbedLLM: DMKDMR on MLPDot-encoded embeddings vs. MLPDot and kNN "
                  "(ports router-eval exp_37)\n")
    lines.append(
        "Source: `experiments/embedllm/{knn-baseline,mlpdot-baseline}/results/selected_metrics.json` "
        "+ `results/dmkdmr_encoded/seed*/{bce,pq_auc}/k*/*.json`, merged by "
        "[`build_comparison.py`](../../../src/experiments/embedllm/bfdkr_sweep/build_comparison.py). "
        f"DMKDMR-encoded rows are seed={HEADLINE_SEED}'s (MLPDot's own selected seed) "
        "val-cost-auc-selected k per objective arm. The 'Reference vs MLPDot' bootstrap (see run "
        "logs / gap_bootstrap.json per fit) is a real in-task paired CI, computed against that "
        "seed's precomputed MLPDot prediction. EmbedLLM has no DMKDMR-raw or kdm_frozen baseline "
        "in this repo, so H1 (vs. raw) and H2 (vs. kdm_frozen), both present in router-eval "
        "exp_37, are not applicable here and are omitted.\n"
    )

    lines.append("## H3: matched-grid (seed, k) paired comparison, bce vs pq_auc (12 points)\n")
    lines.append("| Metric | bce wins | n | mean(bce-pq_auc) | sign-test p | Lower better |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for m, s in h3["summary"].items():
        lines.append(f"| {m.replace('test_', '')} | {s['bce_wins']} | {s['n']} | "
                      f"{s['mean_diff_bce_minus_pqauc']:+.6f} | {s['sign_test_p']:.4f} | "
                      f"{s['lower_is_better']} |")
    lines.append("")
    lines.append(f"**H3** (bce better calibrated, matched grid): "
                  f"{'CONFIRMED' if h3_confirmed else 'NOT CONFIRMED'} "
                  f"(ECE: {ece_s['bce_wins']}/12 wins, sign-test p={ece_s['sign_test_p']}).\n")

    lines.append("## Seed spread (dmkdmr_encoded, bce arm, best-k selected per seed)\n")
    lines.append("| Metric | seed0 | seed1 | seed2 | mean | std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for metric, label in (("cost_auc", "cost_auc"), ("pq_auc", "pq_auc"), ("ece", "ece")):
        d = seed_spread[metric]
        lines.append(f"| {label} | {d['seed0']:.6f} | {d['seed1']:.6f} | {d['seed2']:.6f} | "
                      f"{d['mean']:.6f} | {d['std']:.6f} |")
    lines.append("")

    lines.append("## 4-way comparison (pooled, all test queries)\n")
    pooled_by_router = {r: sources[r]["pooled"] for r in ROUTERS}
    render_metric_table(lines, pooled_by_router, TEST_KEYS_A, "A. Ranking")
    render_metric_table(lines, pooled_by_router, TEST_KEYS_B, "B. Calibration")
    render_metric_table(lines, pooled_by_router, TEST_KEYS_C, "C. Cost-aware (post-hoc)")

    lines.append("## Figures\n")
    lines.append(
        "- 5-router overlay: [`figures/method_comparison_pooled.png`](figures/method_comparison_pooled.png)\n"
        "- bce vs pq_auc (cost_auc, ECE) scatter: "
        "[`figures/bce_vs_pqauc_scatter.png`](figures/bce_vs_pqauc_scatter.png)\n"
        "- k-sweep curve: [`figures/k_sweep_curve.png`](figures/k_sweep_curve.png)\n"
        "- sigma_q x sigma_l sensitivity (winning config): "
        "[`figures/sigma_sensitivity_pq_auc.png`](figures/sigma_sensitivity_pq_auc.png), "
        "[`figures/sigma_sensitivity_ece.png`](figures/sigma_sensitivity_ece.png), "
        "[`figures/sigma_sensitivity_aiq.png`](figures/sigma_sensitivity_aiq.png)\n"
    )

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\nWrote:")
    for p in ("best_config.json", "h3_objective_tradeoff.json",
              "comparison_metrics.json", "seed_spread.json"):
        print(f"  {OUT_RESULTS / p}")
    print(f"  {OUT_REPORTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

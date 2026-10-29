"""
Metadata-representation ablation (EmbedLLM): aggregate the arms, run the paired bootstraps,
and render the hypothesis verdicts.

Reads only what `run_metadata_ablation.py` wrote -- each arm's `metrics.json`,
`other_routers_metrics.json`, `config_audit.json` and the per-router `P_test_*.npy` matrices --
plus each variant's `metadata_coverage.json` from the build step. No refitting.

Pairing across arms needs only a shared test-query order, which the bundle guarantees (every
arm is the same 10,702 test queries in the same order), so the arms do not have to be fit in one
process. That is the one relaxation from the request's section 4; it buys per-(arm, seed) array
tasks that are resumable and keeps worst-task walltime well inside the bfdkr-ablation precedent.

Bootstrap follows `src/experiments/*/bfdkr_ablation/run_ablation_encoded_eval.py`'s
`_bootstrap_distributions`: 1,000 resamples, one shared index matrix per seed, percentile CI on
the delta. Resampling is over test-query rows only; the cost vector is never resampled.
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, reports_dir, results_dir  # noqa: E402

from src import metrics as M
from src.eval import get_spec, load_splits

DATASET = "embedllm"
FAMILY = "embedllm"
EXPERIMENT = "metadata-ablation"

RESULTS_DIR = results_dir(FAMILY, EXPERIMENT)
REPORTS_DIR = reports_dir(FAMILY, EXPERIMENT)
VARIANT_ROOT = dataset_processed(DATASET) / "metadata_variants"

ARMS = ["fp", "meta_struct", "meta_name", "concat"]
REFERENCE = "fp"
SEEDS = [0, 1, 2]
ROUTERS = ["BF-DKR", "MLPDot", "kNN-enc", "kNN-raw"]
N_BOOT = 1000
BOOT_SEED = 42
TIE_BAND_COST_AUC = 0.0035

ARM_LABELS = {
    "fp": "Behavioral fingerprint (2048-d)",
    "meta_struct": "Structured metadata (size + MoE + family/org one-hot)",
    "meta_name": "Model-ID string embedding (2048-d)",
    "concat": "Fingerprint (+) structured metadata",
}


def _read(path: Path):
    if not path.exists():
        sys.exit(f"Missing {path} -- run run_metadata_ablation.py for every (arm, seed) first.")
    print(f"read: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def _pooled(rows):
    return rows if isinstance(rows, dict) else next(r for r in rows if r["group"] == "pooled")


# ── realised-quality decomposition (same construction as the static-policy baseline) ─────
#
# `cost_auc` is the AUC of a curve whose value at cap b is the mean over test queries of the
# realised score of whichever LLM the router picked at that cap. Precomputing the
# (n_queries, n_caps) matrix once makes the paired bootstrap a row-resample plus a column mean
# -- exact, and ~1000x cheaper than recomputing cost_metrics per resample.

def realized_quality_matrix(P, S_true, c):
    P, S_true, c = (np.asarray(x, dtype=np.float64) for x in (P, S_true, c))
    caps = np.unique(c)
    rows = np.arange(S_true.shape[0])
    out = np.empty((S_true.shape[0], caps.size))
    for j, b in enumerate(caps):
        idx = np.flatnonzero(c <= b)
        out[:, j] = S_true[rows, idx[P[:, idx].argmax(axis=1)]]
    return out


def _auc(means, caps, lo, hi):
    return float(M.cap_curve_auc(list(zip(caps.tolist(), means.tolist())), lo, hi))


def bootstrap_vs_reference(realized, caps, lo, hi, reference):
    rng = np.random.default_rng(BOOT_SEED)
    n = next(iter(realized.values())).shape[0]
    idx_matrix = rng.integers(0, n, size=(N_BOOT, n))
    dists, points = {}, {}
    for name, R in realized.items():
        points[name] = _auc(R.mean(0), caps, lo, hi)
        dists[name] = np.array([_auc(R[idx_matrix[b]].mean(0), caps, lo, hi)
                                 for b in range(N_BOOT)])
    out = {"reference": reference, "n_boot": N_BOOT, "n_queries": int(n),
           "boot_seed": BOOT_SEED, "tie_band_cost_auc": TIE_BAND_COST_AUC,
           "point_cost_auc": {k: round(v, 6) for k, v in points.items()}, "gaps": {}}
    for name in realized:
        if name == reference:
            continue
        g = dists[reference] - dists[name]
        ci = [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
        point = points[reference] - points[name]
        out["gaps"][name] = {
            "cost_auc_gap_point": round(point, 6),
            "cost_auc_gap_ci95": [round(ci[0], 6), round(ci[1], 6)],
            "cost_auc_gap_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "outside_tie_band": bool(abs(point) > TIE_BAND_COST_AUC),
        }
    return out


def _verdict(per_seed_gaps, direction="reference_wins"):
    """How many seeds show the reference beating the arm outside the tie-band with a CI
    excluding zero. The house rule from bfdkr-ablation: 3/3 confirms, <=1/3 refutes."""
    wins = sum(1 for g in per_seed_gaps
               if g["cost_auc_gap_point"] > TIE_BAND_COST_AUC and g["cost_auc_gap_excludes_zero"])
    losses = sum(1 for g in per_seed_gaps
                 if g["cost_auc_gap_point"] < -TIE_BAND_COST_AUC and g["cost_auc_gap_excludes_zero"])
    ties = len(per_seed_gaps) - wins - losses
    return {"n_seeds": len(per_seed_gaps), "reference_wins": wins,
            "arm_wins": losses, "ties_within_band": ties,
            "mean_gap": round(float(np.mean([g["cost_auc_gap_point"] for g in per_seed_gaps])), 6)}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    spec = get_spec(DATASET)
    sp = load_splits(spec)
    S_test = sp.S_test.astype(np.float64)
    c_test = np.asarray(sp.c_test, dtype=np.float64)
    caps = np.unique(c_test)
    lo, hi = float(c_test.min()), float(c_test.max())

    rows, boot, coverage = [], {"dataset": DATASET, "per_seed": {}}, {}

    for arm in ARMS:
        if arm == "fp":
            continue
        cov_path = VARIANT_ROOT / arm / "metadata_coverage.json"
        if cov_path.exists():
            coverage[arm] = _read(cov_path)

    per_seed_metrics = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        realized = {}
        for arm in ARMS:
            arm_dir = RESULTS_DIR / f"seed{seed}" / arm
            pooled = _pooled(_read(arm_dir / "metrics.json"))
            others = _read(arm_dir / "other_routers_metrics.json")
            audit = _read(arm_dir / "config_audit.json")
            per_seed_metrics[arm][seed] = {"BF-DKR": pooled, **others,
                                            "k": audit["k_selection"]["selected_k"],
                                            "encoded_dim": audit["llm_representation"]["encoded_dim"],
                                            "raw_dim": audit["llm_representation"]["raw_dim"]}
            P = np.load(arm_dir / "P_test_bfdkr.npy").astype(np.float64)
            realized[arm] = realized_quality_matrix(P, S_test, c_test)
        res = bootstrap_vs_reference(realized, caps, lo, hi, REFERENCE)
        boot["per_seed"][f"seed{seed}"] = res
        for arm, g in res["gaps"].items():
            print(f"  seed{seed} {REFERENCE}-{arm}: {g['cost_auc_gap_point']:+.6f} "
                  f"CI[{g['cost_auc_gap_ci95'][0]:+.6f}, {g['cost_auc_gap_ci95'][1]:+.6f}] "
                  f"excl0={g['cost_auc_gap_excludes_zero']}")

    # ── per-arm, per-router 3-seed aggregates ────────────────────────────────
    for arm in ARMS:
        for router in ROUTERS:
            vals = [per_seed_metrics[arm][s][router]["test_cost_auc"] for s in SEEDS]
            pq = [per_seed_metrics[arm][s][router]["test_pq_auc"] for s in SEEDS]
            rows.append({
                "router": f"{router} / {arm}", "arm": arm, "base_router": router,
                "group": "pooled", "source_exp": f"{FAMILY}/{EXPERIMENT}",
                "arm_label": ARM_LABELS[arm],
                "llm_encoded_dim": per_seed_metrics[arm][SEEDS[0]]["encoded_dim"],
                "llm_raw_dim": per_seed_metrics[arm][SEEDS[0]]["raw_dim"],
                "n_seeds": len(SEEDS),
                "test_cost_auc": round(float(np.mean(vals)), 6),
                "test_cost_auc_std": round(float(np.std(vals)), 6),
                "test_pq_auc": round(float(np.mean(pq)), 6),
                "test_floor_cost_auc": per_seed_metrics[arm][SEEDS[0]][router]["test_floor_cost_auc"],
                "test_oracle_cost_auc": per_seed_metrics[arm][SEEDS[0]][router]["test_oracle_cost_auc"],
                "test_aiq": round(float(np.mean(vals))
                                   - per_seed_metrics[arm][SEEDS[0]][router]["test_floor_cost_auc"], 6),
                "per_seed_cost_auc": [round(v, 6) for v in vals],
            })

    # ── hypothesis verdicts ──────────────────────────────────────────────────
    hypotheses = {}
    for h, arm, statement in (
        ("h1_meta_struct", "meta_struct",
         "fp > meta_struct in cost_auc on 3/3 seeds, CIs excluding 0"),
        ("h1_meta_name", "meta_name",
         "fp > meta_name in cost_auc on 3/3 seeds, CIs excluding 0"),
        ("h4_concat", "concat",
         "concat ties fp -- metadata adds nothing on top of behavior"),
    ):
        gaps = [boot["per_seed"][f"seed{s}"]["gaps"][arm] for s in SEEDS]
        hypotheses[h] = {"statement": statement, "arm": arm, "reference": REFERENCE,
                          "tie_band_cost_auc": TIE_BAND_COST_AUC,
                          "per_seed": gaps, **_verdict(gaps)}

    # H5 is a direction question with no failure mode: which metadata arm recovers more.
    ms = [per_seed_metrics["meta_struct"][s]["BF-DKR"]["test_cost_auc"] for s in SEEDS]
    mn = [per_seed_metrics["meta_name"][s]["BF-DKR"]["test_cost_auc"] for s in SEEDS]
    hypotheses["h5_name_vs_struct"] = {
        "statement": "meta_name > meta_struct (a learned text encoder recovers more than "
                      "hand-parsed fields); either direction is informative",
        "meta_name_mean": round(float(np.mean(mn)), 6),
        "meta_struct_mean": round(float(np.mean(ms)), 6),
        "mean_delta": round(float(np.mean(mn) - np.mean(ms)), 6),
        "n_seeds_meta_name_higher": int(sum(a > b for a, b in zip(mn, ms))),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gap_bootstrap.json").write_text(json.dumps(boot, indent=2))
    (RESULTS_DIR / "comparison_metrics.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "metadata_coverage.json").write_text(json.dumps(coverage, indent=2))
    for name, obj in hypotheses.items():
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2))

    # ── table ────────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# {DATASET}: metadata-representation ablation", "",
             "LLM-side representation swapped at the bundle level; the MLPDot encoder "
             "architecture and every router below it are unchanged across arms.", "",
             "## Pooled test `cost_auc` (3-seed mean +/- std)", "",
             "| Arm | LLM dim (raw -> encoded) | " + " | ".join(ROUTERS) + " |",
             "|---|---|" + "---|" * len(ROUTERS)]
    for arm in ARMS:
        cells = []
        for router in ROUTERS:
            vals = [per_seed_metrics[arm][s][router]["test_cost_auc"] for s in SEEDS]
            cells.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
        d = per_seed_metrics[arm][SEEDS[0]]
        lines.append(f"| `{arm}` | {d['raw_dim']} -> {d['encoded_dim']} | " + " | ".join(cells) + " |")
    lines += ["", f"## BF-DKR gaps vs `{REFERENCE}` (paired bootstrap, {N_BOOT} resamples, "
                  f"tie-band ±{TIE_BAND_COST_AUC})", "",
              "| Arm | Seed | Δcost_auc | 95% CI | Excludes 0? | Outside band? |",
              "|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm == REFERENCE:
            continue
        for s in SEEDS:
            g = boot["per_seed"][f"seed{s}"]["gaps"][arm]
            lines.append(f"| `{arm}` | {s} | {g['cost_auc_gap_point']:+.4f} | "
                          f"[{g['cost_auc_gap_ci95'][0]:+.4f}, {g['cost_auc_gap_ci95'][1]:+.4f}] | "
                          f"{'yes' if g['cost_auc_gap_excludes_zero'] else 'no'} | "
                          f"{'yes' if g['outside_tie_band'] else 'no'} |")
    lines += ["", "## One-hot coverage under cold-cold splits", "",
              "| Variant | Field | Distinct in train | top-K | Test in-vocab | Test UNK | "
              "Test UNK with saturated vocab |", "|---|---|---|---|---|---|---|"]
    for arm, cov in coverage.items():
        for field, c in cov.get("coverage", {}).items():
            lines.append(f"| `{arm}` | {field} | {c['n_distinct_train']} | {c['top_k']} | "
                          f"{c['test_in_vocab_rate']:.3f} | {c['test_unk_rate']:.3f} | "
                          f"{c['saturated_vocab']['test_unk_rate']:.3f} |")
    (REPORTS_DIR / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_DIR} and {REPORTS_DIR / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

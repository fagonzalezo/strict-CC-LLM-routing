"""
Aggregation for the EmbedLLM DMKDMR component ablation on MLPDot-encoded embeddings.

Reads, for each of the 5 arms (full, qk_off, lk_off, bw_off, knn_enc) x 3 encoder
seeds, the outputs of run_ablation_encoded_eval.py:
  - results/<arm>/seed{S}/metrics.json         -- pooled + per-group + macro rows
  - results/<arm>/seed{S}/gap_bootstrap.json   -- paired bootstrap vs `full` (or vs MLPDot for `full`)
  - results/<arm>/seed{S}/config_audit.json    -- sigma_x/sigma_y (or k_q/k_l for knn_enc)

Produces:
  - full_reproduction_gate.json  -- hard gate: our `full` arm (pq_auc, k=4096) must match
    embedllm/bfdkr-sweep's own pq_auc/k4096 cells per seed. If this fails, the knockout
    deltas below are not trustworthy until resolved.
  - h1_query_kernel.json / h2_llm_kernel.json / h3_brent.json  -- knockout arm vs `full`,
    3-seed mean delta + a derived tie-band verdict + bootstrap-CI-exclusion count.
  - h4_sigma_interiority.json  -- no-new-fit readout: how far every sweep-committed and
    this-ablation's-own sigma multiplier sits from the (0.015625, 4.0) search-range bounds
    (replaces router-eval's full_ext-fit-based H4 -- see DESIGN.md).
  - h5_knn.json  -- knn_enc vs `full` (the "ablate everything" reference).
  - attribution_table.json  -- 3-seed mean+-std per arm across the headline metrics, plus
    the derived tie-band and its source path.
  - router_eval_crossref.json  -- our knockout deltas beside router-eval/bfdkr-ablation's
    own committed deltas (directional only -- different representation scale, different
    tie-band, different arm set; see DESIGN.md §7).
  - reports/comparison_table.md

Usage: python build_ablation_comparison.py   (run after all 3 array tasks have completed)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, experiment_dir  # noqa: E402

EMBEDLLM = experiment_dir("embedllm")
ROUTEREVAL = experiment_dir("router-eval")
EXP_DIR = experiment_dir("embedllm", "bfdkr-ablation")
OUT_RESULTS = EXP_DIR / "results"
OUT_REPORTS = EXP_DIR / "reports"

SEEDS = [0, 1, 2]
ARMS = ["full", "qk_off", "lk_off", "bw_off", "knn_enc"]
KNOCKOUT_ARMS = ["qk_off", "lk_off", "bw_off", "knn_enc"]  # all compared vs "full"

REPRO_TOLERANCE = 0.002  # matches router-eval/bfdkr-ablation's own convention
SIGMA_MULT_RANGE = (0.015625, 4.0)  # must match run_ablation_encoded_eval.py's constant

METRICS_FOR_TABLE = ["test_cost_auc", "test_pq_auc", "test_ece", "test_brier",
                     "test_aiq", "test_aiq_deploy"]
LOWER_IS_BETTER = {"test_ece", "test_brier"}


def _require(path: Path, hint: str) -> dict:
    if not path.exists():
        sys.exit(f"Missing {path} -- {hint}")
    return json.loads(path.read_text())


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# Derived tie-band -- from embedllm/bfdkr-sweep's own full-vs-MLPDot bootstrap
# CI half-width at pq_auc/k4096, NOT a hardcoded constant (router-eval's ablation
# hardcoded 0.0035 from its own sweep; this repo's audit-trail convention wants
# every number traceable to a read()-able path instead).
# ---------------------------------------------------------------------------

def _derive_tie_band() -> tuple[float, str]:
    half_widths = []
    for seed in SEEDS:
        gap_path = (EMBEDLLM / "bfdkr-sweep" / "results" / "dmkdmr_encoded"
                    / f"seed{seed}" / "pq_auc" / "k4096" / "gap_bootstrap.json")
        gap = _require(gap_path, "embedllm/bfdkr-sweep must be complete to derive the tie-band.")
        ci = gap["cost_auc_gap_ci95"]
        half_widths.append((ci[1] - ci[0]) / 2.0)
    band = float(np.mean(half_widths))
    source = ("experiments/embedllm/bfdkr-sweep/results/dmkdmr_encoded/seed*/pq_auc/k4096/"
              "gap_bootstrap.json (full-vs-MLPDot cost_auc_gap_ci95 half-width, mean over 3 seeds)")
    return band, source


# ---------------------------------------------------------------------------
# Load the 15 fits (5 arms x 3 seeds)
# ---------------------------------------------------------------------------

def _cdir(arm: str, seed: int) -> Path:
    return OUT_RESULTS / arm / f"seed{seed}"


def _load_all() -> list[dict]:
    """Pooled metrics + gap_bootstrap + config_audit for all 15 (arm, seed) fits."""
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            cdir = _cdir(arm, seed)
            metrics = _require(
                cdir / "metrics.json",
                f"run run_ablation_encoded_eval.py --seed-idx {SEEDS.index(seed)} "
                f"(arm={arm}) first.")
            pooled = next(r for r in metrics if r["group"] == "pooled")
            gap = _require(cdir / "gap_bootstrap.json", f"missing gap_bootstrap.json for {cdir}")
            audit = _require(cdir / "config_audit.json", f"missing config_audit.json for {cdir}")
            row = dict(pooled)
            row["arm"], row["seed"] = arm, seed
            row["gap_bootstrap"] = gap
            row["config"] = audit["config"]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Full-reproduction gate: our `full` arm (pq_auc, k=4096) vs the sweep's own cells
# ---------------------------------------------------------------------------

def _full_reproduction_gate(rows: list[dict]) -> dict:
    per_seed = []
    for seed in SEEDS:
        ours = next(r for r in rows if r["arm"] == "full" and r["seed"] == seed)
        ref_dir = EMBEDLLM / "bfdkr-sweep" / "results" / "dmkdmr_encoded" / f"seed{seed}" / "pq_auc" / "k4096"
        ref_metrics = _require(ref_dir / "metrics.json",
                                f"embedllm/bfdkr-sweep seed{seed}/pq_auc/k4096 must be complete "
                                f"for the reproduction gate.")
        ref_pooled = next(r for r in ref_metrics if r["group"] == "pooled")
        d_cost = ours["test_cost_auc"] - ref_pooled["test_cost_auc"]
        d_pq = ours["test_pq_auc"] - ref_pooled["test_pq_auc"]
        d_ece = ours["test_ece"] - ref_pooled["test_ece"]
        passed = abs(d_cost) <= REPRO_TOLERANCE and abs(d_pq) <= REPRO_TOLERANCE
        per_seed.append({
            "seed": seed,
            "our_full": {"test_cost_auc": ours["test_cost_auc"], "test_pq_auc": ours["test_pq_auc"],
                        "test_ece": ours["test_ece"]},
            "sweep_pq_auc_k4096": {"test_cost_auc": ref_pooled["test_cost_auc"],
                                   "test_pq_auc": ref_pooled["test_pq_auc"],
                                   "test_ece": ref_pooled["test_ece"]},
            "delta_cost_auc": round(d_cost, 6), "delta_pq_auc": round(d_pq, 6),
            "delta_ece": round(d_ece, 6), "passed": passed,
        })
    return {"tolerance": REPRO_TOLERANCE, "per_seed": per_seed,
            "all_passed": all(p["passed"] for p in per_seed)}


# ---------------------------------------------------------------------------
# Generic knockout-arm-vs-full comparison (H1, H2, H3, H5) -- three verdicts
# ---------------------------------------------------------------------------

def _arm_vs_full(rows: list[dict], arm: str, tie_band: float) -> dict:
    per_seed = []
    for seed in SEEDS:
        full_row = next(r for r in rows if r["arm"] == "full" and r["seed"] == seed)
        arm_row = next(r for r in rows if r["arm"] == arm and r["seed"] == seed)
        gap = arm_row["gap_bootstrap"]
        per_seed.append({
            "seed": seed,
            "test_cost_auc": arm_row["test_cost_auc"], "full_test_cost_auc": full_row["test_cost_auc"],
            "delta_cost_auc": round(arm_row["test_cost_auc"] - full_row["test_cost_auc"], 6),
            "delta_pq_auc": round(arm_row["test_pq_auc"] - full_row["test_pq_auc"], 6),
            "delta_ece": round(arm_row["test_ece"] - full_row["test_ece"], 6),
            "delta_aiq_deploy": round(arm_row["test_aiq_deploy"] - full_row["test_aiq_deploy"], 6),
            "cost_auc_gap_ci95": gap["cost_auc_gap_ci95"],
            "cost_auc_gap_excludes_zero": gap["cost_auc_gap_excludes_zero"],
        })
    mean_d_cost = float(np.mean([p["delta_cost_auc"] for p in per_seed]))
    mean_d_pq = float(np.mean([p["delta_pq_auc"] for p in per_seed]))
    mean_d_ece = float(np.mean([p["delta_ece"] for p in per_seed]))
    mean_d_aiqd = float(np.mean([p["delta_aiq_deploy"] for p in per_seed]))
    n_excludes_zero = sum(p["cost_auc_gap_excludes_zero"] for p in per_seed)
    inert = bool(abs(mean_d_cost) < tie_band and n_excludes_zero == 0)
    load_bearing = bool(abs(mean_d_cost) >= tie_band and n_excludes_zero >= 2)
    inconclusive = bool(not inert and not load_bearing)
    return {
        "arm": arm, "per_seed": per_seed,
        "mean_delta_cost_auc": round(mean_d_cost, 6), "mean_delta_pq_auc": round(mean_d_pq, 6),
        "mean_delta_ece": round(mean_d_ece, 6), "mean_delta_aiq_deploy": round(mean_d_aiqd, 6),
        "tie_band_cost_auc": round(tie_band, 6), "n_seeds_ci_excludes_zero": n_excludes_zero,
        "verdict_inert": inert, "verdict_load_bearing": load_bearing,
        "verdict_inconclusive": inconclusive,
    }


# ---------------------------------------------------------------------------
# H4: sigma-interiority readout (no new fit -- see DESIGN.md)
# ---------------------------------------------------------------------------

def _h4_sigma_interiority(rows: list[dict]) -> dict:
    """Read every committed sweep sigma multiplier plus this ablation's own 3
    `full` fits, and report how far each sits from the (0.015625, 4.0) bounds.

    Replaces router-eval's full_ext-fit-based H4: EmbedLLM's own sweep already
    shows no fit anywhere near either bound, so a range-extension fit would be
    a guaranteed no-op (see DESIGN.md H4 for the seed1/pq_auc/k4096 arithmetic).
    """
    lo, hi = SIGMA_MULT_RANGE
    sweep_points = []
    for seed in SEEDS:
        for objective in ("bce", "pq_auc"):
            for k in (128, 512, 2048, 4096):
                audit_path = (EMBEDLLM / "bfdkr-sweep" / "results" / "dmkdmr_encoded"
                              / f"seed{seed}" / objective / f"k{k}" / "config_audit.json")
                if not audit_path.exists():
                    continue
                audit = json.loads(audit_path.read_text())
                cfg = audit["config"]
                sx_mult = cfg["sigma_x"] / cfg["sigma_x_med"] if cfg["sigma_x_med"] else None
                sy_mult = cfg["sigma_y"] / cfg["sigma_y_med"] if cfg["sigma_y_med"] else None
                sweep_points.append({"seed": seed, "objective": objective, "k": k,
                                     "sigma_x_mult": sx_mult, "sigma_y_mult": sy_mult})

    full_points = []
    for seed in SEEDS:
        full_row = next(r for r in rows if r["arm"] == "full" and r["seed"] == seed)
        full_points.append({"seed": seed,
                            "sigma_x_mult": full_row["config"]["sigma_x_mult"],
                            "sigma_y_mult": full_row["config"]["sigma_y_mult"]})

    def _bounds_summary(points, key):
        vals = [p[key] for p in points if p[key] is not None]
        if not vals:
            return {"min": None, "max": None, "min_ratio_to_lo": None, "min_ratio_to_hi": None}
        vmin, vmax = min(vals), max(vals)
        return {
            "min": round(vmin, 6), "max": round(vmax, 6),
            "min_ratio_to_lo": round(vmin / lo, 3),
            "min_ratio_to_hi": round(hi / vmax, 3),
        }

    sweep_sx = _bounds_summary(sweep_points, "sigma_x_mult")
    sweep_sy = _bounds_summary(sweep_points, "sigma_y_mult")
    full_sx = _bounds_summary(full_points, "sigma_x_mult")
    full_sy = _bounds_summary(full_points, "sigma_y_mult")

    # "No pinning" iff every value sits at least 2x away from both bounds.
    def _no_pinning(summary):
        if summary["min_ratio_to_lo"] is None:
            return False
        return summary["min_ratio_to_lo"] >= 2.0 and summary["min_ratio_to_hi"] >= 2.0

    no_pinning = bool(_no_pinning(sweep_sx) and _no_pinning(sweep_sy)
                       and _no_pinning(full_sx) and _no_pinning(full_sy))

    return {
        "note": "No full_ext fit here -- replaces router-eval/bfdkr-ablation's H4. See "
                "DESIGN.md: EmbedLLM's own sweep already shows every sigma multiplier far "
                "from the search-range bounds, so a range-extension fit would be a "
                "guaranteed no-op.",
        "sigma_mult_range": list(SIGMA_MULT_RANGE),
        "n_sweep_points": len(sweep_points),
        "sweep_sigma_x_mult": sweep_sx, "sweep_sigma_y_mult": sweep_sy,
        "full_points": full_points,
        "full_sigma_x_mult": full_sx, "full_sigma_y_mult": full_sy,
        "H4_no_pinning": no_pinning,
    }


# ---------------------------------------------------------------------------
# Attribution table: 3-seed mean +- std per arm
# ---------------------------------------------------------------------------

def _attribution_table(rows: list[dict], tie_band: float, tie_band_source: str) -> dict:
    table = {"tie_band_cost_auc": round(tie_band, 6), "tie_band_source": tie_band_source}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        entry = {}
        for m in METRICS_FOR_TABLE:
            vals = [r[m] for r in arm_rows]
            entry[m] = {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6)}
        table[arm] = entry
    return table


def _router_eval_crossref(h1: dict, h2: dict, h3: dict, h5: dict) -> dict:
    re_path = ROUTEREVAL / "bfdkr-ablation" / "results" / "attribution_table.json"
    re_table = _require(re_path, "router-eval/bfdkr-ablation must be complete for the crossref.")

    def _re_delta(arm: str, metric: str):
        if arm not in re_table or "full" not in re_table:
            return None
        return round(re_table[arm][metric]["mean"] - re_table["full"][metric]["mean"], 6)

    return {
        "note": "Directional only -- router-eval/bfdkr-ablation used a different absolute "
                "cost_auc scale (~0.74 vs. EmbedLLM's ~0.51), a different (hardcoded) "
                "tie-band, and a different arm set (router-eval also fit full_ext). Not a "
                "matched numeric comparison.",
        "source": str(re_path.relative_to(ROOT)),
        "qk_off": {"embedllm_mean_delta_cost_auc": h1["mean_delta_cost_auc"],
                   "router_eval_mean_delta_cost_auc": _re_delta("qk_off", "test_cost_auc")},
        "lk_off": {"embedllm_mean_delta_cost_auc": h2["mean_delta_cost_auc"],
                   "router_eval_mean_delta_cost_auc": _re_delta("lk_off", "test_cost_auc")},
        "bw_off": {"embedllm_mean_delta_cost_auc": h3["mean_delta_cost_auc"],
                   "router_eval_mean_delta_cost_auc": _re_delta("bw_off", "test_cost_auc")},
        "knn_enc": {"embedllm_mean_delta_cost_auc": h5["mean_delta_cost_auc"],
                    "router_eval_mean_delta_cost_auc": _re_delta("knn_enc", "test_cost_auc")},
    }


def main():
    print("Deriving tie-band from embedllm/bfdkr-sweep's full-vs-MLPDot bootstrap CI ...")
    tie_band, tie_band_source = _derive_tie_band()
    print(f"  tie_band_cost_auc={tie_band:.6f}  (source: {tie_band_source})")

    print("\nLoading this ablation's 15 fits (5 arms x 3 seeds) ...")
    rows = _load_all()

    print("\nFull-reproduction gate (our 'full' arm vs the sweep's pq_auc/k4096 cells) ...")
    gate = _full_reproduction_gate(rows)
    for p in gate["per_seed"]:
        status = "PASS" if p["passed"] else "FAIL"
        print(f"  seed={p['seed']}: delta_cost_auc={p['delta_cost_auc']:+.6f}  "
              f"delta_pq_auc={p['delta_pq_auc']:+.6f}  -> {status}")
    if not gate["all_passed"]:
        print("  WARNING: reproduction gate FAILED for >=1 seed -- do NOT trust the knockout "
              "deltas below until this is resolved.")
    (OUT_RESULTS / "full_reproduction_gate.json").write_text(json.dumps(gate, indent=2))

    print("\nH1: query kernel (qk_off vs full) ...")
    h1 = _arm_vs_full(rows, "qk_off", tie_band)
    print(f"  mean_delta_cost_auc={h1['mean_delta_cost_auc']:+.6f}  "
          f"inert={h1['verdict_inert']}  load_bearing={h1['verdict_load_bearing']}  "
          f"inconclusive={h1['verdict_inconclusive']}")

    print("\nH2: LLM kernel (lk_off vs full) ...")
    h2 = _arm_vs_full(rows, "lk_off", tie_band)
    print(f"  mean_delta_cost_auc={h2['mean_delta_cost_auc']:+.6f}  "
          f"mean_delta_aiq_deploy={h2['mean_delta_aiq_deploy']:+.6f}  "
          f"inert={h2['verdict_inert']}  load_bearing={h2['verdict_load_bearing']}  "
          f"inconclusive={h2['verdict_inconclusive']}")

    print("\nH3: Brent bandwidth search (bw_off vs full) ...")
    h3 = _arm_vs_full(rows, "bw_off", tie_band)
    print(f"  mean_delta_cost_auc={h3['mean_delta_cost_auc']:+.6f}  "
          f"mean_delta_ece={h3['mean_delta_ece']:+.6f}  "
          f"inert={h3['verdict_inert']}  load_bearing={h3['verdict_load_bearing']}  "
          f"inconclusive={h3['verdict_inconclusive']}")

    print("\nH4: sigma-interiority readout (no new fit) ...")
    h4 = _h4_sigma_interiority(rows)
    print(f"  H4_no_pinning={h4['H4_no_pinning']}")

    print("\nH5: kNN-on-encoded reference (knn_enc vs full) ...")
    h5 = _arm_vs_full(rows, "knn_enc", tie_band)
    print(f"  mean_delta_cost_auc={h5['mean_delta_cost_auc']:+.6f}  "
          f"inert={h5['verdict_inert']}  load_bearing={h5['verdict_load_bearing']}  "
          f"inconclusive={h5['verdict_inconclusive']}")

    print("\nAttribution table (3-seed mean +- std per arm) ...")
    attribution = _attribution_table(rows, tie_band, tie_band_source)
    crossref = _router_eval_crossref(h1, h2, h3, h5)

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "h1_query_kernel.json").write_text(json.dumps(h1, indent=2))
    (OUT_RESULTS / "h2_llm_kernel.json").write_text(json.dumps(h2, indent=2))
    (OUT_RESULTS / "h3_brent.json").write_text(json.dumps(h3, indent=2))
    (OUT_RESULTS / "h4_sigma_interiority.json").write_text(json.dumps(h4, indent=2))
    (OUT_RESULTS / "h5_knn.json").write_text(json.dumps(h5, indent=2))
    (OUT_RESULTS / "attribution_table.json").write_text(json.dumps(attribution, indent=2))
    (OUT_RESULTS / "router_eval_crossref.json").write_text(json.dumps(crossref, indent=2))

    # --- reports/comparison_table.md ---
    lines = []
    lines.append("# EmbedLLM: DMKDMR component ablation on MLPDot-encoded embeddings\n")
    lines.append(
        "Source: `experiments/embedllm/bfdkr-ablation/results/<arm>/seed*/*.json` (5 arms x "
        "3 seeds), merged by "
        "[`build_ablation_comparison.py`](../../../src/experiments/embedllm/bfdkr_ablation/"
        "build_ablation_comparison.py). All arms use the same MLPDot-encoded representation, "
        "`objective=\"pq_auc\"`, `k=4096`, `sigma_mult_range=(0.015625, 4.0)`. Every knockout "
        "comparison is a real in-task paired bootstrap (all arms' P_test coexist in the same "
        f"process). Tie-band (±{tie_band:.6f}) is derived from `{tie_band_source}`, not "
        "hardcoded.\n"
    )

    lines.append("## Full-reproduction gate\n")
    lines.append("Our `full` arm must match embedllm/bfdkr-sweep's own pq_auc/k=4096 cells "
                  f"per seed (tolerance ±{REPRO_TOLERANCE}).\n")
    lines.append("| Seed | Δcost_auc | Δpq_auc | Δece | Status |")
    lines.append("|---:|---:|---:|---:|---|")
    for p in gate["per_seed"]:
        lines.append(f"| {p['seed']} | {p['delta_cost_auc']:+.6f} | {p['delta_pq_auc']:+.6f} | "
                      f"{p['delta_ece']:+.6f} | {'PASS' if p['passed'] else 'FAIL'} |")
    gate_msg = ("ALL PASSED" if gate["all_passed"]
                else "FAILED for >=1 seed -- do not trust knockout deltas until resolved")
    lines.append(f"\n**Gate**: {gate_msg}.\n")

    def _hypothesis_section(title, h, extra_note=""):
        lines.append(f"## {title}\n")
        lines.append("| Seed | test_cost_auc | Δcost_auc (vs full) | 95% CI | Δpq_auc | Δece | "
                      "Δaiq_deploy |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for p in h["per_seed"]:
            ci = p["cost_auc_gap_ci95"]
            lines.append(f"| {p['seed']} | {p['test_cost_auc']:.6f} | {p['delta_cost_auc']:+.6f} | "
                          f"[{ci[0]:+.4f},{ci[1]:+.4f}] | {p['delta_pq_auc']:+.6f} | "
                          f"{p['delta_ece']:+.6f} | {p['delta_aiq_deploy']:+.6f} |")
        lines.append(f"\nMean Δcost_auc={h['mean_delta_cost_auc']:+.6f} "
                      f"(tie band ±{h['tie_band_cost_auc']}), "
                      f"Δpq_auc={h['mean_delta_pq_auc']:+.6f}, Δece={h['mean_delta_ece']:+.6f}, "
                      f"Δaiq_deploy={h['mean_delta_aiq_deploy']:+.6f}. "
                      f"{h['n_seeds_ci_excludes_zero']}/3 seeds' bootstrap CI excludes 0.\n")
        verdict = ("LOAD-BEARING" if h["verdict_load_bearing"]
                   else "INERT" if h["verdict_inert"] else "INCONCLUSIVE")
        lines.append(f"**Verdict**: {verdict}.{(' ' + extra_note) if extra_note else ''}\n")

    _hypothesis_section("H1: Query kernel (qk_off vs full)", h1)
    _hypothesis_section("H2: LLM kernel (lk_off vs full)", h2)
    _hypothesis_section("H3: Brent bandwidth search (bw_off vs full)", h3)
    _hypothesis_section("H5: kNN-on-encoded reference (knn_enc vs full)", h5)

    lines.append("## H4: sigma-interiority readout (no new fit)\n")
    lines.append(h4["note"] + "\n")
    lines.append(f"Search range: `{tuple(h4['sigma_mult_range'])}`. Across "
                  f"{h4['n_sweep_points']} committed sweep fits (all objectives/k/seeds): "
                  f"σ_x_mult ∈ [{fmt(h4['sweep_sigma_x_mult']['min'])}, "
                  f"{fmt(h4['sweep_sigma_x_mult']['max'])}], "
                  f"σ_y_mult ∈ [{fmt(h4['sweep_sigma_y_mult']['min'])}, "
                  f"{fmt(h4['sweep_sigma_y_mult']['max'])}]. This ablation's 3 `full` fits: "
                  f"σ_x_mult ∈ [{fmt(h4['full_sigma_x_mult']['min'])}, "
                  f"{fmt(h4['full_sigma_x_mult']['max'])}], "
                  f"σ_y_mult ∈ [{fmt(h4['full_sigma_y_mult']['min'])}, "
                  f"{fmt(h4['full_sigma_y_mult']['max'])}].\n")
    lines.append(f"\n**H4** (no pinning at either bound): "
                  f"{'CONFIRMED' if h4['H4_no_pinning'] else 'NOT CONFIRMED'}.\n")

    lines.append("## Attribution table (3-seed mean ± std, pooled test set)\n")
    header = "| Metric | " + " | ".join(ARMS) + " |"
    sep = "|---|" + "---:|" * len(ARMS)
    lines.append(header)
    lines.append(sep)
    for m in METRICS_FOR_TABLE:
        cells = []
        for arm in ARMS:
            e = attribution[arm][m]
            cells.append(f"{e['mean']:.6f} ± {e['std']:.6f}")
        lines.append(f"| {m.replace('test_', '')} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Cross-reference vs router-eval/bfdkr-ablation -- directional only\n")
    lines.append(crossref["note"] + "\n")
    lines.append("| Arm | EmbedLLM mean Δcost_auc | router-eval mean Δcost_auc |")
    lines.append("|---|---:|---:|")
    for arm in ("qk_off", "lk_off", "bw_off", "knn_enc"):
        c = crossref[arm]
        re_d = c["router_eval_mean_delta_cost_auc"]
        lines.append(f"| {arm} | {c['embedllm_mean_delta_cost_auc']:+.6f} | "
                      f"{fmt(re_d) if re_d is not None else '-'} |")
    lines.append("")

    lines.append("## Figures\n")
    lines.append(
        "- Knockout deltas (cost_auc / pq_auc / ece / aiq_deploy vs full): "
        "[`figures/knockout_deltas.png`](figures/knockout_deltas.png)\n"
        "- sigma-interiority scatter (H4 readout): "
        "[`figures/sigma_interiority.png`](figures/sigma_interiority.png)\n"
        "- Cost-quality overlay: "
        "[`figures/cost_quality_overlay.png`](figures/cost_quality_overlay.png)\n"
    )

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\nWrote:")
    for p in ("full_reproduction_gate.json", "h1_query_kernel.json", "h2_llm_kernel.json",
              "h3_brent.json", "h4_sigma_interiority.json", "h5_knn.json",
              "attribution_table.json", "router_eval_crossref.json"):
        print(f"  {OUT_RESULTS / p}")
    print(f"  {OUT_REPORTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

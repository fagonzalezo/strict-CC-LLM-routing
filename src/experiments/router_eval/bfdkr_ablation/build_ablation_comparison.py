"""
Exp 38: aggregation for the DMKDMR component ablation on MLPDot-encoded embeddings.

Reads, for each of the 6 arms (full, qk_off, lk_off, bw_off, full_ext, knn_enc)
x 3 encoder seeds, the outputs of run_ablation_encoded_eval.py:
  - results/<arm>/seed{S}/metrics.json         -- pooled + per-group + macro rows
  - results/<arm>/seed{S}/gap_bootstrap.json   -- paired bootstrap vs `full` (or vs MLPDot for `full`)
  - results/<arm>/seed{S}/config_audit.json    -- sigma_x/sigma_y (or k_q/k_l for knn_enc)

Produces:
  - full_reproduction_gate.json  -- hard gate: our `full` arm (pq_auc, k=4096) must match
    exp_37's own pq_auc/k4096 cells per seed (DESIGN.md §9). If this fails, the knockout
    deltas below are not trustworthy until resolved.
  - h1_query_kernel.json / h2_llm_kernel.json / h3_brent.json  -- knockout arm vs `full`,
    3-seed mean delta + tie-band verdict + bootstrap-CI-exclusion count (DESIGN.md §7).
  - h4_sigma_range.json  -- per-seed `full` vs `full_ext` selected sigma_y_mult; whether
    `full` pins near the old 0.0625 floor and whether `full_ext` drops below it.
  - h5_knn.json  -- knn_enc vs `full` (the "ablate everything" reference).
  - attribution_table.json  -- 3-seed mean+-std per arm across the headline metrics.
  - exp29_crossref.json  -- our knockout deltas beside exp_29's raw-embedding deltas
    (directional only -- exp_29 used a different, pre-src/eval protocol; see DESIGN.md §7).
  - reports/comparison_table.md

Usage: python build_ablation_comparison.py   (run after all 3 exp_38 array tasks have completed)
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import experiment_dir  # noqa: E402

ROUTEREVAL = experiment_dir("router-eval")
EXP_DIR = experiment_dir("router-eval", "bfdkr-ablation")
OUT_RESULTS = EXP_DIR / "results"
OUT_REPORTS = EXP_DIR / "reports"

SEEDS = [0, 1, 2]
ARMS = ["full", "qk_off", "lk_off", "bw_off", "full_ext", "knn_enc"]
KNOCKOUT_ARMS = ["qk_off", "lk_off", "bw_off", "full_ext", "knn_enc"]  # all compared vs "full"

TIE_BAND_COST_AUC = 0.0035  # exp_37 H2's seed-noise reference band
REPRO_TOLERANCE = 0.002     # matches exp_29's own ANCHOR_TOLERANCE convention
OLD_SIGMA_Y_FLOOR = 0.0625  # exp_37's sigma_y_mult_range lower bound

METRICS_FOR_TABLE = ["test_cost_auc", "test_pq_auc", "test_ece", "test_brier",
                     "test_aiq", "test_aiq_deploy"]
LOWER_IS_BETTER = {"test_ece", "test_brier"}

# exp_29's raw-embedding (pre-src/eval, pq_auc-primary, per-benchmark-macro protocol)
# knockout deltas, from experiments/router-eval/INDEX.md's exp_29 row. Directional
# cross-reference only -- NOT a matched numeric comparison (see DESIGN.md §7 risk note).
EXP29_RAW_DELTAS = {
    "qk_off": {"delta_pq_auc": -0.003, "note": "replicates specialist result; inert"},
    "lk_off": {"delta_pq_auc": -0.005, "delta_aiq_deploy": -0.036,
               "note": "inert for full-range ranking; deploy-window AIQ collapses"},
    "bw_off": {"delta_pq_auc": -0.026,
               "note": "ECE x3.2, AIQ -0.067; dominant component on raw embeddings"},
}


def _require(path: Path, hint: str) -> dict:
    if not path.exists():
        sys.exit(f"Missing {path} -- {hint}")
    return json.loads(path.read_text())


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# Load the 18 exp_38 fits (6 arms x 3 seeds)
# ---------------------------------------------------------------------------

def _cdir(arm: str, seed: int) -> Path:
    return OUT_RESULTS / arm / f"seed{seed}"


def _load_all() -> list[dict]:
    """Pooled metrics + gap_bootstrap + config_audit for all 18 (arm, seed) fits."""
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
# Full-reproduction gate: our `full` arm (pq_auc, k=4096) vs exp_37's own cells
# ---------------------------------------------------------------------------

def _full_reproduction_gate(rows: list[dict]) -> dict:
    per_seed = []
    for seed in SEEDS:
        ours = next(r for r in rows if r["arm"] == "full" and r["seed"] == seed)
        ref_dir = ROUTEREVAL / "bfdkr-sweep" / "results" / "dmkdmr_encoded" / f"seed{seed}" / "pq_auc" / "k4096"
        ref_metrics = _require(ref_dir / "metrics.json",
                                f"exp_37 seed{seed}/pq_auc/k4096 must be complete for the "
                                f"reproduction gate.")
        ref_pooled = next(r for r in ref_metrics if r["group"] == "pooled")
        d_cost = ours["test_cost_auc"] - ref_pooled["test_cost_auc"]
        d_pq = ours["test_pq_auc"] - ref_pooled["test_pq_auc"]
        d_ece = ours["test_ece"] - ref_pooled["test_ece"]
        passed = abs(d_cost) <= REPRO_TOLERANCE and abs(d_pq) <= REPRO_TOLERANCE
        per_seed.append({
            "seed": seed,
            "our_full": {"test_cost_auc": ours["test_cost_auc"], "test_pq_auc": ours["test_pq_auc"],
                        "test_ece": ours["test_ece"]},
            "exp37_pq_auc_k4096": {"test_cost_auc": ref_pooled["test_cost_auc"],
                                   "test_pq_auc": ref_pooled["test_pq_auc"],
                                   "test_ece": ref_pooled["test_ece"]},
            "delta_cost_auc": round(d_cost, 6), "delta_pq_auc": round(d_pq, 6),
            "delta_ece": round(d_ece, 6), "passed": passed,
        })
    return {"tolerance": REPRO_TOLERANCE, "per_seed": per_seed,
            "all_passed": all(p["passed"] for p in per_seed)}


# ---------------------------------------------------------------------------
# Generic knockout-arm-vs-full comparison (H1, H2, H3, H5)
# ---------------------------------------------------------------------------

def _arm_vs_full(rows: list[dict], arm: str) -> dict:
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
    inert = bool(abs(mean_d_cost) < TIE_BAND_COST_AUC and n_excludes_zero == 0)
    load_bearing = bool(abs(mean_d_cost) >= TIE_BAND_COST_AUC and n_excludes_zero >= 2)
    return {
        "arm": arm, "per_seed": per_seed,
        "mean_delta_cost_auc": round(mean_d_cost, 6), "mean_delta_pq_auc": round(mean_d_pq, 6),
        "mean_delta_ece": round(mean_d_ece, 6), "mean_delta_aiq_deploy": round(mean_d_aiqd, 6),
        "tie_band_cost_auc": TIE_BAND_COST_AUC, "n_seeds_ci_excludes_zero": n_excludes_zero,
        "verdict_inert": inert, "verdict_load_bearing": load_bearing,
    }


# ---------------------------------------------------------------------------
# H4: sigma_y range extension (full vs full_ext) -- is the exp_37 floor real?
# ---------------------------------------------------------------------------

def _h4_sigma_range(rows: list[dict]) -> dict:
    per_seed = []
    for seed in SEEDS:
        full_row = next(r for r in rows if r["arm"] == "full" and r["seed"] == seed)
        ext_row = next(r for r in rows if r["arm"] == "full_ext" and r["seed"] == seed)
        full_mult = full_row["config"]["sigma_y_mult"]
        ext_mult = ext_row["config"]["sigma_y_mult"]
        pinned_at_old_floor = (full_mult is not None
                                and abs(full_mult - OLD_SIGMA_Y_FLOOR) / OLD_SIGMA_Y_FLOOR < 0.05)
        moved_below_floor = ext_mult is not None and ext_mult < OLD_SIGMA_Y_FLOOR * 0.95
        per_seed.append({
            "seed": seed, "full_sigma_y_mult": full_mult, "full_ext_sigma_y_mult": ext_mult,
            "full_pinned_at_old_floor": pinned_at_old_floor,
            "full_ext_moved_below_floor": moved_below_floor,
            "delta_cost_auc": round(ext_row["test_cost_auc"] - full_row["test_cost_auc"], 6),
            "delta_pq_auc": round(ext_row["test_pq_auc"] - full_row["test_pq_auc"], 6),
        })
    n_pinned = sum(p["full_pinned_at_old_floor"] for p in per_seed)
    n_moved = sum(p["full_ext_moved_below_floor"] for p in per_seed)
    return {
        "per_seed": per_seed, "old_floor": OLD_SIGMA_Y_FLOOR,
        "n_seeds_pinned_at_old_floor": n_pinned, "n_seeds_moved_below_floor": n_moved,
        "H4_range_limited": bool(n_pinned >= 2 and n_moved >= 2),
    }


# ---------------------------------------------------------------------------
# Attribution table: 3-seed mean +- std per arm
# ---------------------------------------------------------------------------

def _attribution_table(rows: list[dict]) -> dict:
    table = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        entry = {}
        for m in METRICS_FOR_TABLE:
            vals = [r[m] for r in arm_rows]
            entry[m] = {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6)}
        table[arm] = entry
    return table


def _exp29_crossref(h1: dict, h2: dict, h3: dict) -> dict:
    return {
        "note": "Directional only -- exp_29 used the old pre-src/eval, per-benchmark-macro, "
                "pq_auc-primary protocol on the RAW sup-EmbedKit representation (different "
                "embeddings, different pipeline). Not a matched numeric comparison.",
        "source": "experiments/router-eval/INDEX.md exp_29 row",
        "qk_off": {"exp38_encoded_mean_delta_pq_auc": h1["mean_delta_pq_auc"],
                   "exp29_raw": EXP29_RAW_DELTAS["qk_off"]},
        "lk_off": {"exp38_encoded_mean_delta_pq_auc": h2["mean_delta_pq_auc"],
                   "exp38_encoded_mean_delta_aiq_deploy": h2["mean_delta_aiq_deploy"],
                   "exp29_raw": EXP29_RAW_DELTAS["lk_off"]},
        "bw_off": {"exp38_encoded_mean_delta_pq_auc": h3["mean_delta_pq_auc"],
                   "exp29_raw": EXP29_RAW_DELTAS["bw_off"]},
    }


def main():
    print("Loading exp_38's 18 fits (6 arms x 3 seeds) ...")
    rows = _load_all()

    print("\nFull-reproduction gate (our 'full' arm vs exp_37's pq_auc/k4096 cells) ...")
    gate = _full_reproduction_gate(rows)
    for p in gate["per_seed"]:
        status = "PASS" if p["passed"] else "FAIL"
        print(f"  seed={p['seed']}: delta_cost_auc={p['delta_cost_auc']:+.6f}  "
              f"delta_pq_auc={p['delta_pq_auc']:+.6f}  -> {status}")
    if not gate["all_passed"]:
        print("  WARNING: reproduction gate FAILED for >=1 seed -- do NOT trust the knockout "
              "deltas below until this is resolved (see IMPLEMENTATION.md verification step 2).")
    (OUT_RESULTS / "full_reproduction_gate.json").write_text(json.dumps(gate, indent=2))

    print("\nH1: query kernel (qk_off vs full) ...")
    h1 = _arm_vs_full(rows, "qk_off")
    print(f"  mean_delta_cost_auc={h1['mean_delta_cost_auc']:+.6f}  "
          f"inert={h1['verdict_inert']}  load_bearing={h1['verdict_load_bearing']}")

    print("\nH2: LLM kernel (lk_off vs full) ...")
    h2 = _arm_vs_full(rows, "lk_off")
    print(f"  mean_delta_cost_auc={h2['mean_delta_cost_auc']:+.6f}  "
          f"mean_delta_aiq_deploy={h2['mean_delta_aiq_deploy']:+.6f}  "
          f"inert={h2['verdict_inert']}  load_bearing={h2['verdict_load_bearing']}")

    print("\nH3: Brent bandwidth search (bw_off vs full) ...")
    h3 = _arm_vs_full(rows, "bw_off")
    print(f"  mean_delta_cost_auc={h3['mean_delta_cost_auc']:+.6f}  "
          f"mean_delta_ece={h3['mean_delta_ece']:+.6f}  "
          f"inert={h3['verdict_inert']}  load_bearing={h3['verdict_load_bearing']}")

    print("\nH4: sigma_y range extension (full vs full_ext) ...")
    h4 = _h4_sigma_range(rows)
    print(f"  n_seeds_pinned_at_old_floor={h4['n_seeds_pinned_at_old_floor']}/3  "
          f"n_seeds_moved_below_floor={h4['n_seeds_moved_below_floor']}/3  "
          f"H4_range_limited={h4['H4_range_limited']}")

    print("\nH5: kNN-on-encoded reference (knn_enc vs full) ...")
    h5 = _arm_vs_full(rows, "knn_enc")
    print(f"  mean_delta_cost_auc={h5['mean_delta_cost_auc']:+.6f}  "
          f"inert={h5['verdict_inert']}  load_bearing={h5['verdict_load_bearing']}")

    print("\nAttribution table (3-seed mean +- std per arm) ...")
    attribution = _attribution_table(rows)
    crossref = _exp29_crossref(h1, h2, h3)

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "h1_query_kernel.json").write_text(json.dumps(h1, indent=2))
    (OUT_RESULTS / "h2_llm_kernel.json").write_text(json.dumps(h2, indent=2))
    (OUT_RESULTS / "h3_brent.json").write_text(json.dumps(h3, indent=2))
    (OUT_RESULTS / "h4_sigma_range.json").write_text(json.dumps(h4, indent=2))
    (OUT_RESULTS / "h5_knn.json").write_text(json.dumps(h5, indent=2))
    (OUT_RESULTS / "attribution_table.json").write_text(json.dumps(attribution, indent=2))
    (OUT_RESULTS / "exp29_crossref.json").write_text(json.dumps(crossref, indent=2))

    # --- reports/comparison_table.md ---
    lines = []
    lines.append("# exp_38 -- router-eval: DMKDMR component ablation on MLPDot-encoded "
                  "embeddings\n")
    lines.append(
        "Source: `experiments/router-eval/exp_38/results/<arm>/seed*/*.json` (6 arms x 3 seeds), "
        "merged by [`scripts/build_ablation_comparison.py`](../scripts/build_ablation_comparison.py). "
        "All arms use the same MLPDot-encoded representation, `objective=\"pq_auc\"`, `k=4096`. "
        "Every knockout comparison IS a real in-task paired bootstrap (all six arms' P_test "
        "coexist in the same process, unlike exp_37's point-estimate-only H1/H2).\n"
    )

    lines.append("## Full-reproduction gate\n")
    lines.append("Our `full` arm must match exp_37's own pq_auc/k=4096 cells per seed "
                  f"(tolerance ±{REPRO_TOLERANCE}).\n")
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
                   else "INERT" if h["verdict_inert"] else "AMBIGUOUS")
        lines.append(f"**Verdict**: {verdict}.{(' ' + extra_note) if extra_note else ''}\n")

    _hypothesis_section("H1: Query kernel (qk_off vs full)", h1)
    _hypothesis_section("H2: LLM kernel (lk_off vs full)", h2,
                        extra_note="Secondary probe: check Δaiq_deploy above against exp_29's "
                        "raw-embedding collapse (Δ=-0.036).")
    _hypothesis_section("H3: Brent bandwidth search (bw_off vs full)", h3)
    _hypothesis_section("H5: kNN-on-encoded reference (knn_enc vs full)", h5)

    lines.append("## H4: sigma_y range extension (full vs full_ext)\n")
    lines.append(f"exp_37's `sigma_y_mult_range` floor was {OLD_SIGMA_Y_FLOOR}; `full_ext` extends "
                  "it to (1/1024, 4.0).\n")
    lines.append("| Seed | full σ_y_mult | full_ext σ_y_mult | Pinned at old floor? | "
                  "Moved below floor? | Δcost_auc | Δpq_auc |")
    lines.append("|---:|---:|---:|---|---|---:|---:|")
    for p in h4["per_seed"]:
        lines.append(f"| {p['seed']} | {fmt(p['full_sigma_y_mult'])} | "
                      f"{fmt(p['full_ext_sigma_y_mult'])} | "
                      f"{'YES' if p['full_pinned_at_old_floor'] else 'no'} | "
                      f"{'YES' if p['full_ext_moved_below_floor'] else 'no'} | "
                      f"{p['delta_cost_auc']:+.6f} | {p['delta_pq_auc']:+.6f} |")
    lines.append(f"\n**H4** (exp_37's optimum was range-limited): "
                  f"{'CONFIRMED' if h4['H4_range_limited'] else 'NOT CONFIRMED'} "
                  f"({h4['n_seeds_pinned_at_old_floor']}/3 seeds pinned at the old floor, "
                  f"{h4['n_seeds_moved_below_floor']}/3 moved below it under the wider range).\n")

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

    lines.append("## Cross-reference vs exp_29 (raw-embedding ablation) -- directional only\n")
    lines.append(crossref["note"] + "\n")
    lines.append("| Arm | exp_38 (encoded) mean Δpq_auc | exp_29 (raw) Δpq_auc | Note |")
    lines.append("|---|---:|---:|---|")
    for arm in ("qk_off", "lk_off", "bw_off"):
        c = crossref[arm]
        enc_d = c["exp38_encoded_mean_delta_pq_auc"]
        raw_d = EXP29_RAW_DELTAS[arm]["delta_pq_auc"]
        note = EXP29_RAW_DELTAS[arm].get("note", "")
        lines.append(f"| {arm} | {enc_d:+.6f} | {raw_d:+.6f} | {note} |")
    lines.append("")

    lines.append("## Figures\n")
    lines.append(
        "- Knockout deltas (cost_auc / pq_auc / ece / aiq_deploy vs full): "
        "[`figures/knockout_deltas.png`](figures/knockout_deltas.png)\n"
        "- sigma_y selected multiplier (full vs full_ext, floor pinning): "
        "[`figures/sigma_y_multiplier.png`](figures/sigma_y_multiplier.png)\n"
        "- Cost-quality overlay: "
        "[`figures/cost_quality_overlay.png`](figures/cost_quality_overlay.png)\n"
    )

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\nWrote:")
    for p in ("full_reproduction_gate.json", "h1_query_kernel.json", "h2_llm_kernel.json",
              "h3_brent.json", "h4_sigma_range.json", "h5_knn.json", "attribution_table.json",
              "exp29_crossref.json"):
        print(f"  {OUT_RESULTS / p}")
    print(f"  {OUT_REPORTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()

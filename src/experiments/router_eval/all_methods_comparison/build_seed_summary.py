"""
RouterEval: every router's headline metrics (cost_auc, aiq, pq_auc, ece) as 3-seed mean +/-
std, plus the test metrics of the val-selected seed, plus a separate BF-DKR component-
ablation table in the same format.

Sibling to `embedllm/all_methods_comparison/build_seed_summary.py` -- same structure, same
self-check philosophy, RouterEval's own artifacts and protocols (see the two RouterEval-
specific differences called out below; everything else mirrors the EmbedLLM script exactly).
Also sibling to `all_methods_comparison/build_comparison.py`, which reports the full A/B/C
metric suite at ONE reference config per router (no seed spread visible) -- this script is
the seed-level complement.

Naming: rows are labeled BF-DKR (the paper's name for the router implemented as `DMKDMRRouter`
in `src/routers/dmkdmr.py` -- see the README's naming table), not DMKDMR/dmkdmr_encoded. Artifact
paths and `read:` provenance lines keep the code name (`dmkdmr_encoded/...`) -- the rename is
presentation-only; `seed_summary.json` carries both `label` and `code_name` per row.

Two differences from the EmbedLLM sibling:

1. **Six ablation arms, not five.** RouterEval's `attribution_table.json` has an extra
   `full_ext` arm -- the H4 sigma-range-robustness check (Brent search extended to
   `(1/1024, 4.0)`), not a component knockout. `h4_sigma_range.json` is a diagnostic (keys:
   `per_seed, old_floor, n_seeds_pinned_at_old_floor, n_seeds_moved_below_floor,
   H4_range_limited`), not the knockout schema (`mean_delta_cost_auc`, `verdict_load_bearing`,
   `per_seed[].cost_auc_gap_excludes_zero`) the other four h-files share. `full_ext` gets its
   own verdict (`"robustness (ties full)"`) with `-` for Delta/CI, and is excluded from the
   cost_auc-sorted ranking the other five arms are sorted by (matching
   `src/paper/gen_tab_ablation.py`, which reports `full_ext` under a separate "sigma-range
   robustness" section, not in its main ARMS table).
2. **Three k protocols, not the EmbedLLM two-vs-one split.** RouterEval's `attribution_table.json`
   lacks the `tie_band_cost_auc`/`tie_band_source` keys the EmbedLLM one carries (tie band is
   read from the four knockout h-files here instead, which each carry their own
   `tie_band_cost_auc` -- asserted equal across all four). The protocol-reconciliation table
   is RouterEval's own three protocols (see `render_protocol_reconciliation` below), not a
   copy of EmbedLLM's k=2048-pinned-to-paper table.

Reads (all already-committed, no recomputation):
  - router-eval/bfdkr-sweep/results/dmkdmr_encoded/seed{S}/{bce,pq_auc}/k{128,512,2048,4096}/
    metrics.json -- BF-DKR, per-seed val-selected k (select_by_cost_auc).
  - router-eval/mlpdot-baseline/results/{selection_audit.json,selected_metrics.json} +
    router-eval/bfdkr-sweep/results/mlpdot_bce_reference_seed{S}.json -- MLPDot, joined on
    (arch, seed): the audit grid has cost_auc/aiq/pq_auc but no ECE; the sweep's reference
    files have ECE. Joined and guarded by matching test_cost_auc per seed.
  - router-eval/dropoutnet-clcrec-baselines/results/{dropoutnet,clcrec}/final/seed{S}/
    {selected_metrics.json,config_audit.json} -- fixed HP-search-winner config, 3 seeds.
  - coldrec-baselines/results/router_eval_finals_summary.json (config indices read from its
    own `config_idx` field, mirroring the EmbedLLM script's convention of never hardcoding
    an index that a committed artifact already carries) -- dict-shaped, no val_* keys
    (winners were chosen by argmax TEST pq_auc on seed0, not by validation).
  - router-eval/knn-baseline/results/selected_metrics.json -- deterministic, no seed axis.
  - router-eval/bfdkr-ablation/results/{full,qk_off,lk_off,bw_off,full_ext,knn_enc}/seed{S}/
    metrics.json + h1_query_kernel.json/h2_llm_kernel.json/h3_brent.json/h5_knn.json --
    BF-DKR component ablation, all knockout arms fixed at objective=pq_auc, k=4096 (knn_enc
    has neither key; full_ext is a robustness arm, not a knockout, see above).

Self-checks (exit non-zero on failure) recompute against every relevant precomputed aggregate
already committed in this repo: router_eval_finals_summary.json, seed_spread.json,
best_config.json, mlpdot selection_audit.json's `selected`, and bfdkr-ablation's
attribution_table.json -- plus the AIQ identity and floor/oracle invariance per seed, and a
reconciliation of the three RouterEval "BF-DKR pq_auc cost_auc" numbers that legitimately
exist in this repo under different k-selection protocols.

Usage: python build_seed_summary.py
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import experiment_dir, relative  # noqa: E402
from src.eval.selection import select_by_cost_auc  # noqa: E402

ROUTER_EVAL = experiment_dir("router-eval")
EXP_DIR = experiment_dir("router-eval", "all-methods-comparison")
OUT_RESULTS = EXP_DIR / "results"
OUT_REPORTS = EXP_DIR / "reports"

METRICS = ["cost_auc", "aiq", "pq_auc", "ece"]
LOWER_IS_BETTER = {"ece"}
SEEDS = [0, 1, 2]
K_CANDIDATES = [128, 512, 2048, 4096]

FLOOR = 0.538449      # RouterEval test_floor_cost_auc -- a property of the test LLM pool,
ORACLE = 0.947298      # invariant across method/seed; verified against every source below.
AIQ_TOL = 2e-6         # committed artifacts round test_aiq to 6dp independently of cost_auc-floor.
AGG_TOL = 1e-6         # tolerance vs. other committed 6dp aggregate files.

BFDKR_SWEEP = ROUTER_EVAL / "bfdkr-sweep" / "results" / "dmkdmr_encoded"
ABLATION_RESULTS = ROUTER_EVAL / "bfdkr-ablation" / "results"

ABLATION_ARMS = [
    ("full", "Full BF-DKR", None),
    ("qk_off", "No query kernel", "h1_query_kernel.json"),
    ("lk_off", "Hard nearest LLM", "h2_llm_kernel.json"),
    ("bw_off", "No bandwidth search", "h3_brent.json"),
    ("knn_enc", "Encoded kNN (memory-only)", "h5_knn.json"),
]
# full_ext is handled separately -- see module docstring point 1.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(path: Path):
    if not path.exists():
        sys.exit(f"Missing {path} -- run the upstream experiment first.")
    print(f"  read: {relative(path)}")
    return json.loads(path.read_text())


def _pooled(path: Path) -> dict:
    """Normalize Shape A (list of group rows) and Shape B (bare pooled dict) alike."""
    data = _load(path)
    rows = data if isinstance(data, list) else [data]
    return next(r for r in rows if r["group"] == "pooled")


def mean_std(values):
    a = np.asarray(values, dtype=float)  # ddof=0, matching every aggregate file in this repo
    return float(a.mean()), float(a.std(ddof=0))


# ---------------------------------------------------------------------------
# Row builder -- shared by every method and every ablation arm
# ---------------------------------------------------------------------------

def _build_row(label, code_name, source_exp, protocol, per_seed):
    n = len(per_seed)

    for r in per_seed:
        implied = r["test_cost_auc"] - r["test_floor_cost_auc"]
        if abs(implied - r["test_aiq"]) > AIQ_TOL:
            sys.exit(f"{label} seed={r['seed']}: AIQ identity broke -- "
                      f"cost_auc-floor={implied:.6f} != test_aiq={r['test_aiq']:.6f}")

    floors = {r["test_floor_cost_auc"] for r in per_seed}
    oracles = {r["test_oracle_cost_auc"] for r in per_seed}
    if len(floors) != 1 or len(oracles) != 1:
        sys.exit(f"{label}: floor/oracle disagree across seeds -- floors={floors} oracles={oracles}; "
                  f"these are pool properties and must be seed-invariant. Do not aggregate.")
    floor, oracle = next(iter(floors)), next(iter(oracles))
    if abs(floor - FLOOR) > AGG_TOL or abs(oracle - ORACLE) > AGG_TOL:
        sys.exit(f"{label}: floor/oracle {floor}/{oracle} != shared RouterEval constants "
                  f"{FLOOR}/{ORACLE}")

    mean, std = {}, {}
    for m in METRICS:
        vals = [r[f"test_{m}"] for r in per_seed]
        mean[m], std[m] = (vals[0], None) if n == 1 else mean_std(vals)

    if n > 1 and std["aiq"] is not None and abs(std["aiq"] - std["cost_auc"]) > AIQ_TOL:
        sys.exit(f"{label}: std(aiq)={std['aiq']:.6f} != std(cost_auc)={std['cost_auc']:.6f} "
                  f"(they should differ only by a constant floor)")

    val_selected = None
    if n > 1 and all(r.get("val_cost_auc") is not None for r in per_seed):
        idx = select_by_cost_auc([r["val_cost_auc"] for r in per_seed],
                                  [r["val_brier"] for r in per_seed])
        sel = per_seed[idx]
        val_selected = {"seed": sel["seed"], "config": sel["config"], "val_cost_auc": sel["val_cost_auc"],
                         **{f"test_{m}": sel[f"test_{m}"] for m in METRICS}}

    return {
        "label": label, "code_name": code_name, "source_exp": source_exp, "protocol": protocol,
        "n_seeds": n, "per_seed": per_seed, "mean": mean, "std": std, "val_selected": val_selected,
        "floor_cost_auc": floor, "oracle_cost_auc": oracle,
    }


def _seed_row(seed, config, test_row, val_cost_auc=None, val_brier=None):
    return {
        "seed": seed, "config": config, "val_cost_auc": val_cost_auc, "val_brier": val_brier,
        "test_cost_auc": test_row["test_cost_auc"], "test_aiq": test_row["test_aiq"],
        "test_pq_auc": test_row["test_pq_auc"], "test_ece": test_row["test_ece"],
        "test_floor_cost_auc": test_row["test_floor_cost_auc"],
        "test_oracle_cost_auc": test_row["test_oracle_cost_auc"],
    }


# ---------------------------------------------------------------------------
# BF-DKR (bce, pq_auc) -- per-seed val-selected k
# ---------------------------------------------------------------------------

def _bfdkr_grid(objective):
    """All 3 seeds x 4 k candidates for one objective, each file read exactly once."""
    return {seed: [dict(_pooled(BFDKR_SWEEP / f"seed{seed}" / objective / f"k{k}" / "metrics.json"), k=k)
                    for k in K_CANDIDATES]
            for seed in SEEDS}


def _bfdkr_per_seed(grid):
    per_seed = []
    for seed in SEEDS:
        cand = grid[seed]
        idx = select_by_cost_auc([c["val_cost_auc"] for c in cand], [c["val_brier"] for c in cand])
        chosen = cand[idx]
        per_seed.append(_seed_row(seed, f"k={chosen['k']}", chosen,
                                   chosen["val_cost_auc"], chosen["val_brier"]))
    return per_seed


def _bfdkr_fixed_k(grid, k):
    vals = [next(c for c in grid[seed] if c["k"] == k)["test_cost_auc"] for seed in SEEDS]
    return mean_std(vals)


# ---------------------------------------------------------------------------
# MLPDot -- join selection_audit.json's grid (cost_auc/aiq/pq_auc) with
# bfdkr-sweep's mlpdot_bce_reference_seed{S}.json (ece), guarded on test_cost_auc agreement
# ---------------------------------------------------------------------------

def _mlpdot_per_seed():
    audit = _load(ROUTER_EVAL / "mlpdot-baseline" / "results" / "selection_audit.json")
    selected = audit["selected"]
    arch = (selected["target_dim"], selected["hidden_dim"], selected["n_layers"])
    floor_row = _pooled(ROUTER_EVAL / "mlpdot-baseline" / "results" / "selected_metrics.json")

    per_seed = []
    for seed in SEEDS:
        grid_row = next(r for r in audit["grid"]
                         if (r["target_dim"], r["hidden_dim"], r["n_layers"]) == arch and r["seed"] == seed)
        ref = _load(ROUTER_EVAL / "bfdkr-sweep" / "results" / f"mlpdot_bce_reference_seed{seed}.json")
        ref_arch = (ref["arch"]["target_dim"], ref["arch"]["hidden_dim"], ref["arch"]["n_layers"])
        if ref_arch != arch:
            sys.exit(f"MLPDot join guard: reference arch {ref_arch} != selected arch {arch} at seed{seed}")
        if abs(ref["test_cost_auc"] - grid_row["test_cost_auc"]) > 1e-9:
            sys.exit(f"MLPDot join guard failed at seed{seed}: "
                      f"reference test_cost_auc={ref['test_cost_auc']} != "
                      f"audit grid test_cost_auc={grid_row['test_cost_auc']}")
        test_row = {
            "test_cost_auc": grid_row["test_cost_auc"], "test_aiq": grid_row["test_aiq"],
            "test_pq_auc": grid_row["test_pq_auc"], "test_ece": ref["test_ece"],
            "test_floor_cost_auc": floor_row["test_floor_cost_auc"],
            "test_oracle_cost_auc": floor_row["test_oracle_cost_auc"],
        }
        arch_str = f"target_dim={arch[0]},hidden_dim={arch[1]},n_layers={arch[2]}"
        per_seed.append(_seed_row(seed, arch_str, test_row, grid_row["val_cost_auc"], grid_row["val_brier"]))
    return per_seed, selected["seed"]


# ---------------------------------------------------------------------------
# DropoutNet / CLCRec -- fixed HP-search-winner config, 3 seeds
# ---------------------------------------------------------------------------

def _final_per_seed(condition):
    per_seed = []
    for seed in SEEDS:
        base = ROUTER_EVAL / "dropoutnet-clcrec-baselines" / "results" / condition / "final" / f"seed{seed}"
        p = _pooled(base / "selected_metrics.json")
        audit = _load(base / "config_audit.json")
        per_seed.append(_seed_row(seed, audit.get("hp_search_selected"), p,
                                   p.get("val_cost_auc"), p.get("val_brier")))
    return per_seed


# ---------------------------------------------------------------------------
# AGNN / FSGNN -- config_idx read from this module's own constants (see docstring: the
# same winners `build_router_eval_finals_summary.py` records). metrics.json is a bare dict
# (Shape B), no val_* keys: winners were chosen by argmax TEST pq_auc on seed0, not by
# validation.
# ---------------------------------------------------------------------------

def _coldrec_per_seed(model, config_idx):
    per_seed = []
    for seed in SEEDS:
        p = _pooled(experiment_dir("coldrec-baselines") / "results" / "router-eval" / model
                     / f"config{config_idx}" / f"seed{seed}" / "metrics.json")
        per_seed.append(_seed_row(seed, f"config{config_idx}", p))
    return per_seed


# ---------------------------------------------------------------------------
# kNN -- deterministic, no seed axis
# ---------------------------------------------------------------------------

def _knn_row():
    p = _pooled(ROUTER_EVAL / "knn-baseline" / "results" / "selected_metrics.json")
    per_seed = [_seed_row(None, f"k_q={p.get('best_k_q')},k_l={p.get('best_k_l')}", p)]
    return _build_row("kNN", "knn-baseline", "router-eval/knn-baseline",
                       "single deterministic run (no seed axis)", per_seed)


# ---------------------------------------------------------------------------
# BF-DKR component ablation -- knockout arms fixed at objective=pq_auc, k=4096 except
# knn_enc. full_ext (H4 sigma-range robustness) is NOT a knockout -- handled separately.
# ---------------------------------------------------------------------------

def _ablation_per_seed(arm):
    per_seed = []
    for seed in SEEDS:
        p = _pooled(ABLATION_RESULTS / arm / f"seed{seed}" / "metrics.json")
        k, objective = p.get("k"), p.get("objective")
        if arm != "knn_enc" and (k != 4096 or objective != "pq_auc"):
            sys.exit(f"Ablation arm {arm} seed{seed}: expected k=4096/objective=pq_auc, "
                      f"got k={k}/objective={objective} -- knockout deltas would not be apples-to-apples.")
        config = f"k={k},objective={objective}" if k is not None else "n/a (no KDM head)"
        per_seed.append(_seed_row(seed, config, p, p.get("val_cost_auc"), p.get("val_brier")))
    return per_seed


def _ablation_row(arm, label, hfile):
    per_seed = _ablation_per_seed(arm)
    protocol = "fixed k=4096, objective=pq_auc" if arm != "knn_enc" else "encoded kNN (no KDM head; no k/objective)"
    row = _build_row(label, f"bfdkr-ablation/{arm}", f"router-eval/bfdkr-ablation/results/{arm}", protocol, per_seed)
    if hfile is None:
        row.update(delta_cost_auc_vs_full=None, n_seeds_ci_excludes_zero=None, verdict="reference", h_file=None)
    else:
        h = _load(ABLATION_RESULTS / hfile)
        n_excl = sum(1 for r in h["per_seed"] if r["cost_auc_gap_excludes_zero"])
        verdict = ("load-bearing" if h["verdict_load_bearing"]
                    else "inert" if h["verdict_inert"] else "inconclusive")
        row.update(delta_cost_auc_vs_full=round(h["mean_delta_cost_auc"], 6),
                    n_seeds_ci_excludes_zero=f"{n_excl}/3", verdict=verdict, h_file=hfile)
    return row


def _full_ext_row():
    """full_ext is the H4 sigma-range-robustness check, not a component knockout -- it gets
    its own verdict and is reported alongside the other arms but excluded from the
    cost_auc-sorted ranking (matching gen_tab_ablation.py, which reports it under a separate
    'sigma-range robustness' section rather than its main ARMS table). Its Delta is the mean
    of h4_sigma_range.json's per-seed *point* deltas (delta_cost_auc), not a paired-bootstrap
    mean_delta_cost_auc like the other four arms -- h4 carries no bootstrap CI, only point
    per-seed deltas (see gen_tab_ablation.py's own D4/H4 section, which reads the same field)."""
    per_seed = _ablation_per_seed("full_ext")
    row = _build_row("Extended sigma-range (H4)", "bfdkr-ablation/full_ext",
                      "router-eval/bfdkr-ablation/results/full_ext",
                      "fixed k=4096, objective=pq_auc (Brent range extended to (1/1024, 4.0))",
                      per_seed)
    h4 = _load(ABLATION_RESULTS / "h4_sigma_range.json")
    delta = round(float(np.mean([r["delta_cost_auc"] for r in h4["per_seed"]])), 6)
    row.update(delta_cost_auc_vs_full=delta, n_seeds_ci_excludes_zero=None,
               verdict="robustness (ties full)", h_file="h4_sigma_range.json")
    return row


# ---------------------------------------------------------------------------
# Cross-checks against every other committed aggregate this table overlaps with
# ---------------------------------------------------------------------------

def _check_coldrec(model, row, finals):
    ref = finals[model]
    for m in METRICS:
        if abs(row["mean"][m] - ref[m]["mean"]) > AGG_TOL or abs(row["std"][m] - ref[m]["std"]) > AGG_TOL:
            sys.exit(f"{model} mismatch vs router_eval_finals_summary.json at {m}: "
                      f"got {row['mean'][m]:.6f}+/-{row['std'][m]:.6f}, "
                      f"expected {ref[m]['mean']:.6f}+/-{ref[m]['std']:.6f}")


def _check_bfdkr_bce(row):
    ss = _load(BFDKR_SWEEP.parent / "seed_spread.json")
    for m in ("cost_auc", "pq_auc", "ece"):
        if abs(row["mean"][m] - ss[m]["mean"]) > AGG_TOL or abs(row["std"][m] - ss[m]["std"]) > AGG_TOL:
            sys.exit(f"BF-DKR (bce) mismatch vs seed_spread.json at {m}")


def _check_bfdkr_pqauc_valselected(row):
    bc = _load(BFDKR_SWEEP.parent / "best_config.json")
    vs = row["val_selected"]
    got_k = int(vs["config"].split("=")[1])
    if vs["seed"] != bc["seed"] or got_k != bc["k"] or abs(vs["test_cost_auc"] - bc["test_cost_auc"]) > AGG_TOL:
        sys.exit(f"BF-DKR (pq_auc) val-selected (seed{vs['seed']}, k{got_k}, {vs['test_cost_auc']:.6f}) "
                  f"!= best_config.json (seed{bc['seed']}, k{bc['k']}, {bc['test_cost_auc']:.6f})")


def _check_mlpdot_valselected(row, audit_selected_seed):
    vs = row["val_selected"]
    if vs["seed"] != audit_selected_seed:
        sys.exit(f"MLPDot val-selected seed{vs['seed']} != selection_audit.json['selected']['seed']="
                  f"{audit_selected_seed}")


def _check_attribution_table(ablation_rows, full_ext_row):
    at = _load(ABLATION_RESULTS / "attribution_table.json")
    for arm, row in ablation_rows.items():
        ref = at[arm]
        for m in METRICS:
            ref_m = ref[f"test_{m}"]
            if abs(row["mean"][m] - ref_m["mean"]) > AGG_TOL or abs(row["std"][m] - ref_m["std"]) > AGG_TOL:
                sys.exit(f"attribution_table.json mismatch for {arm}/{m}: "
                          f"got {row['mean'][m]:.6f}+/-{row['std'][m]:.6f}, "
                          f"expected {ref_m['mean']:.6f}+/-{ref_m['std']:.6f}")
    ref_ext = at["full_ext"]
    for m in METRICS:
        ref_m = ref_ext[f"test_{m}"]
        if (abs(full_ext_row["mean"][m] - ref_m["mean"]) > AGG_TOL
                or abs(full_ext_row["std"][m] - ref_m["std"]) > AGG_TOL):
            sys.exit(f"attribution_table.json mismatch for full_ext/{m}: "
                      f"got {full_ext_row['mean'][m]:.6f}+/-{full_ext_row['std'][m]:.6f}, "
                      f"expected {ref_m['mean']:.6f}+/-{ref_m['std']:.6f}")

    # RouterEval's attribution_table.json carries no tie_band_cost_auc/tie_band_source keys
    # (unlike EmbedLLM's) -- pull the tie band from the four knockout h-files instead, which
    # each carry their own tie_band_cost_auc, and assert they agree.
    tie_bands = set()
    for _, _, hfile in ABLATION_ARMS:
        if hfile is None:
            continue
        h = _load(ABLATION_RESULTS / hfile)
        tie_bands.add(h["tie_band_cost_auc"])
    if len(tie_bands) != 1:
        sys.exit(f"Ablation h-files disagree on tie_band_cost_auc: {tie_bands}")
    tie_band = next(iter(tie_bands))
    tie_band_source = ("experiments/router-eval/bfdkr-ablation/results/h{1,2,3,5}_*.json "
                        "(each h-file's own tie_band_cost_auc; RouterEval's attribution_table.json "
                        "does not carry this key)")
    return tie_band, tie_band_source


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt_mean_std(mean, std):
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def _best_label(methods, key, lower_is_better=False):
    vals = {row["label"]: row["mean"][key] for row in methods if row["mean"][key] is not None}
    if not vals:
        return None
    return (min if lower_is_better else max)(vals, key=vals.get)


def render_table1(methods):
    lines = ["## Table 1 -- 3-seed mean ± std\n"]
    lines.append("| Method | cost_auc | aiq | pq_auc | ece | protocol |")
    lines.append("|---|---:|---:|---:|---:|---|")
    bests = {m: _best_label(methods, m, m in LOWER_IS_BETTER) for m in METRICS}
    for row in methods:
        cells = []
        for m in METRICS:
            s = fmt_mean_std(row["mean"][m], row["std"][m])
            if bests[m] == row["label"]:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {row['label']} | " + " | ".join(cells) + f" | {row['protocol']} |")
    lines.append("")
    return lines


def render_table2(methods):
    lines = ["## Table 2 -- val-selected seed (test metrics)\n"]
    lines.append("| Method | seed | config | val_cost_auc | cost_auc | aiq | pq_auc | ece |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in methods:
        vs = row["val_selected"]
        if vs is None:
            reason = "n/a (deterministic)" if row["n_seeds"] == 1 else "n/a (no val protocol)"
            lines.append(f"| {row['label']} | {reason} | | | | | | |")
            continue
        lines.append(
            f"| {row['label']} | seed{vs['seed']} | {vs['config']} | {vs['val_cost_auc']:.6f} | "
            + " | ".join(f"{vs[f'test_{m}']:.6f}" for m in METRICS) + " |")
    lines.append("")
    return lines


def render_table3(ablation_rows, full_ext_row, tie_band, tie_band_source):
    lines = [f"## Table 3 -- BF-DKR component ablation (equivalence band tie_band_cost_auc = "
             f"{tie_band}; source: {tie_band_source})\n"]
    lines.append("| Arm | cost_auc | aiq | pq_auc | ece | val-sel seed | "
                  "Δ cost_auc vs full | CI≠0 | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---|---|")
    for row in ablation_rows + [full_ext_row]:
        vs = row["val_selected"]
        vs_str = f"seed{vs['seed']}" if vs else "n/a"
        delta = "—" if row["delta_cost_auc_vs_full"] is None else f"{row['delta_cost_auc_vs_full']:+.6f}"
        ci = "—" if row["n_seeds_ci_excludes_zero"] is None else row["n_seeds_ci_excludes_zero"]
        cells = [fmt_mean_std(row["mean"][m], row["std"][m]) for m in METRICS]
        lines.append(f"| {row['label']} | " + " | ".join(cells)
                      + f" | {vs_str} | {delta} | {ci} | {row['verdict']} |")
    lines.append("")
    lines.append("Note: for the four knockout arms, the Δ column is the mean of same-seed "
                  "*paired* bootstrap differences (`mean_delta_cost_auc` in each arm's "
                  "`h*.json`), not the difference of the two arms' 3-seed means shown in the "
                  "metric columns -- both are correct, but they are different estimators (see "
                  "`src/paper/gen_tab_ablation.py`'s own footnote on this). `Extended "
                  "sigma-range (H4)`'s Δ is a *different* estimator again -- the mean of "
                  "`h4_sigma_range.json`'s per-seed **point** deltas (`delta_cost_auc`), since "
                  "H4 carries no paired-bootstrap CI, only point deltas; it is also excluded "
                  "from the cost_auc ranking above (last row, not sorted in) because it is a "
                  "robustness check on `full`, not a component knockout.\n")
    return lines


def render_protocol_reconciliation(valsel, k4096_3seed, seed1_k2048_single):
    lines = ["## Protocol reconciliation -- BF-DKR (pq_auc) cost_auc under three k protocols\n"]
    lines.append(
        "This report selects k per seed by validation cost_auc (Table 1). Two other "
        "defensible protocols are used elsewhere in this repo for the same arm and are listed "
        "here for cross-reference so a reader does not mistake the difference for a "
        "discrepancy. Unlike EmbedLLM (where the paper's Table 2 3-seed row IS the fixed-k2048 "
        "row), RouterEval's paper Table 2 draws its 3-seed row from the `full` ablation arm "
        "(fixed k=4096) and its single-seed row from a *different* protocol "
        "(`best_config.json`, seed1/k=2048) -- three genuinely different numbers, not two:\n")
    lines.append("| k protocol | cost_auc | Used by |")
    lines.append("|---|---:|---|")
    lines.append(f"| per-seed val-selected k (3-seed) | {valsel[0]:.6f} ± {valsel[1]:.6f} | "
                  f"**this report** (Table 1) |")
    lines.append(f"| fixed k=4096 (3-seed, `full` arm) | {k4096_3seed[0]:.6f} ± {k4096_3seed[1]:.6f} | "
                  f"`bfdkr-ablation/results/attribution_table.json[\"full\"]` "
                  f"→ paper Table 2 **3-seed** row, and this report's Table 3 |")
    lines.append(f"| seed1, k=2048 (single seed) | {seed1_k2048_single:.6f} | "
                  f"`bfdkr-sweep/results/best_config.json` → paper Table 2 "
                  f"**single-seed** row |")
    lines.append("")
    return lines


def render_footnotes(methods, agnn_row, fsgnn_row):
    lines = ["## Footnotes\n"]

    a_vals = sorted((r["seed"], r["test_cost_auc"]) for r in agnn_row["per_seed"])
    a_sign = "positive" if agnn_row["mean"]["aiq"] >= 0 else "negative"
    a_min_seed, a_min_val = min(a_vals, key=lambda t: t[1])
    lines.append("- **AGNN** per-seed cost_auc: " + ", ".join(f"seed{s}={v:.6f}" for s, v in a_vals)
                  + f" -- one collapsed seed (seed{a_min_seed}, {a_min_val:.3f} vs. the other "
                    f"two's ~0.66) drives the ±{agnn_row['std']['cost_auc']:.4f} std; mean AIQ "
                    f"is still {a_sign} ({agnn_row['mean']['aiq']:+.4f}) because the two "
                    "uncollapsed seeds sit above floor.")
    f_vals = sorted((r["seed"], r["test_cost_auc"]) for r in fsgnn_row["per_seed"])
    f_above = "above" if fsgnn_row["mean"]["cost_auc"] > FLOOR else "below"
    lines.append("- **FSGNN** per-seed cost_auc: " + ", ".join(f"seed{s}={v:.6f}" for s, v in f_vals)
                  + f" -- std={fsgnn_row['std']['cost_auc']:.6f} (near-deterministic across seeds), "
                    f"mean ({fsgnn_row['mean']['cost_auc']:.6f}) is {f_above} floor ({FLOOR}) → "
                    f"{'positive' if fsgnn_row['mean']['aiq'] >= 0 else 'negative'} AIQ "
                    f"({fsgnn_row['mean']['aiq']:+.4f}).")

    lines.append("- **Val-selected seed vs. 3-seed mean** (validation does not always pick the "
                  "best test seed):")
    for row in methods:
        vs = row["val_selected"]
        if vs is None:
            continue
        seed_vals = sorted(r["test_cost_auc"] for r in row["per_seed"])
        tag = ("best of 3" if abs(vs["test_cost_auc"] - seed_vals[-1]) < 1e-9
               else "worst of 3" if abs(vs["test_cost_auc"] - seed_vals[0]) < 1e-9
               else "middle of 3")
        lines.append(f"  - {row['label']}: val-selected seed{vs['seed']} "
                      f"test_cost_auc={vs['test_cost_auc']:.6f} vs 3-seed mean "
                      f"{row['mean']['cost_auc']:.6f} ({tag}).")

    lines.append("- **AGNN/FSGNN have no val-based selection at any level.** Their winning "
                  "configs (config18, config2) were chosen by argmax *test* pq_auc on seed0 "
                  "(see `build_router_eval_finals_summary.py`'s docstring), so their Table 2 "
                  "`n/a` is not a missing key -- there is no validation protocol behind those "
                  "rows, which makes them not strictly comparable to the other six. AGNN's "
                  "config18 supersedes the old config12 transcription (which did not "
                  "reproduce on rerun); FSGNN's config2 is unchanged from the original "
                  "selection.")
    lines.append(f"- Floor ({FLOOR}) and oracle ({ORACLE}) are shared constants across every "
                  "RouterEval row in both tables (checked per-seed and per-row above).")
    lines.append("- MLPDot's per-seed ECE is read from `mlpdot_bce_reference_seed{S}.json` "
                  "(`val_objective: bce`), while `mlpdot-baseline`'s own architecture selection "
                  "used `val_cost_auc`. The join guard above confirms these are the same three "
                  "fits (`test_cost_auc` matches to <1e-9), but the provenance differs from the "
                  "cost_auc/aiq/pq_auc columns, which come straight from `selection_audit.json`.")
    lines.append("")
    return lines


def render_ablation_footnotes(full_row, full_ext_row):
    lines = ["## Ablation-specific footnotes\n"]
    lines.append(f"- **Table 3's `Full BF-DKR` row ({full_row['mean']['cost_auc']:.6f}) "
                  "is the same fixed-k=4096 arm used for the paper's Table 2 3-seed row, but "
                  "NOT the same protocol as the paper's Table 2 single-seed row (k=2048, "
                  "seed1, via `best_config.json`).** RouterEval's paper Table 2 mixes two "
                  "different k values across its two cells; EmbedLLM's paper Table 2, by "
                  "contrast, uses a single consistent k=2048 protocol for both its 3-seed and "
                  "single-seed cells. See the protocol reconciliation table above. Table 3's "
                  "`Full BF-DKR` row is also NOT this report's own `BF-DKR (pq_auc)` Table 1 "
                  "row (per-seed val-selected k) -- same arm, a third k protocol.")
    lines.append("- `bfdkr-ablation/results/full_reproduction_gate.json` shows the ablation's "
                  "`full` arm is bit-identical (all deltas 0.0) to `bfdkr-sweep/.../pq_auc/"
                  "k4096` on all three seeds -- that identity is why the two experiments can be "
                  "quoted side by side here at all.")
    lines.append("- **`Encoded kNN` (Table 3) is not `kNN` (Table 1).** Table 1's kNN is the "
                  "raw-embedding baseline from `router-eval/knn-baseline`; Table 3's `knn_enc` "
                  "is the memory-only ablation of BF-DKR on MLPDot-encoded representations. "
                  "Different inputs, different numbers, similar names.")
    lines.append(f"- **`Extended sigma-range (H4)` is not a component knockout.** It's the "
                  "H4 robustness check (Brent search range extended to `(1/1024, 4.0)`) and "
                  f"ties `full` (Δcost_auc={full_ext_row['delta_cost_auc_vs_full']:+.6f}, well "
                  "inside the tie band) -- included here for completeness since "
                  "`attribution_table.json` carries it, but excluded from the cost_auc-sorted "
                  "ranking above and from `gen_tab_ablation.py`'s main ARMS table.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building RouterEval seed summary (Tables 1-2: all methods; Table 3: BF-DKR ablation) ...\n")

    print("BF-DKR (bce, pq_auc) -- reading 24 (seed, objective, k) fits ...")
    bce_grid = _bfdkr_grid("bce")
    pqauc_grid = _bfdkr_grid("pq_auc")
    bfdkr_bce = _build_row("BF-DKR (bce)", "dmkdmr_encoded/bce",
                            "router-eval/bfdkr-sweep/results/dmkdmr_encoded/*/bce",
                            "per-seed val-selected k", _bfdkr_per_seed(bce_grid))
    bfdkr_pqauc = _build_row("BF-DKR (pq_auc)", "dmkdmr_encoded/pq_auc",
                              "router-eval/bfdkr-sweep/results/dmkdmr_encoded/*/pq_auc",
                              "per-seed val-selected k", _bfdkr_per_seed(pqauc_grid))
    _check_bfdkr_bce(bfdkr_bce)
    _check_bfdkr_pqauc_valselected(bfdkr_pqauc)
    best_config = _load(BFDKR_SWEEP.parent / "best_config.json")
    seed1_k2048_single = next(c for c in pqauc_grid[1] if c["k"] == 2048)["test_cost_auc"]
    if abs(seed1_k2048_single - best_config["test_cost_auc"]) > AGG_TOL:
        sys.exit(f"Protocol reconciliation check failed: recomputed seed1/k2048 cost_auc "
                  f"{seed1_k2048_single:.6f} != best_config.json's {best_config['test_cost_auc']:.6f}")

    print("\nMLPDot -- joining selection_audit.json grid with mlpdot_bce_reference_seed*.json ...")
    mlpdot_per_seed, mlpdot_audit_seed = _mlpdot_per_seed()
    mlpdot_row = _build_row("MLPDot", "mlpdot-baseline", "router-eval/mlpdot-baseline",
                             "fixed val-selected architecture", mlpdot_per_seed)
    _check_mlpdot_valselected(mlpdot_row, mlpdot_audit_seed)

    print("\nDropoutNet / CLCRec -- fixed HP-search-winner config, 3 seeds ...")
    dropoutnet_row = _build_row("DropoutNet", "dropoutnet", "router-eval/dropoutnet-clcrec-baselines/dropoutnet",
                                 "fixed HP-search winner", _final_per_seed("dropoutnet"))
    clcrec_row = _build_row("CLCRec", "clcrec", "router-eval/dropoutnet-clcrec-baselines/clcrec",
                             "fixed HP-search winner", _final_per_seed("clcrec"))

    print("\nkNN -- deterministic single run ...")
    knn_row = _knn_row()

    print("\nAGNN / FSGNN -- reading router_eval_finals_summary.json for config_idx ...")
    finals = _load(experiment_dir("coldrec-baselines") / "results" / "router_eval_finals_summary.json")
    agnn_row = _build_row("AGNN", f"coldrec/AGNN/config{finals['AGNN']['config_idx']}",
                           "coldrec-baselines (router-eval)", "fixed HP-search winner (argmax test pq_auc)",
                           _coldrec_per_seed("AGNN", finals["AGNN"]["config_idx"]))
    fsgnn_row = _build_row("FSGNN", f"coldrec/FSGNN/config{finals['FSGNN']['config_idx']}",
                            "coldrec-baselines (router-eval)", "fixed HP-search winner (argmax test pq_auc)",
                            _coldrec_per_seed("FSGNN", finals["FSGNN"]["config_idx"]))
    _check_coldrec("AGNN", agnn_row, finals)
    _check_coldrec("FSGNN", fsgnn_row, finals)

    methods = sorted(
        [bfdkr_pqauc, bfdkr_bce, dropoutnet_row, mlpdot_row, knn_row, clcrec_row, agnn_row, fsgnn_row],
        key=lambda r: r["mean"]["cost_auc"], reverse=True)

    print("\nBF-DKR component ablation -- 5 knockout arms (fixed k=4096/pq_auc, knn_enc "
          "excepted) + 1 robustness arm (full_ext) ...")
    ablation_rows_by_arm = {arm: _ablation_row(arm, label, hfile) for arm, label, hfile in ABLATION_ARMS}
    full_ext_row = _full_ext_row()
    ablation_rows = sorted(ablation_rows_by_arm.values(), key=lambda r: r["mean"]["cost_auc"], reverse=True)
    tie_band, tie_band_source = _check_attribution_table(ablation_rows_by_arm, full_ext_row)

    print("\nAll self-checks passed.\n")

    # --- write results JSON ---
    def _round_row(row):
        out = dict(row)
        out["mean"] = {k: (round(v, 6) if v is not None else None) for k, v in row["mean"].items()}
        out["std"] = {k: (round(v, 6) if v is not None else None) for k, v in row["std"].items()}
        return out

    payload = {
        "floor_cost_auc": FLOOR, "oracle_cost_auc": ORACLE,
        "methods": [_round_row(r) for r in methods],
        "ablation": [_round_row(r) for r in ablation_rows],
        "ablation_full_ext_robustness": _round_row(full_ext_row),
        "ablation_tie_band_cost_auc": tie_band,
        "ablation_tie_band_source": tie_band_source,
        "protocol_reconciliation": {
            "bfdkr_pqauc_val_selected_k": {"mean": round(bfdkr_pqauc["mean"]["cost_auc"], 6),
                                            "std": round(bfdkr_pqauc["std"]["cost_auc"], 6)},
            "bfdkr_pqauc_fixed_k4096_3seed": {"mean": round(ablation_rows_by_arm["full"]["mean"]["cost_auc"], 6),
                                               "std": round(ablation_rows_by_arm["full"]["std"]["cost_auc"], 6)},
            "bfdkr_pqauc_seed1_k2048_single": round(seed1_k2048_single, 6),
        },
    }
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "seed_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    # --- write markdown report ---
    lines = ["# RouterEval: all methods, 3-seed mean ± std\n"]
    lines.append(
        "3-seed mean ± std (ddof=0) and val-selected-seed test metrics for every RouterEval "
        "router in this repo, plus a separate BF-DKR component-ablation table. Generated by "
        "[`build_seed_summary.py`](../../../src/experiments/router_eval/all_methods_comparison/"
        "build_seed_summary.py). Complements "
        "[`comparison_table.md`](comparison_table.md), which reports the full A/B/C metric "
        "suite at one reference config with no seed spread.\n")
    lines.append(
        "> **BF-DKR** is the paper's name for the router implemented as `DMKDMRRouter` in "
        "`src/routers/dmkdmr.py`. Artifact paths, condition names and result directories keep "
        "the code name (`dmkdmr_encoded/`), so a `read:` line pointing at "
        "`dmkdmr_encoded/seed1/...` is the BF-DKR row's source, not a different method. See the "
        "naming table in README.md.\n")

    lines += render_table1(methods)
    lines += render_table2(methods)
    lines += render_protocol_reconciliation(
        (bfdkr_pqauc["mean"]["cost_auc"], bfdkr_pqauc["std"]["cost_auc"]),
        (ablation_rows_by_arm["full"]["mean"]["cost_auc"], ablation_rows_by_arm["full"]["std"]["cost_auc"]),
        seed1_k2048_single)
    lines += render_footnotes(methods, agnn_row, fsgnn_row)
    lines += render_table3(ablation_rows, full_ext_row, tie_band, tie_band_source)
    lines += render_ablation_footnotes(ablation_rows_by_arm["full"], full_ext_row)

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "seed_summary.md").write_text("\n".join(lines) + "\n")

    print("Wrote:")
    print(f"  {relative(OUT_RESULTS / 'seed_summary.json')}")
    print(f"  {relative(OUT_REPORTS / 'seed_summary.md')}")


if __name__ == "__main__":
    main()

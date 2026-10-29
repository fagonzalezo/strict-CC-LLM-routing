"""
Static-policy baseline (router-eval), stage 2: S1 / S2 / S2b / S3 and the paired bootstraps.

Answers Part E objection 5 -- *"is per-query routing doing anything a static
best-affordable-model policy wouldn't?"* The paper brackets BF-DKR with a zero-skill `floor`
(a random affordable LLM) and an unreachable `oracle` (per-query max within the cap), but
never reports the policy a pragmatic engineer would deploy: pick one good model per budget and
send every query to it.

Arms
----
S1  largest affordable, `w = c`. Pure metadata, zero training.
S2  learned static quality ranking: a regressor from an LLM's **raw** fingerprint to its mean
    accuracy over train queries, fit on train LLMs only and applied to the cold test LLMs.
    Raw (not encoded) makes S2 seed-free, which is the simpler and better foil. S2 sees
    exactly what BF-DKR sees and differs in one respect only: it cannot condition on the query.
S2b BF-DKR query-marginalised: `w = P_test.mean(0)` from stage 1's matrix. Same model, same
    fit, same information -- only the per-query *variation* removed. No regressor choice, no
    hyperparameter selection, nothing a reviewer can attribute to a weak foil.
S3  hindsight-best static policy, `w = S_test.mean(0)`. The upper bound over *all*
    query-independent policies. **Anchor, never a Table III baseline row** -- it uses test
    labels, and its JSON carries `uses_test_labels: true` so no table generator can pick it up.

Two correctness gates that must pass before any number here means anything
---------------------------------------------------------------------------
* **S1's tie-break.** `np.argmax` takes the first maximal column, and 146 of the 376 test LLMs
  are 7B, so a naive S1 would score as an artifact of column order. S1 is computed instead as
  the mean over the *tied-largest* affordable stratum (`src/routers/static_policy.py`), and
  `s1_order_independence` re-runs it under a permuted column order.
* **S3 two ways.** The rank-1 construction must agree with the running max of
  `src/metrics.py::single_llm_points` within each cap. If those disagree, the rank-1
  construction is wrong somewhere and *every* static arm is suspect -- the script exits.

Usage
-----
  python run_static_policies.py --smoke
  python run_static_policies.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.routers.static_policy import (
    StaticRouter,
    largest_affordable_curve,
    largest_affordable_per_query,
)

DATASET = "router-eval"
FAMILY = "router-eval"
EXPERIMENT = "static-policy-baseline"

RESULTS_DIR = results_dir(FAMILY, EXPERIMENT)
SEEDS = [0, 1, 2]

N_BOOT = 1000
BOOT_SEED = 42
TIE_BAND_COST_AUC = 0.0035

# S2 hyperparameter grids. RouterEval has 374 val LLMs, so val-cost_auc selection under the
# same cap protocol is well-resolved. (EmbedLLM's twin cuts this to a tiny grid -- 16 val LLMs
# is exactly the section VII validation-resolution fragility, and a noisy S2 could manufacture
# a spurious BF-DKR win on the one dataset the paper reports as a tie.)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KNN_KS = [1, 3, 5, 10, 25, 50]
S2_SELECTION = "val_cost_auc"


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# ── realised-quality decomposition ───────────────────────────────────────────
#
# Every arm's `cost_auc` is the AUC of a curve whose value at cap `b` is the *mean over test
# queries* of the realised score of whichever LLM the policy picked at that cap. Precomputing
# the `(n_queries, n_caps)` matrix of those realised scores once turns the paired bootstrap
# into row-resampling plus a column mean -- exact, not an approximation, and ~1000x cheaper
# than recomputing `cost_metrics` per resample. `_assert_decomposition` proves the equivalence
# against `per_query_cap_curve` before anything is reported.

def realized_quality_matrix(P: np.ndarray, S_true: np.ndarray, c: np.ndarray) -> np.ndarray:
    """`(n_queries, n_caps)` realised scores of the LLM `P` routes each query to, per cap."""
    P = np.asarray(P, dtype=np.float64)
    S_true = np.asarray(S_true, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    caps = np.unique(c)
    n = S_true.shape[0]
    out = np.empty((n, caps.size), dtype=np.float64)
    rows = np.arange(n)
    for j, b in enumerate(caps):
        idx = np.flatnonzero(c <= b)
        chosen = idx[P[:, idx].argmax(axis=1)]
        out[:, j] = S_true[rows, chosen]
    return out


def curve_from_matrix(realized: np.ndarray, c: np.ndarray) -> list[tuple[float, float]]:
    return list(zip(np.unique(np.asarray(c, dtype=np.float64)).tolist(),
                    realized.mean(axis=0).tolist()))


def cost_auc_from_matrix(realized: np.ndarray, c: np.ndarray, lo: float, hi: float) -> float:
    return float(M.cap_curve_auc(curve_from_matrix(realized, c), lo, hi))


def _assert_decomposition(P, S_true, c, lo, hi, label):
    """The decomposition must reproduce `per_query_cap_curve`'s own router series exactly."""
    mine = curve_from_matrix(realized_quality_matrix(P, S_true, c), c)
    theirs = M.per_query_cap_curve(P, S_true, c)["router"]
    mine_q = np.array([q for _, q in mine])
    theirs_q = np.array([q for _, q in theirs])
    if not np.allclose(mine_q, theirs_q, atol=1e-12):
        sys.exit(f"{label}: realised-quality decomposition disagrees with per_query_cap_curve "
                  f"(max |diff| = {np.abs(mine_q - theirs_q).max():.3e}). Every bootstrap here "
                  "would be wrong; refusing to continue.")
    direct = M.cost_metrics(P, S_true, c)["scalars"]["cost_auc"]
    mine_auc = float(M.cap_curve_auc(mine, lo, hi))
    if abs(direct - mine_auc) > 1e-12:
        sys.exit(f"{label}: cost_auc via decomposition {mine_auc:.12f} != via cost_metrics "
                  f"{direct:.12f}")


# ── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_gaps(realized: dict[str, np.ndarray], c, lo, hi, reference: str,
                    n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Paired bootstrap of `reference - arm` for every arm, on one shared index matrix.

    Follows `src/experiments/*/bfdkr_ablation/run_ablation_encoded_eval.py`'s
    `_bootstrap_distributions`: one `idx_matrix` drawn once, every arm's distribution computed
    on it, so each pairwise gap is an array subtraction and the pairing is exact. Resampling
    is over test-query rows only; `c` is never resampled.
    """
    rng = np.random.default_rng(seed)
    n = next(iter(realized.values())).shape[0]
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    caps = np.unique(np.asarray(c, dtype=np.float64))

    dists, points = {}, {}
    for name, R in realized.items():
        points[name] = float(M.cap_curve_auc(list(zip(caps.tolist(), R.mean(0).tolist())), lo, hi))
        d = np.empty(n_boot)
        for b in range(n_boot):
            means = R[idx_matrix[b]].mean(axis=0)
            d[b] = M.cap_curve_auc(list(zip(caps.tolist(), means.tolist())), lo, hi)
        dists[name] = d

    out = {"reference": reference, "n_boot": int(n_boot), "n_queries": int(n),
           "boot_seed": int(seed), "tie_band_cost_auc": TIE_BAND_COST_AUC,
           "point_cost_auc": {k: round(v, 6) for k, v in points.items()}, "gaps": {}}
    for name in realized:
        if name == reference:
            continue
        gaps = dists[reference] - dists[name]
        point = points[reference] - points[name]
        ci = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
        out["gaps"][f"{reference}_minus_{name}"] = {
            "cost_auc_gap_point": round(point, 6),
            "cost_auc_gap_ci95": [round(ci[0], 6), round(ci[1], 6)],
            "cost_auc_gap_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "outside_tie_band": bool(abs(point) > TIE_BAND_COST_AUC),
        }
    return out


# ── S2 regressors ────────────────────────────────────────────────────────────

def _ridge_fit_predict(X, y, alpha, X_new):
    """Closed-form ridge on centred features; no sklearn dependency in this repo's stack."""
    Xm, ym = X.mean(0), float(y.mean())
    Xc = X - Xm
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    coef = np.linalg.solve(A, Xc.T @ (y - ym))
    return (X_new - Xm) @ coef + ym


def _knn_fit_predict(X, y, k, X_new):
    """Cosine-similarity kNN regression -- the same geometry `KNNRouter` uses."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Nn = X_new / np.maximum(np.linalg.norm(X_new, axis=1, keepdims=True), 1e-12)
    sim = Nn @ Xn.T
    k = min(k, X.shape[0])
    nbrs = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    return y[nbrs].mean(axis=1)


def fit_s2(L_train, y_train, L_val, S_val, c_val, lo_val, hi_val, alphas, ks):
    """Select S2's regressor and hyperparameter on the **val** LLM pool, never on test."""
    audit = []
    best = None
    for family, grid, fn in (("ridge", alphas, _ridge_fit_predict), ("knn", ks, _knn_fit_predict)):
        for hp in grid:
            w_val = fn(L_train, y_train, hp, L_val)
            P_val = StaticRouter(w_val).predict(np.empty((S_val.shape[0], 0)),
                                                 np.empty((len(w_val), 0)))
            val_ca = float(M.cost_metrics(P_val, S_val, c_val)["scalars"]["cost_auc"])
            audit.append({"family": family, "hp": float(hp), "val_cost_auc": round(val_ca, 6)})
            if best is None or val_ca > best["val_cost_auc"]:
                best = {"family": family, "hp": float(hp), "val_cost_auc": val_ca}
    return best, audit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_root = Path(args.out_dir)
    print(f"Static-policy baseline ({DATASET}) | smoke={args.smoke} | out={out_root}")

    spec = get_spec(DATASET)
    sp = load_splits(spec)
    S_test = sp.S_test.astype(np.float64)
    c_test = np.asarray(sp.c_test, dtype=np.float64)
    S_val = sp.S_val.astype(np.float64)
    c_val = np.asarray(sp.c_val, dtype=np.float64)
    L_train, L_val, L_test = sp.L_train, sp.L_val, sp.L_test
    y_train = sp.S_train.astype(np.float64).mean(axis=0)   # each train LLM's mean accuracy
    lo, hi = float(c_test.min()), float(c_test.max())
    lo_v, hi_v = float(c_val.min()), float(c_val.max())
    n_q = S_test.shape[0]
    print(f"  test: {n_q} queries x {S_test.shape[1]} LLMs | c in [{lo}, {hi}] "
          f"({np.unique(c_test).size} distinct costs)")

    base = M.per_query_cap_curve(S_test, S_test, c_test)
    floor_auc = float(M.cap_curve_auc(base["floor"], lo, hi))
    oracle_auc = float(M.cap_curve_auc(base["oracle"], lo, hi))
    print(f"  floor_cost_auc={floor_auc:.6f}  oracle_cost_auc={oracle_auc:.6f}")

    realized: dict[str, np.ndarray] = {}
    curves: dict[str, list] = {}
    arm_json: dict[str, dict] = {}

    def register(name, R, extra=None, uses_test_labels=False):
        realized[name] = R
        curve = curve_from_matrix(R, c_test)
        curves[name] = [[float(b), float(q)] for b, q in curve]
        ca = float(M.cap_curve_auc(curve, lo, hi))
        row = {
            "arm": name, "dataset": DATASET, "group": "pooled",
            "n_queries": int(n_q), "n_llms_test": int(S_test.shape[1]),
            "test_cost_auc": round(ca, 6),
            "test_floor_cost_auc": round(floor_auc, 6),
            "test_oracle_cost_auc": round(oracle_auc, 6),
            "test_aiq": round(ca - floor_auc, 6),
            "uses_test_labels": bool(uses_test_labels),
        }
        for b in sorted({1.0, 3.0, 7.0, 30.0, hi}):
            row[f"test_quality_at_cap_{b:.0f}B"] = round(float(M.cap_quality_at_budget(curve, b)), 6)
        if extra:
            row.update(extra)
        arm_json[name] = row
        print(f"  {name:>6}: cost_auc={ca:.6f}  aiq={ca - floor_auc:+.6f}")
        return row

    # ── S1: largest affordable, order-independent over the tied size stratum ──
    R_s1 = largest_affordable_per_query(S_test, c_test)
    curve_s1 = largest_affordable_curve(S_test, c_test)
    assert np.allclose(R_s1.mean(0), [q for _, q in curve_s1])
    perm = np.random.default_rng(12345).permutation(S_test.shape[1])
    curve_perm = largest_affordable_curve(S_test[:, perm], c_test[perm])
    order_ok = bool(np.allclose([q for _, q in curve_s1], [q for _, q in curve_perm]))
    if not order_ok:
        sys.exit("S1 is not order-independent -- the tie-break is wrong.")
    n_tied = {float(b): int((c_test[c_test <= b] == c_test[c_test <= b].max()).sum())
              for b in np.unique(c_test)}
    register("S1", R_s1, extra={
        "policy": "largest affordable (w = c), averaged over the tied-largest stratum",
        "s1_order_independence": order_ok,
        "max_tied_stratum_size": int(max(n_tied.values())),
        "test_brier": None, "test_ece": None,
        "calibration_note": "n/a -- w = c is a parameter count, not a probability "
                             "(emitting an ECE from a cost vector would be meaningless)",
    })

    # ── S2: learned static quality ranking, raw fingerprints, seed-free ──
    best, s2_audit = fit_s2(L_train, y_train, L_val, S_val, c_val, lo_v, hi_v,
                             RIDGE_ALPHAS, KNN_KS)
    ridge_best = max((a for a in s2_audit if a["family"] == "ridge"),
                      key=lambda a: a["val_cost_auc"])
    knn_best = max((a for a in s2_audit if a["family"] == "knn"),
                    key=lambda a: a["val_cost_auc"])

    def _s2_test_cost_auc(family, hp):
        fn = _ridge_fit_predict if family == "ridge" else _knn_fit_predict
        hp = float(hp) if family == "ridge" else int(hp)
        w = fn(L_train, y_train, hp, L_test)
        P = StaticRouter(w).predict(np.empty((n_q, 0)), np.empty((len(w), 0)))
        return P, float(M.cost_metrics(P, S_test, c_test)["scalars"]["cost_auc"])

    # Ridge is *the* S2 row; kNN-regression is a robustness sub-row, so the Table III row is
    # unambiguous regardless of which family happened to win on val.
    P_ridge, ridge_test_ca = _s2_test_cost_auc("ridge", ridge_best["hp"])
    _, knn_test_ca = _s2_test_cost_auc("knn", knn_best["hp"])
    _assert_decomposition(P_ridge, S_test, c_test, lo, hi, "S2")
    register("S2", realized_quality_matrix(P_ridge, S_test, c_test), extra={
        "policy": "ridge from raw fingerprint to train-mean accuracy, fit on train LLMs only",
        "representation": "raw fingerprint (seed-free by construction)",
        "selected": {"family": "ridge", "alpha": ridge_best["hp"],
                      "val_cost_auc": ridge_best["val_cost_auc"]},
        "selection_rule": S2_SELECTION,
        "n_val_llms": int(L_val.shape[0]),
        "val_winner_across_families": best,
        "robustness_knn_regression": {
            "k": int(knn_best["hp"]),
            "val_cost_auc": knn_best["val_cost_auc"],
            "test_cost_auc": round(knn_test_ca, 6)},
        "grid_audit": s2_audit,
    })

    # ── S2b: BF-DKR query-marginalised, one per encoder seed ──
    s2b_rows = {}
    for seed in SEEDS:
        pred = out_root / "bfdkr_ref" / f"seed{seed}" / "P_test.npy"
        if not pred.exists():
            if args.smoke:
                # Smoke exercises the CPU arms and every correctness gate before any GPU time
                # is spent on stage 1, so a missing reference matrix is expected here.
                print(f"  [smoke] {pred.name} absent for seed{seed} -- skipping BF-DKR/S2b")
                continue
            sys.exit(f"Missing {pred} -- run run_bfdkr_reference.py --seed-idx {seed} first.")
        print(f"read: {pred.relative_to(ROOT)}")
        P_bf = np.load(pred).astype(np.float64)
        _assert_decomposition(P_bf, S_test, c_test, lo, hi, f"BFDKR/seed{seed}")
        register(f"BFDKR_seed{seed}", realized_quality_matrix(P_bf, S_test, c_test),
                  extra={"policy": "BF-DKR at its per-seed val-selected k", "seed": seed})
        w_s2b = P_bf.mean(axis=0)
        P_s2b = StaticRouter(w_s2b).predict(np.empty((n_q, 0)), np.empty((len(w_s2b), 0)))
        row = register(f"S2b_seed{seed}", realized_quality_matrix(P_s2b, S_test, c_test),
                        extra={"policy": "BF-DKR query-marginalised (w = P_test.mean(0))",
                               "seed": seed,
                               "w": [round(float(v), 8) for v in w_s2b]})
        s2b_rows[seed] = row

    # ── S3: hindsight-best static policy (anchor; uses test labels) ──
    w_s3 = S_test.mean(axis=0)
    P_s3 = StaticRouter(w_s3).predict(np.empty((n_q, 0)), np.empty((len(w_s3), 0)))
    _assert_decomposition(P_s3, S_test, c_test, lo, hi, "S3")
    register("S3", realized_quality_matrix(P_s3, S_test, c_test), extra={
        "policy": "best static policy in hindsight (w = S_test.mean(0))",
        "role": "anchor over all query-independent policies -- alongside floor and oracle, "
                 "never a Table III baseline row",
        "test_brier": None, "test_ece": None,
    }, uses_test_labels=True)

    # ── S3 cross-check: rank-1 vs running max of single_llm_points ──
    pts = M.single_llm_points(S_test, c_test)
    caps = np.unique(c_test)
    running = []
    for b in caps:
        best_q = max(q for cst, q in pts if cst <= b)
        running.append((float(b), float(best_q)))
    rank1 = curve_from_matrix(realized["S3"], c_test)
    diff = np.abs(np.array([q for _, q in rank1]) - np.array([q for _, q in running]))
    cross = {
        "rank1_cost_auc": round(float(M.cap_curve_auc(rank1, lo, hi)), 9),
        "single_llm_running_max_cost_auc": round(float(M.cap_curve_auc(running, lo, hi)), 9),
        "max_abs_curve_diff": float(diff.max()),
        "agree": bool(diff.max() < 1e-9),
        "n_caps": int(caps.size),
        "note": "Two independent derivations of the hindsight-best static policy. If these "
                 "disagree, the rank-1 construction is wrong and every static arm is suspect.",
    }
    print(f"  S3 cross-check: max|diff|={diff.max():.3e} -> {'AGREE' if cross['agree'] else 'DISAGREE'}")
    if not cross["agree"]:
        _atomic_write(out_root / "s3_crosscheck.json", cross)
        sys.exit("S3 cross-check FAILED -- see s3_crosscheck.json.")

    if args.smoke:
        print("[smoke] all gates passed; no artifacts written.")
        return

    # ── paired bootstraps: BF-DKR (per seed) vs each static arm ──
    missing = [s_ for s_ in SEEDS if f"BFDKR_seed{s_}" not in realized]
    if missing:
        sys.exit(f"Missing BF-DKR reference matrices for seeds {missing} -- "
                  "run run_bfdkr_reference.py for each seed first.")
    boot = {"dataset": DATASET, "per_seed": {}}
    for seed in SEEDS:
        subset = {f"BFDKR_seed{seed}": realized[f"BFDKR_seed{seed}"],
                   "S1": realized["S1"], "S2": realized["S2"],
                   "S2b": realized[f"S2b_seed{seed}"], "S3": realized["S3"]}
        res = bootstrap_gaps(subset, c_test, lo, hi, reference=f"BFDKR_seed{seed}")
        boot["per_seed"][f"seed{seed}"] = res
        for name, g in res["gaps"].items():
            print(f"  seed{seed} {name}: {g['cost_auc_gap_point']:+.6f} "
                  f"CI[{g['cost_auc_gap_ci95'][0]:+.6f}, {g['cost_auc_gap_ci95'][1]:+.6f}] "
                  f"excl0={g['cost_auc_gap_excludes_zero']}")

    # ── write ──
    _atomic_write(out_root / "s1_largest_affordable.json", arm_json["S1"])
    _atomic_write(out_root / "s2_learned_static.json", arm_json["S2"])
    _atomic_write(out_root / "s2b_query_marginalized.json",
                   {"per_seed": {f"seed{s}": s2b_rows[s] for s in SEEDS}})
    _atomic_write(out_root / "s3_hindsight_static.json", arm_json["S3"])
    _atomic_write(out_root / "s3_crosscheck.json", cross)
    _atomic_write(out_root / "gap_bootstrap.json", boot)
    _atomic_write(out_root / "cost_curve.json",
                   {"pooled": dict(curves, floor=[[float(b), float(q)] for b, q in base["floor"]],
                                    oracle=[[float(b), float(q)] for b, q in base["oracle"]])})

    # ── comparison_metrics.json, in the all-methods-comparison row schema ─────
    #
    # A list of pooled rows carrying `router`, `group` and `test_*` keys, so these rows can be
    # added to src/experiments/*/all_methods_comparison/build_comparison.py's ROUTERS list
    # later with a four-line edit and no bespoke merge path. S3 is included but carries
    # `uses_test_labels: true`; it is an anchor alongside floor and oracle, never a baseline row.
    bf_mean = float(np.mean([arm_json[f"BFDKR_seed{s_}"]["test_cost_auc"] for s_ in SEEDS]))
    bf_std = float(np.std([arm_json[f"BFDKR_seed{s_}"]["test_cost_auc"] for s_ in SEEDS]))
    s2b_mean = float(np.mean([arm_json[f"S2b_seed{s_}"]["test_cost_auc"] for s_ in SEEDS]))
    s2b_std = float(np.std([arm_json[f"S2b_seed{s_}"]["test_cost_auc"] for s_ in SEEDS]))
    comparison = [
        dict(arm_json["S1"], router="Static: largest affordable (S1)",
              source_exp=f"{FAMILY}/{EXPERIMENT}", n_seeds=1),
        dict(arm_json["S2"], router="Static: learned quality ranking (S2)",
              source_exp=f"{FAMILY}/{EXPERIMENT}", n_seeds=1),
        {"router": "Static: BF-DKR query-marginalised (S2b)", "arm": "S2b", "group": "pooled",
          "source_exp": f"{FAMILY}/{EXPERIMENT}", "n_seeds": len(SEEDS),
          "test_cost_auc": round(s2b_mean, 6), "test_cost_auc_std": round(s2b_std, 6),
          "test_floor_cost_auc": round(floor_auc, 6),
          "test_oracle_cost_auc": round(oracle_auc, 6),
          "test_aiq": round(s2b_mean - floor_auc, 6),
          "per_seed_cost_auc": [arm_json[f"S2b_seed{s_}"]["test_cost_auc"] for s_ in SEEDS],
          "uses_test_labels": False},
        {"router": "BF-DKR (pq_auc, per-seed val-selected k)", "arm": "BFDKR", "group": "pooled",
          "source_exp": f"{FAMILY}/{EXPERIMENT}", "n_seeds": len(SEEDS),
          "test_cost_auc": round(bf_mean, 6), "test_cost_auc_std": round(bf_std, 6),
          "test_floor_cost_auc": round(floor_auc, 6),
          "test_oracle_cost_auc": round(oracle_auc, 6),
          "test_aiq": round(bf_mean - floor_auc, 6),
          "per_seed_cost_auc": [arm_json[f"BFDKR_seed{s_}"]["test_cost_auc"] for s_ in SEEDS],
          "uses_test_labels": False},
        dict(arm_json["S3"], router="Anchor: hindsight-best static policy (S3)",
              source_exp=f"{FAMILY}/{EXPERIMENT}", n_seeds=1),
    ]
    _atomic_write(out_root / "comparison_metrics.json", comparison)

    # ── pre-registered hypothesis verdicts ───────────────────────────────────
    def _gaps(arm):
        return [boot["per_seed"][f"seed{s_}"]["gaps"][f"BFDKR_seed{s_}_minus_{arm}"]
                for s_ in SEEDS]

    def _verdict(arm):
        g = _gaps(arm)
        wins = sum(1 for x in g if x["cost_auc_gap_point"] > TIE_BAND_COST_AUC
                    and x["cost_auc_gap_excludes_zero"])
        losses = sum(1 for x in g if x["cost_auc_gap_point"] < -TIE_BAND_COST_AUC
                      and x["cost_auc_gap_excludes_zero"])
        return {"arm": arm, "per_seed": g, "n_seeds": len(g),
                "bfdkr_wins": wins, "arm_wins": losses,
                "ties_within_band": len(g) - wins - losses,
                "mean_gap": round(float(np.mean([x["cost_auc_gap_point"] for x in g])), 6),
                "tie_band_cost_auc": TIE_BAND_COST_AUC}

    hyp = {
        "H1_vs_S1": {"statement": "BF-DKR > S1 by a wide margin, CI excluding 0. Refuted if S1 "
                                   "ties BF-DKR -- which would mean the cost proxy alone "
                                   "explains the headline.", **_verdict("S1")},
        "H2_vs_S2": {"statement": "BF-DKR > S2 outside the tie-band, CI excluding 0 -> per-query "
                                   "adaptation is doing real work. Refuted if S2 ties BF-DKR: the "
                                   "method's gain would be LLM-quality ranking, not query "
                                   "adaptation. This is the outcome that would most change the "
                                   "paper.", **_verdict("S2")},
        "H2b_vs_S2b": {"statement": "BF-DKR > its own query-marginalised self outside the "
                                     "tie-band, CI excluding 0. A tie means the row-to-row "
                                     "variation in BF-DKR's predictions is noise.",
                        **_verdict("S2b")},
        "H4_vs_S3": {"statement": "BF-DKR clears even the hindsight-best static policy. If "
                                   "BF-DKR < S3, a well-chosen static policy is competitive and "
                                   "section VI-A needs a containment clause.", **_verdict("S3")},
        "H5_S1_vs_S2": {
            "statement": "S1 ~ S2 -> metadata-grade size information is most of what a static "
                          "policy can use. S2 >> S1 means fingerprints carry model-quality "
                          "signal beyond size.",
            "s1_cost_auc": arm_json["S1"]["test_cost_auc"],
            "s2_cost_auc": arm_json["S2"]["test_cost_auc"],
            "delta_s2_minus_s1": round(arm_json["S2"]["test_cost_auc"]
                                        - arm_json["S1"]["test_cost_auc"], 6),
            "within_tie_band": bool(abs(arm_json["S2"]["test_cost_auc"]
                                         - arm_json["S1"]["test_cost_auc"]) <= TIE_BAND_COST_AUC),
        },
    }
    for name, obj in hyp.items():
        _atomic_write(out_root / f"{name.lower()}.json", obj)
    print("\n  Hypothesis verdicts (BF-DKR vs each static arm):")
    for name in ("H1_vs_S1", "H2_vs_S2", "H2b_vs_S2b", "H4_vs_S3"):
        v = hyp[name]
        print(f"    {name:<12} mean gap {v['mean_gap']:+.6f}  "
              f"BF-DKR wins {v['bfdkr_wins']}/{v['n_seeds']}  "
              f"arm wins {v['arm_wins']}/{v['n_seeds']}")

    print(f"\nWrote static-policy artifacts to {out_root}")


if __name__ == "__main__":
    main()

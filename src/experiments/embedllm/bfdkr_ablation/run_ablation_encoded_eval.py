"""
DMKDMR component ablation on MLPDot-encoded embeddings (EmbedLLM).

Ports router-eval/bfdkr-ablation's AblatedDMKDMRRouter
(qk_off / lk_off / bw_off) onto EmbedLLM's own MLPDot-encoded representation
(embedllm/bfdkr-sweep), plus a knn_enc reference (plain kNN on the same encoded
embeddings, EmbedLLM's own cost-aligned kNN-baseline grid) as the "ablate
everything" limit. No full_ext arm: EmbedLLM's own sweep already shows every
selected sigma multiplier far from the (0.015625, 4.0) search-range bounds, so
a range-extension fit would be a guaranteed no-op (see DESIGN.md H4).

One job array task per encoder seed (3 total). Each task fits all four
DMKDMR-family arms in a single process so every arm's per-query P_test
coexists in memory for a shared, paired bootstrap (see router-eval's
IMPLEMENTATION.md "Efficiency audit" -- every predictor's bootstrap
distribution is computed exactly once; every gap is a cheap array
subtraction on shared resample indices). objective is fixed at "pq_auc"
(mirrors router-eval/bfdkr-ablation exactly, for cross-dataset comparability
of knockout deltas); k is fixed at 4096 (max memory budget, makes the `full`
arm reproduce embedllm/bfdkr-sweep's own pq_auc/k4096 cells -- the built-in
reproduction gate); sigma_x_mult_range = sigma_y_mult_range = (0.015625, 4.0)
matches embedllm/bfdkr-sweep's own convention, NOT DMKDMRRouter's
(0.0625, 4.0) code default.

Usage
-----
  python run_ablation_encoded_eval.py --seed-idx 0            # seed=0, all 5 arms
  python run_ablation_encoded_eval.py --seed-idx 1 --smoke --device cpu
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc
from src.routers.dmkdmr import DMKDMRRouter
from src.routers.knn_baseline import predict_from_neighbors, predict_sweep

from ablated_dmkdmr import AblatedDMKDMRRouter  # same-dir import (copy-don't-share pattern)

RESULTS_DIR = results_dir("embedllm", "bfdkr-ablation")
ENCODED_ROOT = dataset_processed("embedllm")

CONDITION = "dmkdmr_ablation"

SEEDS = [0, 1, 2]
OBJECTIVE = "pq_auc"
K = 4096

# EmbedLLM-aligned Brent search range (matches embedllm/bfdkr-sweep's own convention),
# NOT DMKDMRRouter's (0.0625, 4.0) code default. Using the default silently breaks the
# full_reproduction_gate against the sweep's own pq_auc/k4096 cells.
SIGMA_MULT_RANGE = (0.015625, 4.0)

# DMKDMR-family arms fit for every seed; "knn_enc" is fit separately (not a
# DMKDMRRouter/AblatedDMKDMRRouter instance) and appended to the same result set.
# No full_ext arm here -- see module docstring / DESIGN.md H4.
ARMS = ["full", "qk_off", "lk_off", "bw_off"]

# kNN reference grid -- EmbedLLM's own kNN-baseline grid (src/experiments/embedllm/
# knn_baseline/run_knn_pooled_eval.py), NOT router-eval's ablation's grid: EmbedLLM has
# only 70 train LLMs vs. router-eval's ~2,999, hence the much smaller K_L grid.
K_Q_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
K_L_CANDIDATES = [1, 2, 4, 8, 16, 32, 64]
MAX_KQ = 4096

N_BOOT = 1000
BOOT_SEED = 42


# ── Small helpers ─────────────────────────────────────────────────────────────

def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _sigma_to_json(v):
    """Serialize sigma values: inf->'inf', 0.0->0.0, nan->'nan', else float.

    (qk_off's sigma_x_ is float('inf'); lk_off's sigma_y_ is the finite 0.0
    semantic value -- see AblatedDMKDMRRouter's docstring.)
    """
    if v is None:
        return None
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        if v != v:  # NaN
            return "nan"
    return float(v)


def make_k_grid(candidates, max_k):
    seen, grid = set(), []
    for k in sorted(set(candidates) | {max_k}):
        k = min(k, max_k)
        if k not in seen:
            seen.add(k)
            grid.append(k)
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-idx", type=int, required=True, choices=range(len(SEEDS)))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--encoded-root", default=str(ENCODED_ROOT))
    return p.parse_args()


def _load_mlpdot_reference(bundle_dir: Path) -> tuple[np.ndarray, dict]:
    pred_path = bundle_dir / "mlpdot_test_predictions.npy"
    ref_path = bundle_dir / "mlpdot_reference.json"
    if not pred_path.exists() or not ref_path.exists():
        sys.exit(f"Missing {pred_path} or {ref_path} -- run "
                  f"src/data_pipeline/embedllm/encode_embeddings_mlpdot.py --seed N "
                  f"for this seed first.")
    P_mlpdot_test = np.load(pred_path).astype(np.float32)
    reference = json.loads(ref_path.read_text())
    return P_mlpdot_test, reference


# ── Router construction ────────────────────────────────────────────────────────

def _build_router(arm: str, k: int, seed: int, device):
    if arm == "full":
        return DMKDMRRouter(
            k=k, objective=OBJECTIVE, seed=seed, device=device, verbose=False,
            sigma_x_mult_range=SIGMA_MULT_RANGE, sigma_y_mult_range=SIGMA_MULT_RANGE,
        )
    if arm in ("qk_off", "lk_off", "bw_off"):
        return AblatedDMKDMRRouter(
            knockout=arm, k=k, objective=OBJECTIVE, seed=seed, device=device, verbose=False,
            sigma_x_mult_range=SIGMA_MULT_RANGE, sigma_y_mult_range=SIGMA_MULT_RANGE,
        )
    raise ValueError(f"Unknown arm: {arm!r}")


# ── kNN-on-encoded reference (EmbedLLM's own selection protocol) ──────────────

def _fit_knn_enc(Q_train, L_train, S_train, Q_val, L_val, S_val, c_val,
                  Q_test, L_test, smoke: bool):
    """Cost-aligned pooled kNN on the MLPDot-encoded embeddings.

    Mirrors embedllm/knn_baseline's grid/selection exactly (argmax val cost_auc,
    tiebreak val brier) via the amortized `predict_sweep` helper -- one shared
    neighbor-index build, not one KNNRouter.fit()/predict() call per grid point.
    """
    Nq_train, Nl_train = Q_train.shape[0], L_train.shape[0]
    if smoke:
        k_q_run = sorted({1, 64, min(MAX_KQ, Nq_train)})
        k_l_run = sorted({1, 8, Nl_train})
    else:
        k_q_run = make_k_grid(K_Q_CANDIDATES, min(MAX_KQ, Nq_train))
        k_l_run = make_k_grid(K_L_CANDIDATES, Nl_train)
    k_q_max, k_l_max = max(k_q_run), max(k_l_run)

    nn_q = NearestNeighbors(n_neighbors=k_q_max, metric="cosine",
                            algorithm="brute", n_jobs=-1).fit(Q_train)
    nn_l = NearestNeighbors(n_neighbors=k_l_max, metric="cosine",
                            algorithm="brute", n_jobs=-1).fit(L_train)
    val_q_nbrs = nn_q.kneighbors(Q_val, return_distance=False)
    val_l_nbrs = nn_l.kneighbors(L_val, return_distance=False)

    sweep_rows, val_cost_aucs, val_briers = [], [], []
    for k_q, k_l, S_hat in predict_sweep(S_train, val_q_nbrs, val_l_nbrs, k_q_run, k_l_run):
        ca = M.cost_metrics(S_hat, S_val, c_val)["scalars"]["cost_auc"]
        br = M.brier_score(S_hat, S_val)
        val_cost_aucs.append(ca)
        val_briers.append(br)
        sweep_rows.append({"k_q": k_q, "k_l": k_l,
                           "val_cost_auc": round(ca, 6), "val_brier": round(br, 6)})

    best_idx = select_by_cost_auc(val_cost_aucs, val_briers)
    k_q_star, k_l_star = sweep_rows[best_idx]["k_q"], sweep_rows[best_idx]["k_l"]
    if (not smoke) and k_q_star == min(Nq_train, MAX_KQ):
        print("  WARNING: knn_enc selected k_q saturates the grid.")

    test_q_nbrs = nn_q.kneighbors(Q_test, return_distance=False)
    test_l_nbrs = nn_l.kneighbors(L_test, return_distance=False)
    P_val = predict_from_neighbors(S_train, val_q_nbrs[:, :k_q_star], val_l_nbrs[:, :k_l_star])
    P_test = predict_from_neighbors(S_train, test_q_nbrs[:, :k_q_star], test_l_nbrs[:, :k_l_star])
    audit = {"k_q": k_q_star, "k_l": k_l_star, "n_grid_cells": len(sweep_rows),
             "sweep": sweep_rows}
    return P_val, P_test, audit


# ── Shared, efficient paired bootstrap ────────────────────────────────────────

def _bootstrap_distributions(predictors: dict[str, np.ndarray], S_test, c_test,
                              n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Compute each predictor's 1000-sample (cost_auc, per_query_auc) arrays
    exactly once on a shared resample-index matrix, so every pairwise gap
    downstream is a cheap array subtraction instead of a repeated
    `cost_metrics` evaluation.
    """
    rng = np.random.default_rng(seed)
    n = S_test.shape[0]
    idx_matrix = rng.integers(0, n, size=(n_boot, n))

    dists = {}
    for name, P in predictors.items():
        cost_arr = np.empty(n_boot, dtype=np.float64)
        pq_arr = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = idx_matrix[b]
            Sb = S_test[idx]
            cost_arr[b] = M.cost_metrics(P[idx], Sb, c_test)["scalars"]["cost_auc"]
            pq_arr[b] = M.per_query_auc(P[idx], Sb)
        dists[name] = {"cost_auc": cost_arr, "pq_auc": pq_arr}
    return dists


def _gap_json(name_a, name_b, dists, point_cost, point_pq, reference, n_queries):
    cost_gaps = dists[name_a]["cost_auc"] - dists[name_b]["cost_auc"]
    pq_gaps = dists[name_a]["pq_auc"] - dists[name_b]["pq_auc"]
    return {
        "cost_auc_gap_point": round(float(point_cost), 6),
        "cost_auc_gap_ci95": [round(float(np.percentile(cost_gaps, 2.5)), 6),
                               round(float(np.percentile(cost_gaps, 97.5)), 6)],
        "cost_auc_gap_excludes_zero": bool(
            np.percentile(cost_gaps, 2.5) > 0 or np.percentile(cost_gaps, 97.5) < 0),
        "pq_auc_gap_point": round(float(point_pq), 6),
        "pq_auc_gap_ci95": [round(float(np.percentile(pq_gaps, 2.5)), 6),
                            round(float(np.percentile(pq_gaps, 97.5)), 6)],
        "pq_auc_gap_excludes_zero": bool(
            np.percentile(pq_gaps, 2.5) > 0 or np.percentile(pq_gaps, 97.5) < 0),
        "n_boot": N_BOOT, "n_queries": int(n_queries), "reference": reference,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    seed = SEEDS[args.seed_idx]
    out_root = Path(args.out_dir)
    print(f"DMKDMR ablation (EmbedLLM, MLPDot-encoded), seed={seed} "
          f"objective={OBJECTIVE} k={K} sigma_range={SIGMA_MULT_RANGE} | "
          f"smoke={args.smoke} | device={args.device} | out={out_root}")

    base_spec = get_spec("embedllm")
    bundle_path = (Path(args.encoded_root) / f"mlpdot_encoded_seed{seed}"
                   / "llm_perf_dataset_mlpdot_encoded.pkl")
    if not bundle_path.exists():
        sys.exit(f"Missing encoded bundle {bundle_path} -- run "
                  f"src/data_pipeline/embedllm/encode_embeddings_mlpdot.py --seed {seed} first.")
    enc_spec = dataclasses.replace(
        base_spec, name=f"embedllm-mlpdot-abl-seed{seed}", dataset_path=bundle_path,
    )

    print("\nLoading encoded splits via src.eval.load_splits ...")
    sp = load_splits(enc_spec)
    print(f"encoded train_q {sp.Q_train.shape}  train_l {sp.L_train.shape}  "
          f"val_q {sp.Q_val.shape}  test_q {sp.Q_test.shape}")

    P_mlpdot_test, mlpdot_reference = _load_mlpdot_reference(bundle_path.parent)
    print(f"  MLPDot reference (seed={seed}): test_cost_auc={mlpdot_reference['test_cost_auc']:.4f} "
          f"test_pq_auc={mlpdot_reference['test_pq_auc']:.4f}")
    if not args.smoke:
        _atomic_write(out_root / f"mlpdot_bce_reference_seed{seed}.json", mlpdot_reference)

    Q_train, L_train = sp.Q_train, sp.L_train
    Q_val, L_val = sp.Q_val, sp.L_val
    Q_test, L_test = sp.Q_test, sp.L_test
    S_train = sp.S_train.astype(np.float32)
    S_val = sp.S_val.astype(np.float32)
    S_test = sp.S_test.astype(np.float32)
    c_val, c_test = sp.c_val, sp.c_test
    k = K

    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Q_train.shape[0], size=min(300, Q_train.shape[0]), replace=False)
        li = rng.choice(L_train.shape[0], size=min(60, L_train.shape[0]), replace=False)
        Q_train, L_train = Q_train[qi], L_train[li]
        S_train = S_train[np.ix_(qi, li)]
        qv = rng.choice(Q_val.shape[0], size=min(200, Q_val.shape[0]), replace=False)
        lv = rng.choice(L_val.shape[0], size=min(15, L_val.shape[0]), replace=False)
        Q_val, L_val = Q_val[qv], L_val[lv]
        S_val = S_val[np.ix_(qv, lv)]
        c_val = c_val[lv]
        qt = rng.choice(Q_test.shape[0], size=min(200, Q_test.shape[0]), replace=False)
        lt = rng.choice(L_test.shape[0], size=min(21, L_test.shape[0]), replace=False)
        Q_test, L_test = Q_test[qt], L_test[lt]
        S_test = S_test[np.ix_(qt, lt)]
        P_mlpdot_test = P_mlpdot_test[np.ix_(qt, lt)]
        c_test = c_test[lt]
        k = min(k, Q_train.shape[0])
        print(f"  [smoke] train={Q_train.shape[0]}q x {L_train.shape[0]}l  "
              f"val={Q_val.shape[0]}q x {L_val.shape[0]}l  "
              f"test={Q_test.shape[0]}q x {L_test.shape[0]}l  (k capped to {k})")

    # --- Fit the four DMKDMR-family arms (shared process -> shared P_test in memory) ---
    P_val_by_arm, P_test_by_arm, config_rows = {}, {}, {}
    for arm in ARMS:
        print(f"\n--- Fitting arm={arm} (seed={seed}) ---")
        t0 = time.time()
        router = _build_router(arm, k, seed, args.device).fit(
            Q_train, L_train, S_train, Q_val, L_val, S_val)
        fit_s = time.time() - t0
        P_val_by_arm[arm] = router.predict(Q_val, L_val)
        P_test_by_arm[arm] = router.predict(Q_test, L_test)
        sigma_x_mult = (None if router.sigma_x_ in (float("inf"),) or router._sigma_x_med == 0
                         else router.sigma_x_ / router._sigma_x_med)
        sigma_y_mult = (None if router._sigma_y_med == 0
                         else router.sigma_y_ / router._sigma_y_med)
        config_rows[arm] = {
            "arm": arm, "seed": seed, "objective": OBJECTIVE, "k": k,
            "sigma_x": _sigma_to_json(router.sigma_x_),
            "sigma_y": _sigma_to_json(router.sigma_y_),
            "sigma_x_med": round(router._sigma_x_med, 6),
            "sigma_y_med": round(router._sigma_y_med, 6),
            "sigma_x_mult": (round(sigma_x_mult, 6) if sigma_x_mult is not None else None),
            "sigma_y_mult": (round(sigma_y_mult, 6) if sigma_y_mult is not None else None),
            "fit_elapsed_s": round(fit_s, 2),
        }
        print(f"  arm={arm}  fit={fit_s:.1f}s  sigma_x={_sigma_to_json(router.sigma_x_)}  "
              f"sigma_y={_sigma_to_json(router.sigma_y_)}")

    # --- kNN-on-encoded reference ---
    print(f"\n--- Fitting arm=knn_enc (seed={seed}) ---")
    t0 = time.time()
    P_val_knn, P_test_knn, knn_audit = _fit_knn_enc(
        Q_train, L_train, S_train, Q_val, L_val, S_val, c_val, Q_test, L_test, args.smoke)
    fit_s = time.time() - t0
    P_val_by_arm["knn_enc"] = P_val_knn
    P_test_by_arm["knn_enc"] = P_test_knn
    config_rows["knn_enc"] = {
        "arm": "knn_enc", "seed": seed, "objective": None, "k_q": knn_audit["k_q"],
        "k_l": knn_audit["k_l"], "fit_elapsed_s": round(fit_s, 2),
    }
    print(f"  knn_enc  fit={fit_s:.1f}s  k_q={knn_audit['k_q']} k_l={knn_audit['k_l']}")

    all_arms = ARMS + ["knn_enc"]

    # --- Per-arm val/test scalar metrics ---
    for arm in all_arms:
        P_val, P_test = P_val_by_arm[arm], P_test_by_arm[arm]
        val_cost_auc = M.cost_metrics(P_val, S_val, c_val)["scalars"]["cost_auc"]
        val_brier = M.brier_score(P_val, S_val)
        val_pq = M.per_query_auc(P_val, S_val)
        config_rows[arm].update(
            val_cost_auc=round(val_cost_auc, 6),
            val_brier=round(val_brier, 6),
            val_pq_auc=round(val_pq, 6),
        )
        print(f"  [{arm}] val_cost_auc={val_cost_auc:.4f}  "
              f"test_cost_auc={M.cost_metrics(P_test, S_test, c_test)['scalars']['cost_auc']:.4f}  "
              f"test_pq_auc={M.per_query_auc(P_test, S_test):.4f}")

    if args.smoke:
        for arm in all_arms:
            P_test = P_test_by_arm[arm]
            assert P_test.shape == (Q_test.shape[0], L_test.shape[0]), \
                f"{arm}: unexpected P_test shape {P_test.shape}"
            assert np.all(np.isfinite(P_test)), f"{arm}: P_test has non-finite values (NaN/inf)"
            assert P_test.min() >= -1e-5 and P_test.max() <= 1.0 + 1e-5, \
                f"{arm}: P_test out of [0,1] range"
        print("\n[smoke] shape/range/NaN checks passed for all arms.")

    # --- Shared, efficient paired bootstrap over all predictors at once ---
    print(f"\nBootstrap ({N_BOOT} resamples, shared indices across all predictors) ...")
    predictors = {arm: P_test_by_arm[arm] for arm in all_arms}
    predictors["mlpdot"] = P_mlpdot_test
    dists = _bootstrap_distributions(predictors, S_test, c_test)
    n_queries = S_test.shape[0]

    gaps = {}
    for arm in all_arms:
        if arm == "full":
            point_cost = (M.cost_metrics(P_test_by_arm["full"], S_test, c_test)["scalars"]["cost_auc"]
                          - M.cost_metrics(P_mlpdot_test, S_test, c_test)["scalars"]["cost_auc"])
            point_pq = M.per_query_auc(P_test_by_arm["full"], S_test) - M.per_query_auc(P_mlpdot_test, S_test)
            gaps["full"] = _gap_json("full", "mlpdot", dists, point_cost, point_pq,
                                     reference="mlpdot_precomputed", n_queries=n_queries)
        else:
            point_cost = (M.cost_metrics(P_test_by_arm[arm], S_test, c_test)["scalars"]["cost_auc"]
                          - M.cost_metrics(P_test_by_arm["full"], S_test, c_test)["scalars"]["cost_auc"])
            point_pq = M.per_query_auc(P_test_by_arm[arm], S_test) - M.per_query_auc(P_test_by_arm["full"], S_test)
            gaps[arm] = _gap_json(arm, "full", dists, point_cost, point_pq,
                                  reference="full_in_task", n_queries=n_queries)
        print(f"  [{arm} vs {gaps[arm]['reference']}]  "
              f"cost_auc_gap={gaps[arm]['cost_auc_gap_point']:+.4f} "
              f"CI=[{gaps[arm]['cost_auc_gap_ci95'][0]:+.4f},{gaps[arm]['cost_auc_gap_ci95'][1]:+.4f}] "
              f"excl0={gaps[arm]['cost_auc_gap_excludes_zero']}")

    for arm in all_arms:
        arm_dir = out_root / arm / f"seed{seed}"
        _atomic_write(arm_dir / "gap_bootstrap.json", gaps[arm])

    if args.smoke:
        print(f"\n[smoke] done (seed={seed}).")
        return

    # --- Full src.eval report per arm ---
    gm_const = float(S_train.mean())
    for arm in all_arms:
        arm_dir = out_root / arm / f"seed{seed}"
        P_test = P_test_by_arm[arm]
        bundle = evaluate_predictions(
            P_test, enc_spec,
            S_test=sp.S_test, c_test=sp.c_test, test_task_ids=sp.test_task_ids,
            gm_const=gm_const, out_dir=None,
        )
        for r in bundle.rows:
            if r["group"] == "pooled":
                r.update(condition=CONDITION, arm=arm, seed=seed, **{
                    kk: vv for kk, vv in config_rows[arm].items()
                    if kk not in ("arm", "seed")
                })
        io.write_metrics(bundle.rows, arm_dir / "metrics.json")
        io.write_curves(bundle.curves, arm_dir / "cost_curve.json")
        io.write_reliability(bundle.reliability, arm_dir / "reliability.json")
        io.write_cost_report(sp.cost_report_test, arm_dir / "cost_coverage.json")

        config_audit = {
            "condition": CONDITION, "arm": arm, "seed": seed,
            "mlpdot_warm_start_arch": mlpdot_reference["arch"],
            "mlpdot_warm_start_val_objective": "bce",
            "encoded_bundle": str(bundle_path.relative_to(ROOT)),
            "sigma_mult_range": list(SIGMA_MULT_RANGE),
            "config": config_rows[arm],
        }
        if arm == "knn_enc":
            config_audit["knn_sweep_audit"] = knn_audit
        (arm_dir / "config_audit.json").write_text(json.dumps(config_audit, indent=2))

        pooled = next(r for r in bundle.rows if r["group"] == "pooled")
        macro = next((r for r in bundle.rows if r["group"] == "macro"), None)
        print(f"\n=== POOLED [{arm}, seed{seed}] : top1={pooled['test_top1_acc']:.4f} "
              f"pq_auc={pooled['test_pq_auc']:.4f} cost_auc={pooled['test_cost_auc']:.4f} "
              f"aiq={pooled['test_aiq']:.4f} ece={pooled['test_ece']:.4f} ===")
        if macro:
            print(f"=== MACRO(10) [{arm}, seed{seed}]: pq_auc={macro['test_pq_auc']:.4f} "
                  f"aiq={macro['test_aiq']:.4f} ===")

    print(f"\nSaved results for {len(all_arms)} arms to {out_root} (seed={seed}).")


if __name__ == "__main__":
    main()

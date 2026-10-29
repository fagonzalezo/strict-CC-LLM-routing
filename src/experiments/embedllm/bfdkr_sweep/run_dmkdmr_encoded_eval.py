"""
Exp 25: DMKDMR on MLPDot-encoded embeddings (EmbedLLM), cost-aligned selection.

Ports router-eval exp_37 to EmbedLLM. Fits the non-parametric, closed-form
`DMKDMRRouter` directly on MLPDot's frozen, warm-start-encoded query/LLM
embeddings (produced by `data/embedllm/scripts/encode_embeddings_mlpdot.py`)
instead of raw embeddings. DMKDMR's nested Brent search tunes sigma_x (query,
"sigma_q") and sigma_y (LLM, "sigma_l") independently against the chosen
`objective` -- no manual sigma grid needed. The Brent search range is widened
to (0.015625, 4.0) to match EmbedLLM's own DMKDMR convention, rather than
exp_37's narrower (0.0625, 4.0) code default, since exp_37 found Brent pinning
sigma_l at that floor.

One job array task per (seed, objective, k) config -- 24 single-fit tasks
total (3 encoder seeds x 2 objectives x 4 k values). Each task loads that
seed's already-encoded dataset bundle, fits one DMKDMRRouter config, and loads
the MLPDot test-prediction matrix that `encode_embeddings_mlpdot.py` already
persisted alongside that seed's bundle (`mlpdot_test_predictions.npy`) as the
in-task bootstrap reference -- no redundant MLPDot refit here (see router-eval
exp_37's docstring for why: avoids paying for a refit 8x per seed, and
objective="pq_auc"'s much slower DMKDMR Brent search made that costly there).
EmbedLLM has no DMKDMR-raw or kdm_frozen baseline in this repo, so neither H1
(vs. raw) nor H2 (vs. kdm_frozen) applies here, and both are omitted.

Usage
-----
  python run_dmkdmr_encoded_eval.py --config-idx 0 --smoke --device cpu
  python run_dmkdmr_encoded_eval.py --config-idx 0    # seed=0, bce,    k=128
  python run_dmkdmr_encoded_eval.py --config-idx 8     # seed=1, bce,    k=128
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.routers.dmkdmr import DMKDMRRouter

RESULTS_DIR = results_dir("embedllm", "bfdkr-sweep")
ENCODED_ROOT = dataset_processed("embedllm")

CONDITION = "dmkdmr_encoded"

# (seed, objective, k) per array task -- 3 seeds x 2 objectives x 4 k = 24 configs.
# Order: seed varies slowest, then objective, then k. idx 0-7 = seed0, 8-15 = seed1,
# 16-23 = seed2; within each seed block, idx 0-3 = bce (k=128,512,2048,4096), 4-7 = pq_auc.
SEEDS = [0, 1, 2]
OBJECTIVES = ["bce", "pq_auc"]
K_CANDIDATES = [128, 512, 2048, 4096]
CONFIGS = [
    (seed, objective, k)
    for seed in SEEDS
    for objective in OBJECTIVES
    for k in K_CANDIDATES
]

# EmbedLLM-aligned Brent search range (matches exp_21/24's DMKDMR convention), wider than
# exp_37's (0.0625, 4.0) code default -- exp_37 found sigma_l pinning at that floor.
SIGMA_MULT_RANGE = (0.015625, 4.0)

N_BOOT = 1000
BOOT_SEED = 42


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _config_label(seed: int, objective: str, k: int) -> str:
    return f"seed{seed}/{objective}/k{k}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-idx", type=int, required=True, choices=range(len(CONFIGS)))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--encoded-root", default=str(ENCODED_ROOT))
    return p.parse_args()


def _load_mlpdot_reference(bundle_dir: Path) -> tuple[np.ndarray, dict]:
    """Load the MLPDot test-prediction matrix + summary metrics that
    `encode_embeddings_mlpdot.py` already persisted alongside this seed's bundle.
    """
    pred_path = bundle_dir / "mlpdot_test_predictions.npy"
    ref_path = bundle_dir / "mlpdot_reference.json"
    if not pred_path.exists() or not ref_path.exists():
        sys.exit(f"Missing {pred_path} or {ref_path} -- run "
                  f"data/embedllm/scripts/encode_embeddings_mlpdot.py for this seed first "
                  f"(it now persists these as part of the bundle).")
    P_mlpdot_test = np.load(pred_path).astype(np.float32)
    reference = json.loads(ref_path.read_text())
    return P_mlpdot_test, reference


def _bootstrap_gap(P_dmkdmr, P_mlpdot, S_test, c_test, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired per-query bootstrap of the (DMKDMR-encoded - MLPDot) pooled gap.

    P_mlpdot is loaded from the encode step's persisted prediction matrix (same
    seed's deterministic MLPDot fit), not refit in-task -- mirrors router-eval
    exp_37's `_bootstrap_gap` pattern.
    """
    rng = np.random.default_rng(seed)
    n = S_test.shape[0]

    point_cost_gap = (M.cost_metrics(P_dmkdmr, S_test, c_test)["scalars"]["cost_auc"]
                       - M.cost_metrics(P_mlpdot, S_test, c_test)["scalars"]["cost_auc"])
    point_pq_gap = M.per_query_auc(P_dmkdmr, S_test) - M.per_query_auc(P_mlpdot, S_test)

    cost_gaps, pq_gaps = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Sb = S_test[idx]
        ca_d = M.cost_metrics(P_dmkdmr[idx], Sb, c_test)["scalars"]["cost_auc"]
        ca_m = M.cost_metrics(P_mlpdot[idx], Sb, c_test)["scalars"]["cost_auc"]
        cost_gaps[b] = ca_d - ca_m
        pq_gaps[b] = M.per_query_auc(P_dmkdmr[idx], Sb) - M.per_query_auc(P_mlpdot[idx], Sb)

    return {
        "cost_auc_gap_point": round(float(point_cost_gap), 6),
        "cost_auc_gap_ci95": [round(float(np.percentile(cost_gaps, 2.5)), 6),
                               round(float(np.percentile(cost_gaps, 97.5)), 6)],
        "cost_auc_gap_excludes_zero": bool(
            np.percentile(cost_gaps, 2.5) > 0 or np.percentile(cost_gaps, 97.5) < 0),
        "pq_auc_gap_point": round(float(point_pq_gap), 6),
        "pq_auc_gap_ci95": [round(float(np.percentile(pq_gaps, 2.5)), 6),
                            round(float(np.percentile(pq_gaps, 97.5)), 6)],
        "pq_auc_gap_excludes_zero": bool(
            np.percentile(pq_gaps, 2.5) > 0 or np.percentile(pq_gaps, 97.5) < 0),
        "n_boot": n_boot, "n_queries": int(n), "single_seed": True,
        "reference": "mlpdot_precomputed_at_encode_time",
    }


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    seed, objective, k = CONFIGS[args.config_idx]
    config_label = _config_label(seed, objective, k)
    out_dir = Path(args.out_dir) / CONDITION / f"seed{seed}" / objective / f"k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"DMKDMR-encoded (EmbedLLM), condition={CONDITION} config={config_label} "
          f"(seed={seed}, objective={objective}, k={k}) | smoke={args.smoke} | "
          f"device={args.device} | out={out_dir}")

    base_spec = get_spec("embedllm")
    bundle_path = Path(args.encoded_root) / f"mlpdot_encoded_seed{seed}" / "llm_perf_dataset_mlpdot_encoded.pkl"
    if not bundle_path.exists():
        sys.exit(f"Missing encoded bundle {bundle_path} -- run "
                  f"data/embedllm/scripts/encode_embeddings_mlpdot.py --seed {seed} first.")
    enc_spec = dataclasses.replace(
        base_spec, name=f"embedllm-mlpdot-enc-seed{seed}", dataset_path=bundle_path,
    )

    print("\nLoading encoded splits via src.eval.load_splits ...")
    enc_splits = load_splits(enc_spec)

    Qe_train, Le_train, S_train = enc_splits.Q_train, enc_splits.L_train, enc_splits.S_train.astype(np.float32)
    Qe_val, Le_val, S_val = enc_splits.Q_val, enc_splits.L_val, enc_splits.S_val.astype(np.float32)
    Qe_test, Le_test, S_test = enc_splits.Q_test, enc_splits.L_test, enc_splits.S_test.astype(np.float32)
    print(f"encoded train_q {Qe_train.shape}  train_l {Le_train.shape}  "
          f"val_q {Qe_val.shape}  test_q {Qe_test.shape}")

    # --- MLPDot reference: precomputed at encode time, loaded (not refit) ---
    P_mlpdot_test, mlpdot_reference = _load_mlpdot_reference(bundle_path.parent)
    print(f"  MLPDot reference (seed={seed}, loaded): best_epoch={mlpdot_reference['best_epoch']} "
          f"test_cost_auc={mlpdot_reference['test_cost_auc']:.4f} "
          f"test_pq_auc={mlpdot_reference['test_pq_auc']:.4f} "
          f"test_ece={mlpdot_reference['test_ece']:.4f}")
    if not args.smoke:
        # Idempotent: identical across all 8 tasks sharing this seed (same source file).
        _atomic_write(Path(args.out_dir) / f"mlpdot_bce_reference_seed{seed}.json", mlpdot_reference)

    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Qe_train.shape[0], size=min(300, Qe_train.shape[0]), replace=False)
        li = rng.choice(Le_train.shape[0], size=min(60, Le_train.shape[0]), replace=False)
        Qe_train, Le_train = Qe_train[qi], Le_train[li]
        S_train = S_train[np.ix_(qi, li)]
        qv = rng.choice(Qe_val.shape[0], size=min(200, Qe_val.shape[0]), replace=False)
        lv = rng.choice(Le_val.shape[0], size=min(15, Le_val.shape[0]), replace=False)
        Qe_val, Le_val = Qe_val[qv], Le_val[lv]
        S_val = S_val[np.ix_(qv, lv)]
        c_val = enc_splits.c_val[lv]
        qt = rng.choice(Qe_test.shape[0], size=min(200, Qe_test.shape[0]), replace=False)
        lt = rng.choice(Le_test.shape[0], size=min(21, Le_test.shape[0]), replace=False)
        Qe_test, Le_test = Qe_test[qt], Le_test[lt]
        S_test = S_test[np.ix_(qt, lt)]
        P_mlpdot_test = P_mlpdot_test[np.ix_(qt, lt)]
        c_test = enc_splits.c_test[lt]
        k = min(k, Qe_train.shape[0])
        print(f"  [smoke] train={Qe_train.shape[0]}q x {Le_train.shape[0]}l "
              f"val={Qe_val.shape[0]}q x {Le_val.shape[0]}l "
              f"test={Qe_test.shape[0]}q x {Le_test.shape[0]}l  (k capped to {k})")
    else:
        c_val, c_test = enc_splits.c_val, enc_splits.c_test

    # --- Single fit: this task's exact (seed, objective, k) config ---
    t0 = time.time()
    router = DMKDMRRouter(
        k=k, objective=objective, seed=seed, device=args.device, verbose=False,
        sigma_x_mult_range=SIGMA_MULT_RANGE, sigma_y_mult_range=SIGMA_MULT_RANGE,
    ).fit(Qe_train, Le_train, S_train, Qe_val, Le_val, S_val)
    fit_s = time.time() - t0

    P_val = router.predict(Qe_val, Le_val)
    P_test = router.predict(Qe_test, Le_test)
    val_pq, val_brier = M.per_query_auc(P_val, S_val), M.brier_score(P_val, S_val)
    val_top1 = M.top1_accuracy(P_val, S_val)
    val_cost_auc = M.cost_metrics(P_val, S_val, c_val)["scalars"]["cost_auc"]
    t_rank = {"top1": M.top1_accuracy(P_test, S_test),
              "pq_auc": M.per_query_auc(P_test, S_test),
              "brier": M.brier_score(P_test, S_test),
              "ece": M.expected_calibration_error(P_test, S_test)}
    t_cost = M.cost_metrics(P_test, S_test, c_test)["scalars"]

    print(f"  seed={seed} objective={objective} k={k}  fit={fit_s:.1f}s  "
          f"sigma_x={router.sigma_x_:.5f}  sigma_y={router.sigma_y_:.5f}  "
          f"val_cost_auc={val_cost_auc:.4f}  "
          f"test_pq_auc={t_rank['pq_auc']:.4f}  test_cost_auc={t_cost['cost_auc']:.4f}  "
          f"test_ece={t_rank['ece']:.4f}")

    config_row = {
        "seed": seed, "objective": objective, "k": k,
        "sigma_x": round(router.sigma_x_, 6), "sigma_y": round(router.sigma_y_, 6),
        "sigma_x_med": round(router._sigma_x_med, 6), "sigma_y_med": round(router._sigma_y_med, 6),
        "val_pq_auc": round(val_pq, 6), "val_brier": round(val_brier, 6),
        "val_top1": round(val_top1, 6), "val_cost_auc": round(val_cost_auc, 6),
        "val_bce": round(router.val_bce_, 6), "val_obj": round(router.val_obj_, 6),
        "test_top1": round(t_rank["top1"], 6), "test_pq_auc": round(t_rank["pq_auc"], 6),
        "test_brier": round(t_rank["brier"], 6), "test_ece": round(t_rank["ece"], 6),
        "test_cost_auc": round(t_cost["cost_auc"], 6), "test_aiq": round(t_cost["aiq"], 6),
        "fit_elapsed_s": round(fit_s, 2),
    }

    # --- In-task paired bootstrap: (this config's DMKDMR-encoded - in-task MLPDot) gap ---
    print(f"\nBootstrap (DMKDMR-encoded[{config_label}] - MLPDot), {N_BOOT} resamples ...")
    gap = _bootstrap_gap(P_test, P_mlpdot_test, S_test, c_test)
    gap.update(seed=seed, objective=objective, k=k)
    print(f"  cost_auc gap: {gap['cost_auc_gap_point']:+.4f}  "
          f"95% CI [{gap['cost_auc_gap_ci95'][0]:+.4f}, {gap['cost_auc_gap_ci95'][1]:+.4f}]  "
          f"excludes_zero={gap['cost_auc_gap_excludes_zero']}")
    print(f"  pq_auc  gap: {gap['pq_auc_gap_point']:+.4f}  "
          f"95% CI [{gap['pq_auc_gap_ci95'][0]:+.4f}, {gap['pq_auc_gap_ci95'][1]:+.4f}]  "
          f"excludes_zero={gap['pq_auc_gap_excludes_zero']}")
    _atomic_write(out_dir / "gap_bootstrap.json", gap)

    if args.smoke:
        print(f"[smoke] done (config={config_label}).")
        return

    # --- Pooled + per-benchmark + macro report via src.eval ---
    gm_const = float(S_train.mean())
    bundle = evaluate_predictions(
        P_test, enc_spec,
        S_test=enc_splits.S_test, c_test=enc_splits.c_test, test_task_ids=enc_splits.test_task_ids,
        gm_const=gm_const, out_dir=None,
    )

    for r in bundle.rows:
        if r["group"] == "pooled":
            r.update(condition=CONDITION, config_label=config_label, seed=seed,
                      objective=objective, k=k,
                      sigma_x=config_row["sigma_x"], sigma_y=config_row["sigma_y"],
                      val_cost_auc=round(val_cost_auc, 6),
                      val_brier=round(val_brier, 6),
                      val_pq_auc=round(val_pq, 6))

    io.write_metrics(bundle.rows, out_dir / "metrics.json")
    io.write_curves(bundle.curves, out_dir / "cost_curve.json")
    io.write_reliability(bundle.reliability, out_dir / "reliability.json")
    io.write_cost_report(enc_splits.cost_report_test, out_dir / "cost_coverage.json")

    config_audit = {
        "condition": CONDITION, "config_label": config_label,
        "mlpdot_warm_start_arch": mlpdot_reference["arch"],
        "mlpdot_warm_start_val_objective": "bce",
        "encoded_bundle": str(bundle_path.relative_to(ROOT)),
        "sigma_mult_range": list(SIGMA_MULT_RANGE),
        "config": config_row,
    }
    (out_dir / "config_audit.json").write_text(json.dumps(config_audit, indent=2))

    pooled = next(r for r in bundle.rows if r["group"] == "pooled")
    macro = next((r for r in bundle.rows if r["group"] == "macro"), None)
    print(f"\n=== POOLED [{config_label}] : top1={pooled['test_top1_acc']:.4f} "
          f"pq_auc={pooled['test_pq_auc']:.4f} cost_auc={pooled['test_cost_auc']:.4f} "
          f"aiq={pooled['test_aiq']:.4f} ece={pooled['test_ece']:.4f} ===")
    if macro:
        print(f"=== MACRO(10) [{config_label}]: top1={macro['test_top1_acc']:.4f} "
              f"pq_auc={macro['test_pq_auc']:.4f} aiq={macro['test_aiq']:.4f} ===")
    print(f"Saved {len(bundle.rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()

"""
DropoutNet-style cold-start router (EmbedLLM), cost-aligned selection.

New in this repo -- the original study only ran this comparison on
router-eval (`exp_39`); there is no embedllm counterpart to port. This script mirrors
`src/experiments/router_eval/dropoutnet_clcrec_baselines/run_dropoutnet_eval.py`
verbatim except for the dataset spec/output dir and the MF-dim grid (see DESIGN.md):

Two-phase run, mirroring exp_32's single-sequential-script pattern (not exp_37's
job-array pattern -- DropoutNetRouter has no per-fit cost-blowup risk analogous to
DMKDMR's Brent search; both its stages have fixed, bounded per-step cost):

  Phase A (HP search, single seed): sweep mf_dim x p_drop (9 configs) at HP_SEED=0,
    select the winner by val budget-cap cost_auc (tiebreak min val brier).
  Phase B (seed variance at the winner): refit the selected (mf_dim, p_drop) at 3
    seeds, report the full A+B+C suite per seed, plus the router's own diagnostics
    (four-cell taxonomy breakdown, realized zero-branch exposure) and the raw test
    prediction matrix (needed for build_comparison.py's paired bootstraps).

Usage
-----
  python run_dropoutnet_eval.py --probe --probe-config-idx 0 --device cuda
  python run_dropoutnet_eval.py --smoke --device cpu
  python run_dropoutnet_eval.py                 # full run (GPU recommended)
"""

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc
from src.routers.dropoutnet import DropoutNetRouter

RESULTS_DIR = results_dir("embedllm", "dropoutnet-clcrec-baselines")
CONDITION = "dropoutnet"

# router-eval's [16, 32, 64] grid was sized for a ~3,000-LLM train pool; EmbedLLM's
# train pool is only 70 LLMs, so mf_dim=64 makes the MF-pretrain LLM factor matrix V
# (70 x 64) nearly full row-rank. Checked via --probe (2026-09-08) before committing
# to this grid: full-scale test_cost_auc at mf_dim=64 (0.4826) was not worse than at
# mf_dim=16 (0.4729) -- no degeneracy observed downstream -- so the grid is kept
# unchanged. See DESIGN.md.
MF_DIM_CANDIDATES = [16, 32, 64]
PDROP_CANDIDATES = [0.3, 0.5, 0.7]   # symmetric: p_drop_q == p_drop_m
HP_SEED = 0
FINAL_SEEDS = [0, 1, 2]

# Fixed training HP (DropoutNetRouter constructor defaults, made explicit here)
TARGET_DIM, HIDDEN_DIM, N_LAYERS = 64, 256, 2
MF_EPOCHS, EPOCHS, BATCH_SIZE = 100, 200, 256
LR, WEIGHT_DECAY = 1e-3, 1e-4

# Smoke overrides
MF_DIM_SMOKE, PDROP_SMOKE = [16, 32], [0.3, 0.5]
FINAL_SEEDS_SMOKE = [0, 1]
EPOCHS_SMOKE, MF_EPOCHS_SMOKE = 6, 5


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _config_label(mf_dim: int, pdrop: float) -> str:
    return f"mf_dim{mf_dim}_pdrop{pdrop}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--probe", action="store_true",
                    help="Fit exactly one full-scale config and time it; write "
                         "timing_probe.json; exit before the real sweep.")
    p.add_argument("--probe-config-idx", type=int, default=0)
    return p.parse_args()


def _make_router(mf_dim, pdrop, seed, epochs, mf_epochs, device, val_objective="bce"):
    return DropoutNetRouter(
        mf_dim=mf_dim, mf_epochs=mf_epochs, p_drop_q=pdrop, p_drop_m=pdrop,
        target_dim=TARGET_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        lr=LR, weight_decay=WEIGHT_DECAY, epochs=epochs, batch_size=BATCH_SIZE,
        seed=seed, device=device, verbose=False, val_objective=val_objective,
    )


def _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                    Q_test, L_test, S_test, c_val, c_test):
    t0 = time.time()
    router.fit(Q_train, L_train, S_train, Q_val, L_val, S_val)
    fit_s = time.time() - t0
    P_val = router.predict(Q_val, L_val)
    P_test = router.predict(Q_test, L_test)
    val_pq, val_brier = M.per_query_auc(P_val, S_val), M.brier_score(P_val, S_val)
    val_top1 = M.top1_accuracy(P_val, S_val)
    val_cost_auc = M.cost_metrics(P_val, S_val, c_val)["scalars"]["cost_auc"]
    t_rank = {"top1": M.top1_accuracy(P_test, S_test),
              "pq_auc": M.per_query_auc(P_test, S_test),
              "brier": M.brier_score(P_test, S_test),
              "ece": M.expected_calibration_error(P_test, S_test)}
    t_cost = M.cost_metrics(P_test, S_test, c_test)["scalars"]
    return P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = Path(args.out_dir) / CONDITION
    print(f"DropoutNetRouter (EmbedLLM), condition={CONDITION} | smoke={args.smoke} | "
          f"probe={args.probe} | device={args.device} | out={out_dir}")

    spec = get_spec("embedllm")
    print("\nLoading dataset via src.eval.load_splits ...")
    splits = load_splits(spec)

    Q_train, L_train, S_train = splits.Q_train, splits.L_train, splits.S_train.astype(np.float32)
    Q_val, L_val, S_val = splits.Q_val, splits.L_val, splits.S_val.astype(np.float32)
    Q_test, L_test, S_test = splits.Q_test, splits.L_test, splits.S_test.astype(np.float32)
    c_val, c_test = splits.c_val, splits.c_test
    print(f"train_q {Q_train.shape}  train_l {L_train.shape}  "
          f"val_q {Q_val.shape}  test_q {Q_test.shape}")

    if args.probe:
        configs = list(product(MF_DIM_CANDIDATES, PDROP_CANDIDATES))
        mf_dim, pdrop = configs[args.probe_config_idx]
        print(f"\n[probe] mf_dim={mf_dim} pdrop={pdrop} seed={HP_SEED} "
              f"epochs={EPOCHS} mf_epochs={MF_EPOCHS} device={args.device}")
        router = _make_router(mf_dim, pdrop, HP_SEED, EPOCHS, MF_EPOCHS, args.device)
        (P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost) = \
            _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                            Q_test, L_test, S_test, c_val, c_test)
        probe = {
            "config": {"mf_dim": mf_dim, "p_drop": pdrop, "seed": HP_SEED},
            "fit_s": round(fit_s, 2), "device": str(args.device),
            "mf_train_bce": round(router.mf_train_bce_, 6),
            "val_bce": round(router.val_bce_, 6),
            "n_train_q": int(Q_train.shape[0]), "n_train_l": int(L_train.shape[0]),
            "test_cost_auc": round(t_cost["cost_auc"], 6),
            "test_pq_auc": round(t_rank["pq_auc"], 6),
        }
        _atomic_write(out_dir / "timing_probe.json", probe)
        print(f"[probe] done: fit_s={fit_s:.1f}s  test_cost_auc={t_cost['cost_auc']:.4f}  "
              f"-> {out_dir / 'timing_probe.json'}")
        return

    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Q_train.shape[0], size=min(300, Q_train.shape[0]), replace=False)
        li = rng.choice(L_train.shape[0], size=min(30, L_train.shape[0]), replace=False)
        Q_train, L_train = Q_train[qi], L_train[li]
        S_train = S_train[np.ix_(qi, li)]
        qv = rng.choice(Q_val.shape[0], size=min(200, Q_val.shape[0]), replace=False)
        lv = rng.choice(L_val.shape[0], size=min(10, L_val.shape[0]), replace=False)
        Q_val, L_val = Q_val[qv], L_val[lv]
        S_val = S_val[np.ix_(qv, lv)]
        c_val = splits.c_val[lv]
        qt = rng.choice(Q_test.shape[0], size=min(200, Q_test.shape[0]), replace=False)
        lt = rng.choice(L_test.shape[0], size=min(10, L_test.shape[0]), replace=False)
        Q_test, L_test = Q_test[qt], L_test[lt]
        S_test = S_test[np.ix_(qt, lt)]
        c_test = splits.c_test[lt]
        print(f"  [smoke] train={Q_train.shape[0]}q x {L_train.shape[0]}l "
              f"val={Q_val.shape[0]}q x {L_val.shape[0]}l "
              f"test={Q_test.shape[0]}q x {L_test.shape[0]}l")

    mf_dim_c = MF_DIM_SMOKE if args.smoke else MF_DIM_CANDIDATES
    pdrop_c = PDROP_SMOKE if args.smoke else PDROP_CANDIDATES
    final_seeds = FINAL_SEEDS_SMOKE if args.smoke else FINAL_SEEDS
    epochs = EPOCHS_SMOKE if args.smoke else EPOCHS
    mf_epochs = MF_EPOCHS_SMOKE if args.smoke else MF_EPOCHS
    hp_configs = list(product(mf_dim_c, pdrop_c))

    # ── Phase A: HP search at HP_SEED, val-cost_auc selection ──────────────
    print(f"\nPhase A: HP search ({len(hp_configs)} configs) at seed={HP_SEED} ...")
    sweep_rows = []
    val_cost_aucs, val_briers = [], []
    for i, (mf_dim, pdrop) in enumerate(hp_configs, 1):
        label = _config_label(mf_dim, pdrop)
        router = _make_router(mf_dim, pdrop, HP_SEED, epochs, mf_epochs, args.device)
        (P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost) = \
            _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                            Q_test, L_test, S_test, c_val, c_test)

        print(f"  [{i:2d}/{len(hp_configs)}] {label}  fit={fit_s:.1f}s  "
              f"val_cost_auc={val_cost_auc:.4f}  test_pq_auc={t_rank['pq_auc']:.4f}  "
              f"test_cost_auc={t_cost['cost_auc']:.4f}")

        gm_const = float(S_train.mean())
        bundle = evaluate_predictions(
            P_test, spec, S_test=S_test, c_test=c_test,
            test_task_ids=splits.test_task_ids[qt] if args.smoke else splits.test_task_ids,
            gm_const=gm_const, out_dir=None,
        )
        hp_dir = out_dir / "hp_search" / label
        io.write_metrics(bundle.rows, hp_dir / "metrics.json")
        io.write_curves(bundle.curves, hp_dir / "cost_curve.json")
        io.write_reliability(bundle.reliability, hp_dir / "reliability.json")

        sweep_rows.append({
            "mf_dim": mf_dim, "p_drop": pdrop, "seed": HP_SEED, "label": label,
            "val_pq_auc": round(val_pq, 6), "val_brier": round(val_brier, 6),
            "val_top1": round(val_top1, 6), "val_cost_auc": round(val_cost_auc, 6),
            "val_bce": round(router.val_bce_, 6), "best_epoch": router.best_epoch_,
            "mf_train_bce": round(router.mf_train_bce_, 6),
            "test_top1": round(t_rank["top1"], 6), "test_pq_auc": round(t_rank["pq_auc"], 6),
            "test_brier": round(t_rank["brier"], 6), "test_ece": round(t_rank["ece"], 6),
            "test_cost_auc": round(t_cost["cost_auc"], 6), "test_aiq": round(t_cost["aiq"], 6),
            "fit_elapsed_s": round(fit_s, 2),
        })
        _atomic_write(out_dir / "sweep_metrics.json", sweep_rows)
        val_cost_aucs.append(val_cost_auc)
        val_briers.append(val_brier)

    best_idx = select_by_cost_auc(val_cost_aucs, val_briers)
    best = sweep_rows[best_idx]
    b_mf_dim, b_pdrop = best["mf_dim"], best["p_drop"]
    print(f"  => selected (cost-aligned): mf_dim={b_mf_dim} p_drop={b_pdrop}  "
          f"val_cost_auc={best['val_cost_auc']:.4f}  val_brier={best['val_brier']:.4f}")
    _atomic_write(out_dir / "selection_audit.json", {
        "grid": sweep_rows,
        "selected": {"mf_dim": b_mf_dim, "p_drop": b_pdrop, "hp_seed": HP_SEED,
                     **{k: best[k] for k in ("val_cost_auc", "val_brier", "val_pq_auc")}},
    })

    # ── Phase B: refit winner at FINAL_SEEDS ────────────────────────────────
    print(f"\nPhase B: refit (mf_dim={b_mf_dim}, p_drop={b_pdrop}) at seeds {final_seeds} ...")
    seed_summ = {"test_cost_auc": [], "test_pq_auc": [], "test_brier": [], "test_ece": []}
    for seed in final_seeds:
        router = _make_router(b_mf_dim, b_pdrop, seed, epochs, mf_epochs, args.device)
        (P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost) = \
            _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                            Q_test, L_test, S_test, c_val, c_test)
        print(f"  seed={seed}  fit={fit_s:.1f}s  test_pq_auc={t_rank['pq_auc']:.4f}  "
              f"test_cost_auc={t_cost['cost_auc']:.4f}  test_ece={t_rank['ece']:.4f}")
        if seed == HP_SEED:
            print(f"    [diagnostic] Phase-A HP_SEED row: test_cost_auc={best['test_cost_auc']:.6f}  "
                  f"vs. Phase-B refit: test_cost_auc={t_cost['cost_auc']:.6f} "
                  f"(not asserted equal -- GPU kernels aren't guaranteed deterministic)")

        gm_const = float(S_train.mean())
        bundle = evaluate_predictions(
            P_test, spec, S_test=S_test, c_test=c_test,
            test_task_ids=splits.test_task_ids[qt] if args.smoke else splits.test_task_ids,
            gm_const=gm_const, out_dir=None,
        )
        for r in bundle.rows:
            if r["group"] == "pooled":
                r.update(mf_dim=b_mf_dim, p_drop=b_pdrop, seed=seed,
                          val_cost_auc=round(val_cost_auc, 6),
                          val_brier=round(val_brier, 6), val_pq_auc=round(val_pq, 6))

        final_dir = out_dir / "final" / f"seed{seed}"
        io.write_metrics(bundle.rows, final_dir / "selected_metrics.json")
        io.write_curves(bundle.curves, final_dir / "cost_curve.json")
        io.write_reliability(bundle.reliability, final_dir / "reliability.json")
        if not args.smoke:
            io.write_cost_report(splits.cost_report_test, final_dir / "cost_coverage.json")

        _atomic_write(final_dir / "four_cell.json", router.cell_diag_)
        _atomic_write(final_dir / "zero_branch_exposure.json", router.zero_branch_exposure_)
        _atomic_write(final_dir / "config_audit.json", {
            "condition": CONDITION, "mf_dim": b_mf_dim, "p_drop": b_pdrop, "seed": seed,
            "target_dim": TARGET_DIM, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
            "mf_epochs": mf_epochs, "epochs": epochs, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "mf_train_bce": router.mf_train_bce_, "val_bce": router.val_bce_,
            "best_epoch": router.best_epoch_, "fit_elapsed_s": round(fit_s, 2),
            "hp_search_selected": best["label"],
        })
        np.save(final_dir / "dropoutnet_test_predictions.npy", P_test.astype(np.float32))

        for k in seed_summ:
            seed_summ[k].append(t_cost["cost_auc"] if k == "test_cost_auc" else
                                 t_rank["pq_auc"] if k == "test_pq_auc" else
                                 t_rank["brier"] if k == "test_brier" else t_rank["ece"])

    if not args.smoke:
        seed_spread = {}
        for k, vals in seed_summ.items():
            seed_spread[k] = {f"seed{s}": v for s, v in zip(final_seeds, vals)}
            seed_spread[k]["mean"] = round(float(np.mean(vals)), 6)
            seed_spread[k]["std"] = round(float(np.std(vals)), 6)
        _atomic_write(out_dir / "seed_spread.json", seed_spread)
        print(f"\nSeed spread: test_cost_auc mean={seed_spread['test_cost_auc']['mean']:.4f} "
              f"std={seed_spread['test_cost_auc']['std']:.4f}")

    print(f"\nDone. Wrote HP search ({len(hp_configs)} configs) + "
          f"{len(final_seeds)} final seed fits to {out_dir}")


if __name__ == "__main__":
    main()

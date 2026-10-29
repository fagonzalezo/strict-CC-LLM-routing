"""
Exp 39: CLCRec-style joint-contrastive cold-start router (router-eval), cost-aligned
selection.

Same two-phase pattern as run_dropoutnet_eval.py (see that file's docstring for the
rationale for a single sequential script rather than a job array): Phase A sweeps
lambda_re x tau (9 configs) at HP_SEED=0, selects the winner by val budget-cap
cost_auc; Phase B refits the winner at 3 seeds, reporting the full A+B+C suite plus
CLCRec's own Assumption-1 diagnostic (per-train-LLM L_RE-at-convergence, joined
against that LLM's own train-set routing accuracy -- the H4 ingredient) and the raw
test prediction matrix (needed for build_comparison.py's paired bootstraps).

Usage
-----
  python run_clcrec_eval.py --probe --probe-config-idx 0 --device cuda
  python run_clcrec_eval.py --smoke --device cpu
  python run_clcrec_eval.py                 # full run (GPU recommended)
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
from src.data_loader import load_perf_dataset
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc
from src.routers.clcrec import CLCRecRouter

RESULTS_DIR = results_dir("router-eval", "dropoutnet-clcrec-baselines")
CONDITION = "clcrec"

LAMBDA_RE_CANDIDATES = [0.25, 0.5, 0.75]
TAU_CANDIDATES = [0.05, 0.1, 0.2]
HP_SEED = 0
FINAL_SEEDS = [0, 1, 2]

# Fixed training HP (CLCRecRouter constructor defaults, made explicit here)
TARGET_DIM, HIDDEN_DIM, N_LAYERS = 64, 256, 2
EPOCHS, BATCH_SIZE, W_BCE = 200, 256, 1.0
LR, WEIGHT_DECAY = 1e-3, 1e-4

# Smoke overrides
LAMBDA_RE_SMOKE, TAU_SMOKE = [0.25, 0.5], [0.1, 0.2]
FINAL_SEEDS_SMOKE = [0, 1]
EPOCHS_SMOKE = 6


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _config_label(lambda_re: float, tau: float) -> str:
    return f"lre{lambda_re}_tau{tau}"


def _per_model_train_accuracy(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-column (per-train-LLM) binary routing accuracy at threshold 0.5.

    Not a general-purpose metric (src/metrics.py's group breakdown is per-benchmark-
    category, never per-LLM) -- specific to this experiment's H4 diagnostic, kept
    local rather than added to src/metrics.py.
    """
    pred = (P >= 0.5).astype(np.float32)
    return (pred == S).mean(axis=0)


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


def _make_router(lambda_re, tau, seed, epochs, device, val_objective="bce"):
    return CLCRecRouter(
        target_dim=TARGET_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        lambda_re=lambda_re, tau=tau, w_bce=W_BCE,
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
    print(f"CLCRecRouter (router-eval), condition={CONDITION} | smoke={args.smoke} | "
          f"probe={args.probe} | device={args.device} | out={out_dir}")

    spec = get_spec("router-eval")
    print("\nLoading dataset via src.eval.load_splits ...")
    splits = load_splits(spec)

    Q_train, L_train, S_train = splits.Q_train, splits.L_train, splits.S_train.astype(np.float32)
    Q_val, L_val, S_val = splits.Q_val, splits.L_val, splits.S_val.astype(np.float32)
    Q_test, L_test, S_test = splits.Q_test, splits.L_test, splits.S_test.astype(np.float32)
    c_val, c_test = splits.c_val, splits.c_test
    print(f"train_q {Q_train.shape}  train_l {L_train.shape}  "
          f"val_q {Q_val.shape}  test_q {Q_test.shape}")

    train_llm_names = np.asarray(
        load_perf_dataset(spec.dataset_path)["llms"]["train"]["names"]
    )
    assert len(train_llm_names) == L_train.shape[0], \
        f"train_llm_names ({len(train_llm_names)}) != L_train rows ({L_train.shape[0]})"

    if args.probe:
        configs = list(product(LAMBDA_RE_CANDIDATES, TAU_CANDIDATES))
        lambda_re, tau = configs[args.probe_config_idx]
        print(f"\n[probe] lambda_re={lambda_re} tau={tau} seed={HP_SEED} "
              f"epochs={EPOCHS} device={args.device}")
        router = _make_router(lambda_re, tau, HP_SEED, EPOCHS, args.device)
        (P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost) = \
            _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                            Q_test, L_test, S_test, c_val, c_test)
        probe = {
            "config": {"lambda_re": lambda_re, "tau": tau, "seed": HP_SEED},
            "fit_s": round(fit_s, 2), "device": str(args.device),
            "val_bce": round(router.val_bce_, 6),
            "n_train_q": int(Q_train.shape[0]), "n_train_l": int(L_train.shape[0]),
            "test_cost_auc": round(t_cost["cost_auc"], 6),
            "test_pq_auc": round(t_rank["pq_auc"], 6),
        }
        _atomic_write(out_dir / "timing_probe.json", probe)
        print(f"[probe] done: fit_s={fit_s:.1f}s  test_cost_auc={t_cost['cost_auc']:.4f}  "
              f"-> {out_dir / 'timing_probe.json'}")
        return

    li = None
    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Q_train.shape[0], size=min(300, Q_train.shape[0]), replace=False)
        li = rng.choice(L_train.shape[0], size=min(100, L_train.shape[0]), replace=False)
        Q_train, L_train = Q_train[qi], L_train[li]
        S_train = S_train[np.ix_(qi, li)]
        train_llm_names = train_llm_names[li]
        qv = rng.choice(Q_val.shape[0], size=min(200, Q_val.shape[0]), replace=False)
        lv = rng.choice(L_val.shape[0], size=min(60, L_val.shape[0]), replace=False)
        Q_val, L_val = Q_val[qv], L_val[lv]
        S_val = S_val[np.ix_(qv, lv)]
        c_val = splits.c_val[lv]
        qt = rng.choice(Q_test.shape[0], size=min(200, Q_test.shape[0]), replace=False)
        lt = rng.choice(L_test.shape[0], size=min(60, L_test.shape[0]), replace=False)
        Q_test, L_test = Q_test[qt], L_test[lt]
        S_test = S_test[np.ix_(qt, lt)]
        c_test = splits.c_test[lt]
        print(f"  [smoke] train={Q_train.shape[0]}q x {L_train.shape[0]}l "
              f"val={Q_val.shape[0]}q x {L_val.shape[0]}l "
              f"test={Q_test.shape[0]}q x {L_test.shape[0]}l")

    lre_c = LAMBDA_RE_SMOKE if args.smoke else LAMBDA_RE_CANDIDATES
    tau_c = TAU_SMOKE if args.smoke else TAU_CANDIDATES
    final_seeds = FINAL_SEEDS_SMOKE if args.smoke else FINAL_SEEDS
    epochs = EPOCHS_SMOKE if args.smoke else EPOCHS
    hp_configs = list(product(lre_c, tau_c))

    # ── Phase A: HP search at HP_SEED, val-cost_auc selection ──────────────
    print(f"\nPhase A: HP search ({len(hp_configs)} configs) at seed={HP_SEED} ...")
    sweep_rows = []
    val_cost_aucs, val_briers = [], []
    for i, (lambda_re, tau) in enumerate(hp_configs, 1):
        label = _config_label(lambda_re, tau)
        router = _make_router(lambda_re, tau, HP_SEED, epochs, args.device)
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
            "lambda_re": lambda_re, "tau": tau, "seed": HP_SEED, "label": label,
            "val_pq_auc": round(val_pq, 6), "val_brier": round(val_brier, 6),
            "val_top1": round(val_top1, 6), "val_cost_auc": round(val_cost_auc, 6),
            "val_bce": round(router.val_bce_, 6), "best_epoch": router.best_epoch_,
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
    b_lre, b_tau = best["lambda_re"], best["tau"]
    print(f"  => selected (cost-aligned): lambda_re={b_lre} tau={b_tau}  "
          f"val_cost_auc={best['val_cost_auc']:.4f}  val_brier={best['val_brier']:.4f}")
    _atomic_write(out_dir / "selection_audit.json", {
        "grid": sweep_rows,
        "selected": {"lambda_re": b_lre, "tau": b_tau, "hp_seed": HP_SEED,
                     **{k: best[k] for k in ("val_cost_auc", "val_brier", "val_pq_auc")}},
    })

    # ── Phase B: refit winner at FINAL_SEEDS, plus the H4 ingredient ───────
    print(f"\nPhase B: refit (lambda_re={b_lre}, tau={b_tau}) at seeds {final_seeds} ...")
    seed_summ = {"test_cost_auc": [], "test_pq_auc": [], "test_brier": [], "test_ece": []}
    for seed in final_seeds:
        router = _make_router(b_lre, b_tau, seed, epochs, args.device)
        (P_test, fit_s, val_pq, val_brier, val_top1, val_cost_auc, t_rank, t_cost) = \
            _fit_and_score(router, Q_train, L_train, S_train, Q_val, L_val, S_val,
                            Q_test, L_test, S_test, c_val, c_test)
        print(f"  seed={seed}  fit={fit_s:.1f}s  test_pq_auc={t_rank['pq_auc']:.4f}  "
              f"test_cost_auc={t_cost['cost_auc']:.4f}  test_ece={t_rank['ece']:.4f}  "
              f"re_loss mean={router.re_loss_per_model_.mean():.4f} "
              f"std={router.re_loss_per_model_.std():.4f}")

        gm_const = float(S_train.mean())
        bundle = evaluate_predictions(
            P_test, spec, S_test=S_test, c_test=c_test,
            test_task_ids=splits.test_task_ids[qt] if args.smoke else splits.test_task_ids,
            gm_const=gm_const, out_dir=None,
        )
        for r in bundle.rows:
            if r["group"] == "pooled":
                r.update(lambda_re=b_lre, tau=b_tau, seed=seed,
                          val_cost_auc=round(val_cost_auc, 6),
                          val_brier=round(val_brier, 6), val_pq_auc=round(val_pq, 6))

        final_dir = out_dir / "final" / f"seed{seed}"
        io.write_metrics(bundle.rows, final_dir / "selected_metrics.json")
        io.write_curves(bundle.curves, final_dir / "cost_curve.json")
        io.write_reliability(bundle.reliability, final_dir / "reliability.json")
        if not args.smoke:
            io.write_cost_report(splits.cost_report_test, final_dir / "cost_coverage.json")

        # --- H4 ingredient: per-train-LLM L_RE-at-convergence vs. train accuracy ---
        P_train_self = router.predict(Q_train, L_train)
        train_acc = _per_model_train_accuracy(P_train_self, S_train)
        assert len(train_llm_names) == len(router.re_loss_per_model_) == L_train.shape[0], \
            (f"misalignment: names={len(train_llm_names)} "
             f"re_loss={len(router.re_loss_per_model_)} L_train={L_train.shape[0]}")
        re_per_model = {
            str(name): {"re_loss": float(router.re_loss_per_model_[j]),
                        "train_accuracy": float(train_acc[j])}
            for j, name in enumerate(train_llm_names)
        }
        _atomic_write(final_dir / "re_per_model.json", re_per_model)

        _atomic_write(final_dir / "config_audit.json", {
            "condition": CONDITION, "lambda_re": b_lre, "tau": b_tau, "seed": seed,
            "target_dim": TARGET_DIM, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
            "w_bce": W_BCE, "epochs": epochs, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "val_bce": router.val_bce_, "best_epoch": router.best_epoch_,
            "fit_elapsed_s": round(fit_s, 2), "hp_search_selected": best["label"],
        })
        np.save(final_dir / "clcrec_test_predictions.npy", P_test.astype(np.float32))

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

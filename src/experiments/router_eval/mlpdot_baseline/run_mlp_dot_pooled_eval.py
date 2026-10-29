"""
Exp 32: Pooled (generalist) MLPDot router, cost-aligned selection (router-eval).

Re-runs exp_27's pooled MLPDot architecture sweep (same 36-config grid, same training
loop) but selects the winning fit by validation budget-cap **cost_auc** instead of val
pq_auc:

  - Train : all 22,906 train queries x all ~2,999 train LLMs -> one MLPDot training set.
  - Val   : all 2,859 val queries x all 374 val LLMs -- one global architecture sweep
            (12 arch x 3 seeds = 36 fits); select ONE config by argmax val budget-cap
            cost_auc (tiebreak min val brier).
  - Test  : all 2,894 test queries x all 376 test LLMs -- the single selected model
            predicts the full matrix; `src.eval.evaluate_predictions` reports the full
            A+B+C suite on the pooled matrix AND each of the 6 benchmark slices + macro.

Cost enters selection only (val cost vector), never fitting; early stopping stays on val
BCE (unchanged from exp_27) -- cost never enters training.

H2 diagnostic: val<->test cost_auc rank agreement (Spearman) across the 36-fit grid,
computed post-hoc from the sweep rows (test_cost_auc is already logged per fit, as in
exp_27).

Usage
-----
  python run_mlp_dot_pooled_eval.py                 # full pooled run (GPU recommended)
  python run_mlp_dot_pooled_eval.py --smoke --device cpu
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
from src.eval.selection import select_by_cost_auc, val_test_agreement
from src.routers.mlp_dot import MLPDotRouter

RESULTS_DIR = results_dir("router-eval", "mlpdot-baseline")

# Architecture grid (unchanged from exp_27)
TD_CANDIDATES = [32, 64, 128]
HD_CANDIDATES = [128, 256]
NL_CANDIDATES = [2, 3]
SEEDS = [0, 1, 2]

# Smoke overrides
TD_SMOKE, HD_SMOKE, NL_SMOKE, SEEDS_SMOKE, EPOCHS_SMOKE = [32], [128], [2], [0], 20

# Fixed training HP (unchanged from exp_27)
EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, DROPOUT = 200, 256, 1e-3, 1e-4, 0.1

# exp_27's val-pq_auc-selected config (re-anchored in exp_30/DESIGN.md).
ANCHOR_EXP27 = {"target_dim": 128, "hidden_dim": 128, "n_layers": 3, "seed": 1,
                "test_pq_auc": 0.765275, "test_cost_auc": 0.7126, "test_ece": 0.0111}


def _atomic_write(path: Path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pooled MLPDot (router-eval), cost-aligned selection | smoke={args.smoke} | "
          f"device={args.device} | out={out_dir}")

    spec = get_spec("router-eval")
    print("\nLoading dataset via src.eval.load_splits ...")
    splits = load_splits(spec)

    Q_train, L_train, S_train = splits.Q_train, splits.L_train, splits.S_train.astype(np.float32)
    Q_val, L_val, S_val = splits.Q_val, splits.L_val, splits.S_val.astype(np.float32)
    Q_test, L_test, S_test = splits.Q_test, splits.L_test, splits.S_test.astype(np.float32)
    Nq_train, Nl_train = Q_train.shape[0], L_train.shape[0]
    print(f"train_q {Q_train.shape}  train_l {L_train.shape}  "
          f"val_q {Q_val.shape}  test_q {Q_test.shape}")

    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Nq_train, size=min(300, Nq_train), replace=False)
        li = rng.choice(Nl_train, size=min(100, Nl_train), replace=False)
        Q_train, L_train = Q_train[qi], L_train[li]
        S_train = S_train[np.ix_(qi, li)]
        qv = rng.choice(Q_val.shape[0], size=min(200, Q_val.shape[0]), replace=False)
        lv = rng.choice(L_val.shape[0], size=min(60, L_val.shape[0]), replace=False)
        Q_val, L_val, S_val = Q_val[qv], L_val[lv], S_val[np.ix_(qv, lv)]
        c_val = splits.c_val[lv]
        print(f"  [smoke] train={Q_train.shape[0]}q x {L_train.shape[0]}l "
              f"val={Q_val.shape[0]}q x {L_val.shape[0]}l")
    else:
        c_val = splits.c_val

    td_c = TD_SMOKE if args.smoke else TD_CANDIDATES
    hd_c = HD_SMOKE if args.smoke else HD_CANDIDATES
    nl_c = NL_SMOKE if args.smoke else NL_CANDIDATES
    seeds = SEEDS_SMOKE if args.smoke else SEEDS
    epochs = EPOCHS_SMOKE if args.smoke else EPOCHS
    configs = list(product(td_c, hd_c, nl_c, seeds))
    n_cfgs = len(configs)

    # --- Single global architecture sweep, selected by val budget-cap cost_auc ---
    sweep_rows = []
    val_cost_aucs, val_briers = [], []
    best_P_val = best_P_test = None
    P_val_by_cfg, P_test_by_cfg = [], []

    for i, (td, hd, nl, seed) in enumerate(configs, 1):
        t0 = time.time()
        router = MLPDotRouter(
            target_dim=td, hidden_dim=hd, n_layers=nl,
            dropout=DROPOUT, normalize_output=True,
            temperature_init=10.0, learn_temperature=True,
            lr=LR, weight_decay=WEIGHT_DECAY,
            epochs=epochs, batch_size=BATCH_SIZE,
            seed=seed, device=args.device, verbose=False,
        ).fit(Q_train, L_train, S_train, Q_val, L_val, S_val)
        fit_s = time.time() - t0

        P_val, P_test = router.predict(Q_val, L_val), router.predict(Q_test, L_test)
        val_pq, val_brier = M.per_query_auc(P_val, S_val), M.brier_score(P_val, S_val)
        val_top1 = M.top1_accuracy(P_val, S_val)
        val_cost_auc = M.cost_metrics(P_val, S_val, c_val)["scalars"]["cost_auc"]
        t_rank = {"top1": M.top1_accuracy(P_test, S_test),
                  "pq_auc": M.per_query_auc(P_test, S_test),
                  "brier": M.brier_score(P_test, S_test)}
        t_cost = M.cost_metrics(P_test, S_test, splits.c_test)["scalars"]

        print(f"  [{i:2d}/{n_cfgs}] td{td}_hd{hd}_nl{nl}_s{seed}  fit={fit_s:.1f}s  "
              f"val_cost_auc={val_cost_auc:.4f}  test_pq_auc={t_rank['pq_auc']:.4f}  "
              f"test_cost_auc={t_cost['cost_auc']:.4f}")

        sweep_rows.append({
            "target_dim": td, "hidden_dim": hd, "n_layers": nl, "seed": seed,
            "val_pq_auc": round(val_pq, 6), "val_brier": round(val_brier, 6),
            "val_top1": round(val_top1, 6), "val_cost_auc": round(val_cost_auc, 6),
            "val_bce": round(router.val_bce_, 6), "best_epoch": router.best_epoch_,
            "test_top1": round(t_rank["top1"], 6), "test_pq_auc": round(t_rank["pq_auc"], 6),
            "test_brier": round(t_rank["brier"], 6),
            "test_cost_auc": round(t_cost["cost_auc"], 6), "test_aiq": round(t_cost["aiq"], 6),
            "fit_elapsed_s": round(fit_s, 2),
        })
        _atomic_write(out_dir / "sweep_metrics.json", sweep_rows)
        val_cost_aucs.append(val_cost_auc)
        val_briers.append(val_brier)
        P_val_by_cfg.append(P_val)
        P_test_by_cfg.append(P_test)

    best_idx = select_by_cost_auc(val_cost_aucs, val_briers)
    best = sweep_rows[best_idx]
    btd, bhd, bnl, bseed = best["target_dim"], best["hidden_dim"], best["n_layers"], best["seed"]
    best_P_test = P_test_by_cfg[best_idx]
    print(f"  => selected (cost-aligned): td={btd} hd={bhd} nl={bnl} s={bseed}  "
          f"val_cost_auc={best['val_cost_auc']:.4f}  val_brier={best['val_brier']:.4f}")

    h2 = val_test_agreement([r["val_cost_auc"] for r in sweep_rows],
                             [r["test_cost_auc"] for r in sweep_rows])
    print(f"H2 val<->test cost_auc Spearman rho={h2['spearman']:.3f} (n={h2['n']})")

    if args.smoke:
        print(f"[smoke] done ({len(sweep_rows)} sweep rows).")
        return

    # --- Pooled + per-benchmark + macro report via src.eval ---
    gm_const = float(S_train.mean())
    bundle = evaluate_predictions(
        best_P_test, spec,
        S_test=splits.S_test, c_test=splits.c_test, test_task_ids=splits.test_task_ids,
        gm_const=gm_const, out_dir=None,
    )

    # Seed variance at the selected (td,hd,nl), from the sweep rows.
    seed_rows = [r for r in sweep_rows
                 if (r["target_dim"], r["hidden_dim"], r["n_layers"]) == (btd, bhd, bnl)]
    seed_agg = {}
    for k in ("test_pq_auc", "test_top1", "test_brier", "test_cost_auc"):
        vals = [r[k] for r in seed_rows]
        seed_agg[f"{k}_seed_mean"] = round(float(np.mean(vals)), 6)
        seed_agg[f"{k}_seed_std"] = round(float(np.std(vals)), 6)

    for r in bundle.rows:
        if r["group"] == "pooled":
            r.update(target_dim=btd, hidden_dim=bhd, n_layers=bnl, seed=bseed,
                     Nq_train=int(Nq_train), Nl_train=int(Nl_train),
                     val_cost_auc=round(best["val_cost_auc"], 6),
                     val_brier=round(best["val_brier"], 6),
                     val_pq_auc=round(best["val_pq_auc"], 6),
                     n_seeds=len(seed_rows), **seed_agg)

    io.write_metrics(bundle.rows, out_dir / "selected_metrics.json")
    io.write_curves(bundle.curves, out_dir / "cost_curve.json")
    io.write_reliability(bundle.reliability, out_dir / "reliability.json")
    io.write_cost_report(splits.cost_report_test, out_dir / "cost_coverage.json")

    audit = {
        "grid": sweep_rows,
        "selected": {"target_dim": btd, "hidden_dim": bhd, "n_layers": bnl, "seed": bseed,
                     **{k: best[k] for k in ("val_cost_auc", "val_brier", "val_pq_auc")}},
        "anchor_exp27": ANCHOR_EXP27,
        "h2_val_test_agreement": h2,
    }
    (out_dir / "selection_audit.json").write_text(json.dumps(audit, indent=2))

    pooled = next(r for r in bundle.rows if r["group"] == "pooled")
    macro = next((r for r in bundle.rows if r["group"] == "macro"), None)
    print(f"\n=== POOLED  : top1={pooled['test_top1_acc']:.4f} "
          f"pq_auc={pooled['test_pq_auc']:.4f} cost_auc={pooled['test_cost_auc']:.4f} "
          f"aiq={pooled['test_aiq']:.4f} (seed pq_auc {seed_agg['test_pq_auc_seed_mean']}"
          f"+-{seed_agg['test_pq_auc_seed_std']}) ===")
    if macro:
        print(f"=== MACRO(6): top1={macro['test_top1_acc']:.4f} "
              f"pq_auc={macro['test_pq_auc']:.4f} aiq={macro['test_aiq']:.4f} ===")
    delta = pooled["test_cost_auc"] - ANCHOR_EXP27["test_cost_auc"]
    print(f"H1: selected (td,hd,nl,s)=({btd},{bhd},{bnl},{bseed}) vs anchor "
          f"({ANCHOR_EXP27['target_dim']},{ANCHOR_EXP27['hidden_dim']},"
          f"{ANCHOR_EXP27['n_layers']},{ANCHOR_EXP27['seed']}); "
          f"delta test_cost_auc={delta:+.4f} (threshold +0.005 with a different fit)")
    print(f"Saved {len(bundle.rows)} rows + {len(sweep_rows)} sweep rows.")


if __name__ == "__main__":
    main()

"""
Exp 22: Pooled (generalist) kNN router, cost-aligned selection (EmbedLLM).

Re-runs exp_19's pooled kNN sweep (same grid, same deterministic fit) but selects the
global (k_q*, k_l*) by validation budget-cap **cost_auc** instead of val pq_auc:

  - Train : all 21,404 train queries x all 66 train LLMs -> one kNN reference matrix.
  - Val   : all 3,567 val queries x all 15 val LLMs -- sweep (k_q, k_l), select ONE global
            (k_q*, k_l*) by argmax val budget-cap cost_auc (tiebreak min val brier).
  - Test  : all 10,702 test queries x all 21 test LLMs -- the single model predicts the
            full matrix; `src.eval.evaluate_predictions` reports the full A+B+C suite on
            the pooled matrix AND each of the 10 EmbedLLM benchmark slices + a macro row.

Cost enters selection only (val cost vector), never fitting; test cost stays evaluation-only.

H2 diagnostic: a second sweep over the identical (k_q, k_l) grid using TEST neighbors,
to check the val<->test rank agreement of cost_auc across the grid (Spearman).

Usage
-----
  python run_knn_pooled_eval.py            # full pooled run
  python run_knn_pooled_eval.py --smoke    # subsampled val, tiny grid
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc, val_test_agreement
from src.routers.knn_baseline import predict_from_neighbors, predict_sweep

RESULTS_DIR = results_dir("embedllm", "knn-baseline")

# Grid unchanged from exp_19 (k_q extended to 4096 per project guidance; EmbedLLM has only
# 66 train LLMs, so K_L stops there).
K_Q_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
K_L_CANDIDATES = [1, 2, 4, 8, 16, 32, 64]

MAX_KQ = 4096  # excluded full-N (21,404) sentinel -- degenerate, see exp_19 rationale.

# exp_19's val-pq_auc-selected config.
ANCHOR_EXP19 = {"k_q": 8, "k_l": 16, "test_pq_auc": 0.600588, "test_cost_auc": 0.458152}


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
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pooled kNN (EmbedLLM), cost-aligned selection | smoke={args.smoke} | out={out_dir}")

    spec = get_spec("embedllm")
    print("\nLoading dataset via src.eval.load_splits ...")
    splits = load_splits(spec)

    Nq_train, Nl_train = splits.Q_train.shape[0], splits.L_train.shape[0]
    print(f"train_q {splits.Q_train.shape} train_l {splits.L_train.shape} "
          f"val_q {splits.Q_val.shape} test_q {splits.Q_test.shape}")
    print(f"Cost proxy (test LLMs): n={splits.cost_report_test['n']} "
          f"coverage={splits.cost_report_test['coverage']:.0%} "
          f"params[B] min/median/max="
          f"{splits.cost_report_test['min_billions']:.3f}/"
          f"{splits.cost_report_test['median_billions']:.1f}/"
          f"{splits.cost_report_test['max_billions']:.1f}")

    S_train = splits.S_train.astype(np.float32)

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        qi = rng.choice(splits.Q_val.shape[0], size=min(200, splits.Q_val.shape[0]),
                        replace=False)
        val_q, S_val_s = splits.Q_val[qi], splits.S_val[qi]
        c_val = splits.c_val
        k_q_run = sorted(set([1, 64, min(MAX_KQ, Nq_train)]))
        k_l_run = sorted(set([1, 16, Nl_train]))
    else:
        val_q, S_val_s = splits.Q_val, splits.S_val
        c_val = splits.c_val
        k_q_run = make_k_grid(K_Q_CANDIDATES, min(MAX_KQ, Nq_train))
        k_l_run = make_k_grid(K_L_CANDIDATES, Nl_train)

    # --- Val sweep: select global (k_q*, k_l*) by val budget-cap cost_auc, tiebreak brier ---
    k_q_max, k_l_max = max(k_q_run), max(k_l_run)
    t0 = time.time()
    nn_q = NearestNeighbors(n_neighbors=k_q_max, metric="cosine",
                            algorithm="brute", n_jobs=-1).fit(splits.Q_train)
    nn_l = NearestNeighbors(n_neighbors=k_l_max, metric="cosine",
                            algorithm="brute", n_jobs=-1).fit(splits.L_train)
    val_q_nbrs = nn_q.kneighbors(val_q, return_distance=False)
    val_l_nbrs = nn_l.kneighbors(splits.L_val, return_distance=False)
    print(f"val neighbors in {time.time()-t0:.1f}s (k_q_max={k_q_max}, k_l_max={k_l_max})")

    sweep_rows = []
    val_cost_aucs, val_briers = [], []
    t0 = time.time()
    for k_q, k_l, S_hat in predict_sweep(S_train, val_q_nbrs, val_l_nbrs, k_q_run, k_l_run):
        pq = M.per_query_auc(S_hat, S_val_s)
        br = M.brier_score(S_hat, S_val_s)
        t1 = M.top1_accuracy(S_hat, S_val_s)
        ca = M.cost_metrics(S_hat, S_val_s, c_val)["scalars"]["cost_auc"]
        sweep_rows.append({"k_q": k_q, "k_l": k_l,
                           "val_pq_auc": round(pq, 6), "val_top1": round(t1, 6),
                           "val_brier": round(br, 6), "val_cost_auc": round(ca, 6)})
        val_cost_aucs.append(ca)
        val_briers.append(br)
    print(f"val sweep in {time.time()-t0:.1f}s ({len(k_q_run)}x{len(k_l_run)} cells)")

    best_idx = select_by_cost_auc(val_cost_aucs, val_briers)
    best = sweep_rows[best_idx]
    k_q, k_l = best["k_q"], best["k_l"]
    print(f"selected (cost-aligned): k_q={k_q} k_l={k_l} "
          f"val_cost_auc={best['val_cost_auc']:.4f} val_brier={best['val_brier']:.4f} "
          f"val_pq_auc={best['val_pq_auc']:.4f}")

    (out_dir / "sweep_metrics.json").write_text(json.dumps(sweep_rows, indent=2))
    if k_q == min(Nq_train, MAX_KQ):
        print("  WARNING: selected k_q saturates the grid -- consider extending K_Q_CANDIDATES.")

    if args.smoke:
        print(f"[smoke] done ({len(sweep_rows)} sweep rows).")
        return

    # --- H2: val<->test cost_auc rank agreement across the (k_q, k_l) grid ---
    test_q_nbrs_full = nn_q.kneighbors(splits.Q_test, return_distance=False)
    test_l_nbrs_full = nn_l.kneighbors(splits.L_test, return_distance=False)
    test_cost_aucs = []
    t0 = time.time()
    for _, _, S_hat_te in predict_sweep(S_train, test_q_nbrs_full, test_l_nbrs_full,
                                         k_q_run, k_l_run):
        test_cost_aucs.append(
            M.cost_metrics(S_hat_te, splits.S_test, splits.c_test)["scalars"]["cost_auc"])
    print(f"test sweep (H2) in {time.time()-t0:.1f}s")
    h2 = val_test_agreement(val_cost_aucs, test_cost_aucs)
    print(f"H2 val<->test cost_auc Spearman rho={h2['spearman']:.3f} (n={h2['n']})")

    # --- Test prediction at the selected (k_q*, k_l*) -- slice the already-computed
    # full-width test neighbors rather than re-fitting NearestNeighbors. ---
    P_test = predict_from_neighbors(S_train, test_q_nbrs_full[:, :k_q], test_l_nbrs_full[:, :k_l])

    gm_const = float(S_train.mean())
    bundle = evaluate_predictions(
        P_test, spec,
        S_test=splits.S_test, c_test=splits.c_test, test_task_ids=splits.test_task_ids,
        gm_const=gm_const, out_dir=None,
    )

    for r in bundle.rows:
        if r["group"] == "pooled":
            r.update(best_k_q=int(k_q), best_k_l=int(k_l),
                     Nq_train=int(Nq_train), Nl_train=int(Nl_train),
                     val_cost_auc=round(best["val_cost_auc"], 6),
                     val_brier=round(best["val_brier"], 6),
                     val_pq_auc=round(best["val_pq_auc"], 6))

    io.write_metrics(bundle.rows, out_dir / "selected_metrics.json")
    io.write_curves(bundle.curves, out_dir / "cost_curve.json")
    io.write_reliability(bundle.reliability, out_dir / "reliability.json")
    io.write_cost_report(splits.cost_report_test, out_dir / "cost_coverage.json")

    audit = {
        "grid": sweep_rows,
        "selected": {"k_q": k_q, "k_l": k_l, **best},
        "anchor_exp19": ANCHOR_EXP19,
        "h2_val_test_agreement": h2,
    }
    (out_dir / "selection_audit.json").write_text(json.dumps(audit, indent=2))

    pooled = next(r for r in bundle.rows if r["group"] == "pooled")
    macro = next((r for r in bundle.rows if r["group"] == "macro"), None)
    print(f"\n=== POOLED  : top1={pooled['test_top1_acc']:.4f} "
          f"pq_auc={pooled['test_pq_auc']:.4f} cost_auc={pooled['test_cost_auc']:.4f} "
          f"aiq={pooled['test_aiq']:.4f} ===")
    if macro:
        print(f"=== MACRO(10): top1={macro['test_top1_acc']:.4f} "
              f"pq_auc={macro['test_pq_auc']:.4f} aiq={macro['test_aiq']:.4f} ===")
    delta = pooled["test_cost_auc"] - ANCHOR_EXP19["test_cost_auc"]
    print(f"H1: selected (k_q,k_l)=({k_q},{k_l}) vs anchor "
          f"({ANCHOR_EXP19['k_q']},{ANCHOR_EXP19['k_l']}); "
          f"delta test_cost_auc={delta:+.4f} (threshold +0.005 with a different config)")
    print(f"Saved {len(bundle.rows)} rows (pooled + per-benchmark + macro), curves, "
          f"reliability, selection_audit.")


if __name__ == "__main__":
    main()

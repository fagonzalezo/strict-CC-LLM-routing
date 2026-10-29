"""
Metadata-representation ablation (router-eval): is the reported `cost_auc` column-order
dependent?

`meta_struct` is a low-dimensional vector of indicators plus one size feature, so distinct
LLMs collapse onto identical rows -- and an identical raw row gives an identical encoded row
(the encoder is a deterministic per-row map), hence bitwise-identical prediction columns.
`src/metrics.py::per_query_cap_curve` resolves its within-cap `argmax` by taking the lowest
column index, so any duplicate group is broken by roster order rather than by the data. That
is the same hazard `static-policy-baseline`'s S1 arm is built to avoid, and here it sits
inside a reported number.

This script measures the exposure rather than assuming it is small:

  * how many DISTINCT LLM-side vectors each variant bundle really has, per split;
  * how many distinct prediction columns each arm's `P_test_*.npy` has;
  * `cost_auc` recomputed under `N_PERM` joint permutations of (P, S_test, c) -- the spread
    is the order dependence;
  * `cost_auc` under a TIE-AVERAGED reading (mean realised score over every column attaining
    the within-cap max), which is order-independent by construction.

Writes `results/representation_degeneracy.json`. Reads only committed artifacts; no refitting.
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, results_dir  # noqa: E402

from src import metrics as M
from src.eval import get_spec, load_splits

DATASET = "router-eval"
FAMILY = "router-eval"
EXPERIMENT = "metadata-ablation"

RESULTS_DIR = results_dir(FAMILY, EXPERIMENT)
VARIANT_ROOT = dataset_processed(DATASET) / "metadata_variants"

ARMS = ["fp", "meta_struct", "meta_name", "concat"]
SEEDS = [0, 1, 2]
ROUTER_FILES = {"BF-DKR": "P_test_bfdkr.npy", "MLPDot": "P_test_mlpdot.npy",
                "kNN-enc": "P_test_knn_enc.npy", "kNN-raw": "P_test_knn_raw.npy"}
N_PERM = 20
PERM_SEED = 42
TIE_BAND_COST_AUC = 0.0035


def _distinct_rows(a: np.ndarray) -> int:
    return int(len(np.unique(np.round(np.asarray(a, dtype=np.float64), 6), axis=0)))


def _realized(P, S_true, c, order=None):
    """(n_queries, n_caps) realised score, lowest-index argmax -- the house convention."""
    if order is not None:
        P, S_true, c = P[:, order], S_true[:, order], c[order]
    caps = np.unique(c)
    out = np.empty((S_true.shape[0], caps.size), dtype=np.float64)
    rows = np.arange(S_true.shape[0])
    for j, b in enumerate(caps):
        idx = np.flatnonzero(c <= b)
        out[:, j] = S_true[rows, idx[P[:, idx].argmax(axis=1)]]
    return out, caps


def _realized_tie_averaged(P, S_true, c):
    """Order-independent: mean realised score over every column attaining the within-cap max."""
    caps = np.unique(c)
    out = np.empty((S_true.shape[0], caps.size), dtype=np.float64)
    for j, b in enumerate(caps):
        idx = np.flatnonzero(c <= b)
        sub = P[:, idx]
        tied = sub >= sub.max(axis=1, keepdims=True) - 1e-12
        out[:, j] = (S_true[:, idx] * tied).sum(axis=1) / tied.sum(axis=1)
    return out, caps


def _auc(realized, caps, lo, hi):
    return float(M.cap_curve_auc(list(zip(caps.tolist(), realized.mean(axis=0).tolist())), lo, hi))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    spec = get_spec(DATASET)
    sp = load_splits(spec)
    S_test = np.asarray(sp.S_test, dtype=np.float64)
    c_test = np.asarray(sp.c_test, dtype=np.float64)
    lo, hi = float(c_test.min()), float(c_test.max())
    n_l = S_test.shape[1]

    out = {"dataset": DATASET, "n_test_llms": int(n_l), "n_perm": N_PERM,
           "perm_seed": PERM_SEED, "tie_band_cost_auc": TIE_BAND_COST_AUC,
           "bundle_distinct_llm_vectors": {}, "arms": {}}

    # ── (1) how distinguishable are the LLMs in each variant bundle at all? ──
    import pickle
    for arm in ARMS:
        path = (dataset_processed(DATASET) / "llm_perf_dataset.pkl") if arm == "fp" \
            else (VARIANT_ROOT / arm / "llm_perf_dataset.pkl")
        if not path.exists():
            continue
        print(f"read: {path.relative_to(ROOT)}")
        d = pickle.load(open(path, "rb"))
        rec = {}
        for s in ["train", "val", "test"]:
            E = np.asarray(d["llms"][s]["embeddings"], dtype=np.float64)
            nd = _distinct_rows(E)
            rec[s] = {"n_llms": int(len(E)), "n_distinct": nd,
                      "distinct_frac": round(nd / len(E), 4)}
        out["bundle_distinct_llm_vectors"][arm] = rec
        print(f"  {arm:<12} " + "  ".join(
            f"{s} {rec[s]['n_distinct']}/{rec[s]['n_llms']}" for s in rec))

    # ── (2) column degeneracy + order dependence of the reported cost_auc ────
    rng = np.random.default_rng(PERM_SEED)
    perms = [rng.permutation(n_l) for _ in range(N_PERM)]

    for arm in ARMS:
        out["arms"][arm] = {}
        for router, fname in ROUTER_FILES.items():
            per_seed = []
            for seed in SEEDS:
                p = RESULTS_DIR / f"seed{seed}" / arm / fname
                if not p.exists():
                    continue
                P = np.load(p).astype(np.float64)
                n_dist_cols = _distinct_rows(P.T)
                base, caps = _realized(P, S_test, c_test)
                auc_base = _auc(base, caps, lo, hi)
                tie_r, tie_caps = _realized_tie_averaged(P, S_test, c_test)
                auc_tie = _auc(tie_r, tie_caps, lo, hi)
                if n_dist_cols < n_l:
                    aucs = []
                    for order in perms:
                        r, cp = _realized(P, S_test, c_test, order)
                        aucs.append(_auc(r, cp, lo, hi))
                    spread = float(max(aucs) - min(aucs))
                    perm_stats = {"min": round(float(min(aucs)), 6),
                                  "max": round(float(max(aucs)), 6),
                                  "mean": round(float(np.mean(aucs)), 6),
                                  "spread": round(spread, 6),
                                  "spread_exceeds_tie_band": bool(spread > TIE_BAND_COST_AUC)}
                else:
                    perm_stats = None  # unique columns -> argmax has no tie to break
                per_seed.append({
                    "seed": seed,
                    "n_distinct_pred_columns": n_dist_cols,
                    "has_duplicate_columns": bool(n_dist_cols < n_l),
                    "cost_auc_as_reported": round(auc_base, 6),
                    "cost_auc_tie_averaged": round(auc_tie, 6),
                    "tie_averaged_delta": round(auc_tie - auc_base, 6),
                    "cost_auc_under_permutation": perm_stats,
                })
                extra = (f"  perm[{perm_stats['min']:.6f}, {perm_stats['max']:.6f}] "
                         f"spread {perm_stats['spread']:.6f}") if perm_stats else "  (unique columns)"
                print(f"  {arm:<12} {router:<8} seed{seed}  cols {n_dist_cols}/{n_l}  "
                      f"reported {auc_base:.6f}  tie-avg {auc_tie:.6f}{extra}")
            out["arms"][arm][router] = per_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / "representation_degeneracy.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()

"""
Dataset-independent pooled/generalist reporting.

Generalizes the per-experiment pooled reporters this project used before it: a single
global model predicts the full test matrix, and per-benchmark numbers are
recovered by *slicing* that matrix by `task_ids`, using whatever `groups` map
a `DatasetSpec` provides instead of a hard-coded 6-benchmark taxonomy.

The val-tuned `router_envelope` curve (`M.cap_router_envelope`, tuned via a
val-split cap curve) has been removed project-wide: it was the only place val
predictions entered *reporting* (the router still uses val at fit time for
bandwidth/hyperparameter tuning), so `report_pooled` takes only test arrays.
"""

from __future__ import annotations

import numpy as np

from src import metrics as M
from src.eval.spec import DatasetSpec


def _round_metrics(prefix, d):
    return {f"{prefix}{k}": round(float(v), 6) for k, v in d.items()}


def _view_row(group, S_hat_te, S_te, c_cost, gm_const, recall_ks, cap_budgets):
    """Full A+B+C metric suite for one view (pooled or a benchmark slice).

    Returns (row, curves, reliability). No val-tuned envelope: `curves`
    contains only router / floor / oracle / single_llm.
    """
    rank = M.ranking_metrics(S_hat_te, S_te, ks=recall_ks)
    cal = M.calibration_metrics(S_hat_te, S_te)
    cost = M.cost_metrics(S_hat_te, S_te, c_cost, cap_budgets=cap_budgets)
    cost["curves"].pop("router_envelope", None)

    reliability = M.reliability_curve(S_hat_te, S_te)

    row = {"group": group, "Nq_test": int(S_hat_te.shape[0])}
    row.update(_round_metrics("test_", rank))
    row.update(_round_metrics("test_", cal))
    row.update(_round_metrics("test_", cost["scalars"]))
    if gm_const is not None:
        gm = np.full(S_te.shape, float(gm_const), dtype=np.float32)
        gm_rank = M.ranking_metrics(gm, S_te, ks=recall_ks)
        row["gm_test_top1_acc"] = round(float(gm_rank["top1_acc"]), 6)
        row["gm_test_pq_auc"] = round(float(gm_rank["pq_auc"]), 6)
    return row, cost["curves"], reliability


def build_breakdown_masks(task_ids, groups) -> dict[str, np.ndarray]:
    """One boolean row-mask per benchmark over a split's task_ids."""
    return {g: np.array([pred(t) for t in task_ids], dtype=bool)
            for g, pred in groups.items()}


def report_pooled(
    S_hat_test, S_test, c_test, test_task_ids, spec: DatasetSpec, gm_const=None,
):
    """Pooled + per-benchmark + macro reporting from a single global model's
    test predictions, parameterized by `spec.groups` / `spec.cap_budgets` /
    `spec.recall_ks`.

    Parameters
    ----------
    S_hat_test, S_test : (Nq_test, Nl_test)  predicted / true test scores (full pool).
    c_test              : (Nl_test,)          test-LLM cost vector (params, B).
    test_task_ids       : per-query benchmark task_ids for slicing.
    spec                : DatasetSpec supplying groups, cap_budgets, recall_ks.
    gm_const            : optional constant s-hat for the global-mean baseline
                           (typically S_train.mean()).

    Returns
    -------
    rows        : [pooled_row, <group>_row, ..., macro_row]
    curves      : {"pooled": ..., "<group>": ...}   (router/floor/oracle/single_llm)
    reliability : {"pooled": ..., "<group>": ...}
    """
    S_hat_test = np.asarray(S_hat_test, dtype=np.float32)
    S_test = np.asarray(S_test)

    test_masks = build_breakdown_masks(test_task_ids, spec.groups)

    rows, curves, reliability = [], {}, {}

    # --- Pooled view (whole matrix) ---
    pooled_row, pooled_curves, pooled_rel = _view_row(
        "pooled", S_hat_test, S_test, c_test, gm_const, spec.recall_ks, spec.cap_budgets)
    rows.append(pooled_row)
    curves["pooled"] = pooled_curves
    reliability["pooled"] = pooled_rel

    # --- Per-benchmark slices of the single pooled model ---
    group_rows = []
    for g in spec.groups:
        mte = test_masks[g]
        if not mte.any():
            continue
        row, crv, rel = _view_row(
            g, S_hat_test[mte], S_test[mte], c_test, gm_const, spec.recall_ks, spec.cap_budgets)
        rows.append(row)
        group_rows.append(row)
        curves[g] = crv
        reliability[g] = rel

    # --- Macro row: mean over the benchmark slices of every numeric field ---
    if group_rows:
        numeric_keys = [k for k, v in group_rows[0].items()
                        if isinstance(v, (int, float)) and k != "Nq_test"]
        macro = {"group": "macro", "n_groups": len(group_rows)}
        for k in numeric_keys:
            macro[k] = round(float(np.mean([r[k] for r in group_rows])), 6)
        rows.append(macro)

    return rows, curves, reliability

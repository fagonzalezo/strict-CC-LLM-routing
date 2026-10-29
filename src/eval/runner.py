"""
End-to-end evaluation driver for `src/eval`.

`evaluate` takes a router (or router factory) and a DatasetSpec, and runs the
whole pipeline: load splits -> fit -> predict -> report -> optional write.
`evaluate_predictions` skips fit/predict for cases where the test prediction
matrix was already produced elsewhere (e.g. after an EmbedKit hyperparameter
sweep an experiment script ran itself) and only need scoring.

The router still receives the val split at `fit()` time (bandwidth / HP
tuning stays the router's job); only *reporting* no longer consumes val
predictions, since the val-tuned router_envelope curve has been removed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eval import io
from src.eval.data import Splits, load_splits
from src.eval.report import report_pooled
from src.eval.spec import DatasetSpec


@dataclass
class Bundle:
    rows: list[dict]
    curves: dict
    reliability: dict
    splits: Splits


def _fit_router(router, splits: Splits) -> None:
    """Call `router.fit(...)`, passing the val split only if `fit` accepts it.

    Most routers (DMKDMRRouter, MLPDotRouter, KDMRouter) take
    `fit(Qtr, Ltr, Str, Qva, Lva, Sva)` and use val for bandwidth/HP tuning.
    `KNNRouter` is train-only: `fit(Qtr, Ltr, Str)`. Introspecting the
    signature keeps `evaluate` working for both without a special case per
    router class.
    """
    n_params = len(inspect.signature(router.fit).parameters)
    if n_params >= 6:
        router.fit(
            splits.Q_train, splits.L_train, splits.S_train,
            splits.Q_val, splits.L_val, splits.S_val,
        )
    else:
        router.fit(splits.Q_train, splits.L_train, splits.S_train)


def evaluate(
    router,
    spec: DatasetSpec,
    *,
    gm_from_train: bool = True,
    out_dir: str | Path | None = None,
) -> Bundle:
    """Load `spec`'s splits, fit `router` on train/val, predict on test, report.

    `router` must implement the de-facto project contract:
        fit(Q_train, L_train, S_train, Q_val, L_val, S_val) -> self
        predict(Q, L) -> (Nq, Nl) float32 in [0, 1]
    """
    splits = load_splits(spec)

    _fit_router(router, splits)
    P_test = router.predict(splits.Q_test, splits.L_test)

    gm_const = float(np.asarray(splits.S_train).mean()) if gm_from_train else None
    rows, curves, reliability = report_pooled(
        P_test, splits.S_test, splits.c_test, splits.test_task_ids, spec, gm_const=gm_const,
    )

    if out_dir is not None:
        _write_bundle(rows, curves, reliability, splits.cost_report_test, out_dir)

    return Bundle(rows=rows, curves=curves, reliability=reliability, splits=splits)


def evaluate_predictions(
    P_test: np.ndarray,
    spec: DatasetSpec,
    *,
    S_test: np.ndarray | None = None,
    c_test: np.ndarray | None = None,
    test_task_ids=None,
    gm_const: float | None = None,
    out_dir: str | Path | None = None,
) -> Bundle:
    """Score an already-computed test prediction matrix against `spec`.

    If `S_test`/`c_test`/`test_task_ids` are omitted, they are loaded fresh
    from `spec` (the common case: the caller already has P_test from its own
    router-fitting / HP-sweep code and just wants the standard report).
    """
    splits = None
    if S_test is None or c_test is None or test_task_ids is None:
        splits = load_splits(spec)
        S_test = splits.S_test if S_test is None else S_test
        c_test = splits.c_test if c_test is None else c_test
        test_task_ids = splits.test_task_ids if test_task_ids is None else test_task_ids
        if gm_const is None:
            gm_const = float(np.asarray(splits.S_train).mean())

    rows, curves, reliability = report_pooled(
        P_test, S_test, c_test, test_task_ids, spec, gm_const=gm_const,
    )

    if out_dir is not None:
        cost_report = splits.cost_report_test if splits is not None else None
        _write_bundle(rows, curves, reliability, cost_report, out_dir)

    return Bundle(rows=rows, curves=curves, reliability=reliability, splits=splits)


def _write_bundle(rows, curves, reliability, cost_report, out_dir) -> None:
    out_dir = Path(out_dir)
    io.write_metrics(rows, out_dir / "selected_metrics.json")
    io.write_curves(curves, out_dir / "cost_curve.json")
    io.write_reliability(reliability, out_dir / "reliability.json")
    if cost_report is not None:
        io.write_cost_report(cost_report, out_dir / "cost_coverage.json")

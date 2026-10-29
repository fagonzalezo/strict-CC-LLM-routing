"""
Cost-aligned model-selection primitives, shared by the cost-aligned re-run
campaign (embedllm knn-baseline/mlpdot-baseline/bfdkr-sweep, router-eval
knn-baseline/mlpdot-baseline/bfdkr-sweep).

Predecessor experiments selected every router by validation `pq_auc` — a
pure ranking metric that is structurally blind to the paper's primary cost
metric, per-query budget-cap `cost_auc` (`src.metrics.cost_metrics` ->
`per_query_cap_curve`). This module provides the replacement selection rule
(argmax val cost_auc, tiebreak min val brier) and a val<->test rank-agreement
diagnostic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def select_by_cost_auc(val_cost_aucs, val_briers) -> int:
    """argmax val cost_auc; ties (np.isclose on the max) broken by min val brier.

    Returns the winning index into the input sequences.
    """
    val_cost_aucs = np.asarray(val_cost_aucs, dtype=float)
    val_briers = np.asarray(val_briers, dtype=float)
    if val_cost_aucs.shape != val_briers.shape or val_cost_aucs.ndim != 1 or len(val_cost_aucs) == 0:
        raise ValueError("val_cost_aucs and val_briers must be equal-length 1-D non-empty sequences")
    best_ca = val_cost_aucs.max()
    tied = np.flatnonzero(np.isclose(val_cost_aucs, best_ca))
    return int(tied[np.argmin(val_briers[tied])])


def val_test_agreement(val_cost_aucs, test_cost_aucs) -> dict:
    """Spearman rank correlation between val and test cost_auc across a config grid."""
    val_cost_aucs = np.asarray(val_cost_aucs, dtype=float)
    test_cost_aucs = np.asarray(test_cost_aucs, dtype=float)
    if val_cost_aucs.shape != test_cost_aucs.shape:
        raise ValueError("val_cost_aucs and test_cost_aucs must have the same shape")
    rho, p = spearmanr(val_cost_aucs, test_cost_aucs)
    return {"spearman": float(rho), "p_value": float(p), "n": int(len(val_cost_aucs))}

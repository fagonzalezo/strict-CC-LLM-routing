"""
Dataset-independent LLM-routing evaluation framework.

Public API
----------
DatasetSpec, get_spec, ROUTER_EVAL_SPEC, EMBEDLLM_SPEC
    Per-dataset configuration: cost proxy, benchmark groups, dataset path.
load_splits, Splits
    Load train/val/test arrays + cost vectors for a DatasetSpec.
report_pooled
    Turn a (P_test, S_test, c_test) triple into pooled + per-group + macro
    metric rows, curves, and reliability diagrams. No val args — the
    val-tuned router_envelope curve has been removed project-wide.
evaluate, evaluate_predictions, Bundle
    End-to-end drivers: fit+predict+report a router, or report an
    already-computed prediction matrix.
select_by_cost_auc, val_test_agreement
    Cost-aligned model-selection primitives (argmax val cost_auc selection,
    val<->test rank agreement).
plot_cost_quality, plot_reliability, grid_figure, plot_method_comparison
    Plotting primitives generalized from the per-experiment plot scripts.
"""

from src.eval.spec import DatasetSpec, EMBEDLLM_SPEC, ROUTER_EVAL_SPEC, get_spec
from src.eval.data import Splits, load_splits
from src.eval.report import build_breakdown_masks, report_pooled
from src.eval.runner import Bundle, evaluate, evaluate_predictions
from src.eval.selection import select_by_cost_auc, val_test_agreement
from src.eval.plots import (
    grid_figure,
    plot_cost_quality,
    plot_method_comparison,
    plot_reliability,
)

__all__ = [
    "DatasetSpec", "EMBEDLLM_SPEC", "ROUTER_EVAL_SPEC", "get_spec",
    "Splits", "load_splits",
    "build_breakdown_masks", "report_pooled",
    "Bundle", "evaluate", "evaluate_predictions",
    "select_by_cost_auc", "val_test_agreement",
    "grid_figure", "plot_cost_quality", "plot_method_comparison", "plot_reliability",
]

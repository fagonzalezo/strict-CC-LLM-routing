"""
Dataset-independent data access for `src/eval`, built on `src.data_loader`.

Loads the standard cold-cold split convention shared by both datasets:
train/val/test queries AND LLMs are independently split, so
`dataset["scores"][query_split][llm_split]` is read with matching split names
(`scores["train"]["train"]`, `["val"]["val"]`, `["test"]["test"]`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_loader import get_arrays, load_perf_dataset
from src.eval.spec import DatasetSpec


@dataclass
class Splits:
    """Arrays for one DatasetSpec, ready to feed a router + the reporter."""

    Q_train: np.ndarray
    L_train: np.ndarray
    S_train: np.ndarray
    Q_val: np.ndarray
    L_val: np.ndarray
    S_val: np.ndarray
    Q_test: np.ndarray
    L_test: np.ndarray
    S_test: np.ndarray
    c_test: np.ndarray
    c_val: np.ndarray
    test_task_ids: np.ndarray
    val_task_ids: np.ndarray
    cost_report_test: dict
    cost_report_val: dict


def load_splits(spec: DatasetSpec) -> Splits:
    """Load train/val/test arrays + cost vectors for `spec` from its dataset_path."""
    dataset = load_perf_dataset(spec.dataset_path)

    Q_train, L_train, S_train = get_arrays(dataset, "train", "train")
    Q_val, L_val, S_val = get_arrays(dataset, "val", "val")
    Q_test, L_test, S_test = get_arrays(dataset, "test", "test")

    c_test, report_test = spec.cost_fn(dataset, "test")
    c_val, report_val = spec.cost_fn(dataset, "val")

    test_task_ids = np.asarray(dataset["queries"]["test"]["task_ids"])
    val_task_ids = np.asarray(dataset["queries"]["val"]["task_ids"])

    return Splits(
        Q_train=Q_train, L_train=L_train, S_train=S_train,
        Q_val=Q_val, L_val=L_val, S_val=S_val,
        Q_test=Q_test, L_test=L_test, S_test=S_test,
        c_test=c_test, c_val=c_val,
        test_task_ids=test_task_ids, val_task_ids=val_task_ids,
        cost_report_test=report_test, cost_report_val=report_val,
    )

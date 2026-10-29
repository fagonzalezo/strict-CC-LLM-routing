"""
Parameter-count cost proxy for cost-aware routing evaluation.

The router-eval dataset has no $/token or token-count cost data. Instead, cost is
proxied by **model parameter count (in billions)**. Every model in the dataset was
pre-filtered to carry a size token by src/data_pipeline/router_eval/partition_models.py,
and the resulting size strings ("7B", "13B", "4X7B", "248M", ...) are stored in
data/router-eval/processed/model_partition.pkl under subset["size_strings"], aligned
to subset["indices"] (positions in the original 5000-model array). Coverage is 100%
by construction, so this module reuses that artifact rather than re-parsing names.

Cost is a compute/size proxy, NOT a dollar cost.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.paths import dataset_processed

DEFAULT_PARTITION_PATH = dataset_processed("router-eval") / "model_partition.pkl"

_NUM_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)([BM])$")


def _num_unit_to_billions(tok: str) -> float:
    m = _NUM_UNIT_RE.match(tok)
    if m is None:
        raise ValueError(f"Unparseable size token: {tok!r}")
    val = float(m.group(1))
    return val if m.group(2) == "B" else val / 1000.0


def size_string_to_billions(s: str) -> float:
    """Convert a stored size string to parameter count in billions.

    Examples
    --------
    "7B"    -> 7.0
    "1.5B"  -> 1.5
    "248M"  -> 0.248
    "4X7B"  -> 28.0   (MoE: n_experts × expert size)
    """
    s = s.strip().upper().replace(" ", "")
    if "X" in s:
        left, right = s.split("X", 1)
        return float(left) * _num_unit_to_billions(right)
    return _num_unit_to_billions(s)


def build_cost_vector(
    dataset: dict, split: str, partition_path: str | Path = DEFAULT_PARTITION_PATH
):
    """Return (c, report) for the LLMs of `split`.

    c : (Nl,) float64 — parameter count in billions, aligned to
        dataset["llms"][split] column order.
    report : dict — coverage (asserted 1.0), size histogram, min/median/max.
    """
    with open(partition_path, "rb") as f:
        part = pickle.load(f)
    idx2size = {
        int(i): str(s)
        for i, s in zip(part["subset"]["indices"], part["subset"]["size_strings"])
    }

    idx5000 = dataset["llms"][split]["indices_in_5000"]
    n = len(idx5000)
    c = np.full(n, np.nan, dtype=np.float64)
    missing = []
    sizes = []
    for j, gi in enumerate(idx5000):
        gi = int(gi)
        if gi not in idx2size:
            missing.append(gi)
            continue
        sz = idx2size[gi]
        sizes.append(sz.upper())
        c[j] = size_string_to_billions(sz)

    coverage = 1.0 - len(missing) / n
    assert coverage == 1.0, (
        f"Cost coverage {coverage:.4f} < 1.0 for split={split}: "
        f"{len(missing)} LLMs missing from model_partition.pkl (artifact drift?)"
    )

    report = {
        "split": split,
        "n": int(n),
        "coverage": float(coverage),
        "min_billions": float(np.nanmin(c)),
        "median_billions": float(np.nanmedian(c)),
        "max_billions": float(np.nanmax(c)),
        "size_histogram": dict(Counter(sizes).most_common()),
    }
    return c, report

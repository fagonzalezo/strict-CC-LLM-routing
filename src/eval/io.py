"""
JSON I/O for `src/eval` reports, on one canonical schema so `plots.py` and
downstream tooling have a single input contract.

Historical outputs are inconsistent: `cost_curve.json` (singular, router-eval)
vs `cost_curves.json` (plural, embedllm), and pooled data nested under a
condition key, under `"pooled"`, or flat (see the three loader shapes in the
old `plot_method_comparison.py` scripts). New writes standardize on the
singular `cost_curve.json` with a flat `{group: {router, floor, oracle,
single_llm}}` shape and no `router_envelope` field. `load_curves` still
normalizes the historical variants for reading old results.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_metrics(rows: list[dict], path) -> None:
    """Write the `rows` list (pooled + per-group + macro) from report_pooled."""
    _write_json(rows, path)


def write_curves(curves: dict, path) -> None:
    """Write the `curves` dict from report_pooled. Never contains router_envelope."""
    for group_curves in curves.values():
        group_curves.pop("router_envelope", None)
    _write_json(curves, path)


def write_reliability(reliability: dict, path) -> None:
    _write_json(reliability, path)


def write_cost_report(report: dict, path) -> None:
    _write_json(report, path)


def load_curves(path) -> dict:
    """Load a cost-curve JSON, normalizing historical shapes to a flat
    `{group: {router, floor, oracle, single_llm, ...}}` dict.

    Handles:
      - flat (new convention / exp_17 single-scenario file): {group: {...}}
      - nested under a condition key (old exp_13 style): {condition: {group: {...}}}
    A file is "flat" if its top-level values already look like curve dicts
    (contain a "router" key); otherwise the first nested value is unwrapped.
    """
    raw = json.loads(Path(path).read_text())
    if not raw:
        return raw
    first = next(iter(raw.values()))
    if isinstance(first, dict) and "router" in first:
        return raw
    # Nested one level under a condition key — unwrap it.
    condition = next(iter(raw))
    return raw[condition]


def load_reliability(path) -> dict:
    return json.loads(Path(path).read_text())

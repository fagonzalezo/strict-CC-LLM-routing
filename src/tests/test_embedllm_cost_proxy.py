"""
Coverage regression for `src/embedllm_cost_proxy.py` over the real EmbedLLM model roster.

Not a test of the parsed values themselves (those are spot-checked against HuggingFace model
cards by inspection, per the module docstring) -- this guards the one thing that silently breaks
at eval time: `build_cost_vector` asserts every model in a split has a resolvable size, and that
assertion has no fallback. A model added to `data/embedllm/raw/model_list.csv` whose name carries
no parseable size token, and that isn't in `_MANUAL_SIZES_B`, doesn't fail here -- it fails much
later, inside a real eval run, as an opaque AssertionError.

Run: `python -m pytest src/tests -q`  (from anywhere).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.embedllm_cost_proxy import _MANUAL_SIZES_B, _parse_size  # noqa: E402
from src.paths import dataset_raw  # noqa: E402


def test_every_model_list_name_resolves_to_a_size():
    """All 107 model_list.csv models go into a routing split, including `bigcode__octocoder`,
    whose name carries no size token (see docs/dataset_construction.md §6.2). Coverage must be
    107/107."""
    with open(dataset_raw("embedllm") / "model_list.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 107

    unresolved = [r["model_name"] for r in rows
                  if r["model_name"] not in _MANUAL_SIZES_B
                  and _parse_size(r["model_name"]) is None]
    assert unresolved == [], (
        f"{unresolved} have no parseable size and aren't in _MANUAL_SIZES_B -- "
        "build_cost_vector's coverage assert will fail on a fresh eval run. "
        "Add them to _MANUAL_SIZES_B in src/embedllm_cost_proxy.py."
    )


def test_octocoder_size_is_in_the_manual_lookup():
    """bigcode/octocoder is StarCoder-based (15.5B) and its name carries no size token."""
    assert _parse_size("bigcode__octocoder") is None
    assert _MANUAL_SIZES_B["bigcode__octocoder"] == 15.5

"""
Parameter-count cost proxy for the EmbedLLM routing experiments.

Model parameter counts are extracted from HuggingFace model IDs via a regex
parser (handles standard tokens like "7B", "13b", "6.7b", "10.7B",
"bloom-7b1", MoE lead "4x7B") plus a 15-entry static lookup table for models
whose names do not carry a parseable size token.  Lookup-table values were
verified against HuggingFace model cards by the experiment authors.

All 107 EmbedLLM LLMs are covered — coverage is asserted at runtime and logged
to results/cost_coverage.json.  Cost = parameter count in billions (a compute
proxy, not $/token).

Usage
-----
    from src.embedllm_cost_proxy import build_cost_vector
    c, report = build_cost_vector(dataset, "test")
    # c      : (Nl,) float64 — billions, aligned to dataset["llms"]["test"]
    # report : dict  — coverage info + size histogram
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

# ── Static lookup for 15 models whose names lack parseable size tokens ────────
# Values verified against HuggingFace model cards.
_MANUAL_SIZES_B: dict[str, float] = {
    "AdaptLLM__medicine-LLM":                        7.0,   # LLaMA-1 7B base
    "AdaptLLM__medicine-chat":                        7.0,   # LLaMA-1 7B base
    "Biomimicry-AI__ANIMA-Nectar-v2":                7.0,   # Mistral-7B base
    "CultriX__NeuralTrix-bf16":                      7.0,   # Mistral-7B merge
    "bigcode__octocoder":                            15.5,  # 15.5B (HF model card: StarCoder base, instruction-tuned)
    "golaxy__gowizardlm":                            7.0,   # WizardLM 7B base
    "kevin009__llamaRAGdrama":                        7.0,   # LLaMA-2 7B base
    "kyujinpy__Sakura-SOLRCA-Math-Instruct-DPO-v1":  10.7,  # SOLAR-10.7B base
    "microsoft__phi-1_5":                            1.3,   # Phi-1.5 (name = version)
    "microsoft__phi-2":                              2.7,   # Phi-2   (name = version)
    "openchat__openchat-3.5-0106":                   7.0,   # Mistral-7B base
    "openchat__openchat_3.5":                        7.0,   # Mistral-7B base
    "rishiraj__CatPPT-base":                         7.0,   # LLaMA-2 7B base
    "shleeeee__mistral-ko-tech-science-v1":          7.0,   # Mistral-7B base
    "zhengr__MixTAO-7Bx2-MoE-v8.1":                 14.0,  # 2 × 7 B MoE (7B×2)
}

# ── Regex patterns ─────────────────────────────────────────────────────────────

# MoE lead: 4x7B, 4X7b  (count × size)
_MOE_LEAD_RE = re.compile(r"(\d+)[xX](\d+(?:\.\d+)?)([BbMm])")

# Simple size: 7B, 13b, 6.7b, 10.7B, 0.5B, bloom-7b1
# Preceded by: separator (-, _, start); unit B/M must NOT be followed by a letter.
# Trailing digit is allowed so that "7b1" (bloom-7b1) parses as 7B.
_SIMPLE_RE = re.compile(r"(?:^|[-_])(\d+(?:\.\d+)?)([BbMm])(?:[^a-zA-Z]|$)")


def _parse_size(name: str) -> float | None:
    """Extract parameter count (B) from a HuggingFace model ID, or return None."""
    model = name.split("__")[-1] if "__" in name else name

    # MoE lead (4x7B → 28 B) takes priority over simple matches.
    m = _MOE_LEAD_RE.search(model)
    if m:
        count = int(m.group(1))
        val = float(m.group(2))
        unit = m.group(3).upper()
        return count * (val if unit == "B" else val / 1000.0)

    # Simple: collect all matches, take the rightmost (most specific for "Qwen1.5-7B").
    hits = _SIMPLE_RE.findall(model)
    if hits:
        val_str, unit = hits[-1]
        val = float(val_str)
        return val if unit.upper() == "B" else val / 1000.0

    return None


def build_cost_vector(dataset: dict, split: str) -> tuple[np.ndarray, dict]:
    """Return (c, report) for the LLMs of `split`.

    Parameters
    ----------
    dataset : dict   loaded by src.data_loader.load_perf_dataset
    split   : str    one of "train", "val", "test"

    Returns
    -------
    c      : (Nl,) float64  parameter count in billions, aligned to
             dataset["llms"][split]["names"] column order.
    report : dict   coverage (asserted 1.0), size histogram, min/median/max,
             per-model costs.
    """
    names = list(dataset["llms"][split]["names"])
    n = len(names)
    c = np.full(n, np.nan, dtype=np.float64)
    size_labels: list[str] = []
    missing: list[str] = []

    for j, name in enumerate(names):
        # Lookup table takes priority (manually verified values).
        if name in _MANUAL_SIZES_B:
            b = _MANUAL_SIZES_B[name]
        else:
            b = _parse_size(name)

        if b is None:
            missing.append(name)
        else:
            c[j] = b
            size_labels.append(f"{b:.1f}B")

    coverage = 1.0 - len(missing) / n
    if coverage < 1.0:
        raise AssertionError(
            f"Cost coverage {coverage:.4f} < 1.0 for split={split!r}. "
            f"Missing {len(missing)} LLMs: {missing}. "
            "Add them to _MANUAL_SIZES_B in src/embedllm_cost_proxy.py."
        )

    report: dict = {
        "split":            split,
        "n":                int(n),
        "coverage":         float(coverage),
        "min_billions":     float(np.nanmin(c)),
        "median_billions":  float(np.nanmedian(c)),
        "max_billions":     float(np.nanmax(c)),
        "size_histogram":   dict(Counter(size_labels).most_common()),
        "model_costs":      {name: float(c[j]) for j, name in enumerate(names)},
    }
    return c, report

"""
DatasetSpec: the single object that captures everything dataset-specific about
an LLM-routing evaluation, so the rest of `src/eval` (data loading, reporting,
plotting) is dataset-independent.

Comparing the generalist DMKDMR experiments on both datasets (embedllm/exp_17,
router-eval/exp_28) showed the data representation and evaluation protocol are
already identical on the critical path (same pickle schema, same router
contract `fit(...)->predict(Q,L)->P (Nq,Nl) in [0,1]`, same `src.metrics` A+B+C
suite). Only three things differ per dataset:

  1. cost proxy      — how (Nl,) param-count-in-billions is derived
  2. benchmark groups — task_id -> predicate map used to slice pooled matrices
  3. dataset path     — which llm_perf_dataset.pkl to load

`cap_budgets=None` lets `src.metrics.cost_metrics` fall back to its default
{1, 3, 7, 30, c_max} grid, which both datasets' cost ranges (~0.1-86B) already
suit — no dataset-specific override needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src import cost_proxy
from src import embedllm_cost_proxy
from src.paths import dataset_path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_path: Path
    cost_fn: Callable[..., tuple]            # (dataset, split) -> (c, report)
    groups: dict[str, Callable[[str], bool]]  # benchmark name -> task_id predicate
    display_names: dict[str, str] = field(default_factory=dict)
    cap_budgets: tuple[float, ...] | None = None
    recall_ks: tuple[int, ...] = (1, 3, 5, 10)


# ── router-eval: 6 benchmarks ─────────────────────────────────────────

ROUTER_EVAL_GROUPS: dict[str, Callable[[str], bool]] = {
    "arc":        lambda t: t == "arc",
    "gsm8k":      lambda t: t == "gsm8k",
    "hellaswag":  lambda t: t == "hellaswag",
    "winogrande": lambda t: t == "winogrande",
    "truthfulqa": lambda t: t == "harness_truthfulqa_mc_0",
    "mmlu":       lambda t: t.startswith("harness_hendrycksTest_"),
}

ROUTER_EVAL_SPEC = DatasetSpec(
    name="router-eval",
    dataset_path=dataset_path("router-eval"),
    cost_fn=cost_proxy.build_cost_vector,
    groups=ROUTER_EVAL_GROUPS,
    cap_budgets=None,
)

# ── embedllm: 10 benchmarks (lifted from exp_17's BENCHMARKS map) ────────────

EMBEDLLM_GROUPS: dict[str, Callable[[str], bool]] = {
    "asdiv":      lambda t: t == "asdiv",
    "gpqa":       lambda t: t.startswith("gpqa_"),
    "gsm8k":      lambda t: t == "gsm8k",
    "logiqa":     lambda t: t == "logiqa",
    "mathqa":     lambda t: t == "mathqa",
    "medmcqa":    lambda t: t == "medmcqa",
    "mmlu":       lambda t: t.startswith("mmlu_"),
    "piqa":       lambda t: t == "piqa",
    "socialqa":   lambda t: t == "social_iqa",
    "truthfulqa": lambda t: t == "truthfulqa_mc1",
}

EMBEDLLM_DISPLAY_NAMES: dict[str, str] = {
    "asdiv": "ASDiv", "gpqa": "GPQA", "gsm8k": "GSM8K", "logiqa": "LogiQA",
    "mathqa": "MathQA", "medmcqa": "MedMCQA", "mmlu": "MMLU", "piqa": "PIQA",
    "socialqa": "SocialQA", "truthfulqa": "TruthfulQA",
}

EMBEDLLM_SPEC = DatasetSpec(
    name="embedllm",
    dataset_path=dataset_path("embedllm"),
    cost_fn=embedllm_cost_proxy.build_cost_vector,
    groups=EMBEDLLM_GROUPS,
    display_names=EMBEDLLM_DISPLAY_NAMES,
    cap_budgets=None,
)

_REGISTRY: dict[str, DatasetSpec] = {
    "router-eval": ROUTER_EVAL_SPEC,
    "embedllm": EMBEDLLM_SPEC,
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset spec {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None

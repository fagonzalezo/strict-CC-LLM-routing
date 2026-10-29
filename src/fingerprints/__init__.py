"""
Dataset-independent LLM behavioral fingerprinting.

Both `router-eval` and `embedllm` represent an LLM the same way: run it on a fixed set of
probe questions, embed each answer with a sentence encoder truncated to a small dimension,
and concatenate the per-probe embeddings in probe order. This package is the one place that
recipe is implemented — see docs/dataset_construction.md.

    probe_set.py    ProbeSet — the frozen, committed question set both datasets are probed with
    answers.py      AnswerStore — persists/reads raw per-model probe answers under data/
    encode.py       build_fingerprints() — probe answers -> (N, n_probe * embed_dim) array
    sources/        one AnswerSource implementation per dataset
"""

from src.fingerprints.probe_set import ProbeSet, load_probe_set, write_probe_set
from src.fingerprints.answers import AnswerSource, AnswerStore
from src.fingerprints.encode import EncodeConfig, ModelResult, build_fingerprints

__all__ = [
    "ProbeSet", "load_probe_set", "write_probe_set",
    "AnswerSource", "AnswerStore",
    "EncodeConfig", "ModelResult", "build_fingerprints",
]

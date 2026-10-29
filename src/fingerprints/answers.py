"""Persisting and reading raw per-model probe answers.

This is the "LLM answers" artifact the fingerprint recipe used to throw away: both legacy
scripts pulled generated text, embedded it, and kept only the embedding. Persisting the answers
themselves is what makes a fingerprint auditable and re-encodable without re-downloading or
re-generating anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.paths import answers_dir

__all__ = ["AnswerRecord", "AnswerSource", "AnswerStore", "AnswerCollectionFailed",
           "classify_exception"]


class AnswerSource(Protocol):
    """Anything that can produce one model's probe answers, in probe order.

    `None` at position `i` means "no answer for probe `i`" — the encoder writes explicit zeros
    for that 32-d slot rather than embedding a placeholder string (see encode.py).
    """

    def answers_for(self, model_key: str) -> list[str | None]: ...


@dataclass
class AnswerRecord:
    model_key: str
    status: str  # "ok" | "missing_probes:N" | "missing" | "error: <msg>"
    answers: list[str | None]  # len == probe_set.n, in probe order; all-None for a failed record


class AnswerCollectionFailed(RuntimeError):
    """Raised by `AnswerStore.answers_for` when the persisted record recorded a collection
    failure. Carries that `status` through unchanged so a later classification step (see
    `encode.py`) reports exactly what `collect_answers.py` found, instead of re-deriving a
    coarser guess from a generic exception message."""

    def __init__(self, model_key: str, status: str):
        super().__init__(f"{model_key}: {status}")
        self.model_key = model_key
        self.status = status


def classify_exception(exc: Exception) -> str:
    """`"missing"` (model/file genuinely absent) vs `"error: <msg>"` (anything else, worth a
    look). Shared by `collect_answers.py` (deciding what status to persist) and `encode.py`
    (classifying any other `AnswerSource` failure the same way)."""
    if isinstance(exc, AnswerCollectionFailed):
        return exc.status
    msg = str(exc).replace("\n", " ")[:120]
    lowered = msg.lower()
    if isinstance(exc, FileNotFoundError) or "not found" in lowered or "404" in msg:
        return "missing"
    return f"error: {msg}"


def _safe_stem(model_key: str) -> str:
    # Both sources key by a numeric id (RouterEval's leaderboard position, EmbedLLM's task_id),
    # never a "org/model" name, so this is a guard against a future source misusing the
    # protocol, not a working escape scheme — a lossy escape would make all_model_keys()
    # unreliable.
    if "/" in model_key:
        raise ValueError(
            f"model_key {model_key!r} contains '/'. AnswerStore keys must be filesystem-safe "
            "(both sources should key by numeric id, not a slashed model name)."
        )
    return model_key


class AnswerStore:
    """One JSON file per model under `data/fingerprints/answers/<dataset>/<model_key>.json`.

    Deliberately one file per model rather than a single consolidated log: an interrupted
    collection run resumes by skipping whatever files already exist, matching the checkpoint
    discipline the two legacy scripts each hand-rolled, and a job array job (one task per
    model, as EmbedLLM generation needs) writes without any inter-task coordination.

    Satisfies `AnswerSource` — this is what `build_fingerprints.py` reads from for both
    datasets; `collect_answers.py` is the only thing that ever writes to it.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.dir: Path = answers_dir(dataset)

    def path_for(self, model_key: str) -> Path:
        return self.dir / f"{_safe_stem(model_key)}.json"

    def has(self, model_key: str) -> bool:
        return self.path_for(model_key).is_file()

    def write(self, record: AnswerRecord) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(record.model_key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "model_key": record.model_key,
            "status": record.status,
            "answers": record.answers,
        }))
        tmp.replace(path)

    def read(self, model_key: str) -> AnswerRecord:
        obj = json.loads(self.path_for(model_key).read_text())
        return AnswerRecord(model_key=obj["model_key"], status=obj["status"],
                             answers=obj["answers"])

    def answers_for(self, model_key: str) -> list[str | None]:
        """Satisfies `AnswerSource`. Raises `FileNotFoundError` if `model_key` was never
        collected, or `AnswerCollectionFailed` if it was attempted and failed."""
        record = self.read(model_key)
        if record.status != "ok" and not record.status.startswith("missing_probes"):
            raise AnswerCollectionFailed(model_key, record.status)
        return record.answers

    def all_model_keys(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

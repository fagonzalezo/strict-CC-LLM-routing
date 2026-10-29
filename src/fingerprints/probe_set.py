"""The probe set both datasets are fingerprinted against.

A `ProbeSet` is nothing more than an ordered list of questions: `probe_index` fixes the
concatenation order that makes an `(n_probe * embed_dim,)` fingerprint well defined, and
`prompt` is the exact text a model must see (RouterEval's models already saw it — lighteval's
5-shot `full_prompt`; EmbedLLM's models must be generated against the same string). See
`docs/dataset_construction.md` §2.2 for why this is now frozen to a file instead of re-sampled
by `numpy.random.default_rng(42).choice(...)` at run time.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.paths import probe_set_path

__all__ = ["ProbeSet", "load_probe_set", "write_probe_set"]

_FIELDS = ["probe_index", "gsm8k_index", "question", "prompt"]


@dataclass(frozen=True)
class ProbeSet:
    """`n` probes, ordered. `meta` carries provenance (sample_seed, task, source, sha256, ...)."""
    name: str
    gsm8k_index: list[int]
    question: list[str]
    prompt: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.gsm8k_index)
        if not (len(self.question) == len(self.prompt) == n):
            raise ValueError(
                f"ProbeSet {self.name!r}: column length mismatch "
                f"(gsm8k_index={n}, question={len(self.question)}, prompt={len(self.prompt)})"
            )

    @property
    def n(self) -> int:
        return len(self.gsm8k_index)


def _csv_bytes(gsm8k_index: list[int], question: list[str], prompt: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_FIELDS)
    for i, (gi, q, p) in enumerate(zip(gsm8k_index, question, prompt)):
        w.writerow([i, gi, q, p])
    return buf.getvalue().encode("utf-8")


def write_probe_set(
    name: str,
    gsm8k_index: list[int],
    question: list[str],
    prompt: list[str],
    meta: dict,
) -> ProbeSet:
    """Write `data/fingerprints/probes/<name>.csv` + `<name>.meta.json` (both tracked in git).

    `probe_index` (the row order) is assigned here — callers pass columns in the order they
    want frozen. The written meta.json's `sha256` is the loader's integrity check.
    """
    csv_path = probe_set_path(name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _csv_bytes(gsm8k_index, question, prompt)
    digest = hashlib.sha256(payload).hexdigest()

    csv_path.write_bytes(payload)

    meta_out = dict(meta)
    meta_out["name"] = name
    meta_out["n"] = len(gsm8k_index)
    meta_out["sha256"] = digest
    meta_path = csv_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True) + "\n")

    return ProbeSet(name=name, gsm8k_index=list(gsm8k_index), question=list(question),
                     prompt=list(prompt), meta=meta_out)


def load_probe_set(name: str) -> ProbeSet:
    """Load and integrity-check the committed probe set `name`.

    Raises if the CSV's sha256 no longer matches `<name>.meta.json` — the probe set is the one
    thing every fingerprint (past or future) must trace back to unambiguously, so a silent edit
    to the CSV must not pass unnoticed.
    """
    csv_path = probe_set_path(name)
    meta_path = csv_path.with_suffix(".meta.json")
    if not csv_path.is_file():
        raise FileNotFoundError(f"No probe set named {name!r} at {csv_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Probe set {name!r} has no meta file at {meta_path}")

    raw = csv_path.read_bytes()
    meta = json.loads(meta_path.read_text())
    expected = meta.get("sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if expected != actual:
        raise ValueError(
            f"Probe set {name!r} failed its sha256 check: "
            f"{csv_path} has been modified since {meta_path} was written "
            f"(meta says {expected}, csv hashes to {actual})."
        )

    # NOT `raw.decode(...).splitlines()`: prompts embed real newlines inside quoted CSV
    # fields, and splitlines() breaks those apart before csv.DictReader ever sees them,
    # silently corrupting multi-line prompts. io.StringIO preserves line structure the same
    # way iterating an open(..., newline="") file would.
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    rows.sort(key=lambda r: int(r["probe_index"]))
    gsm8k_index = [int(r["gsm8k_index"]) for r in rows]
    question = [r["question"] for r in rows]
    prompt = [r["prompt"] for r in rows]

    if meta.get("n") != len(rows):
        raise ValueError(
            f"Probe set {name!r}: meta says n={meta.get('n')} but csv has {len(rows)} rows."
        )

    return ProbeSet(name=name, gsm8k_index=gsm8k_index, question=question, prompt=prompt,
                     meta=meta)

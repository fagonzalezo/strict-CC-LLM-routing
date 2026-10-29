"""The shared fingerprint-encoding loop.

`build_fingerprints` is everything `encode_model_outputs.py` and `encode_outputs.py` used to
duplicate: pull each model's probe answers from an `AnswerSource`, encode the present ones with
a sentence encoder truncated to `embed_dim`, concatenate in probe order into one
`(n_probe * embed_dim,)` row per model, checkpoint periodically, and support `--resume`. The
only thing that differs between datasets is the `AnswerSource` (see `sources/`).

A missing probe answer leaves that probe's `embed_dim`-wide slot as **explicit zeros** — this is
what `docs/dataset_construction.md` promised and what the pre-refactor `encode_outputs.py` did
not actually do (it embedded `""`, which SentenceTransformer turns into a non-zero, shared
vector across every model with that gap).
"""

from __future__ import annotations

import datetime
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from src.fingerprints.answers import AnswerSource, classify_exception
from src.fingerprints.probe_set import ProbeSet

__all__ = ["EncodeConfig", "ModelResult", "Encoder", "build_fingerprints"]

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBED_DIM = 32
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHECKPOINT_EVERY = 100


class Encoder(Protocol):
    """The slice of `sentence_transformers.SentenceTransformer` this module needs — narrow
    enough that tests can substitute a deterministic fake with no model download."""

    def encode(self, texts: list[str], batch_size: int, convert_to_numpy: bool,
               normalize_embeddings: bool, show_progress_bar: bool) -> np.ndarray: ...


@dataclass(frozen=True)
class EncodeConfig:
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    device: str | None = None  # None => "cuda" if available else "cpu"


@dataclass
class ModelResult:
    probe_set_name: str
    config: EncodeConfig
    model_keys: list[str]
    embeddings: np.ndarray            # (N, n_probe * embed_dim) float32
    stats: dict[str, str] = field(default_factory=dict)  # model_key -> "ok"/"missing"/"missing_probes:N"/"error: ..."


def _log_default(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_default_encoder(config: EncodeConfig) -> Encoder:
    import torch
    from sentence_transformers import SentenceTransformer
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(
        config.embed_model,
        model_kwargs={"torch_dtype": torch.float16},
        truncate_dim=config.embed_dim,
        device=device,
    )


def _atomic_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def build_fingerprints(
    model_keys: list[str],
    probe_set: ProbeSet,
    source: AnswerSource,
    config: EncodeConfig = EncodeConfig(),
    *,
    out_path: Path | None = None,
    resume: bool = False,
    encoder: Encoder | None = None,
    log: Callable[[str], None] = _log_default,
) -> ModelResult:
    """Encode `model_keys` against `probe_set` via `source`, in probe-index order.

    If `out_path` is given, checkpoints every `config.checkpoint_every` models (atomic pickle)
    and can `resume` from it — a resumed run must be called with the same `model_keys`,
    `probe_set`, and `config`, or it raises rather than silently mixing runs.
    """
    n_probe = probe_set.n
    emb_dim = n_probe * config.embed_dim
    N = len(model_keys)

    embeddings = np.zeros((N, emb_dim), dtype=np.float32)
    stats: dict[str, str] = {}

    if resume and out_path is not None and out_path.exists():
        log(f"--resume: loading checkpoint {out_path} ...")
        with open(out_path, "rb") as f:
            prev = pickle.load(f)
        mismatch = []
        if prev.get("probe_set_name") != probe_set.name:
            mismatch.append(f"probe_set {prev.get('probe_set_name')!r} -> {probe_set.name!r}")
        if prev.get("config") != config:
            mismatch.append(f"config {prev.get('config')} -> {config}")
        if prev.get("model_keys") != model_keys:
            mismatch.append("model_keys differ")
        if mismatch:
            raise ValueError(
                "Resume mismatch: " + "; ".join(mismatch) +
                f". Delete {out_path} or omit resume to start fresh."
            )
        embeddings = prev["embeddings"].copy()
        stats = dict(prev["stats"])
        log(f"  resuming: {sum(1 for v in stats.values() if v.startswith('ok') or v.startswith('missing_probes'))}/{N} already done")

    encoder = encoder or _build_default_encoder(config)

    n_ok = sum(1 for v in stats.values() if v == "ok" or v.startswith("missing_probes"))
    n_missing = sum(1 for v in stats.values() if v == "missing")
    n_error = sum(1 for v in stats.values() if v.startswith("error"))

    def _checkpoint() -> None:
        if out_path is None:
            return
        _atomic_pickle({
            "probe_set_name": probe_set.name,
            "config": config,
            "model_keys": model_keys,
            "embeddings": embeddings,
            "stats": stats,
        }, out_path)

    t_start = time.time()
    for i, model_key in enumerate(model_keys):
        if stats.get(model_key, "").startswith(("ok", "missing_probes")):
            continue

        try:
            answers = source.answers_for(model_key)
            if len(answers) != n_probe:
                raise ValueError(
                    f"{model_key}: expected {n_probe} answers, got {len(answers)}"
                )
            present = [j for j, a in enumerate(answers) if a]
            n_missing_probes = n_probe - len(present)
            if n_missing_probes == n_probe:
                raise ValueError(f"{model_key}: all {n_probe} probe answers missing")

            texts = [answers[j] for j in present]
            enc = encoder.encode(texts, batch_size=config.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
            row = np.zeros(emb_dim, dtype=np.float32)
            for k, j in enumerate(present):
                row[j * config.embed_dim:(j + 1) * config.embed_dim] = enc[k]
            embeddings[i] = row

            status = "ok" if n_missing_probes == 0 else f"missing_probes:{n_missing_probes}"
            stats[model_key] = status
            n_ok += 1

        except Exception as e:  # noqa: BLE001 - classify and continue, mirrors legacy scripts
            status = classify_exception(e)
            stats[model_key] = status
            if status == "missing":
                n_missing += 1
            else:
                n_error += 1
            log(f"  [{i + 1:5d}/{N}] {model_key}  {status}")

        if (i + 1) % config.checkpoint_every == 0 or (i + 1) == N:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed / 60, 1e-9)
            log(f"  -- [{i + 1}/{N}] ok={n_ok} missing={n_missing} error={n_error} "
                f"rate={rate:.1f} models/min --")
            _checkpoint()

    _checkpoint()
    return ModelResult(probe_set_name=probe_set.name, config=config, model_keys=list(model_keys),
                        embeddings=embeddings, stats=stats)

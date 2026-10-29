"""
Tests for the shared fingerprint builder (src/fingerprints/).

None of these download a model or hit the network: `build_fingerprints` is tested against a
deterministic fake encoder and an in-memory `AnswerSource`, which is exactly what the encode
loop is decoupled to allow. The network/model-dependent path (RouterEvalLightevalSource against
the local HF cache) is a separate, opt-in slow test — see test_fingerprints_slow.py.

Run: `python -m pytest src/tests -q`  (from anywhere).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.fingerprints.encode import EncodeConfig, build_fingerprints  # noqa: E402
from src.fingerprints.probe_set import ProbeSet, load_probe_set, write_probe_set  # noqa: E402


class _FakeEncoder:
    """Deterministic stand-in for SentenceTransformer: a text's embedding is a hash of its own
    content, so two calls on the same text always agree without downloading anything."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings,
               show_progress_bar):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32)
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            out[i] = v
        return out


class _FakeSource:
    def __init__(self, mapping: dict[str, list[str | None]]):
        self.mapping = mapping

    def answers_for(self, model_key: str) -> list[str | None]:
        if model_key not in self.mapping:
            raise FileNotFoundError(model_key)
        return self.mapping[model_key]


class _CountingSource:
    def __init__(self, inner):
        self.inner = inner
        self.calls: list[str] = []

    def answers_for(self, model_key: str) -> list[str | None]:
        self.calls.append(model_key)
        return self.inner.answers_for(model_key)


def _write_checkpoint(path: Path, probe_set: ProbeSet, config: EncodeConfig,
                       model_keys: list[str], embeddings: np.ndarray, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"probe_set_name": probe_set.name, "config": config,
                     "model_keys": model_keys, "embeddings": embeddings, "stats": stats}, f)


# ── probe_set.py ────────────────────────────────────────────────────────────

def test_committed_probe_set_loads_and_verifies():
    """The real, tracked probe set every fingerprint traces back to."""
    ps = load_probe_set("gsm8k_seed42_n64")
    assert ps.n == 64
    assert ps.meta["task"] == "harness|gsm8k|5"
    assert ps.meta["sample_seed"] == 42
    assert len(ps.gsm8k_index) == len(ps.prompt) == len(ps.question) == 64
    assert len(set(ps.gsm8k_index)) == 64  # no duplicate probes


def test_committed_probe_set_multiline_prompts_are_not_corrupted():
    """Regression: prompts embed real newlines inside a quoted CSV field. An earlier version of
    `load_probe_set` split the raw bytes into lines with `str.splitlines()` before handing them
    to `csv.DictReader`, which breaks a multi-line quoted field apart before the CSV parser ever
    sees it and silently truncates/corrupts the prompt. Compare against a plain
    `open(..., newline="")` + `csv.DictReader` read, which is unambiguously correct."""
    import csv as _csv
    from src.paths import probe_set_path

    ps = load_probe_set("gsm8k_seed42_n64")
    with open(probe_set_path("gsm8k_seed42_n64"), newline="", encoding="utf-8") as f:
        rows = sorted(_csv.DictReader(f), key=lambda r: int(r["probe_index"]))

    assert ps.prompt == [r["prompt"] for r in rows]
    assert ps.gsm8k_index == [int(r["gsm8k_index"]) for r in rows]
    assert any("\n" in p for p in ps.prompt), "fixture should exercise the multi-line case"


def test_probe_set_roundtrip_and_tamper_detection(tmp_path, monkeypatch):
    import src.paths as paths
    monkeypatch.setattr(paths, "FINGERPRINTS", tmp_path)

    multiline = "line one\nline two\n\nline four"
    write_probe_set("t", gsm8k_index=[1, 2], question=["q1", "q2"],
                     prompt=[multiline, "p2"], meta={"task": "x"})
    loaded = load_probe_set("t")
    assert loaded.gsm8k_index == [1, 2]
    assert loaded.prompt == [multiline, "p2"]

    csv_path = paths.probe_set_path("t")
    csv_path.write_text(csv_path.read_text() + "\n99,99,tampered,tampered\n")
    with pytest.raises(ValueError):
        load_probe_set("t")


# ── encode.py ─────────────────────────────────────────────────────────────────

def test_probe_order_and_zero_padding_for_missing_answers():
    probe_set = ProbeSet(name="t", gsm8k_index=[10, 20, 30],
                          question=["q0", "q1", "q2"], prompt=["p0", "p1", "p2"])
    config = EncodeConfig(embed_dim=4)
    encoder = _FakeEncoder(4)
    source = _FakeSource({"m0": ["hello", None, "world"]})

    result = build_fingerprints(["m0"], probe_set, source, config, encoder=encoder)

    row = result.embeddings[0]
    assert row.shape == (12,)
    expected0 = encoder.encode(["hello"], 32, True, True, False)[0]
    expected2 = encoder.encode(["world"], 32, True, True, False)[0]
    assert np.allclose(row[0:4], expected0)
    assert np.array_equal(row[4:8], np.zeros(4, dtype=np.float32))  # missing probe -> zeros
    assert np.allclose(row[8:12], expected2)
    assert result.stats["m0"] == "missing_probes:1"


def test_all_answers_missing_is_classified_as_error_not_a_zero_ok_row():
    probe_set = ProbeSet(name="t", gsm8k_index=[0, 1], question=["q0", "q1"],
                          prompt=["p0", "p1"])
    source = _FakeSource({"m0": [None, None]})
    result = build_fingerprints(["m0"], probe_set, source, EncodeConfig(embed_dim=4),
                                 encoder=_FakeEncoder(4))
    assert result.stats["m0"].startswith("error")
    assert np.array_equal(result.embeddings[0], np.zeros(8, dtype=np.float32))


def test_unknown_model_key_is_classified_missing():
    probe_set = ProbeSet(name="t", gsm8k_index=[0], question=["q0"], prompt=["p0"])
    source = _FakeSource({})  # "m0" not present -> FileNotFoundError
    result = build_fingerprints(["m0"], probe_set, source, EncodeConfig(embed_dim=4),
                                 encoder=_FakeEncoder(4))
    assert result.stats["m0"] == "missing"


def test_resume_skips_completed_models_and_keeps_their_rows(tmp_path):
    probe_set = ProbeSet(name="t", gsm8k_index=[0, 1], question=["q0", "q1"],
                          prompt=["p0", "p1"])
    mapping = {"m0": ["a0", "a1"], "m1": ["b0", "b1"], "m2": ["c0", "c1"]}
    config = EncodeConfig(embed_dim=4, checkpoint_every=1)
    emb_dim = probe_set.n * config.embed_dim
    encoder = _FakeEncoder(4)

    out = tmp_path / "ckpt.pkl"
    partial_embeddings = np.zeros((3, emb_dim), dtype=np.float32)
    partial_embeddings[0] = encoder.encode(mapping["m0"], 32, True, True, False).flatten()
    _write_checkpoint(out, probe_set, config, ["m0", "m1", "m2"], partial_embeddings,
                       {"m0": "ok"})

    counting = _CountingSource(_FakeSource(mapping))
    result = build_fingerprints(["m0", "m1", "m2"], probe_set, counting, config,
                                 out_path=out, resume=True, encoder=encoder)

    assert counting.calls == ["m1", "m2"]  # m0 was skipped, not re-fetched
    assert result.stats == {"m0": "ok", "m1": "ok", "m2": "ok"}
    assert np.array_equal(result.embeddings[0], partial_embeddings[0])  # m0's row untouched


def test_resume_config_mismatch_raises(tmp_path):
    probe_set = ProbeSet(name="t", gsm8k_index=[0], question=["q0"], prompt=["p0"])
    config = EncodeConfig(embed_dim=4)
    out = tmp_path / "ckpt.pkl"
    _write_checkpoint(out, probe_set, config, ["m0"], np.zeros((1, 4), dtype=np.float32),
                       {"m0": "ok"})

    other_config = EncodeConfig(embed_dim=8)
    with pytest.raises(ValueError):
        build_fingerprints(["m0"], probe_set, _FakeSource({"m0": ["x"]}), other_config,
                            out_path=out, resume=True, encoder=_FakeEncoder(8))


# ── sources/embedllm.py ────────────────────────────────────────────────────────

def test_embedllm_generator_truncates_at_next_question_boundary():
    """Regression: unconstrained generate() has no stop condition, so a decoded answer would
    otherwise run past its own '#### <n>' line into a fabricated next few-shot turn -- a
    verbosity artifact of decoding config, not of model behaviour, which the shared probe set
    exists to rule out. Checked against real RouterEval predictions, which never do this."""
    from src.fingerprints.sources.embedllm import _truncate_at_stop

    # A clean answer's own trailing blank line is itself a "\n\n" occurrence -- cut there too;
    # the caller's .strip() (see answers_for) normalizes the result either way.
    real_style = "The answer is 42.\n#### 42\n\n"
    assert _truncate_at_stop(real_style).strip() == "The answer is 42.\n#### 42"

    rambling = "The answer is 42.\n#### 42\n\nQuestion: A new question here.\nAnswer: made up"
    assert _truncate_at_stop(rambling) == "The answer is 42.\n#### 42"

    no_blank_line = "The answer is 42.\n#### 42\nQuestion: fabricated next turn"
    assert _truncate_at_stop(no_blank_line) == "The answer is 42.\n#### 42"


def _real_model_list() -> list[dict]:
    import csv

    from src.paths import dataset_raw

    with open(dataset_raw("embedllm") / "model_list.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_embedllm_local_maps_task_id_to_model_id_filename():
    """task_id and model_id diverge for 78/107 rows, and the committed model_outputs/ files are
    keyed by model_id, not task_id. task_id 29 must resolve to model_id 30, Qwen1.5-32B-Chat,
    and come back "ok"."""
    from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs

    models = _real_model_list()
    row = next(m for m in models if m["task_id"] == "29")
    assert row["model_id"] == "30"
    assert row["model_name"] == "Qwen__Qwen1.5-32B-Chat"

    probe_set = load_probe_set("gsm8k_seed42_n64")
    model_ids = {m["task_id"]: m["model_id"] for m in models}
    source = EmbedLLMLocalOutputs(probe_set, model_ids)
    answers = source.answers_for("29")
    assert len(answers) == probe_set.n
    assert sum(1 for a in answers if a) == probe_set.n  # this model has no gaps


def test_embedllm_local_answers_align_with_probe_set():
    """Compressed alignment check against the real committed data: probe 0 asks about "Ray"
    swimming and probe 63 about an avocado tree, so a real on-probe answer should mention that
    keyword far more often than chance. Guards against a future off-by-one or re-keyed file
    set silently reading the wrong probe's answer."""
    from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs

    models = _real_model_list()
    probe_set = load_probe_set("gsm8k_seed42_n64")
    model_ids = {m["task_id"]: m["model_id"] for m in models}
    source = EmbedLLMLocalOutputs(probe_set, model_ids)

    task_ids = [m["task_id"] for m in models[:20]]
    hits_probe0 = hits_probe63 = 0
    for task_id in task_ids:
        answers = source.answers_for(task_id)
        hits_probe0 += "ray" in (answers[0] or "").lower()
        hits_probe63 += "avocado" in (answers[63] or "").lower()
    assert hits_probe0 >= 0.8 * len(task_ids)
    assert hits_probe63 >= 0.6 * len(task_ids)


def test_embedllm_local_empty_output_becomes_none():
    """task_id 100 (model_id 104, Qwen1.5-0.5B-Chat) has 2 of 64 probe answers that are empty or
    become empty after truncation -- these must surface as None so encode.py zero-pads them,
    matching what a real collect_answers.py run against this file produces
    (status "missing_probes:2")."""
    from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs

    models = _real_model_list()
    probe_set = load_probe_set("gsm8k_seed42_n64")
    model_ids = {m["task_id"]: m["model_id"] for m in models}
    source = EmbedLLMLocalOutputs(probe_set, model_ids)

    answers = source.answers_for("100")
    n_missing = sum(1 for a in answers if a is None)
    assert n_missing == 2


def test_embedllm_local_truncates_rambling_output():
    """A shipped answer known to ramble past its own '#### <n>' line into a fabricated next
    question comes back cut at the boundary, exactly as EmbedLLMGenerator's live-generation
    path would truncate it -- both call the same shared _normalize_answer."""
    from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs, _normalize_answer

    models = _real_model_list()
    probe_set = load_probe_set("gsm8k_seed42_n64")
    model_ids = {m["task_id"]: m["model_id"] for m in models}
    source = EmbedLLMLocalOutputs(probe_set, model_ids)

    # 44% of the committed answers contain "\nQuestion:"; find one from a small sample rather
    # than hardcoding a task_id whose raw text could shift if the shipped files are ever
    # re-fetched.
    import json

    from src.paths import dataset_raw

    outputs_dir = dataset_raw("embedllm") / "model_outputs"
    for m in models[:10]:
        raw = json.loads((outputs_dir / f"{m['model_id']}.json").read_text())
        rambling = next((r for r in raw if "\nQuestion:" in r["output"]), None)
        if rambling is None:
            continue
        answers = source.answers_for(m["task_id"])
        expected = _normalize_answer(rambling["output"])
        assert answers[rambling["prompt_id"]] == expected
        assert answers[rambling["prompt_id"]] is None or "\nQuestion:" not in answers[rambling["prompt_id"]]
        return
    pytest.skip("no rambling answer found in the first 10 sampled models")


def test_embedllm_normalize_answer_survives_a_leading_blank_line():
    """Regression: truncating before stripping would find the leading "\\n\\n" at index 0-1 and
    discard the entire answer, not just a trailing artifact. All 64 of microsoft/Orca-2-13b's
    committed answers (task_id 75) open this way -- naive truncate-then-strip wrongly empties
    every one of them; strip-first (this function) empties none."""
    from src.fingerprints.sources.embedllm import _normalize_answer

    assert (_normalize_answer(" \n\nWorking backward, we know the answer.\n#### 42")
            == "Working backward, we know the answer.\n#### 42")
    assert _normalize_answer("   \n\n  ") is None


def test_embedllm_roster_is_all_107_models():
    """model_list.csv is already the 107 valid models -- collect_answers.py's roster must be
    exactly that, unfiltered."""
    models = _real_model_list()
    assert len(models) == 107
    assert {m["task_id"] for m in models}.issuperset({"29", "38", "70", "76", "105"})


def test_embedllm_local_rejects_bad_prompt_id_set(tmp_path):
    """A model_outputs file whose prompt_ids don't cover exactly 0..n-1 is a corrupted or
    re-keyed file, not silently-misaligned data -- answers_for must raise rather than guess."""
    from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs

    probe_set = ProbeSet(name="tiny", gsm8k_index=[0, 1], question=["q0", "q1"],
                          prompt=["p0", "p1"], meta={})
    outputs_dir = tmp_path / "model_outputs"
    outputs_dir.mkdir()
    (outputs_dir / "0.json").write_text('[{"prompt_id": 0, "output": "a"}, '
                                         '{"prompt_id": 2, "output": "b"}]')

    source = EmbedLLMLocalOutputs(probe_set, {"tid0": "0"}, outputs_dir=outputs_dir)
    with pytest.raises(ValueError, match="prompt_id"):
        source.answers_for("tid0")

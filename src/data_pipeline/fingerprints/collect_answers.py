#!/usr/bin/env python3
"""Persist raw per-model probe answers under `data/fingerprints/answers/<dataset>/`.

This is stage 1 of the shared fingerprint pipeline (stage 2 is `build_fingerprints.py`).
Splitting collection from encoding means a network hiccup or a bad GPU node only costs the one
model it hit, and the encoding step never needs to touch a network or a generation model again.

router-eval: downloads each model's lighteval GSM8K zip from HuggingFace and slices it at the
frozen probe set's positions. CPU + network; ~3,749 downloads for the full run.

    python src/data_pipeline/fingerprints/collect_answers.py --dataset router-eval \\
        --model-partition data/router-eval/processed/model_partition.pkl \\
        --leaderboard data/router-eval/raw/leaderboard_score/leaderboard_old.pkl \\
        [--n-models N] [--force]

embedllm: by default (`--source local`), reads each model's answers straight from the shipped
`data/embedllm/raw/model_outputs/<model_id>.json` (107 files, committed) — no GPU, no network,
seconds not days. `--source generate` regenerates them locally with the named HuggingFace causal
LM instead (GPU; one model per invocation is the intended entry point for a job array via
`--model-idx`), kept for provenance and for the day the probe set changes.

    python src/data_pipeline/fingerprints/collect_answers.py --dataset embedllm \\
        --model-list data/embedllm/raw/model_list.csv [--force]

    python src/data_pipeline/fingerprints/collect_answers.py --dataset embedllm \\
        --source generate --model-list data/embedllm/raw/model_list.csv \\
        --model-idx 0 [--force]

Both variants are resumable: an existing `answers/<dataset>/<model_key>.json` is skipped unless
`--force`.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.fingerprints.answers import AnswerRecord, AnswerStore, classify_exception  # noqa: E402
from src.fingerprints.probe_set import load_probe_set  # noqa: E402
from src.paths import relative  # noqa: E402

DEFAULT_PROBE_SET = "gsm8k_seed42_n64"


def _collect_one(store: AnswerStore, key: str, label: str, n_probe: int,
                  answers_for) -> str:
    """Try `answers_for()`, persist whatever happened, return the status string."""
    try:
        answers = answers_for()
        n_missing_probes = sum(1 for a in answers if not a)
        status = "ok" if n_missing_probes == 0 else f"missing_probes:{n_missing_probes}"
        store.write(AnswerRecord(model_key=key, status=status, answers=answers))
    except Exception as e:  # noqa: BLE001 - persist the failure, keep going
        status = classify_exception(e)
        store.write(AnswerRecord(model_key=key, status=status, answers=[None] * n_probe))
    return status


def _run_router_eval(args) -> None:
    from src.fingerprints.sources.router_eval import RouterEvalLightevalSource

    probe_set = load_probe_set(args.probe_set)
    print(f"probe set: {args.probe_set} ({probe_set.n} probes)")

    with open(args.model_partition, "rb") as f:
        partition = pickle.load(f)
    print(f"read: {relative(args.model_partition)}")
    with open(args.leaderboard, "rb") as f:
        leaderboard = pickle.load(f)
    print(f"read: {relative(args.leaderboard)}")

    subset_indices = [int(m) for m in partition["subset"]["indices"]]
    if args.n_models is not None:
        subset_indices = subset_indices[:args.n_models]
    model_names = {str(m): leaderboard["model"][m] for m in subset_indices}
    model_keys = [str(m) for m in subset_indices]

    source = RouterEvalLightevalSource(probe_set, model_names)
    store = AnswerStore("router-eval")

    n_ok = n_missing = n_error = n_skip = 0
    t0 = time.time()
    N = len(model_keys)
    for i, key in enumerate(model_keys):
        if not args.force and store.has(key):
            n_skip += 1
            continue
        status = _collect_one(store, key, model_names[key], probe_set.n,
                               lambda k=key: source.answers_for(k))
        if status == "ok" or status.startswith("missing_probes"):
            n_ok += 1
        elif status == "missing":
            n_missing += 1
        else:
            n_error += 1

        if (i + 1) % 100 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{N}] ok={n_ok} missing={n_missing} error={n_error} "
                  f"skip={n_skip}  ({elapsed:.0f}s)")

    print(f"done: dataset=router-eval ok={n_ok} missing={n_missing} error={n_error} "
          f"skip={n_skip} total={N}")


def _run_embedllm(args) -> None:
    probe_set = load_probe_set(args.probe_set)
    print(f"probe set: {args.probe_set} ({probe_set.n} probes)")

    with open(args.model_list, newline="", encoding="utf-8") as f:
        models = list(csv.DictReader(f))
    print(f"read: {relative(args.model_list)}  ({len(models)} models)")

    # model_list.csv is already the 107 valid models -- no separate exclusion list to apply.
    # Its task_id is a dense 0..106 re-index, distinct from the original 0..111 id space (see
    # docs/dataset_construction.md).
    roster = models
    if args.model_idx is not None:
        roster = [roster[args.model_idx]]

    if args.source == "local":
        from src.fingerprints.sources.embedllm import EmbedLLMLocalOutputs
        model_ids = {m["task_id"]: m["model_id"] for m in models}
        source = EmbedLLMLocalOutputs(probe_set, model_ids)
    else:
        from src.fingerprints.sources.embedllm import EmbedLLMGenerator
        model_names = {m["task_id"]: m["hf_name"] for m in models}
        quantize = {m["task_id"]: m["quantize"] for m in models if m.get("quantize")}
        source = EmbedLLMGenerator(probe_set, model_names, quantize=quantize)

    store = AnswerStore("embedllm")

    for m in roster:
        key = m["task_id"]
        if not args.force and store.has(key):
            print(f"  skip (exists): task_id={key}  {m['model_name']}")
            continue
        t0 = time.time()
        status = _collect_one(store, key, m["model_name"], probe_set.n,
                               lambda k=key: source.answers_for(k))
        print(f"  [task_id={key}] {m['model_name']}  {status}  ({time.time() - t0:.0f}s)")

    print(f"done: dataset=embedllm  {len(roster)} model(s) attempted "
          f"(roster is all {len(models)} model_list.csv rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["router-eval", "embedllm"])
    ap.add_argument("--probe-set", default=DEFAULT_PROBE_SET)
    ap.add_argument("--force", action="store_true", help="Re-collect even if already on disk")

    ap.add_argument("--model-partition", type=Path, help="[router-eval] model_partition.pkl")
    ap.add_argument("--leaderboard", type=Path,
                     help="[router-eval] leaderboard_score/leaderboard_old.pkl")
    ap.add_argument("--n-models", type=int, default=None,
                     help="[router-eval] limit to first N models (smoke testing)")

    ap.add_argument("--model-list", type=Path, help="[embedllm] raw/model_list.csv")
    ap.add_argument("--source", choices=["local", "generate"], default="local",
                     help="[embedllm] local: read shipped raw/model_outputs/ (default, no GPU). "
                          "generate: regenerate on a GPU with the named HF causal LM.")
    ap.add_argument("--model-idx", type=int, default=None,
                     help="[embedllm] collect a single roster position (job array task)")

    args = ap.parse_args()

    if args.dataset == "router-eval":
        if not (args.model_partition and args.leaderboard):
            ap.error("--dataset router-eval requires --model-partition and --leaderboard")
        _run_router_eval(args)
    else:
        if not args.model_list:
            ap.error("--dataset embedllm requires --model-list")
        _run_embedllm(args)


if __name__ == "__main__":
    main()

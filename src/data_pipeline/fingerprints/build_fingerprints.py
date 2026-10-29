#!/usr/bin/env python3
"""Encode collected probe answers into a fingerprint array.

Stage 2 of the shared fingerprint pipeline (stage 1 is `collect_answers.py`). Reads back from
`data/fingerprints/answers/<dataset>/` — never a network or a generation model — and writes:

  1. The canonical artifact, `data/fingerprints/<dataset>/fingerprints.pkl` (gitignored).
  2. For `router-eval` only, a **legacy-schema** remap at
     `data/router-eval/processed/gsm8k_model_embeddings.pkl` — the path and keys
     `build_llm_perf_dataset.py` already reads, so that script needs no changes. EmbedLLM has no
     equivalent standalone legacy artifact: its Phase A output was always consumed in-process by
     `encode_outputs.py`, which now loads straight from the canonical artifact instead.

Usage:
    python src/data_pipeline/fingerprints/build_fingerprints.py --dataset router-eval \\
        --model-partition data/router-eval/processed/model_partition.pkl \\
        --leaderboard data/router-eval/raw/leaderboard_score/leaderboard_old.pkl \\
        [--resume] [--device cpu]

    python src/data_pipeline/fingerprints/build_fingerprints.py --dataset embedllm \\
        --model-list data/embedllm/raw/model_list.csv \\
        [--resume] [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.fingerprints.answers import AnswerStore  # noqa: E402
from src.fingerprints.encode import EncodeConfig, build_fingerprints  # noqa: E402
from src.fingerprints.probe_set import load_probe_set  # noqa: E402
from src.paths import ROOT, dataset_processed, fingerprints_path, relative  # noqa: E402

DEFAULT_PROBE_SET = "gsm8k_seed42_n64"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        text=True).strip()
    except Exception:
        return "unknown"


def _atomic_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def _write_canonical(dataset: str, result, probe_set_name: str) -> Path:
    out = fingerprints_path(dataset)
    _atomic_pickle({
        "dataset": dataset,
        "probe_set_name": probe_set_name,
        "config": result.config,
        "model_keys": result.model_keys,
        "embeddings": result.embeddings,
        "stats": result.stats,
        "git_commit": _git_commit(),
        "created_at": datetime.datetime.now().isoformat(),
    }, out)
    return out


def _run_router_eval(args) -> None:
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
    model_keys = [str(m) for m in subset_indices]
    model_names = [leaderboard["model"][m] for m in subset_indices]

    config = EncodeConfig(embed_model=args.embed_model, embed_dim=args.embed_dim,
                           batch_size=args.batch_size, device=args.device)
    store = AnswerStore("router-eval")
    canonical_out = fingerprints_path("router-eval")

    result = build_fingerprints(model_keys, probe_set, store, config,
                                 out_path=canonical_out, resume=args.resume)
    _write_canonical("router-eval", result, args.probe_set)
    print(f"wrote: {relative(canonical_out)}")

    n_ok = sum(1 for v in result.stats.values() if v == "ok")
    print(f"stats: ok={n_ok}/{len(model_keys)}  "
          f"missing={sum(1 for v in result.stats.values() if v == 'missing')}  "
          f"error={sum(1 for v in result.stats.values() if v.startswith('error'))}")

    # Legacy remap — exact path and keys build_llm_perf_dataset.py already reads.
    legacy_out = dataset_processed("router-eval") / "gsm8k_model_embeddings.pkl"
    legacy = {
        "config": {
            "embed_model": config.embed_model,
            "embed_dim": config.embed_dim,
            "n_sample_questions": probe_set.n,
            "sample_seed": probe_set.meta.get("sample_seed"),
            "task": probe_set.meta.get("task"),
            "probe_set": args.probe_set,
            "source_partition": relative(args.model_partition),
            "git_commit": _git_commit(),
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "sampled_questions": {
            "indices_in_full_gsm8k": np.array(probe_set.gsm8k_index),
            "prompts": np.array(probe_set.prompt, dtype=object),
        },
        "model_indices": np.array(subset_indices),
        "model_names": np.array(model_names, dtype=object),
        "embeddings": result.embeddings,
        "stats": result.stats,
        "splits": {k: partition["splits"][k] for k in ("train", "val", "test")},
    }
    _atomic_pickle(legacy, legacy_out)
    print(f"wrote: {relative(legacy_out)}  (legacy schema, for build_llm_perf_dataset.py)")

    # This is a tracked, doc-referenced artifact (data/router-eval/processed/*.md cites it) —
    # keep it live, not just the pickle it used to accompany.
    stats_out = dataset_processed("router-eval") / "gsm8k_encode_stats.json"
    stats_out.write_text(json.dumps(result.stats, indent=2) + "\n")
    print(f"wrote: {relative(stats_out)}")

    if not any(v == "ok" for v in result.stats.values()):
        raise SystemExit(
            "Every model came back non-'ok'. Did collect_answers.py --dataset router-eval run "
            "first? (fingerprints.pkl was still written, but it would be all zeros.)"
        )


def _run_embedllm(args) -> None:
    probe_set = load_probe_set(args.probe_set)
    print(f"probe set: {args.probe_set} ({probe_set.n} probes)")

    with open(args.model_list, newline="", encoding="utf-8") as f:
        models = list(csv.DictReader(f))
    print(f"read: {relative(args.model_list)}  ({len(models)} models)")

    # All 107 listed models, in model_list.csv row order -- matches what encode_outputs.py's
    # Phase D expects as `model_ids`. A model can still come back "missing"/"error" if its own
    # answer collection failed.
    # --n-models truncates this the same way on both sides -- pass the same N to
    # encode_outputs.py --n-models for a matching smoke test.
    if args.n_models is not None:
        models = models[:args.n_models]
    model_keys = [m["task_id"] for m in models]

    config = EncodeConfig(embed_model=args.embed_model, embed_dim=args.embed_dim,
                           batch_size=args.batch_size, device=args.device)
    store = AnswerStore("embedllm")
    canonical_out = fingerprints_path("embedllm")

    result = build_fingerprints(model_keys, probe_set, store, config,
                                 out_path=canonical_out, resume=args.resume)
    _write_canonical("embedllm", result, args.probe_set)
    print(f"wrote: {relative(canonical_out)}")

    n_ok = sum(1 for v in result.stats.values() if v == "ok")
    n_collected = sum(1 for v in result.stats.values()
                       if v == "ok" or v.startswith("missing_probes"))
    print(f"stats: ok={n_ok}/{len(model_keys)}  "
          f"missing={sum(1 for v in result.stats.values() if v == 'missing')}  "
          f"error={sum(1 for v in result.stats.values() if v.startswith('error'))}")
    print("Next: python src/data_pipeline/embedllm/encode_outputs.py "
          "(Phase A now loads this artifact; Phases B-D unchanged)")

    if n_collected == 0:
        raise SystemExit(
            "No model was successfully collected. Did collect_answers.py --dataset embedllm "
            "run first? (fingerprints.pkl was still written, but it would be all zeros.)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["router-eval", "embedllm"])
    ap.add_argument("--probe-set", default=DEFAULT_PROBE_SET)
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--model-partition", type=Path, help="[router-eval] model_partition.pkl")
    ap.add_argument("--leaderboard", type=Path,
                     help="[router-eval] leaderboard_score/leaderboard_old.pkl")
    ap.add_argument("--n-models", type=int, default=None,
                     help="Limit to first N models (smoke testing); pass the same N to "
                          "encode_outputs.py --n-models for embedllm")

    ap.add_argument("--model-list", type=Path, help="[embedllm] raw/model_list.csv")

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

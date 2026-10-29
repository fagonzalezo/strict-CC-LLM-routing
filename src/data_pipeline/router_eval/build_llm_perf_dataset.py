#!/usr/bin/env python3
"""Build the LLM performance-prediction dataset.

Assembles query embeddings, LLM embeddings, and binarized correctness scores
from leaderboard_old across 6 in-scope tasks (arc, gsm8k, hellaswag,
winogrande, harness_truthfulqa_mc_0, mmlu). Uses pre-computed train/val/test
splits for both queries (from router_dataset pkls) and LLMs (model_partition.pkl).

All paths are derived from this file's own location — the script can be run
from any working directory.

Usage:
    # Default (RoBERTa query embeddings):
    python build_llm_perf_dataset.py [--output PATH] [--threshold FLOAT]

    # Alternate query embeddings (e.g. Qwen3-Embedding-4B, 2560-d):
    python build_llm_perf_dataset.py \\
        --query-embed-file data/router-eval/processed/Qwen3-Embedding-4B_query_result.pkl \\
        --query-embed-name Qwen3-Embedding-4B \\
        --output data/router-eval/processed/llm_perf_dataset_qwen3.pkl \\
        --report data/router-eval/processed/llm_perf_dataset_qwen3_report.md

The --query-embed-file must be a pickle with an "embeddings" key mapping
task names to (n_q, D) float32 arrays, row-aligned with leaderboard_old.
RoBERTa embeddings are always loaded as the index-alignment oracle; only the
emitted query embeddings change.
"""
import argparse
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path constants (all relative to project root, derived from __file__)
# ---------------------------------------------------------------------------
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, dataset_raw  # noqa: E402
PROJECT_ROOT = ROOT

SCORE_FILE = dataset_raw("router-eval") / "leaderboard_score/leaderboard_old.pkl"
EMBED_FILE = dataset_raw("router-eval") / "leaderboard_embed/leaderboard_old_embed/RoBERTa_result.pkl"
ROUTER_DIR = dataset_raw("router-eval") / "router_dataset"
MODEL_PARTITION = dataset_processed("router-eval") / "model_partition.pkl"
MODEL_EMBEDDINGS = dataset_processed("router-eval") / "gsm8k_model_embeddings.pkl"
DEFAULT_OUTPUT = dataset_processed("router-eval") / "llm_perf_dataset.pkl"
DEFAULT_REPORT = dataset_processed("router-eval") / "llm_perf_dataset_report.md"

# Mapping: router_dataset filename stem → list of leaderboard task names
TASK_MAP = {
    "arc": ["harness_arc_challenge_25"],
    "gsm8k": ["harness_gsm8k_5"],
    "hellaswag": ["harness_hellaswag_10"],
    "winogrande": ["harness_winogrande_5"],
    "harness_truthfulqa_mc_0": ["harness_truthfulqa_mc_0"],
    "mmlu": None,  # resolved dynamically (57 hendrycksTest tasks)
}

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# MMLU ordering resolution
# ---------------------------------------------------------------------------

def resolve_mmlu_order(hendrycks_tasks: list[str], roberta_emb: dict, mmlu_rd: dict) -> tuple[list[str], np.ndarray]:
    """Determine the concatenation order of the 57 hendrycksTest tasks.

    Tries alphabetical sort then insertion order; verifies via embedding match
    against the router_dataset's pre-split embeddings.
    Returns (ordered_task_list, cumulative_size_array).
    """
    train_idx = mmlu_rd["split_index"]["train_indices"]
    ref_emb = mmlu_rd["embedding"]["train_embed"]
    n_probe = min(200, len(train_idx))
    sample_idx = train_idx[:n_probe]
    ref_sample = ref_emb[:n_probe]

    candidates = [
        sorted(hendrycks_tasks),
        list(dict.fromkeys(hendrycks_tasks)),  # insertion order from score file
    ]
    candidate_names = ["alphabetical", "insertion_order"]

    for name, order in zip(candidate_names, candidates):
        sizes = [roberta_emb[t].shape[0] for t in order]
        cumsum = np.concatenate([[0], np.cumsum(sizes)])
        concat_emb = np.concatenate([roberta_emb[t] for t in order], axis=0)
        if np.allclose(concat_emb[sample_idx], ref_sample, atol=1e-4):
            print(f"  MMLU ordering resolved: {name} ({len(order)} tasks, {int(cumsum[-1])} questions)")
            return order, cumsum, concat_emb

    raise RuntimeError(
        "Could not determine MMLU concatenation order — neither alphabetical nor "
        "insertion-order of the score file matched the router_dataset embeddings."
    )


def global_to_subtask(global_indices: np.ndarray, mmlu_order: list[str], mmlu_cumsum: np.ndarray):
    """Decompose global MMLU indices into (subtask_names, local_indices)."""
    task_k = np.searchsorted(mmlu_cumsum[1:], global_indices, side="right")
    local_idx = global_indices - mmlu_cumsum[task_k]
    subtask_names = np.array([mmlu_order[k] for k in task_k], dtype=object)
    return subtask_names, local_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Core data loading per task
# ---------------------------------------------------------------------------

def load_task_data(
    task_key: str,
    leaderboard_tasks: list[str],
    roberta_emb: dict,
    score_data: dict,
    router_rd: dict,
    mmlu_order: list[str] | None,
    mmlu_cumsum: np.ndarray | None,
    mmlu_concat_emb: np.ndarray | None,
    mmlu_concat_score: np.ndarray | None,
    threshold: float,
    alt_emb: dict | None = None,
    mmlu_concat_alt_emb: np.ndarray | None = None,
) -> dict:
    """Return per-split dicts with query_embeddings, task_ids, original_indices, scores (pre-binarize).

    RoBERTa embeddings are always used to verify index alignment against router_dataset.
    If alt_emb is provided, the *emitted* query_embeddings use alt_emb instead of RoBERTa.
    """
    is_mmlu = task_key == "mmlu"
    result = {}

    for split in SPLITS:
        idx = router_rd["split_index"][f"{split}_indices"]
        ref_emb = router_rd["embedding"][f"{split}_embed"]

        if is_mmlu:
            q_emb_roberta = mmlu_concat_emb[idx]
            scores_rows = mmlu_concat_score[idx]  # (N, 5000)
            subtask_names, local_idx = global_to_subtask(idx, mmlu_order, mmlu_cumsum)
            task_ids = subtask_names
            orig_idx = local_idx
            # Alt embeddings for emission (if provided)
            q_emb_out = mmlu_concat_alt_emb[idx] if mmlu_concat_alt_emb is not None else q_emb_roberta
        else:
            assert len(leaderboard_tasks) == 1
            lb_task = leaderboard_tasks[0]
            q_emb_roberta = roberta_emb[lb_task][idx]
            scores_rows = score_data[lb_task]["correctness"][idx]  # (N, 5000)
            task_ids = np.array([task_key] * len(idx), dtype=object)
            orig_idx = idx.copy()
            # Alt embeddings for emission (if provided)
            q_emb_out = alt_emb[lb_task][idx] if alt_emb is not None else q_emb_roberta

        # Verify RoBERTa embeddings match router_dataset (index alignment oracle)
        if not np.allclose(q_emb_roberta, ref_emb, atol=1e-4):
            raise AssertionError(
                f"Embedding mismatch for task={task_key} split={split}. "
                "Index alignment between router_dataset and RoBERTa_result is broken."
            )

        result[split] = {
            "query_embeddings": q_emb_out.astype(np.float32),
            "task_ids": task_ids,
            "original_indices": orig_idx,
            "scores_raw": scores_rows,  # float, not yet binarized
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output pickle path")
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="Output report path")
    p.add_argument(
        "--threshold", type=float, default=0.95,
        help="Binarization threshold: score >= threshold → 1 (default: 0.95)",
    )
    p.add_argument(
        "--query-embed-file", default=None,
        help=(
            "Path to an alternate query-embedding pickle (e.g. "
            "Qwen3-Embedding-4B_query_result.pkl). Must have an 'embeddings' key "
            "mapping task names to (n_q, D) float32 arrays. When omitted, the "
            "default RoBERTa embeddings are used."
        ),
    )
    p.add_argument(
        "--query-embed-name", default=None,
        help="Human-readable label for the alternate embedding model (written to metadata).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    report_path = Path(args.report)
    threshold = args.threshold

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output       : {output_path}")
    print(f"Threshold    : {threshold}")
    if args.query_embed_file:
        print(f"Alt query emb: {args.query_embed_file}")
        print(f"Alt emb name : {args.query_embed_name or '(unset)'}")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n[1/7] Loading source files...")
    for p in (SCORE_FILE, EMBED_FILE, MODEL_PARTITION, MODEL_EMBEDDINGS):
        if not p.exists():
            sys.exit(f"Missing: {p}")

    score_raw = load_pkl(SCORE_FILE)          # {model: [...], data: {task: {correctness: ndarray}}}
    roberta_emb = load_pkl(EMBED_FILE)        # {task_name: ndarray(n_q, 768)}
    model_part = load_pkl(MODEL_PARTITION)    # {subset: {indices, names}, splits: {train, val, test}}
    model_emb = load_pkl(MODEL_EMBEDDINGS)   # {embeddings: (3749, 2048), model_indices: (3749,), ...}

    # Optional alternate query embeddings (e.g. Qwen3-Embedding-4B)
    alt_emb: dict | None = None
    alt_emb_label: str = "RoBERTa (leaderboard_old_embed/RoBERTa_result.pkl)"
    alt_emb_source: str = str(EMBED_FILE.relative_to(PROJECT_ROOT))

    if args.query_embed_file:
        alt_path = Path(args.query_embed_file)
        if not alt_path.is_absolute():
            alt_path = PROJECT_ROOT / alt_path
        if not alt_path.exists():
            sys.exit(f"Missing alt query-embed file: {alt_path}")
        raw_alt = load_pkl(alt_path)
        if "embeddings" not in raw_alt:
            sys.exit(
                f"Alt query-embed file {alt_path} has no 'embeddings' key. "
                "Expected a dict with 'embeddings': {task_name: ndarray(n_q, D)}."
            )
        alt_emb = raw_alt["embeddings"]   # {task_name: ndarray(n_q, D)}
        alt_emb_label = args.query_embed_name or str(alt_path.name)
        alt_emb_source = str(alt_path.relative_to(PROJECT_ROOT) if alt_path.is_relative_to(PROJECT_ROOT) else alt_path)
        print(f"  Alt query embeddings loaded: {len(alt_emb)} tasks")

    score_data = score_raw["data"]           # {task_name: {correctness: ndarray(n_q, 5000)}}

    # ------------------------------------------------------------------
    # Verify model index alignment and build reverse lookup
    # ------------------------------------------------------------------
    subset_indices = model_part["subset"]["indices"]  # (3749,) positions into 5000
    assert np.array_equal(model_emb["model_indices"], subset_indices), (
        "model_embeddings.model_indices does not match model_partition.subset.indices"
    )
    # splits['train'/'val'/'test'] are positions directly into the 5000-model array.
    # Build a reverse map: position_in_5000 → row_in_3749 (for embedding lookup).
    idx5k_to_3749 = np.full(len(score_raw["model"]), -1, dtype=np.int64)
    for row_3749, pos_5k in enumerate(model_emb["model_indices"]):
        idx5k_to_3749[pos_5k] = row_3749

    # ------------------------------------------------------------------
    # Identify MMLU tasks and resolve concatenation order
    # ------------------------------------------------------------------
    print("[2/7] Resolving MMLU concatenation order...")
    hendrycks_tasks = [k for k in score_data if k.startswith("harness_hendrycksTest_")]
    mmlu_rd_path = ROUTER_DIR / "mmlu_router_dataset.pkl"
    if not mmlu_rd_path.exists():
        sys.exit(f"Missing: {mmlu_rd_path}")
    mmlu_rd = load_pkl(mmlu_rd_path)

    # Always use RoBERTa embeddings to resolve/verify MMLU order (oracle).
    mmlu_order, mmlu_cumsum, mmlu_concat_emb = resolve_mmlu_order(
        hendrycks_tasks, roberta_emb, mmlu_rd
    )

    # If an alternate embedding is provided, validate row counts and build its MMLU concat.
    mmlu_concat_alt_emb: np.ndarray | None = None
    if alt_emb is not None:
        # Verify row counts for all in-scope tasks (including MMLU subtasks)
        for lb_task in mmlu_order:
            n_roberta = roberta_emb[lb_task].shape[0]
            if lb_task not in alt_emb:
                sys.exit(
                    f"Alt query-embed file is missing MMLU subtask: {lb_task}. "
                    "Ensure the file covers all 62 leaderboard tasks."
                )
            n_alt = alt_emb[lb_task].shape[0]
            if n_alt != n_roberta:
                sys.exit(
                    f"Row count mismatch for MMLU subtask {lb_task}: "
                    f"RoBERTa={n_roberta}, alt={n_alt}."
                )
        for task_key, lb_tasks in TASK_MAP.items():
            if task_key == "mmlu" or lb_tasks is None:
                continue
            lb_task = lb_tasks[0]
            n_roberta = roberta_emb[lb_task].shape[0]
            if lb_task not in alt_emb:
                sys.exit(
                    f"Alt query-embed file is missing task: {lb_task}. "
                    "Ensure the file covers all 62 leaderboard tasks."
                )
            n_alt = alt_emb[lb_task].shape[0]
            if n_alt != n_roberta:
                sys.exit(
                    f"Row count mismatch for task {lb_task}: "
                    f"RoBERTa={n_roberta}, alt={n_alt}."
                )
        mmlu_concat_alt_emb = np.concatenate(
            [alt_emb[t] for t in mmlu_order], axis=0
        )
        print(f"  Alt MMLU concat: {mmlu_concat_alt_emb.shape}  "
              f"(RoBERTa oracle: {mmlu_concat_emb.shape})")

    print("  Building MMLU concatenated score matrix (14042 × 5000)...")
    mmlu_concat_score = np.concatenate(
        [score_data[t]["correctness"] for t in mmlu_order], axis=0
    )  # (14042, 5000)

    # ------------------------------------------------------------------
    # Audit score ranges before binarization
    # ------------------------------------------------------------------
    print("[3/7] Auditing score ranges...")
    continuous_tasks = []
    score_range_info = {}
    for tk in list(TASK_MAP.keys()):
        if tk == "mmlu":
            arr = mmlu_concat_score
            label = "mmlu (all hendrycksTest)"
        else:
            lb_task = TASK_MAP[tk][0]
            arr = score_data[lb_task]["correctness"]
            label = lb_task

        uniq = np.unique(arr)
        mn, mx = float(arr.min()), float(arr.max())
        score_range_info[tk] = {"min": mn, "max": mx, "n_unique": len(uniq)}
        is_binary = set(uniq.tolist()).issubset({0.0, 1.0})
        if not is_binary:
            continuous_tasks.append(tk)
            print(f"  {label}: continuous values — min={mn:.4f} max={mx:.4f} n_unique={len(uniq)}")
        else:
            print(f"  {label}: binary — OK")

    # ------------------------------------------------------------------
    # Load and validate router_dataset splits per task, gather raw arrays
    # ------------------------------------------------------------------
    print("[4/7] Loading router_dataset files and extracting per-task splits...")
    task_split_data = {}

    for task_key in sorted(TASK_MAP.keys()):
        rd_name = task_key + "_router_dataset.pkl"
        rd_path = ROUTER_DIR / rd_name
        if not rd_path.exists():
            sys.exit(f"Missing router_dataset: {rd_path}")
        rd = load_pkl(rd_path)

        print(f"  {task_key}: ", end="", flush=True)
        data = load_task_data(
            task_key=task_key,
            leaderboard_tasks=TASK_MAP[task_key],
            roberta_emb=roberta_emb,
            score_data=score_data,
            router_rd=rd if task_key != "mmlu" else mmlu_rd,
            mmlu_order=mmlu_order if task_key == "mmlu" else None,
            mmlu_cumsum=mmlu_cumsum if task_key == "mmlu" else None,
            mmlu_concat_emb=mmlu_concat_emb if task_key == "mmlu" else None,
            mmlu_concat_score=mmlu_concat_score if task_key == "mmlu" else None,
            threshold=threshold,
            alt_emb=alt_emb,
            mmlu_concat_alt_emb=mmlu_concat_alt_emb if task_key == "mmlu" else None,
        )
        for split in SPLITS:
            n = len(data[split]["query_embeddings"])
            print(f"{split}={n} ", end="", flush=True)
        print("✓")
        task_split_data[task_key] = data

    # ------------------------------------------------------------------
    # Concatenate across tasks (task alphabetical order = sorted TASK_MAP keys)
    # ------------------------------------------------------------------
    print("[5/7] Concatenating tasks and binarizing scores...")
    combined = {}
    for split in SPLITS:
        q_emb_list, task_id_list, orig_idx_list, scores_list = [], [], [], []
        for task_key in sorted(TASK_MAP.keys()):
            d = task_split_data[task_key][split]
            q_emb_list.append(d["query_embeddings"])
            task_id_list.append(d["task_ids"])
            orig_idx_list.append(d["original_indices"])
            scores_list.append(d["scores_raw"])

        q_emb = np.concatenate(q_emb_list, axis=0)       # (Nq, 768) float32
        task_ids = np.concatenate(task_id_list, axis=0)  # (Nq,) object
        orig_idx = np.concatenate(orig_idx_list, axis=0) # (Nq,) int64
        scores_full = np.concatenate(scores_list, axis=0) # (Nq, 5000) float

        # Binarize
        scores_bin = (scores_full >= threshold).astype(np.uint8)  # (Nq, 5000)
        del scores_full

        combined[split] = {
            "query_embeddings": q_emb,
            "task_ids": task_ids,
            "original_indices": orig_idx,
            "scores_bin": scores_bin,  # (Nq, 5000) uint8
        }
        print(f"  {split}: {len(q_emb)} queries, accuracy_after_binarize={scores_bin.mean():.4f}")

    # ------------------------------------------------------------------
    # Leakage checks
    # ------------------------------------------------------------------
    print("[6/7] Running leakage checks...")
    for task_key in sorted(TASK_MAP.keys()):
        if task_key == "mmlu":
            continue
        for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
            idx1 = task_split_data[task_key][s1]["original_indices"]
            idx2 = task_split_data[task_key][s2]["original_indices"]
            overlap = len(set(idx1.tolist()) & set(idx2.tolist()))
            assert overlap == 0, f"Query leakage: task={task_key} {s1}∩{s2}={overlap}"

    # LLM split leakage (splits already hold positions into 5000)
    llm_splits_5k = {split: model_part["splits"][split] for split in SPLITS}
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = len(set(llm_splits_5k[s1].tolist()) & set(llm_splits_5k[s2].tolist()))
        assert overlap == 0, f"LLM leakage: {s1}∩{s2}={overlap}"
    print("  No query or LLM leakage detected.")

    # ------------------------------------------------------------------
    # Build 9 score blocks and assemble output
    # ------------------------------------------------------------------
    queries_out = {}
    for split in SPLITS:
        queries_out[split] = {
            "embeddings": combined[split]["query_embeddings"],
            "task_ids": combined[split]["task_ids"],
            "original_indices": combined[split]["original_indices"],
        }

    llms_out = {}
    for split in SPLITS:
        idx_5k = model_part["splits"][split]           # positions into 5000 (direct)
        rows_3749 = idx5k_to_3749[idx_5k]             # corresponding rows in embeddings array
        assert np.all(rows_3749 >= 0), f"LLM split={split} has indices outside the 3749 subset"
        llms_out[split] = {
            "embeddings": model_emb["embeddings"][rows_3749].astype(np.float32),
            "names": model_part["subset"]["names"][rows_3749],
            "indices_in_5000": idx_5k,
        }

    scores_out = {}
    for qs in SPLITS:
        scores_out[qs] = {}
        for ls in SPLITS:
            col_idx = llms_out[ls]["indices_in_5000"]
            scores_out[qs][ls] = combined[qs]["scores_bin"][:, col_idx]  # (Nq, Nl) uint8

    metadata = {
        "tasks_in_scope": sorted(TASK_MAP.keys()),
        "task_to_leaderboard_keys": {k: v for k, v in TASK_MAP.items() if k != "mmlu"},
        "mmlu_concat_order": mmlu_order,
        "mmlu_task_offsets": {t: int(mmlu_cumsum[i]) for i, t in enumerate(mmlu_order)},
        "embedding_model_query": alt_emb_label,
        "embedding_model_llm": model_emb.get("config", {}).get("model_name", "see gsm8k_model_embeddings['config']"),
        "binarization_threshold": threshold,
        "continuous_score_tasks": continuous_tasks,
        "query_split_sizes": {s: int(len(queries_out[s]["embeddings"])) for s in SPLITS},
        "llm_split_sizes": {s: int(len(llms_out[s]["embeddings"])) for s in SPLITS},
        "query_embed_dim": int(next(iter(queries_out.values()))["embeddings"].shape[1]),
        "source_files": {
            "scores": str(SCORE_FILE.relative_to(PROJECT_ROOT)),
            "query_embeddings": alt_emb_source,
            "query_embeddings_alignment_oracle": str(EMBED_FILE.relative_to(PROJECT_ROOT)),
            "model_partition": str(MODEL_PARTITION.relative_to(PROJECT_ROOT)),
            "model_embeddings": str(MODEL_EMBEDDINGS.relative_to(PROJECT_ROOT)),
        },
        "partition_seed": model_part.get("seed", 42),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }

    dataset = {
        "queries": queries_out,
        "llms": llms_out,
        "scores": scores_out,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nDataset saved → {output_path}")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    _write_report(report_path, dataset, score_range_info, threshold)
    print(f"Report saved  → {report_path}")

    # Summary
    print("\nShape summary:")
    for qs in SPLITS:
        for ls in SPLITS:
            s = scores_out[qs][ls]
            print(f"  scores[{qs}][{ls}]: {s.shape}  acc={s.mean():.4f}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(report_path: Path, dataset: dict, score_range_info: dict, threshold: float):
    meta = dataset["metadata"]
    qs_sizes = meta["query_split_sizes"]
    ls_sizes = meta["llm_split_sizes"]
    scores = dataset["scores"]

    task_rows = "\n".join(
        f"| `{t}` | {meta['task_to_leaderboard_keys'].get(t, '57 × harness_hendrycksTest_*')} |"
        for t in meta["tasks_in_scope"]
    )

    query_rows = "\n".join(
        f"| {s} | {qs_sizes[s]:,} |" for s in SPLITS
    )
    llm_rows = "\n".join(
        f"| {s} | {ls_sizes[s]:,} |" for s in SPLITS
    )

    score_rows = "\n".join(
        f"| scores[{qs}][{ls}] | ({qs_sizes[qs]:,}, {ls_sizes[ls]:,}) | {scores[qs][ls].mean():.4f} |"
        for qs in SPLITS for ls in SPLITS
    )

    range_rows = "\n".join(
        f"| `{t}` | {v['min']:.4f} | {v['max']:.4f} | {v['n_unique']:,} | {'Yes' if t in meta['continuous_score_tasks'] else 'No'} |"
        for t, v in score_range_info.items()
    )

    report = f"""\
# LLM Performance Prediction Dataset Report

Generated: {meta['created_utc']}
Git commit: `{meta['git_commit']}`

## In-scope tasks

| Router key | Leaderboard task(s) |
|---|---|
{task_rows}

## Binarization

Threshold: **{threshold}** (score >= threshold → 1)

Tasks with continuous scores (pre-binarization):
{', '.join(f'`{t}`' for t in meta['continuous_score_tasks']) or 'None'}

### Score range audit (pre-binarization)

| Task | Min | Max | Unique values | Continuous? |
|------|-----|-----|---------------|-------------|
{range_rows}

## MMLU concatenation order

The 57 `harness_hendrycksTest_*` tasks are concatenated in this order to form
the 14 042-row MMLU block. The ordering matched the **{
'alphabetical' if meta['mmlu_concat_order'] == sorted(meta['mmlu_concat_order']) else 'insertion'
}** sort of task names.

First 5 tasks: {', '.join(f'`{t}`' for t in meta['mmlu_concat_order'][:5])}, ...

## Query splits

| Split | Queries |
|-------|---------|
{query_rows}

## LLM splits

| Split | LLMs |
|-------|------|
{llm_rows}

## Score blocks (query_split × llm_split)

Shape: (n_queries, n_llms), dtype uint8, values {{0, 1}}

| Block | Shape | Mean accuracy |
|-------|-------|---------------|
{score_rows}

## Source files

| Role | Path |
|------|------|
{chr(10).join(f'| {k} | `{v}` |' for k, v in meta['source_files'].items())}

## LLM embedding model

{meta['embedding_model_llm']}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Partition RouterEval models that have size info in their names.

Produces a pickle with the filtered subset (original indices + names) and a
reproducible 80/10/10 train/val/test split, plus a markdown report.

Run from project root:
    python src/data_pipeline/router_eval/partition_models.py
"""
import argparse
import math
import pickle
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import dataset_processed, dataset_raw  # noqa: E402

# Matches numeric size suffixes: 7B, 13b, 1.5B, 70B, 350M, 8x7B, 1.1B, etc.
# The (?<![A-Za-z0-9]) negative lookbehind prevents matching inside longer words.
SIZE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[xX]?\d*\.?\d*[BbMm](?![A-Za-z])")
# Simpler capture just for display: grab the first NNN[BbMm] or NxNNN[BbMm] chunk
SIZE_DISPLAY_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:[xX]\d+)?(?:\.\d+)?[BbMm])(?![A-Za-z])")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        default=str(dataset_raw("router-eval") / "leaderboard_score/leaderboard_old.pkl"),
        help="Path to leaderboard_old.pkl (relative to project root or absolute)",
    )
    p.add_argument(
        "--out-pickle",
        default=str(dataset_processed("router-eval") / "model_partition.pkl"),
    )
    p.add_argument(
        "--out-report",
        default=str(dataset_processed("router-eval") / "model_partition_report.md"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    return p.parse_args()


def extract_size_string(name: str) -> str | None:
    m = SIZE_DISPLAY_RE.search(name)
    return m.group(1) if m else None


def main():
    args = parse_args()
    source_path = Path(args.source)
    out_pickle = Path(args.out_pickle)
    out_report = Path(args.out_report)

    if not source_path.exists():
        sys.exit(f"Source not found: {source_path}")

    with open(source_path, "rb") as f:
        raw = pickle.load(f)

    models: list[str] = raw["model"]
    n_total = len(models)

    # Filter: keep models whose name contains a size token
    kept_indices = []
    kept_names = []
    kept_sizes = []
    dropped_names = []

    for i, name in enumerate(models):
        size = extract_size_string(name)
        if size is not None:
            kept_indices.append(i)
            kept_names.append(name)
            kept_sizes.append(size.upper())
        else:
            dropped_names.append(name)

    n_subset = len(kept_indices)
    kept_indices = np.array(kept_indices, dtype=np.int64)
    kept_names = np.array(kept_names, dtype=object)

    # Reproducible shuffle + split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_subset)

    n_train = math.floor(args.train_frac * n_subset)
    n_val = math.floor(args.val_frac * n_subset)
    n_test = n_subset - n_train - n_val

    shuffled = kept_indices[perm]
    train_idx = np.sort(shuffled[:n_train])
    val_idx = np.sort(shuffled[n_train : n_train + n_val])
    test_idx = np.sort(shuffled[n_train + n_val :])

    # Sanity checks
    all_split = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_split) == n_subset, "Split sizes don't sum to subset size"
    assert set(all_split.tolist()) == set(kept_indices.tolist()), "Split is not a partition of subset"
    assert int(all_split.min()) >= 0 and int(all_split.max()) < n_total, "Index out of range"

    result = {
        "source_file": str(source_path),
        "seed": args.seed,
        "size_regex": SIZE_DISPLAY_RE.pattern,
        "n_total": n_total,
        "subset": {
            "indices": kept_indices,
            "names": kept_names,
            "size_strings": kept_sizes,
        },
        "splits": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
    }

    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pickle, "wb") as f:
        pickle.dump(result, f)

    _write_report(
        out_report=out_report,
        source_path=source_path,
        n_total=n_total,
        n_subset=n_subset,
        kept_names=kept_names,
        kept_sizes=kept_sizes,
        dropped_names=dropped_names,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        args=args,
        out_pickle=out_pickle,
    )

    print(
        f"Done. {n_subset}/{n_total} models kept "
        f"({n_subset/n_total:.1%}) → "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
        f"Pickle: {out_pickle}"
    )


def _short(name: str, max_len: int = 80) -> str:
    """Strip HF prefix and truncate for display."""
    if "details_" in name:
        name = name.split("details_", 1)[1]
    return name[:max_len] + ("…" if len(name) > max_len else "")


def _write_report(
    out_report, source_path, n_total, n_subset, kept_names, kept_sizes,
    dropped_names, train_idx, val_idx, test_idx, args, out_pickle,
):
    n_dropped = n_total - n_subset
    size_counts = Counter(kept_sizes).most_common(20)

    kept_examples = "\n".join(f"- `{_short(n)}`" for n in kept_names[:10])
    dropped_examples = "\n".join(f"- `{_short(n)}`" for n in dropped_names[:10])

    size_table_rows = "\n".join(
        f"| {s} | {c} | {c/n_subset:.1%} |" for s, c in size_counts
    )

    out_report.parent.mkdir(parents=True, exist_ok=True)
    report = f"""\
# RouterEval Model Partition Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Source

| Field | Value |
|-------|-------|
| File | `{source_path}` |
| Total models | {n_total} |

## Configuration

| Parameter | Value |
|-----------|-------|
| Seed | {args.seed} |
| Size regex | `{SIZE_DISPLAY_RE.pattern}` |
| Train fraction | {args.train_frac} |
| Val fraction | {args.val_frac} |
| Test fraction | {1 - args.train_frac - args.val_frac:.1f} |

## Filtering

| Category | Count | % of total |
|----------|-------|------------|
| Total models | {n_total} | 100% |
| Models **with** size info | {n_subset} | {n_subset/n_total:.1%} |
| Models **without** size info | {n_dropped} | {n_dropped/n_total:.1%} |

### Top 20 size strings (among kept models)

| Size | Count | % of subset |
|------|-------|-------------|
{size_table_rows}

### Example kept models (first 10)

{kept_examples}

### Example dropped models (first 10)

{dropped_examples}

## Train / Val / Test Split

| Split | Count | % of subset |
|-------|-------|-------------|
| Train | {len(train_idx)} | {len(train_idx)/n_subset:.1%} |
| Val | {len(val_idx)} | {len(val_idx)/n_subset:.1%} |
| Test | {len(test_idx)} | {len(test_idx)/n_subset:.1%} |
| **Total** | **{n_subset}** | **100%** |

Indices in the pickle are positions in the original {n_total}-model array and align
directly with columns of every `data[task]['correctness']` matrix.

Output pickle: `{out_pickle}`
"""
    out_report.write_text(report)
    print(f"Report written to {out_report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the EmbedLLM files the pipeline reads from HuggingFace (RZ412/EmbedLLM, Apache-2.0).

Fetched: the question order, the train/val/test correctness CSVs, the released query embeddings
(Sentence-BERT all-mpnet-base-v2) and the released MF model embeddings. Everything else in the
upstream release (scalability/, *_x/y.pth) is unused and skipped. The probe answers
(data/embedllm/raw/model_outputs/) and the model roster (model_list.csv) are this repo's own
data and are committed, not downloaded.

Usage:
    python src/data_pipeline/embedllm/download_data.py [--dry-run]
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import dataset_raw  # noqa: E402

REPO_ID     = "RZ412/EmbedLLM"
RAW_DIR     = dataset_raw("embedllm")
NEEDED      = (
    "question_order.csv", "train.csv", "val.csv", "test.csv",
    "question_embeddings.pth", "optimal_mf_model_embeddings.pth",
)


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be fetched without downloading")
    args = parser.parse_args()

    _log(f"Listing files in {REPO_ID} ...")
    all_files = sorted(list_repo_files(REPO_ID, repo_type="dataset"))
    _log(f"  {len(all_files)} files found")

    fetch_files = [f for f in all_files if f in NEEDED]
    missing = sorted(set(NEEDED) - set(fetch_files))
    if missing:
        _log(f"[ERROR] not found upstream: {missing}")
        sys.exit(1)

    _log(f"To fetch: {len(fetch_files)} of {len(all_files)} files")
    for f in fetch_files:
        _log(f"  {f}")

    if args.dry_run:
        _log("\n--dry-run: exiting without downloading.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_downloaded = 0
    n_skipped    = 0

    for hf_path in fetch_files:
        out_path = RAW_DIR / hf_path
        if out_path.exists():
            _log(f"  skip (exists): {out_path.relative_to(RAW_DIR)}")
            n_skipped += 1
            continue
        _log(f"  downloading: {hf_path} ...")
        cached = hf_hub_download(repo_id=REPO_ID, filename=hf_path, repo_type="dataset")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, out_path)
        _log(f"  saved → {out_path.relative_to(RAW_DIR)}")
        n_downloaded += 1

    _log("=" * 60)
    _log(f"Done.  downloaded={n_downloaded}  skipped(exists)={n_skipped}")


if __name__ == "__main__":
    main()

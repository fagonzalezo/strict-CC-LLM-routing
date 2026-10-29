#!/usr/bin/env python3
"""Download the RouterEval files the pipeline reads from HuggingFace (linggm/RouterEval, MIT).

Fetches three archives (~0.8 GB) and extracts only what is used: the old-leaderboard score
matrix (5,000 models), the released RoBERTa query embeddings, and the router_dataset splits of
the six benchmarks in RouterEval-CC (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande).

The per-model GSM8K generations used to build fingerprints are fetched separately, one zip per
model, by `src/data_pipeline/fingerprints/collect_answers.py --dataset router-eval`.

Can be run from any directory:
    python src/data_pipeline/router_eval/download_data.py
"""

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

import sys
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import dataset_raw  # noqa: E402

REPO_ID = "linggm/RouterEval"
RAW_DIR = dataset_raw("router-eval")

# Benchmarks kept in RouterEval-CC (keys of build_llm_perf_dataset.TASK_MAP).
TASKS = ["arc", "gsm8k", "hellaswag", "winogrande", "harness_truthfulqa_mc_0", "mmlu"]

# (zip_filename, members_to_extract); members_to_extract=None means extract everything
DOWNLOADS = [
    ("leaderboard_embed.zip", ["leaderboard_embed/leaderboard_old_embed/RoBERTa_result.pkl"]),
    ("router_dataset.zip", [f"router_dataset/{t}_router_dataset.pkl" for t in TASKS]),
    ("leaderboard_score.zip", ["leaderboard_score/leaderboard_old.pkl"]),
]


def extract_members(zf: zipfile.ZipFile, members: list[str] | None, dest: Path) -> None:
    targets = zf.namelist() if members is None else members
    for member in targets:
        if member.endswith("/"):
            continue
        out_path = dest / member
        if out_path.exists():
            print(f"  skip (exists): {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        print(f"  extracted: {out_path}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for zip_name, members in DOWNLOADS:
        print(f"\nDownloading {zip_name} ...")
        cached = hf_hub_download(
            repo_id=REPO_ID,
            filename=zip_name,
            repo_type="dataset",
        )
        print(f"Extracting from {zip_name} ...")
        with zipfile.ZipFile(cached) as zf:
            extract_members(zf, members, RAW_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()

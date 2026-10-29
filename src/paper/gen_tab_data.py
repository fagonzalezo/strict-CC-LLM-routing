"""Derive Table II (`tab:data`): statistics of the two Cold--Cold datasets.

Unlike the other generators this one reads the dataset bundles
`data/{router-eval,embedllm}/processed/llm_perf_dataset.pkl`, so it needs the tier-B data
pipeline to have run first (see README). Expected output:

    RouterEval-CC  d_q=768  d_m=2048  queries 22906/2859/2894   LLMs 2999/374/376
    EmbedLLM-CC    d_q=768  d_m=2048  queries 21404/3567/10702  LLMs 70/16/21

Run: `python src/paper/gen_tab_data.py`.
"""

import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.data_loader import load_perf_dataset  # noqa: E402
from src.paths import dataset_path, relative  # noqa: E402

SPLITS = ("train", "val", "test")

print("=" * 78)
print("Table II: statistics of the Cold--Cold datasets")
print("=" * 78)
missing = [f for f in ("router-eval", "embedllm") if not dataset_path(f).is_file()]
if missing:
    for f in missing:
        print(f"  missing: {relative(dataset_path(f))} -- build it with the tier-B pipeline (README)")
    sys.exit(1)

for family, name in (("router-eval", "RouterEval-CC"), ("embedllm", "EmbedLLM-CC")):
    path = dataset_path(family)
    print(f"  read: {relative(path)}")
    ds = load_perf_dataset(path)
    d_q = ds["queries"]["train"]["embeddings"].shape[1]
    d_m = ds["llms"]["train"]["embeddings"].shape[1]
    nq = "/".join(str(len(ds["queries"][s]["embeddings"])) for s in SPLITS)
    nl = "/".join(str(len(ds["llms"][s]["embeddings"])) for s in SPLITS)
    print(f"  {name:14s} d_q={d_q}  d_m={d_m}  queries {nq:18s} LLMs {nl}")

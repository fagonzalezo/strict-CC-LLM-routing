#!/usr/bin/env python3
"""Assemble EmbedLLM's `embedllm_fingerprints.pkl`.

Model fingerprints (what used to be this script's Phase A) are no longer built here. They come
from the shared pipeline in `src/data_pipeline/fingerprints/` — `collect_answers.py --dataset
embedllm` generates each model's answers to the probe set both datasets now share, and
`build_fingerprints.py --dataset embedllm` encodes them into
`data/fingerprints/embedllm/fingerprints.pkl`, which this script loads. See
`docs/dataset_construction.md` for why the two datasets used to be fingerprinted on different
questions and are not anymore.

Phase B: Query embeddings — load HF precomputed file (--hf-query-embeddings) or
         encode all prompts from --query-file at full Qwen dimension.
Phase C: Deterministic train/val/test splits: models 65/15/20, queries 60/10/30.
Phase D: Atomic save -> data/embedllm/processed/embedllm_fingerprints.pkl

Usage:
    python src/data_pipeline/fingerprints/collect_answers.py --dataset embedllm ...   # GPU
    python src/data_pipeline/fingerprints/build_fingerprints.py --dataset embedllm ...
    python src/data_pipeline/embedllm/encode_outputs.py \\
        --hf-query-embeddings data/embedllm/raw/question_embeddings.pth
"""

import argparse
import csv
import datetime
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, fingerprints_path  # noqa: E402
REPO_ROOT = ROOT
DATA_DIR  = REPO_ROOT / "data" / "embedllm"
PROC_DIR  = DATA_DIR / "processed"
RAW_DIR   = DATA_DIR / "raw"

DEFAULT_EMBED_MODEL    = "Qwen/Qwen3-Embedding-4B"
DEFAULT_BATCH_SIZE     = 32
DEFAULT_PARTITION_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_splits(ids: list, seed: int, train_frac: float, val_frac: float) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    n       = len(ids)
    n_train = round(train_frac * n)
    n_val   = round(val_frac * n)
    return {
        "train": [ids[i] for i in sorted(idx[:n_train])],
        "val":   [ids[i] for i in sorted(idx[n_train:n_train + n_val])],
        "test":  [ids[i] for i in sorted(idx[n_train + n_val:])],
    }


def _atomic_pkl(data: object, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=4)
    tmp.replace(path)


def _load_fingerprints(expected_keys: list[str]) -> tuple[np.ndarray, dict[str, str], str]:
    """Load the shared-pipeline artifact and check it lines up with `model_list.csv`'s
    `task_id` order — the two are built from independent runs (this script vs.
    `build_fingerprints.py`), so a stale or mismatched artifact must fail loudly rather than
    silently misalign embeddings with models."""
    path = fingerprints_path("embedllm")
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Run first:\n"
            "  python src/data_pipeline/fingerprints/collect_answers.py --dataset embedllm ...\n"
            "  python src/data_pipeline/fingerprints/build_fingerprints.py --dataset embedllm ..."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    model_keys = bundle["model_keys"]
    if model_keys[:len(expected_keys)] != expected_keys:
        raise SystemExit(
            f"{path}: model_keys do not match model_list.csv's task_id order. Rebuild it with "
            "build_fingerprints.py --dataset embedllm using the same model_list.csv."
        )
    embeddings = bundle["embeddings"][:len(expected_keys)].astype(np.float32)
    stats      = bundle["stats"]

    # build_fingerprints.py checkpoints to this same path/schema while it runs, so an
    # interrupted or still-running build leaves a file with the full-height `embeddings` array
    # but a `stats` dict covering only whichever models finished. Phase C below treats
    # `stats.get(tid) != "missing"` as "keep this model" -- an ABSENT key would pass that check
    # and silently enter the splits as an all-zero row, while build_perf_dataset.py's own
    # `stats.get(tid, "missing")` would then treat the same model as excluded. Catch the
    # disagreement here instead of letting it happen downstream.
    unfinished = [k for k in expected_keys if k not in stats]
    if unfinished:
        raise SystemExit(
            f"{path}: {len(unfinished)} model(s) have no stats entry at all (e.g. "
            f"{unfinished[:5]}) -- this looks like a partial/interrupted build_fingerprints.py "
            "run, not a finished one. Re-run build_fingerprints.py --dataset embedllm "
            "(--resume if it was interrupted) before assembling."
        )

    return embeddings, stats, bundle.get("probe_set_name", "?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed-model",  default=DEFAULT_EMBED_MODEL,
                        help="Used only for Phase B's --query-file fallback encoder")
    parser.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-models",     type=int, default=None,
                        help="Limit to first N models (smoke test)")
    parser.add_argument("--hf-query-embeddings", metavar="PATH", default=None,
                        help="Path to HF-precomputed query embeddings (.npy or .pkl)")
    parser.add_argument("--query-file",   metavar="PATH", default=None,
                        help="CSV or pkl with all prompts for Phase B encoding")
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    args = parser.parse_args()

    _log("=" * 60)
    _log("encode_outputs.py — start")
    _log(f"  n_models_cap     : {args.n_models or 'all'}")
    _log(f"  partition_seed   : {args.partition_seed}")
    _log(f"  git_commit       : {_git_commit()}")
    _log("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load metadata ─────────────────────────────────────────────────────────
    _log("[meta] Loading model_list.csv ...")
    models = _load_csv(RAW_DIR / "model_list.csv")
    _log(f"  models : {len(models)}")

    target_models = models if args.n_models is None else models[: args.n_models]
    N = len(target_models)
    expected_keys = [str(m["task_id"]) for m in target_models]

    # ── Fingerprints (built by src/data_pipeline/fingerprints/, not here) ─────
    _log(f"[A] Loading fingerprints for {N} models from "
         f"{fingerprints_path('embedllm')} ...")
    embeddings, stats, probe_set_name = _load_fingerprints(expected_keys)
    n_ok      = sum(1 for v in stats.values() if v == "ok" or v.startswith("missing_probes"))
    n_missing = sum(1 for v in stats.values() if v == "missing")
    n_error   = sum(1 for v in stats.values() if v.startswith("error"))
    _log(f"  probe_set={probe_set_name}  embeddings={embeddings.shape}  "
         f"OK={n_ok}  missing={n_missing}  errors={n_error}")

    # ── Phase B: Query embeddings ─────────────────────────────────────────────
    _log("[B] Query embeddings ...")
    query_embeddings   = None
    query_embed_dim    = None
    query_embed_source = None
    all_query_ids      = None

    if args.hf_query_embeddings:
        path = Path(args.hf_query_embeddings)
        _log(f"  Loading precomputed embeddings from {path} ...")
        if path.suffix == ".npy":
            query_embeddings = np.load(path).astype(np.float32)
        elif path.suffix == ".pth":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                query_embeddings = obj.numpy().astype(np.float32)
            elif isinstance(obj, dict):
                all_query_ids    = obj.get("query_ids")
                arr = obj.get("embeddings") or obj.get("query_embeddings")
                query_embeddings = np.array(arr, dtype=np.float32) if arr is not None else None
            else:
                query_embeddings = np.array(obj, dtype=np.float32)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, np.ndarray):
                query_embeddings = obj.astype(np.float32)
            elif isinstance(obj, dict):
                all_query_ids    = obj.get("query_ids")
                arr = obj.get("embeddings") or obj.get("query_embeddings")
                query_embeddings = np.array(arr, dtype=np.float32) if arr is not None else None
        if query_embeddings is None:
            _log("  [WARN] Could not extract array from file — skipping Phase B.")
        else:
            query_embed_dim    = query_embeddings.shape[1]
            query_embed_source = "hf_precomputed"
            _log(f"  shape={query_embeddings.shape}  dim={query_embed_dim}")

    elif args.query_file:
        path = Path(args.query_file)
        _log(f"  Loading prompts from {path} ...")
        prompts = None
        if path.suffix == ".csv":
            rows     = _load_csv(path)
            id_col   = next((k for k in rows[0] if "id" in k.lower()), list(rows[0].keys())[0])
            text_col = next((k for k in rows[0]
                             if k.lower() in ("prompt", "text", "question", "query")),
                            list(rows[0].keys())[1])
            all_query_ids = [r[id_col] for r in rows]
            prompts       = [r[text_col] for r in rows]
        elif path.suffix in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                all_query_ids = obj.get("query_ids") or obj.get("ids")
                prompts = obj.get("prompts") or obj.get("texts") or obj.get("queries")
            else:
                prompts       = list(obj)
                all_query_ids = [str(i) for i in range(len(prompts))]
        else:
            _log(f"  [WARN] Unsupported format {path.suffix} — skipping Phase B.")

        if prompts is not None:
            _log(f"  {len(prompts)} prompts — encoding at full Qwen dim ...")
            sbert_full = SentenceTransformer(
                args.embed_model,
                model_kwargs={"torch_dtype": torch.float16},
                device=device,
            )
            query_embeddings = sbert_full.encode(
                prompts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            query_embed_dim    = query_embeddings.shape[1]
            query_embed_source = "qwen_computed"
            _log(f"  shape={query_embeddings.shape}  dim={query_embed_dim}")

    else:
        _log("  No --hf-query-embeddings or --query-file provided — skipping Phase B.")

    # ── Phase C: Partitions ───────────────────────────────────────────────────
    _log("[C] Building splits ...")
    valid_task_ids = [int(m["task_id"]) for m in target_models if stats.get(str(m["task_id"])) != "missing"]
    model_splits   = _make_splits(valid_task_ids, args.partition_seed, 0.65, 0.15)
    _log(f"  models  train={len(model_splits['train'])}  val={len(model_splits['val'])}  test={len(model_splits['test'])}  (valid only, missing excluded)")

    if query_embeddings is not None:
        q_indices    = list(range(len(query_embeddings)))
        query_splits = _make_splits(q_indices, args.partition_seed, 0.60, 0.10)
        _log(f"  queries train={len(query_splits['train'])}  val={len(query_splits['val'])}  test={len(query_splits['test'])}  (indices into query_embeddings)")
    else:
        query_splits = {"train": [], "val": [], "test": []}
        _log("  queries: no query_embeddings — splits empty")

    # ── Phase D: Save ─────────────────────────────────────────────────────────
    _log("[D] Saving final pickle ...")
    config = {
        "probe_set":      probe_set_name,
        "fingerprints_source": str(fingerprints_path("embedllm")),
        "partition_seed": args.partition_seed,
        "git_commit":     _git_commit(),
        "created_at":     datetime.datetime.now().isoformat(),
        "query_embed_dim":    query_embed_dim,
        "query_embed_source": query_embed_source,
    }
    out_pkl = PROC_DIR / "embedllm_fingerprints.pkl"
    result = {
        "config":            config,
        "model_ids":         [int(m["task_id"]) for m in target_models],
        "model_metadata":    [dict(m) for m in target_models],
        "embeddings":        embeddings,
        "stats":             stats,
        "query_ids":         all_query_ids,
        "query_embeddings":  query_embeddings,
        "splits": {
            "models":  model_splits,
            "queries": query_splits,
        },
    }
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_pkl(result, out_pkl)

    pkl_mb = out_pkl.stat().st_size / 1e6
    _log("=" * 60)
    _log("DONE")
    _log(f"  OK={n_ok}  missing={n_missing}  errors={n_error}  total={N}")
    _log(f"  embeddings shape     : {embeddings.shape}")
    _log(f"  query_embeddings     : {query_embeddings.shape if query_embeddings is not None else 'not computed'}")
    _log(f"  pickle               : {out_pkl}  ({pkl_mb:.1f} MB)")
    _log("=" * 60)


if __name__ == "__main__":
    main()

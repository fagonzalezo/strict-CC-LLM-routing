"""
Build llm_perf_dataset.pkl for EmbedLLM, mirroring the router-eval schema.

Sources (on HPC):
  data/embedllm/processed/embedllm_fingerprints.pkl   — embeddings + model/query splits
  data/embedllm/raw/question_order.csv                — row_index -> prompt_id
  data/embedllm/raw/train.csv / val.csv / test.csv    — (prompt_id, model_id, label) pairs
  data/embedllm/raw/optimal_mf_model_embeddings.pth   — EmbedLLM paper MF model embeddings

Output:
  data/embedllm/processed/llm_perf_dataset.pkl

Output schema (mirrors src/data_loader.py):
  dataset["queries"][split]["embeddings"]        (Nq, 768)  float32
  dataset["queries"][split]["task_ids"]          (Nq,)      object  (category strings)
  dataset["queries"][split]["original_indices"]  (Nq,)      int64   (row in question_order.csv)
  dataset["llms"][split]["embeddings"]           (Nl, 2048) float32 — orig (Qwen3 fingerprint)
  dataset["llms"][split]["embeddings_mf"]        (Nl, D_mf) float32 — refined (MF paper embed)
  dataset["llms"][split]["names"]                (Nl,)      object
  dataset["llms"][split]["task_ids"]             (Nl,)      int32
  dataset["llms"][split]["indices_in_valid"]     (Nl,)      int32   (col in full score matrix)
  dataset["scores"][query_split][llm_split]      (Nq, Nl)   uint8   (binary correctness)
  dataset["metadata"]                            dict

Cold-cold splits come from embedllm_fingerprints.pkl (seed=42):
  models:  train=70, val=16, test=21  (task_ids of valid models)
  queries: train=21404, val=3567, test=10702 (indices into question_order.csv)
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import dataset_processed, dataset_raw  # noqa: E402

FP_PKL     = dataset_processed("embedllm") / "embedllm_fingerprints.pkl"
QORDER_CSV = dataset_raw("embedllm") / "question_order.csv"
TRAIN_CSV  = dataset_raw("embedllm") / "train.csv"
VAL_CSV    = dataset_raw("embedllm") / "val.csv"
TEST_CSV   = dataset_raw("embedllm") / "test.csv"
MF_EMB     = dataset_raw("embedllm") / "optimal_mf_model_embeddings.pth"
OUT_PKL    = dataset_processed("embedllm") / "llm_perf_dataset.pkl"


def _atomic_pkl(obj, path: Path):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load embedllm_fingerprints.pkl
    # ------------------------------------------------------------------
    print("1/7  Loading embedllm_fingerprints.pkl ...")
    with open(FP_PKL, "rb") as f:
        fp = pickle.load(f)

    q_emb_all   = fp["query_embeddings"]   # (35673, 768) float32
    m_emb_all   = fp["embeddings"]         # (107, 2048)  float32
    m_metadata  = fp["model_metadata"]     # list[dict{task_id,model_id,model_name,...}]
    m_row_ids   = fp["model_ids"]          # list[int] task_ids in row order (len=107)
    stats       = fp["stats"]              # {str(task_id): "ok"|"missing..."|"missing"}
    splits      = fp["splits"]             # {models:{train/val/test}, queries:{train/val/test}}

    print(f"     query_embeddings: {q_emb_all.shape}")
    print(f"     model_embeddings: {m_emb_all.shape}")
    print(f"     model splits: { {k: len(v) for k, v in splits['models'].items()} }")
    print(f"     query splits: { {k: len(v) for k, v in splits['queries'].items()} }")

    # Build mappings
    task_id_to_pkl_row  = {int(tid): i for i, tid in enumerate(m_row_ids)}
    task_id_to_model_id = {int(m["task_id"]): int(m["model_id"]) for m in m_metadata}
    task_id_to_name     = {int(m["task_id"]): m["model_name"] for m in m_metadata}

    # ------------------------------------------------------------------
    # 2. Load optional MF model embeddings (refined)
    # ------------------------------------------------------------------
    mf_emb_all = None
    if MF_EMB.exists():
        print("2/7  Loading optimal_mf_model_embeddings.pth ...")
        obj = torch.load(MF_EMB, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            mf_emb_all = obj.detach().numpy().astype(np.float32)
        print(f"     mf_embeddings: {mf_emb_all.shape}")
    else:
        print("2/7  optional_mf_model_embeddings.pth not found — skipping refined embeddings.")

    # ------------------------------------------------------------------
    # 3. Load question_order.csv: row_index -> prompt_id
    # ------------------------------------------------------------------
    print("3/7  Loading question_order.csv ...")
    qorder = pd.read_csv(QORDER_CSV, usecols=["prompt_id"], dtype={"prompt_id": np.int32})
    row_to_prompt_id = qorder["prompt_id"].values          # shape (35673,)
    prompt_id_to_row = {int(pid): i for i, pid in enumerate(row_to_prompt_id)}
    print(f"     {len(row_to_prompt_id)} queries indexed")

    # ------------------------------------------------------------------
    # 4. Load label CSVs (train + val + test)
    #    Only prompt_id, model_id, label, category — skip large prompt text.
    #    Python engine required because prompt field has embedded newlines.
    # ------------------------------------------------------------------
    print("4/7  Loading label CSVs (prompt field is multi-line — may take several minutes) ...")
    frames = []
    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        t1 = time.time()
        df = pd.read_csv(
            csv_path,
            usecols=["prompt_id", "model_id", "label", "category"],
            dtype={"prompt_id": np.int32, "model_id": np.int32, "label": np.uint8},
            engine="python",
        )
        print(f"     {csv_path.name}: {len(df):,} rows  ({time.time()-t1:.0f}s)")
        frames.append(df)

    all_labels = pd.concat(frames, ignore_index=True)
    del frames
    print(f"     total: {len(all_labels):,} rows")

    # ------------------------------------------------------------------
    # 5. Build full score matrix S[N_queries x N_valid_models]
    # ------------------------------------------------------------------
    print("5/7  Building score matrix ...")

    # Valid models: non-missing, in order of task_id
    all_task_ids = sorted(task_id_to_model_id.keys())
    valid_task_ids = [
        tid for tid in all_task_ids
        if stats.get(str(tid), "missing") != "missing"
    ]
    valid_model_ids = [task_id_to_model_id[tid] for tid in valid_task_ids]
    N_q = len(row_to_prompt_id)   # 35673
    N_m = len(valid_model_ids)    # 107 valid models

    # model_id -> column index in S
    model_id_to_col = {mid: j for j, mid in enumerate(valid_model_ids)}

    S = np.zeros((N_q, N_m), dtype=np.uint8)

    # Vectorised assignment
    pids = all_labels["prompt_id"].values
    mids = all_labels["model_id"].values
    lbls = all_labels["label"].values

    max_pid = int(pids.max())
    max_mid = int(mids.max())

    pid_lookup = np.full(max_pid + 1, -1, dtype=np.int32)
    for pid, row in prompt_id_to_row.items():
        if pid <= max_pid:
            pid_lookup[pid] = row

    mid_lookup = np.full(max_mid + 1, -1, dtype=np.int32)
    for mid, col in model_id_to_col.items():
        if mid <= max_mid:
            mid_lookup[mid] = col

    rows_idx = pid_lookup[pids]
    cols_idx = mid_lookup[mids]
    valid = (rows_idx >= 0) & (cols_idx >= 0)
    S[rows_idx[valid], cols_idx[valid]] = lbls[valid]

    n_filled = int(valid.sum())
    coverage = n_filled / (N_q * N_m) * 100
    print(f"     S shape: {S.shape}  coverage: {n_filled:,}/{N_q*N_m:,} = {coverage:.1f}%")
    print(f"     overall mean label: {S.mean():.4f}")

    # ------------------------------------------------------------------
    # 6. Build prompt_id -> category map for task_id labels
    # ------------------------------------------------------------------
    cat_map = (
        all_labels.drop_duplicates("prompt_id")
        .set_index("prompt_id")["category"]
        .to_dict()
    )
    query_categories = np.array(
        [cat_map.get(int(pid), "unknown") for pid in row_to_prompt_id],
        dtype=object,
    )
    del all_labels, cat_map

    # ------------------------------------------------------------------
    # 7. Assemble dataset dict with cold-cold sub-matrices
    # ------------------------------------------------------------------
    print("6/7  Assembling dataset splits ...")

    def _llm_block(split_task_ids):
        """Return (pkl_rows, valid_cols, task_ids, names) for the split."""
        pkl_rows, vcols, tids, names = [], [], [], []
        for tid in split_task_ids:
            mid = task_id_to_model_id.get(tid)
            if mid is None:
                continue
            col = model_id_to_col.get(mid)
            if col is None:
                continue
            pkl_rows.append(task_id_to_pkl_row[tid])
            vcols.append(col)
            tids.append(tid)
            names.append(task_id_to_name[tid])
        return (
            np.array(pkl_rows, dtype=np.int32),
            np.array(vcols, dtype=np.int32),
            np.array(tids, dtype=np.int32),
            np.array(names, dtype=object),
        )

    dataset = {
        "queries":  {},
        "llms":     {},
        "scores":   {},
        "metadata": {
            "source_pkl":    str(FP_PKL.name),
            "source_labels": "train.csv + val.csv + test.csv",
            "n_queries":     N_q,
            "n_valid_models": N_m,
            "split_seed":    42,
            "score_dtype":   "uint8 binary (0/1 correctness)",
            "coverage_pct":  round(coverage, 2),
        },
    }

    for qsplit in ("train", "val", "test"):
        q_idx = np.array(splits["queries"][qsplit], dtype=np.int64)
        dataset["queries"][qsplit] = {
            "embeddings":       q_emb_all[q_idx].astype(np.float32),
            "task_ids":         query_categories[q_idx],
            "original_indices": q_idx,
        }
        dataset["scores"][qsplit] = {}

    for lsplit in ("train", "val", "test"):
        pkl_rows, vcols, tids, names = _llm_block(splits["models"][lsplit])
        d = {
            "embeddings":       m_emb_all[pkl_rows].astype(np.float32),
            "names":            names,
            "task_ids":         tids,
            "indices_in_valid": vcols,
        }
        if mf_emb_all is not None:
            # MF embedding order follows model_order.csv which uses model_id index
            model_ids_for_split = np.array(
                [task_id_to_model_id[int(tid)] for tid in tids], dtype=np.int32
            )
            d["embeddings_mf"] = mf_emb_all[model_ids_for_split].astype(np.float32)
        dataset["llms"][lsplit] = d

    for qsplit in ("train", "val", "test"):
        for lsplit in ("train", "val", "test"):
            q_idx = np.array(splits["queries"][qsplit], dtype=np.int64)
            l_vcols = dataset["llms"][lsplit]["indices_in_valid"]
            dataset["scores"][qsplit][lsplit] = S[np.ix_(q_idx, l_vcols)].copy()

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    print(f"7/7  Saving to {OUT_PKL} ...")
    _atomic_pkl(dataset, OUT_PKL)
    sz = OUT_PKL.stat().st_size / 1024**2
    print(f"     saved {sz:.1f} MB")
    print(f"     elapsed: {time.time()-t0:.0f}s")

    # Summary
    print("\n=== Summary ===")
    for qsplit in ("train", "val", "test"):
        for lsplit in ("train", "val", "test"):
            sc = dataset["scores"][qsplit][lsplit]
            print(f"  scores[{qsplit}][{lsplit}]: {sc.shape}  mean={sc.mean():.4f}")
    print()
    for split in ("train", "val", "test"):
        q = dataset["queries"][split]
        l = dataset["llms"][split]
        cats, counts = np.unique(q["task_ids"], return_counts=True)
        print(f"  {split} queries: {q['embeddings'].shape}  categories: {dict(zip(cats, counts))}")
        print(f"  {split} llms:    {l['embeddings'].shape}  names: {l['names'][:3].tolist()} ...")
    print("Done.")


if __name__ == "__main__":
    main()

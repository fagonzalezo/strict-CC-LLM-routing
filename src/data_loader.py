import pickle
from pathlib import Path
import numpy as np


def load_perf_dataset(path: str | Path) -> dict:
    """Load llm_perf_dataset.pkl and return the raw dict.

    Schema
    ------
    dataset["queries"][split]["embeddings"]       (Nq, 768)  float32
    dataset["queries"][split]["task_ids"]         (Nq,)      object
    dataset["queries"][split]["original_indices"] (Nq,)      int64
    dataset["llms"][split]["embeddings"]          (Nl, 2048) float32
    dataset["llms"][split]["names"]               (Nl,)      object
    dataset["llms"][split]["indices_in_5000"]     (Nl,)      int
    dataset["scores"][query_split][llm_split]     (Nq, Nl)   uint8  (binary, threshold 0.95)
    dataset["metadata"]                           dict
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def get_arrays(dataset: dict, query_split: str, llm_split: str):
    """Return (q_emb, l_emb, scores) for the requested splits as float32 numpy arrays."""
    q_emb = dataset["queries"][query_split]["embeddings"].astype(np.float32)
    l_emb = dataset["llms"][llm_split]["embeddings"].astype(np.float32)
    scores = dataset["scores"][query_split][llm_split]  # uint8
    return q_emb, l_emb, scores

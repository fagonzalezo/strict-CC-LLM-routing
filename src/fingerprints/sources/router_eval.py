"""RouterEval answer source: pulls each model's own GSM8K generations from HuggingFace.

RouterEval's 3,749 leaderboard models cannot be re-run — this is the only place their text
exists. `_extract_text` / `load_gsm8k_outputs` are moved verbatim from the pre-refactor
`encode_model_outputs.py`; the only change is slicing by the frozen `ProbeSet.gsm8k_index`
instead of a freshly-sampled `rng(42).choice(...)`.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

from src.fingerprints.probe_set import ProbeSet

__all__ = ["RouterEvalLightevalSource"]

HF_REPO = "linggm/RouterEval"
TASK = "gsm8k"
LIGHTEVAL_KEY = "harness|gsm8k|5"


def _extract_text(pred) -> str:
    if isinstance(pred, str):
        return pred
    if isinstance(pred, (list, tuple)):
        return _extract_text(pred[0]) if pred else ""
    if isinstance(pred, np.ndarray):
        if pred.dtype.kind in ("U", "S", "O"):
            val = pred.flat[0]
            return val.decode() if isinstance(val, bytes) else str(val)
        return ""
    return str(pred)


def load_gsm8k_outputs(zip_path: Path) -> list[str]:
    """Return all generated output strings from a lighteval GSM8K zip, in eval-set row order."""
    with zipfile.ZipFile(zip_path) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            data = pickle.load(f)
    sub = data.get(LIGHTEVAL_KEY) or data[next(iter(data))]
    predictions = sub["predictions"]
    return [_extract_text(predictions[i]) for i in range(len(predictions))]


class RouterEvalLightevalSource:
    """`answers_for(model_key)` downloads that model's lighteval GSM8K zip and slices it at the
    probe set's `gsm8k_index` positions. `model_key` is the model's position in the 5,000-row
    leaderboard array (as a string) — the same key `model_partition.pkl` and
    `build_llm_perf_dataset.py` use — and `model_names` maps it to the HuggingFace leaderboard
    name needed to build the download path.
    """

    def __init__(self, probe_set: ProbeSet, model_names: dict[str, str]):
        if probe_set.n == 0:
            raise ValueError("probe_set is empty")
        self.probe_set = probe_set
        self.model_names = model_names
        self._max_gsm8k_index = max(probe_set.gsm8k_index)

    def answers_for(self, model_key: str) -> list[str | None]:
        from huggingface_hub import hf_hub_download

        name = self.model_names[model_key]
        slug = name.split("details_", 1)[-1] if "details_" in name else name
        hf_file = f"full_data/{TASK}/details_{slug}.zip"

        local_zip = hf_hub_download(repo_id=HF_REPO, filename=hf_file, repo_type="dataset")
        all_outputs = load_gsm8k_outputs(Path(local_zip))

        if len(all_outputs) < self._max_gsm8k_index + 1:
            raise ValueError(
                f"{model_key}: expected >= {self._max_gsm8k_index + 1} GSM8K outputs, "
                f"got {len(all_outputs)}"
            )
        return [all_outputs[gi] for gi in self.probe_set.gsm8k_index]

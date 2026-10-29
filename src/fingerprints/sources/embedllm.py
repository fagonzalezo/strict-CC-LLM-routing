"""EmbedLLM answer sources: `EmbedLLMLocalOutputs` reads shipped answers, `EmbedLLMGenerator`
regenerates them on a GPU.

`data/embedllm/raw/model_outputs/<model_id>.json` (107 files, committed) already answer the
shared GSM8K probe set (`gsm8k_seed42_n64`), in `probe_index` order — verified empirically
(probe-unique-token alignment 94.4% at the correct offset vs 2.5-4.7% at four control offsets,
across all 107 files). `EmbedLLMLocalOutputs` is Tier-A (no GPU, no network) and is the default
answer source. `EmbedLLMGenerator` remains for provenance and for the day the probe set changes.

Ported from the authors' original EmbedLLM generation script (`generate_model_outputs.py`), with four
deliberate deviations (see docs/dataset_construction.md):

  1. **No `truncation=True, max_length=512`.** A 5-shot GSM8K prompt with full chain-of-thought
     solutions runs roughly 700-1000 tokens; HF's default right-truncation would delete the
     question being asked, turning every answer into a continuation of the few-shot examples.
     Prompts are passed untruncated.
  2. Prompts come from the canonical `ProbeSet.prompt` column (lighteval's own `full_prompt`),
     not a dataset-specific CSV — this is what makes the two datasets' fingerprints comparable.
  3. **Greedy decoding, `max_new_tokens=256`** — matching lighteval's `harness|gsm8k|5`
     generation config, so EmbedLLM's models are elicited the same way RouterEval's were.
  4. **Stop at the next few-shot boundary.** lighteval's own harness run used stop sequences —
     checked directly against the cached `full_data/gsm8k/details_*.zip` predictions: every one
     ends at or just past `#### <answer>`, never rambling into a fabricated `Question:` /
     `Answer:` pair, with mean length ~300 characters against a `max_new_tokens=256` budget that
     would allow far more. Plain `generate()` has no such stop condition, so without truncating
     the decoded text, RouterEval's answers (short, on-topic) and EmbedLLM's (long, running past
     the answer into invented continuations) would differ in a way that reflects *this decoding
     config*, not model behaviour — precisely the confound the shared probe set exists to
     remove. `_truncate_at_stop` below cuts each decoded answer at the first `"\nQuestion:"` or
     blank-line boundary, the cheapest correct proxy for a real stopping criterion.

Also sets `tokenizer.padding_side = "left"`, which the ported script left at its (often right)
default — batched `generate()` on a decoder-only model needs left padding for the per-row
`input_len` slice below to isolate the generated continuation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.fingerprints.probe_set import ProbeSet
from src.paths import dataset_raw

__all__ = ["EmbedLLMLocalOutputs", "EmbedLLMGenerator"]

_GEN_BATCH_SIZE = 8
_MAX_NEW_TOKENS = 256

# Checked against the cached RouterEval gsm8k predictions (deviation 4 above): a real
# lighteval|gsm8k|5 answer never continues past its own "#### <n>" line into the next few-shot
# turn. Cut at whichever of these appears first. Applied by both sources below, so a shipped
# answer and a freshly-generated one are normalized identically.
_STOP_MARKERS = ("\nQuestion:", "\n\n")


def _truncate_at_stop(text: str) -> str:
    cut = len(text)
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def _normalize_answer(text: str) -> str | None:
    """The one place a raw decoded/stored answer becomes a probe answer, shared by both sources
    so they can never diverge in how they clean generated text.

    `.strip()` runs *before* `_truncate_at_stop`, not just after: some models' decoded text opens
    with a blank line (e.g. `" \n\nWorking backward, ..."`), and without a leading strip,
    `_truncate_at_stop` finds `"\n\n"` at index 0-1 and discards the entire real answer, not just
    a trailing artifact. Stripping first removes that leading whitespace so the stop-marker search
    only ever fires on a genuine boundary. Verified against microsoft/Orca-2-13b's committed
    outputs, all 64 of which open with a blank line: strip-after wrongly empties 4 of them,
    strip-first empties 0.
    """
    return _truncate_at_stop(text.strip()).strip() or None


class EmbedLLMLocalOutputs:
    """`answers_for(task_id)` reads the shipped `raw/model_outputs/<model_id>.json` for that
    model — no GPU, no network. Tier-A; the default EmbedLLM answer source.

    `model_key` is the EmbedLLM `task_id` (the key `AnswerStore` and `model_list.csv` use);
    `model_ids` maps it to the `model_id` the *filename* uses instead. These are different id
    spaces — `task_id` is a dense 0..106 re-index, `model_id` the original 0..111 numbering with
    gaps — and 78 of 107 `model_list.csv` rows have `task_id != model_id`, so this map is
    mandatory, not defensive.

    Each file is a JSON array of `{"prompt_id": int, "output": str}` records, one per probe, in
    no guaranteed order — answers are placed by `prompt_id`, not list position. A file whose
    `prompt_id`s aren't exactly `0..probe_set.n-1` is rejected rather than silently misaligned.
    Every answer is passed through `_normalize_answer`, shared with `EmbedLLMGenerator`.
    """

    def __init__(self, probe_set: ProbeSet, model_ids: dict[str, str],
                 outputs_dir: Path | None = None):
        if probe_set.n == 0:
            raise ValueError("probe_set is empty")
        self.probe_set = probe_set
        self.model_ids = model_ids
        self.outputs_dir = outputs_dir or (dataset_raw("embedllm") / "model_outputs")

    def answers_for(self, model_key: str) -> list[str | None]:
        model_id = self.model_ids[model_key]
        path = self.outputs_dir / f"{model_id}.json"
        records = json.loads(path.read_text())  # FileNotFoundError -> classify_exception "missing"

        n = self.probe_set.n
        by_prompt_id = {rec["prompt_id"]: rec["output"] for rec in records}
        if set(by_prompt_id) != set(range(n)):
            raise ValueError(
                f"{path}: prompt_ids {sorted(by_prompt_id)} != expected 0..{n - 1} "
                f"for probe set {self.probe_set.name!r}"
            )

        return [_normalize_answer(by_prompt_id[i]) for i in range(n)]


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except ValueError:
        # Some older fine-tunes trigger a broken tiktoken conversion path in recent
        # `transformers`; fall back to the slow (SentencePiece) tokenizer.
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)


class EmbedLLMGenerator:
    """`answers_for(model_key)` loads that one model, generates on the whole probe set, and
    unloads it. `model_key` is the EmbedLLM `task_id` (as a string) — the same key
    `model_list.csv` / `stats` use — and `model_names` maps it to the HF `hf_name` to load.
    `quantize` optionally maps a `model_key` to `"4bit"` / `"8bit"`, taken from
    `model_list.csv`'s own `quantize` column, for models too large to load in fp16.

    One model per call by design: EmbedLLM's roster spans 7B-72B parameters, so nothing here
    keeps more than one model resident. `collect_answers.py --dataset embedllm --model-idx N`
    is the intended entry point, one process per model (a job array task per model_idx).
    """

    def __init__(self, probe_set: ProbeSet, model_names: dict[str, str],
                 quantize: dict[str, str] | None = None,
                 max_new_tokens: int = _MAX_NEW_TOKENS, device_map: str = "auto"):
        if probe_set.n == 0:
            raise ValueError("probe_set is empty")
        self.probe_set = probe_set
        self.model_names = model_names
        self.quantize = quantize or {}
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map

    def answers_for(self, model_key: str) -> list[str | None]:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        model_name = self.model_names[model_key]
        quantize = self.quantize.get(model_key)

        tokenizer = _load_tokenizer(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_kwargs = {"device_map": self.device_map, "trust_remote_code": True,
                        "torch_dtype": torch.float16}
        if quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        elif quantize == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        try:
            prompts = self.probe_set.prompt
            answers: list[str | None] = []
            for i in range(0, len(prompts), _GEN_BATCH_SIZE):
                batch = prompts[i:i + _GEN_BATCH_SIZE]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=False).to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # greedy, for reproducibility
                        pad_token_id=tokenizer.pad_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]  # uniform: left-padded to batch max
                for gen in generated:
                    text = tokenizer.decode(gen[input_len:], skip_special_tokens=True)
                    answers.append(_normalize_answer(text))
            return answers
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

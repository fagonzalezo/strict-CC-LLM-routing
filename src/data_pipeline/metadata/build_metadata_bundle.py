"""
Build a metadata-representation variant of an `llm_perf_dataset.pkl` bundle.

The paper's distinctive claim is that routing to unseen LLMs works from a *behavioral*
fingerprint, and that it works **because it is behavioral, not because it is an LLM
descriptor**. No experiment has ever fed the routers a metadata descriptor instead, so the
claim is an argument rather than a table. This script produces the bundles that make it one.

The swap happens **at the dataset-bundle level**, upstream of everything else:

    llm_perf_dataset.pkl                        llms[split]["embeddings"] = 2048-d fingerprint
      +-> <variant>/llm_perf_dataset.pkl        SAME schema, embeddings replaced, rows aligned
            +-> encode_embeddings_mlpdot.py     MLPDot two-tower encoder, seeds {0,1,2}
                  +-> the existing BF-DKR / kNN / MLPDot eval scripts, unchanged

`MLPDotRouter.__init__` carries no input-dimension argument -- both tower dims are inferred at
`fit` time from the data (`src/routers/mlp_dot.py:250-252`) -- and `DMKDMRRouter` only ever sees
the encoder's `target_dim`-d output. So **no router code changes and nothing below the encoder
differs between arms**, which is what makes the comparison like-for-like.

Variants
--------
meta_struct  log10(params in B), MoE flag, family one-hot, org one-hot -- the cheap metadata a
             deployment actually has.
meta_name    the model-ID string embedded with the same `Qwen/Qwen3-Embedding-4B` and truncated
             to 2048-d. The *strong* metadata arm, dimension-matched so capacity is not the
             confound.
concat       fingerprint (+) meta_struct -- does metadata add anything *on top of* behavior?

Name parsing
------------
Both datasets reduce to `org__model` after one substitution. RouterEval names are
`open-llm-leaderboard-old/details_<ORG>__<MODEL>`; stripping the `<leaderboard>/details_`
prefix leaves EmbedLLM's own `<ORG>__<MODEL>` shape, so one parser serves both. A naive split
on "/" would find a single "org" (the constant leaderboard prefix) on RouterEval.

Cold-cold bites the one-hots, and that is the finding
-----------------------------------------------------
Test LLMs are disjoint from train LLMs, so test orgs and families include values unseen in
training. Unseen values go to an explicit `UNK` bucket rather than an all-zero row, and the
test-pool UNK rate is reported in `metadata_coverage.json`. It is high -- roughly 3 of every 4
RouterEval test LLMs fall outside the top-50 training orgs. That is not a defect to paper over:
metadata generalizes to a new model only through categories the training pool already had,
which is precisely the failure mode a behavioral fingerprint is designed to avoid.

Usage
-----
  python build_metadata_bundle.py --dataset router-eval --variant meta_struct --dry-run
  python build_metadata_bundle.py --dataset router-eval --variant meta_struct
  python build_metadata_bundle.py --dataset embedllm  --variant meta_name --device cuda
"""

import argparse
import collections
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_path, dataset_processed  # noqa: E402

VARIANTS = ["meta_struct", "meta_name", "concat"]
SPLITS = ["train", "val", "test"]

# `open-llm-leaderboard-old/details_` on RouterEval; a no-op on EmbedLLM.
_LEADERBOARD_PREFIX = re.compile(r"^[^/]+/details_")
# "4x7B", "8X7B" -- a mixture-of-experts parameter token.
_MOE_TOKEN = re.compile(r"(?<![A-Za-z0-9])\d+[xX]\d+(?:\.\d+)?[BbMm](?![A-Za-z])")

# Default one-hot caps. RouterEval's 2,999 training LLMs span 828 orgs, most singletons; a full
# one-hot would be a sparse near-identity matrix and the arm would measure memorization rather
# than metadata. EmbedLLM's 70 training LLMs span 45 orgs, so its cap has to be much smaller --
# these are the defaults, and --top-k-* overrides them from the actual frequency table.
DEFAULT_TOP_K = {"router-eval": {"org": 50, "family": 50},
                 "embedllm": {"org": 10, "family": 10}}

MODEL_ID_DIM = 2048   # meta_name is dimension-matched to the fingerprint


def normalize_name(name: str) -> str:
    """Reduce either dataset's LLM name to the shared `org__model` form."""
    return _LEADERBOARD_PREFIX.sub("", str(name))


def split_org_model(name: str) -> tuple[str, str]:
    norm = normalize_name(name)
    org, _, model = norm.partition("__")
    return (org, model) if model else ("UNK", norm)


def family_of(name: str) -> str:
    """Heuristic model family: the first alphabetic token of the model slug.

    **Family is not a field in either dataset.** RouterEval's population is dominated by merges
    and slerps whose "family" is genuinely ill-defined (`neurotic-crown-clown-7b-tak-stack-dpo`),
    so this is a stated heuristic, not a recovered attribute -- see DESIGN.md section 2.
    """
    _, model = split_org_model(name)
    tokens = re.split(r"[-_.0-9]+", model.lower())
    return next((t for t in tokens if t), "unk")


def _atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _atomic_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def cost_vectors(dataset: str, data: dict) -> dict[str, np.ndarray]:
    """Parameter counts in billions per split, from the project's existing cost proxies.

    Deliberately not a new parser: RouterEval joins `model_partition.pkl`'s `size_strings` on
    `indices_in_5000`, and EmbedLLM's proxy carries the 15-entry `_MANUAL_SIZES_B` table for the
    models whose names have no parseable size token. Both assert full coverage.
    """
    if dataset == "router-eval":
        from src import cost_proxy as cp
    else:
        from src import embedllm_cost_proxy as cp
    return {s: np.asarray(cp.build_cost_vector(data, s)[0], dtype=np.float64) for s in SPLITS}


def build_meta_struct(data: dict, dataset: str, top_k: dict[str, int]):
    """Structured metadata features, vocabularies fit on **train** LLMs only."""
    names = {s: [str(n) for n in data["llms"][s]["names"]] for s in SPLITS}
    costs = cost_vectors(dataset, data)
    orgs = {s: [split_org_model(n)[0] for n in names[s]] for s in SPLITS}
    fams = {s: [family_of(n) for n in names[s]] for s in SPLITS}
    moe = {s: np.array([1.0 if _MOE_TOKEN.search(normalize_name(n)) else 0.0 for n in names[s]])
           for s in SPLITS}

    vocab, coverage = {}, {}
    for field, values in (("org", orgs), ("family", fams)):
        counter = collections.Counter(values["train"])
        keep = [v for v, _ in counter.most_common(top_k[field])]
        vocab[field] = keep
        coverage[field] = {
            "n_distinct_train": len(counter),
            "top_k": int(top_k[field]),
            "vocab_size_with_unk": len(keep) + 1,
            **{f"{s}_in_vocab_rate": round(float(np.mean([v in set(keep) for v in values[s]])), 6)
               for s in SPLITS},
            **{f"{s}_unk_rate": round(float(np.mean([v not in set(keep) for v in values[s]])), 6)
               for s in SPLITS},
            # The ceiling: what a one-hot over EVERY value seen in training would cover. This
            # is the strongest form of the finding -- the residual UNK rate here is not a
            # consequence of capping K, it is cold-cold itself. Metadata reaches a new model
            # only through categories the training pool already had.
            "saturated_vocab": {
                "size": len(counter),
                **{f"{s}_unk_rate": round(
                    float(np.mean([v not in set(counter) for v in values[s]])), 6)
                   for s in SPLITS},
            },
        }

    def one_hot(values, keep):
        index = {v: i for i, v in enumerate(keep)}
        out = np.zeros((len(values), len(keep) + 1), dtype=np.float64)  # last column = UNK
        for i, v in enumerate(values):
            out[i, index.get(v, len(keep))] = 1.0
        return out

    # The MoE flag is constant on EmbedLLM (0 of 70 training models carry an NxM token), so it
    # would be a zero-variance column. Drop it there rather than feeding the encoder a constant.
    include_moe = bool(moe["train"].std() > 0)

    log_size = {s: np.log10(np.maximum(costs[s], 1e-6)) for s in SPLITS}
    mu, sd = float(log_size["train"].mean()), float(log_size["train"].std())
    sd = sd if sd > 0 else 1.0

    feats, field_names = {}, []
    for s in SPLITS:
        cols = [((log_size[s] - mu) / sd).reshape(-1, 1)]
        if s == "train":
            field_names.append("log10_params_b_z")
        if include_moe:
            cols.append(moe[s].reshape(-1, 1))
            if s == "train":
                field_names.append("moe_flag")
        cols.append(one_hot(fams[s], vocab["family"]))
        cols.append(one_hot(orgs[s], vocab["org"]))
        if s == "train":
            field_names += [f"family={v}" for v in vocab["family"]] + ["family=UNK"]
            field_names += [f"org={v}" for v in vocab["org"]] + ["org=UNK"]
        feats[s] = np.hstack(cols).astype(np.float32)

    audit = {
        "fields": field_names,
        "dim": int(feats["train"].shape[1]),
        "log10_params_b_z": {"train_mean": round(mu, 6), "train_std": round(sd, 6),
                              "note": "z-scored on train LLMs"},
        "moe_flag": {"included": include_moe,
                      "train_positives": int(moe["train"].sum()),
                      "note": None if include_moe else
                              "dropped: zero variance on this dataset's train pool"},
        "one_hot_scaling": (
            "Indicator columns are left as {0, 1} rather than z-scored. The request specifies "
            "z-scoring all of meta_struct; z-scoring a one-hot whose train frequency is ~0.003 "
            "(the 50th most common RouterEval org) produces inputs of magnitude ~17 into an "
            "encoder architecture selected on unit-scale 2048-d fingerprints, which would "
            "confound the representation comparison with an input-scaling artifact. The "
            "continuous size feature is z-scored as specified."),
        "coverage": coverage,
        "size_coverage": {s: {"n": int(costs[s].size),
                               "min_b": float(costs[s].min()),
                               "median_b": float(np.median(costs[s])),
                               "max_b": float(costs[s].max())} for s in SPLITS},
    }
    return feats, audit


def build_meta_name(data: dict, device: str | None, batch_size: int):
    """Embed the `org__model` string with the fingerprints' own encoder, truncated to 2048-d."""
    from src.fingerprints.encode import EncodeConfig, _build_default_encoder
    config = EncodeConfig(embed_dim=MODEL_ID_DIM, batch_size=batch_size, device=device)
    encoder = _build_default_encoder(config)
    feats = {}
    for s in SPLITS:
        texts = [normalize_name(n) for n in data["llms"][s]["names"]]
        emb = encoder.encode(texts, batch_size=config.batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        feats[s] = np.asarray(emb, dtype=np.float32)
        print(f"  {s}: encoded {len(texts)} model IDs -> {feats[s].shape}")
    audit = {"embed_model": config.embed_model, "truncate_dim": MODEL_ID_DIM,
             "normalized_embeddings": True,
             "dim": int(feats["train"].shape[1]),
             "example_inputs": [normalize_name(n) for n in data["llms"]["train"]["names"][:5]]}
    return feats, audit


def assert_schema_preserved(source: dict, out: dict, feats: dict):
    """Only `llms[split]["embeddings"]` may differ. Everything else is asserted identical."""
    assert set(source.keys()) == set(out.keys()), "top-level keys changed"
    for s in SPLITS:
        src_l, out_l = source["llms"][s], out["llms"][s]
        assert set(src_l.keys()) == set(out_l.keys()), f"llms[{s}] keys changed"
        for key in src_l:
            if key == "embeddings":
                continue
            assert np.array_equal(np.asarray(src_l[key]), np.asarray(out_l[key])), \
                f"llms[{s}][{key}] changed -- row order or membership is not preserved"
        assert out_l["embeddings"].shape[0] == len(src_l["names"]), \
            f"llms[{s}] row count {out_l['embeddings'].shape[0]} != {len(src_l['names'])}"
        assert out_l["embeddings"].shape[0] == feats[s].shape[0]
        assert np.isfinite(out_l["embeddings"]).all(), f"llms[{s}] embeddings not finite"
        for qk in ("embeddings", "task_ids", "original_indices"):
            if qk in source["queries"][s]:
                assert np.array_equal(np.asarray(source["queries"][s][qk]),
                                       np.asarray(out["queries"][s][qk])), f"queries[{s}][{qk}] changed"
    for qs in SPLITS:
        for ls in SPLITS:
            assert np.array_equal(source["scores"][qs][ls], out["scores"][qs][ls]), \
                f"scores[{qs}][{ls}] changed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["router-eval", "embedllm"])
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--source", default=None, help="defaults to paths.dataset_path(--dataset)")
    p.add_argument("--out-root", default=None,
                   help="defaults to data/<dataset>/processed/metadata_variants")
    p.add_argument("--top-k-org", type=int, default=None)
    p.add_argument("--top-k-family", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    source = Path(args.source) if args.source else dataset_path(args.dataset)
    out_root = (Path(args.out_root) if args.out_root
                else dataset_processed(args.dataset) / "metadata_variants")
    out_dir = out_root / args.variant
    if not source.exists():
        sys.exit(f"Missing source bundle {source} -- run the tier-B pipeline for "
                  f"{args.dataset} first.")
    print(f"read: {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}")
    with open(source, "rb") as fh:
        data = pickle.load(fh)

    top_k = dict(DEFAULT_TOP_K[args.dataset])
    if args.top_k_org is not None:
        top_k["org"] = args.top_k_org
    if args.top_k_family is not None:
        top_k["family"] = args.top_k_family

    fp = {s: np.asarray(data["llms"][s]["embeddings"], dtype=np.float32) for s in SPLITS}
    print(f"source fingerprint dims: " + "  ".join(f"{s}={fp[s].shape}" for s in SPLITS))

    if args.variant == "meta_name":
        if args.dry_run:
            print(f"[dry-run] would encode {sum(len(data['llms'][s]['names']) for s in SPLITS)} "
                  f"model IDs to {MODEL_ID_DIM}-d with the fingerprint encoder")
            return
        feats, audit = build_meta_name(data, args.device, args.batch_size)
    else:
        feats, audit = build_meta_struct(data, args.dataset, top_k)
        if args.variant == "concat":
            feats = {s: np.hstack([fp[s], feats[s]]).astype(np.float32) for s in SPLITS}
            audit = {"components": ["fingerprint", "meta_struct"],
                     "fingerprint_dim": int(fp["train"].shape[1]),
                     "meta_struct_dim": int(audit["dim"]),
                     "dim": int(feats["train"].shape[1]),
                     "meta_struct": audit}

    print(f"variant={args.variant} LLM-side dims: " + "  ".join(f"{s}={feats[s].shape}" for s in SPLITS))
    if args.variant != "meta_name":
        cov = audit["coverage"] if args.variant == "meta_struct" else audit["meta_struct"]["coverage"]
        for field, c in cov.items():
            print(f"  {field}: {c['n_distinct_train']} distinct in train, top-{c['top_k']} "
                  f"covers train={c['train_in_vocab_rate']:.3f} test={c['test_in_vocab_rate']:.3f} "
                  f"(test UNK rate {c['test_unk_rate']:.3f})")

    if args.dry_run:
        print("[dry-run] no bundle written.")
        return

    out = dict(data)
    out["llms"] = {s: dict(data["llms"][s]) for s in SPLITS}
    for s in SPLITS:
        out["llms"][s]["embeddings"] = feats[s]
    out["metadata"] = dict(data.get("metadata", {}))
    out["metadata"].update({
        "llm_representation_variant": args.variant,
        "llm_representation_dim": int(feats["train"].shape[1]),
        "source_dataset_path": str(source),
        "variant_builder": "src/data_pipeline/metadata/build_metadata_bundle.py",
    })
    assert_schema_preserved(data, out, feats)
    print("schema guard: names / indices / task_ids / scores / column order all preserved")

    _atomic_pickle(out, out_dir / "llm_perf_dataset.pkl")
    _atomic_write_json(out_dir / "metadata_fields.json",
                        {"dataset": args.dataset, "variant": args.variant, **audit})
    if args.variant in ("meta_struct", "concat"):
        cov = audit["coverage"] if args.variant == "meta_struct" else audit["meta_struct"]["coverage"]
        _atomic_write_json(out_dir / "metadata_coverage.json", {
            "dataset": args.dataset, "variant": args.variant,
            "dim": int(feats["train"].shape[1]),
            "n_llms": {s: int(feats[s].shape[0]) for s in SPLITS},
            "coverage": cov,
            "size_coverage": (audit if args.variant == "meta_struct"
                              else audit["meta_struct"])["size_coverage"],
            "moe_flag": (audit if args.variant == "meta_struct" else audit["meta_struct"])["moe_flag"],
        })
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

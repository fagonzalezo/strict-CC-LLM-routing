#!/usr/bin/env python3
"""Encode router-eval query/LLM embeddings with a frozen MLPDot two-tower encoder.

Refits `MLPDotRouter` at exp_32's cost-aligned-selected architecture
(`target_dim, hidden_dim, n_layers`, read from
`experiments/router-eval/exp_32/results/selection_audit.json`) with
`val_objective="bce"` (MLPDotRouter's default warm-start convention), extracts the
frozen encoders `f_q`/`f_l`, and encodes every split's
*raw* query/LLM embeddings through them (`.eval()` mode -- the encoders contain
BatchNorm1d, so eval uses the trained running stats, not per-batch stats).

Writes a drop-in replacement pickle -- same schema as `llm_perf_dataset.pkl`,
only `["embeddings"]` swapped per split -- so it plugs into `src.eval.load_splits`
unchanged (task_ids, original_indices, names, indices_in_5000, scores all
preserved). Modeled directly on `refine_embeddings_embedkit.py`'s bundle-build
pattern (see that script for the EmbedKit sibling).

Also persists `mlpdot_test_predictions.npy` (the fitted MLPDot's raw-embedding
test-set prediction matrix) alongside the bundle, so downstream sweep scripts that
need an MLPDot reference for a paired bootstrap can load it directly instead of
redundantly refitting MLPDot (deterministic given seed+arch+data) in every one of
their own per-config tasks.

One encoder seed per invocation (exp_37 sweeps seed in {0, 1, 2}); the MLPDot
architecture is fixed (exp_32's selection), only the warm-start seed varies.

All paths are derived from this file's own location -- runnable from any cwd.

Usage:
    python encode_embeddings_mlpdot.py --seed 0
    python encode_embeddings_mlpdot.py --seed 1 --device cuda:0
    python encode_embeddings_mlpdot.py --seed 0 --dry-run

    --source PATH        Input llm_perf_dataset.pkl (default: processed/llm_perf_dataset.pkl)
    --out-root PATH       Output root dir; bundle written to <out-root>/mlpdot_encoded_seed{seed}/
                          (default: processed/)
    --reports-dir PATH   Output directory for the markdown report (default: reports/)
    --mlpdot-audit PATH  exp_32 selection_audit.json (arch source)
    --seed INT           MLPDot warm-start encoder seed (required)
    --device STR         torch device string, e.g. 'cuda:0' or 'cpu' (default: auto)
    --out-pkl-name STR   Filename for output pickle (default: llm_perf_dataset_mlpdot_encoded.pkl)
    --verbose            Print MLPDot per-epoch training progress
    --dry-run            Print planned shapes/arch without training and exit
"""
import argparse
import copy
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path constants (all relative to project root, derived from __file__)
# ---------------------------------------------------------------------------
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import DATA, ROOT, dataset_processed, results_dir  # noqa: E402
PROJECT_ROOT = ROOT

from src import metrics as M
from src import cost_proxy
from src.data_loader import load_perf_dataset
from src.routers.mlp_dot import MLPDotRouter

DEFAULT_SOURCE = dataset_processed("router-eval") / "llm_perf_dataset.pkl"
DEFAULT_OUT_ROOT = dataset_processed("router-eval")
DEFAULT_REPORTS_DIR = DATA / "router-eval" / "reports"
DEFAULT_MLPDOT_AUDIT = results_dir("router-eval", "mlpdot-baseline") / "selection_audit.json"

SPLITS = ("train", "val", "test")

# MLPDot's fixed training HP -- matches exp_32's warm-start refit, INCLUDING
# val_objective="bce" (MLPDotRouter's own default).
MLPDOT_FIXED_KWARGS = dict(
    dropout=0.1, normalize_output=True,
    temperature_init=10.0, learn_temperature=True,
    lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=256,
    val_objective="bce",
)

# exp_32's official BCE-refit reference row (pooled test), used only as a soft
# consistency guard when --seed matches exp_32's own selected seed.
EXP32_OFFICIAL = {"test_cost_auc": 0.719871, "test_ece": 0.020498}
EXP32_GUARD_TOL = 0.01


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    p.add_argument("--mlpdot-audit", default=str(DEFAULT_MLPDOT_AUDIT))
    p.add_argument("--seed", type=int, required=True, help="MLPDot warm-start encoder seed")
    p.add_argument("--device", default=None)
    p.add_argument("--out-pkl-name", default="llm_perf_dataset_mlpdot_encoded.pkl")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_mlpdot_selected_arch(audit_path: Path) -> dict:
    """Read exp_32's selected (target_dim, hidden_dim, n_layers, seed)."""
    if not audit_path.exists():
        sys.exit(f"Missing {audit_path} -- run exp_32's run_mlp_dot_pooled_eval.py first.")
    audit = json.loads(audit_path.read_text())
    sel = audit["selected"]
    return {"target_dim": sel["target_dim"], "hidden_dim": sel["hidden_dim"],
            "n_layers": sel["n_layers"], "exp32_selected_seed": sel["seed"]}


def _refit_mlpdot(Q_train, L_train, S_train, Q_val, L_val, S_val,
                   arch: dict, seed: int, device: str | None, verbose: bool = False) -> MLPDotRouter:
    """Refit MLPDot at exp_32's selected architecture, this task's warm-start seed."""
    print(f"  refitting MLPDot warm-start encoder (val_objective=bce): "
          f"td={arch['target_dim']} hd={arch['hidden_dim']} nl={arch['n_layers']} seed={seed}")
    router = MLPDotRouter(
        target_dim=arch["target_dim"], hidden_dim=arch["hidden_dim"],
        n_layers=arch["n_layers"], seed=seed,
        device=device, verbose=verbose, **MLPDOT_FIXED_KWARGS,
    ).fit(Q_train, L_train, S_train.astype(np.float32),
          Q_val, L_val, S_val.astype(np.float32))
    return router


def _encode_split(f_q, f_l, q_emb: np.ndarray, l_emb: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Encode raw (query, llm) embeddings through frozen, eval-mode encoders."""
    import torch
    f_q.eval()
    f_l.eval()
    with torch.no_grad():
        z_q = f_q(torch.from_numpy(q_emb.astype(np.float32)).to(device)).cpu().numpy().astype(np.float32)
        z_l = f_l(torch.from_numpy(l_emb.astype(np.float32)).to(device)).cpu().numpy().astype(np.float32)
    return z_q, z_l


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    source_path = Path(args.source).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    mlpdot_audit = Path(args.mlpdot_audit).resolve()
    out_dir = Path(args.out_root).resolve() / f"mlpdot_encoded_seed{args.seed}"

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source       : {source_path}")
    print(f"Out dir      : {out_dir}")
    print(f"Reports dir  : {reports_dir}")
    print(f"MLPDot audit : {mlpdot_audit}")
    print(f"Seed         : {args.seed}")
    print(f"Device       : {args.device or 'auto'}")

    if not source_path.exists():
        sys.exit(f"Missing source file: {source_path}")

    print("\n[1/5] Loading source dataset...")
    dataset = load_pkl(source_path)
    q_emb = {s: dataset["queries"][s]["embeddings"] for s in SPLITS}
    l_emb = {s: dataset["llms"][s]["embeddings"] for s in SPLITS}
    print(f"  Queries — train:{q_emb['train'].shape} val:{q_emb['val'].shape} test:{q_emb['test'].shape}")
    print(f"  LLMs    — train:{l_emb['train'].shape} val:{l_emb['val'].shape} test:{l_emb['test'].shape}")

    arch = _load_mlpdot_selected_arch(mlpdot_audit)
    print(f"  MLPDot arch (exp_32-selected): target_dim={arch['target_dim']} "
          f"hidden_dim={arch['hidden_dim']} n_layers={arch['n_layers']} "
          f"(exp_32's own selected seed: {arch['exp32_selected_seed']})")

    if args.dry_run:
        print("\nDry run complete — no training performed.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Refit MLPDot warm-start encoder at this task's seed
    # ------------------------------------------------------------------
    print("\n[2/5] Refitting MLPDot warm-start encoder...")
    mlpdot_router = _refit_mlpdot(
        q_emb["train"], l_emb["train"], dataset["scores"]["train"]["train"],
        q_emb["val"], l_emb["val"], dataset["scores"]["val"]["val"],
        arch, args.seed, args.device, verbose=args.verbose,
    )
    torch_device = mlpdot_router._device

    # ------------------------------------------------------------------
    # Reference metrics + consistency guard (soft, only meaningful at exp_32's own seed)
    # ------------------------------------------------------------------
    P_test_mlpdot = mlpdot_router.predict(q_emb["test"], l_emb["test"])
    S_test = dataset["scores"]["test"]["test"].astype(np.float32)
    c_test, _ = cost_proxy.build_cost_vector(dataset, "test")
    test_cost_auc = float(M.cost_metrics(P_test_mlpdot, S_test, c_test)["scalars"]["cost_auc"])
    test_ece = float(M.expected_calibration_error(P_test_mlpdot, S_test))
    test_pq_auc = float(M.per_query_auc(P_test_mlpdot, S_test))
    print(f"  MLPDot reference (seed={args.seed}): best_epoch={mlpdot_router.best_epoch_}  "
          f"val_bce={mlpdot_router.val_bce_:.5f}  "
          f"test_cost_auc={test_cost_auc:.6f}  test_pq_auc={test_pq_auc:.6f}  test_ece={test_ece:.6f}")

    if args.seed == arch["exp32_selected_seed"]:
        d_cost = abs(test_cost_auc - EXP32_OFFICIAL["test_cost_auc"])
        d_ece = abs(test_ece - EXP32_OFFICIAL["test_ece"])
        status = "PASS" if (d_cost < EXP32_GUARD_TOL and d_ece < EXP32_GUARD_TOL) else "WARN"
        print(f"  [consistency guard] seed={args.seed} matches exp_32's selected seed -- {status}: "
              f"test_cost_auc Δ={d_cost:.6f} (tol {EXP32_GUARD_TOL}), "
              f"test_ece Δ={d_ece:.6f} (tol {EXP32_GUARD_TOL}), "
              f"vs. exp_32 official cost_auc={EXP32_OFFICIAL['test_cost_auc']} ece={EXP32_OFFICIAL['test_ece']}")

    # ------------------------------------------------------------------
    # Encode every split through the frozen, eval-mode encoders
    # ------------------------------------------------------------------
    print("\n[3/5] Encoding QUERY and LLM embeddings...")
    f_q = copy.deepcopy(mlpdot_router._model.f_q).to(torch_device).eval()
    f_l = copy.deepcopy(mlpdot_router._model.f_l).to(torch_device).eval()

    encoded_q: dict[str, np.ndarray] = {}
    encoded_l: dict[str, np.ndarray] = {}
    for split in SPLITS:
        zq, zl = _encode_split(f_q, f_l, q_emb[split], l_emb[split], torch_device)
        assert np.isfinite(zq).all(), f"non-finite query encoding in split={split}"
        assert np.isfinite(zl).all(), f"non-finite llm encoding in split={split}"
        encoded_q[split], encoded_l[split] = zq, zl
        print(f"  [{split}] queries {q_emb[split].shape} -> {zq.shape}   "
              f"llms {l_emb[split].shape} -> {zl.shape}   "
              f"|z_q| mean={np.linalg.norm(zq, axis=1).mean():.4f}  "
              f"|z_l| mean={np.linalg.norm(zl, axis=1).mean():.4f}")

    # ------------------------------------------------------------------
    # Assemble output dataset (drop-in replacement, EmbedKit bundle pattern)
    # ------------------------------------------------------------------
    print("\n[4/5] Assembling encoded dataset bundle...")

    queries_out = {}
    for split in SPLITS:
        queries_out[split] = dict(dataset["queries"][split])  # copy task_ids, original_indices
        queries_out[split]["embeddings"] = encoded_q[split]

    llms_out = {}
    for split in SPLITS:
        llms_out[split] = dict(dataset["llms"][split])  # copy names, indices_in_5000
        llms_out[split]["embeddings"] = encoded_l[split]

    meta = dict(dataset["metadata"])
    meta.update({
        "encoded_with": "mlpdot",
        "encoder_arch": {k: v for k, v in arch.items() if k != "exp32_selected_seed"},
        "encoder_seed": args.seed,
        "encoder_val_objective": "bce",
        "encoder_best_epoch": mlpdot_router.best_epoch_,
        "encoder_val_bce": round(float(mlpdot_router.val_bce_), 6),
        "query_encoded_dim": int(encoded_q["train"].shape[1]),
        "llm_encoded_dim": int(encoded_l["train"].shape[1]),
        "source_dataset_path": str(source_path.relative_to(PROJECT_ROOT)),
        "encoded_git_commit": _git_commit(),
        "encoded_utc": datetime.now(timezone.utc).isoformat(),
        "mlpdot_reference_test_cost_auc": round(test_cost_auc, 6),
        "mlpdot_reference_test_pq_auc": round(test_pq_auc, 6),
        "mlpdot_reference_test_ece": round(test_ece, 6),
    })

    encoded_dataset = {
        "queries": queries_out,
        "llms": llms_out,
        "scores": dataset["scores"],  # scores block is unchanged
        "metadata": meta,
    }

    out_pkl = out_dir / args.out_pkl_name
    with open(out_pkl, "wb") as f:
        pickle.dump(encoded_dataset, f)
    print(f"  Encoded dataset saved → {out_pkl}")

    # Persist the MLPDot test-prediction matrix alongside the bundle. Downstream sweep scripts
    # (e.g. exp_37's run_dmkdmr_encoded_eval.py) need this as an in-task bootstrap reference;
    # without it, every one of the sweep's per-config tasks would otherwise have to redundantly
    # refit MLPDot from raw embeddings just to reproduce this same deterministic prediction.
    pred_path = out_dir / "mlpdot_test_predictions.npy"
    np.save(pred_path, P_test_mlpdot.astype(np.float32))
    print(f"  MLPDot test predictions saved → {pred_path}")

    # ------------------------------------------------------------------
    # Save encoders + reference json
    # ------------------------------------------------------------------
    import torch
    for space, module in (("queries", f_q), ("llms", f_l)):
        enc_dir = out_dir / f"encoder_{space}"
        enc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(module.state_dict(), enc_dir / "model.pt")
        (enc_dir / "config.json").write_text(json.dumps(
            {**{k: v for k, v in arch.items() if k != "exp32_selected_seed"},
             "seed": args.seed, "space": space, "source": "MLPDotRouter._model"},
            indent=2))
        print(f"  [{space}] encoder saved → {enc_dir}")

    reference = {
        "seed": args.seed, "arch": {k: v for k, v in arch.items() if k != "exp32_selected_seed"},
        "val_objective": "bce",
        "best_epoch": mlpdot_router.best_epoch_,
        "val_bce": round(float(mlpdot_router.val_bce_), 6),
        "test_cost_auc": round(test_cost_auc, 6),
        "test_pq_auc": round(test_pq_auc, 6),
        "test_ece": round(test_ece, 6),
    }
    (out_dir / "mlpdot_reference.json").write_text(json.dumps(reference, indent=2))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n[5/5] Writing report...")
    _write_report(
        reports_dir / f"mlpdot_encoding_seed{args.seed}_report.md",
        seed=args.seed, arch=arch, meta=meta, reference=reference,
        q_shapes={"train": q_emb["train"].shape, "encoded_train": encoded_q["train"].shape},
        l_shapes={"train": l_emb["train"].shape, "encoded_train": encoded_l["train"].shape},
        out_pkl=out_pkl,
    )
    print("\nDone.")


def _write_report(path: Path, seed: int, arch: dict, meta: dict, reference: dict,
                   q_shapes: dict, l_shapes: dict, out_pkl: Path):
    report = f"""\
# MLPDot Embedding Encoding — seed {seed}

Generated: {meta['encoded_utc']}
Git commit: `{meta['encoded_git_commit']}`
Source: `{meta['source_dataset_path']}`

## Overview

Frozen MLPDot two-tower encoders (`f_q`, `f_l`), refit in-task at exp_32's
cost-aligned-selected architecture with `val_objective="bce"`, applied to encode
the query embedding space (RoBERTa 768-d) and the LLM embedding space (Qwen3-MRL
2048-d) into a shared, L2-normalised `target_dim`-d space. The encoder was fit on
train(+val, per MLPDotRouter's own protocol); all three splits are encoded via the
frozen, eval-mode forward pass (no leakage — BatchNorm running stats are fixed
from training, not recomputed per split).

## Architecture

| Parameter | Value |
|-----------|-------|
| target_dim | {arch['target_dim']} |
| hidden_dim | {arch['hidden_dim']} |
| n_layers | {arch['n_layers']} |
| warm-start seed | {seed} |
| val_objective | bce |
| best_epoch | {reference['best_epoch']} |
| val_bce | {reference['val_bce']} |

## Shape changes

| Space | Original dim | Encoded dim | Train samples |
|-------|-------------|------------|---------------|
| Queries | {q_shapes['train'][1]} | {q_shapes['encoded_train'][1]} | {q_shapes['train'][0]:,} |
| LLMs | {l_shapes['train'][1]} | {l_shapes['encoded_train'][1]} | {l_shapes['train'][0]:,} |

## MLPDot reference (pooled test, this seed's encoder)

| Metric | Value |
|--------|-------|
| test_cost_auc | {reference['test_cost_auc']} |
| test_pq_auc | {reference['test_pq_auc']} |
| test_ece | {reference['test_ece']} |

exp_32 official (seed={arch['exp32_selected_seed']}): test_cost_auc=0.719871,
test_pq_auc=0.764832, test_ece=0.020498. This seed's numbers should closely match
when `seed == {arch['exp32_selected_seed']}`; otherwise they characterize
encoder-seed variance.

## Output files

| Artifact | Path |
|----------|------|
| Encoded dataset | `data/router-eval/processed/mlpdot_encoded_seed{seed}/{out_pkl.name}` |
| Query encoder | `data/router-eval/processed/mlpdot_encoded_seed{seed}/encoder_queries/` |
| LLM encoder | `data/router-eval/processed/mlpdot_encoded_seed{seed}/encoder_llms/` |
| Reference metrics | `data/router-eval/processed/mlpdot_encoded_seed{seed}/mlpdot_reference.json` |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(f"  Report → {path}")


if __name__ == "__main__":
    main()

"""
run_coldrec.py — AGNN / FSGNN baselines for strict doubly-cold-start routing.

Bypasses ColdRec's main.py and CSV pipeline entirely.  Builds ColdStartDataBuilder
directly from llm_perf_dataset.pkl, trains the chosen model, extracts embeddings,
and writes routing metrics to:

  experiments/coldrec-baselines/results/<dataset>/<model>/config<N>/seed<S>/

cold_object='both': train queries × train LLMs → training,
                    val   queries × val   LLMs → cold_val,
                    test  queries × test  LLMs → cold_test.

Source-ID convention (train-first, monotone order so source_id == dense_id):
  Users (queries): 0..nq_tr-1 | nq_tr..nq_tr+nq_va-1 | nq_tr+nq_va..N_q-1
  Items (LLMs):    0..nl_tr-1 | nl_tr..nl_tr+nl_va-1  | nl_tr+nl_va..N_l-1

To guarantee source_id == dense_id for all entities, each split includes a
"registration" phase: min(nq, nl) diagonal pairs (k, k+offset, 0.0) plus
one-sided extras.  These add negligible BPR noise (order of min(nq,nl) pairs
vs ~millions of real positives in the full run).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import SRC, dataset_path, dataset_processed, results_dir  # noqa: E402

# coldrec's own modules import each other by bare name (upstream convention),
# so its package dir goes on the path too.
sys.path.insert(0, str(SRC / "baselines" / "coldrec"))

import src.metrics as M
from src.data_loader import load_perf_dataset

# ---------------------------------------------------------------------------
# HP grids
# ---------------------------------------------------------------------------
AGNN_GRID = [
    {"emb_size": e, "lr": lr, "agnn_knn_k": k, "agnn_dropout": d}
    for e in [32, 64]
    for lr in [1e-4, 5e-4]
    for k in [5, 10, 20]
    for d in [0.3, 0.5]
]  # 24 configs

FSGNN_GRID = [
    {"emb_size": e, "lr": lr, "fsgnn_lambda_fc": lfc, "fsgnn_mu_sc": msc}
    for e in [32, 64]
    for lr in [1e-3, 5e-3]
    for lfc in [0.1, 0.5]
    for msc in [0.1, 0.5]
]  # 16 configs

GRIDS = {"AGNN": AGNN_GRID, "FSGNN": FSGNN_GRID}

DATASET_PKL = {
    "embedllm":    dataset_path("embedllm"),
    "router-eval": dataset_path("router-eval"),
}

MAX_TRAIN_INTERACTIONS = 5_000_000

# Benchmark groups (embedllm — 10 groups, matching exp_13/14/17)
BENCHMARKS = {
    "ASDiv":      lambda t: t == "asdiv",
    "GPQA":       lambda t: t.startswith("gpqa_"),
    "GSM8K":      lambda t: t == "gsm8k",
    "LogiQA":     lambda t: t == "logiqa",
    "MathQA":     lambda t: t == "mathqa",
    "MedMCQA":    lambda t: t == "medmcqa",
    "MMLU":       lambda t: t.startswith("mmlu_"),
    "PIQA":       lambda t: t == "piqa",
    "SocialQA":   lambda t: t == "social_iqa",
    "TruthfulQA": lambda t: t == "truthfulqa_mc1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def _registration_pairs(nq: int, nl: int, u_off: int = 0, i_off: int = 0):
    """
    Return min(nq, nl) + extras dummy 0.0-rated pairs that guarantee all
    nq users and nl items appear in the data in ID order.
    """
    pairs = []
    for k in range(min(nq, nl)):
        pairs.append([u_off + k, i_off + k, 0.0])
    for u in range(nl, nq):
        pairs.append([u_off + u, i_off, 0.0])
    for i in range(nq, nl):
        pairs.append([u_off, i_off + i, 0.0])
    return pairs


def _pos_pairs(score_mat, nq: int, nl: int, u_off: int = 0, i_off: int = 0):
    u_ids, i_ids = np.where(score_mat[:nq, :nl] == 1)
    return [[int(u + u_off), int(i + i_off), 1.0] for u, i in zip(u_ids, i_ids)]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_coldrec_data(ds: dict, dataset_name: str, smoke: bool = False, seed: int = 0):
    """
    Construct ColdStartDataBuilder and attach all attributes expected by AGNN/FSGNN.
    """
    from util.databuilder import ColdStartDataBuilder

    Q = ds["queries"]
    L = ds["llms"]
    S = ds["scores"]

    nq_tr = len(Q["train"]["embeddings"])
    nq_va = len(Q["val"]["embeddings"])
    nq_te = len(Q["test"]["embeddings"])
    nl_tr = len(L["train"]["embeddings"])
    nl_va = len(L["val"]["embeddings"])
    nl_te = len(L["test"]["embeddings"])

    if smoke:
        nq_tr = min(nq_tr, 200)
        nq_va = min(nq_va, 50)
        nq_te = min(nq_te, 50)
        nl_tr = min(nl_tr, 15)
        nl_va = min(nl_va, 8)
        nl_te = min(nl_te, 8)

    q_off_va = nq_tr
    q_off_te = nq_tr + nq_va
    l_off_va = nl_tr
    l_off_te = nl_tr + nl_va
    N_q = nq_tr + nq_va + nq_te
    N_l = nl_tr + nl_va + nl_te

    # --- Training data: registration + positives
    train_reg = _registration_pairs(nq_tr, nl_tr)
    train_pos = _pos_pairs(S["train"]["train"], nq_tr, nl_tr)
    if dataset_name == "router-eval" and len(train_pos) > MAX_TRAIN_INTERACTIONS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_pos), MAX_TRAIN_INTERACTIONS, replace=False)
        train_pos = [train_pos[i] for i in sorted(idx)]
        print(f"  [router-eval] capped train positives to {len(train_pos):,}")
    training_data = train_reg + train_pos

    # --- Val data: registration + positives (both cold_valid and overall_valid)
    va_reg = _registration_pairs(nq_va, nl_va, q_off_va, l_off_va)
    va_pos = _pos_pairs(S["val"]["val"], nq_va, nl_va, q_off_va, l_off_va)
    cold_va = va_reg + va_pos

    # --- Test data: registration + positives
    te_reg = _registration_pairs(nq_te, nl_te, q_off_te, l_off_te)
    te_pos = _pos_pairs(S["test"]["test"], nq_te, nl_te, q_off_te, l_off_te)
    cold_te = te_reg + te_pos

    # --- Content arrays (row k = content for source ID k)
    q_content = np.concatenate([
        Q["train"]["embeddings"][:nq_tr].astype(np.float32),
        Q["val"]["embeddings"][:nq_va].astype(np.float32),
        Q["test"]["embeddings"][:nq_te].astype(np.float32),
    ], axis=0)
    l_content = np.concatenate([
        L["train"]["embeddings"][:nl_tr].astype(np.float32),
        L["val"]["embeddings"][:nl_va].astype(np.float32),
        L["test"]["embeddings"][:nl_te].astype(np.float32),
    ], axis=0)

    builder = ColdStartDataBuilder(
        training_data=training_data,
        warm_valid_data=[],
        cold_valid_data=cold_va,
        overall_valid_data=cold_va,   # cold_both: overall == cold
        warm_test_data=[],
        cold_test_data=cold_te,
        overall_test_data=cold_te,
        user_num=N_q,
        item_num=N_l,
        warm_user_idx=list(range(nq_tr)),
        warm_item_idx=list(range(nl_tr)),
        cold_user_idx=list(range(nq_tr, N_q)),
        cold_item_idx=list(range(nl_tr, N_l)),
        user_content=q_content,
        item_content=l_content,
    )

    # Verify dense ID assignment is monotone (sanity check from registration phase)
    for k in range(min(nq_tr, nl_tr)):
        assert builder.user.get(k) == k, f"user source_id {k} != dense_id {builder.user.get(k)}"
        assert builder.item.get(k) == k, f"item source_id {k} != dense_id {builder.item.get(k)}"

    # --- Attach attributes expected by AGNN/FSGNN (not set by ColdStartDataBuilder)

    # Content aligned by dense ID (user_content/item_content from ColdStartDataBuilder._align_content)
    builder.mapped_user_content = builder.user_content   # [N_q, feat_dim]
    builder.mapped_item_content = builder.item_content   # [N_l, feat_dim]

    # Dense IDs of warm/cold entities  (source_id == dense_id by construction)
    builder.mapped_warm_user_idx = np.array([builder.user[s] for s in range(nq_tr)          if s in builder.user], dtype=np.int64)
    builder.mapped_warm_item_idx = np.array([builder.item[s] for s in range(nl_tr)          if s in builder.item], dtype=np.int64)
    builder.mapped_cold_user_idx = np.array([builder.user[s] for s in range(nq_tr, N_q)     if s in builder.user], dtype=np.int64)
    builder.mapped_cold_item_idx = np.array([builder.item[s] for s in range(nl_tr, N_l)     if s in builder.item], dtype=np.int64)

    # Evaluation sets for fast_evaluation / BaseRecommender
    builder.warm_valid_set = {}
    builder.cold_valid_set    = dict(builder.valid_set_u)
    builder.overall_valid_set = dict(builder.valid_set_u)
    builder.warm_test_set = {}
    builder.cold_test_set    = dict(builder.test_set_u)
    builder.overall_test_set = dict(builder.test_set_u)

    # Reverse mapping dense_id → source_id
    builder.id2item = {v: k for k, v in builder.item.items()}
    builder.id2user = {v: k for k, v in builder.user.items()}

    # Lookup methods
    builder.get_user_id_list = lambda users: [builder.user.get(u, 0) for u in users]
    builder.get_item_id_list = lambda items: [builder.item.get(i, 0) for i in items]

    # Sparse interaction matrix [N_q, N_l] needed by FSGNN
    builder.interaction_mat = builder.get_adj_mat()

    # FSGNN needs these two methods on the data object
    def _create_bipartite(self_connection=False):
        R = builder.interaction_mat.astype(np.float32)
        n_u, n_i = R.shape
        z_uu = sp.csr_matrix((n_u, n_u), dtype=np.float32)
        z_ii = sp.csr_matrix((n_i, n_i), dtype=np.float32)
        adj = sp.bmat([[z_uu, R], [R.T, z_ii]], format="csr")
        if self_connection:
            adj = adj + sp.eye(n_u + n_i, format="csr")
        return adj

    def _normalize_graph(adj_mat):
        adj_mat = adj_mat.tocsr().astype(np.float32)
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.zeros_like(rowsum, dtype=np.float32)
        np.power(rowsum, -0.5, out=d_inv, where=rowsum != 0)
        d_mat = sp.diags(d_inv)
        return (d_mat.dot(adj_mat)).dot(d_mat).tocsr()

    builder.create_sparse_complete_bipartite_adjacency = _create_bipartite
    builder.normalize_graph_mat = _normalize_graph

    sizes = {
        "nq_tr": nq_tr, "nq_va": nq_va, "nq_te": nq_te,
        "nl_tr": nl_tr, "nl_va": nl_va, "nl_te": nl_te,
        "q_off_va": q_off_va, "q_off_te": q_off_te,
        "l_off_va": l_off_va, "l_off_te": l_off_te,
        "N_q": N_q, "N_l": N_l,
    }
    return builder, sizes


# ---------------------------------------------------------------------------
# Args namespace
# ---------------------------------------------------------------------------

def build_args(model: str, dataset: str, hp: dict, seed: int, use_gpu: bool):
    """Build argparse.Namespace with all attributes required by AGNN / FSGNN."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    base = dict(
        model=model,
        dataset=dataset,
        cold_object="both",
        topN="10,20",
        seed=seed,
        use_gpu=use_gpu,
        gpu_id=0,
        save_emb=False,
        # AGNN
        agnn_no_backbone=True,
        agnn_finetune=False,
        agnn_rank_weight=1.0,
        agnn_align_weight=1.0,
        agnn_vae_lambda=1.0,
        agnn_eval_chunk=512,
        agnn_knn_k=10,
        agnn_dropout=0.3,
        # FSGNN
        fsgnn_fallback_dim=32,
        fsgnn_gat_hidden=64,
        fsgnn_p_drop=0.3,
        fsgnn_dropout_cold_side_only=True,
        fsgnn_lfc_cold_side_only=True,
        fsgnn_fc_decoder_layers=2,
        fsgnn_ppr_delta=0.15,
        fsgnn_ppr_iter=30,
        fsgnn_ppr_topk=15,
        fsgnn_ppr_weighted=True,
        fsgnn_knn_k=15,
        fsgnn_knn_weighted=True,
        fsgnn_sc_layers=2,
        fsgnn_id_residual=False,
        fsgnn_light_layers=2,
        fsgnn_sc_max_edges=4096,
        fsgnn_weight_decay=5e-4,
        fsgnn_lambda_fc=0.1,
        fsgnn_mu_sc=0.1,
    )
    if model == "AGNN":
        base.update(dict(epochs=20, bs=4096, lr=1e-4, reg=1e-4, early_stop=5, eval_every=1))
    else:  # FSGNN
        base.update(dict(epochs=200, bs=4096, lr=1e-3, reg=5e-4, early_stop=20, eval_every=5))
    if dataset == "router-eval":
        base["fsgnn_ppr_iter"] = 5
        base["fsgnn_ppr_topk"] = 5
    base.update(hp)
    args = types.SimpleNamespace(**base)
    args.device = device
    return args


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_scores(trainer, builder, sizes, split: str):
    """
    Return S_hat [nq, nl] float32 for a given split ("va" or "te").
    Entities not in builder.user/builder.item get zero embedding → sigmoid(0)=0.5 score.
    """
    q_off = sizes[f"q_off_{split}"]
    l_off = sizes[f"l_off_{split}"]
    nq    = sizes[f"nq_{split}"]
    nl    = sizes[f"nl_{split}"]

    emb_dim = trainer.user_emb.shape[1]
    u_emb = np.zeros((nq, emb_dim), dtype=np.float32)
    i_emb = np.zeros((nl, emb_dim), dtype=np.float32)

    for j, src in enumerate(range(q_off, q_off + nq)):
        d = builder.user.get(src)
        if d is not None:
            u_emb[j] = trainer.user_emb[d].cpu().float().numpy()

    for j, src in enumerate(range(l_off, l_off + nl)):
        d = builder.item.get(src)
        if d is not None:
            i_emb[j] = trainer.item_emb[d].cpu().float().numpy()

    logits = (u_emb @ i_emb.T).astype(np.float64)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["AGNN", "FSGNN"], required=True)
    parser.add_argument("--dataset",    choices=["embedllm", "router-eval"], required=True)
    parser.add_argument("--config-idx", type=int, required=True)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--smoke",      action="store_true")
    cli = parser.parse_args()

    grid = GRIDS[cli.model]
    if cli.config_idx >= len(grid):
        raise ValueError(f"--config-idx {cli.config_idx} out of range (grid size={len(grid)})")
    hp = grid[cli.config_idx]

    out_dir = (results_dir("coldrec-baselines")
               / cli.dataset / cli.model
               / f"config{cli.config_idx}" / f"seed{cli.seed}")
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists():
        print(f"Already exists, skipping: {metrics_path}")
        return

    print(f"[run_coldrec] model={cli.model} dataset={cli.dataset} "
          f"config={cli.config_idx} seed={cli.seed} smoke={cli.smoke}")
    print(f"  hp={hp}")

    # Seed
    from util.utils import set_seed
    use_gpu = torch.cuda.is_available()
    set_seed(cli.seed, cuda=use_gpu)

    # Load raw dataset
    ds = load_perf_dataset(DATASET_PKL[cli.dataset])

    # Cost vectors
    if cli.dataset == "embedllm":
        from src.embedllm_cost_proxy import build_cost_vector as _build_cost
        c_te_full, _ = _build_cost(ds, "test")
        c_va_full, _ = _build_cost(ds, "val")
    else:
        from src.cost_proxy import build_cost_vector as _build_cost
        pp = dataset_processed("router-eval") / "model_partition.pkl"
        c_te_full, _ = _build_cost(ds, "test", pp)
        c_va_full, _ = _build_cost(ds, "val", pp)

    # Build ColdStartDataBuilder
    builder, sizes = build_coldrec_data(
        ds, cli.dataset, smoke=cli.smoke, seed=cli.seed
    )
    nq_tr, nl_tr = sizes["nq_tr"], sizes["nl_tr"]
    n_reg = max(nq_tr, nl_tr)
    print(f"  data: {builder.user_num} users, {builder.item_num} items, "
          f"{len(builder.training_data):,} training entries (incl. ~{n_reg} reg pairs)")

    # Args + ColdRec Config object
    args = build_args(cli.model, cli.dataset, hp, cli.seed, use_gpu)

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.args   = args
    cfg.data   = builder
    cfg.device = args.device

    # Instantiate and train
    if cli.model == "AGNN":
        from model.AGNN import AGNN
        trainer = AGNN(cfg)
    else:
        from model.FSGNN import FSGNN
        trainer = FSGNN(cfg)

    trainer.train()

    # Guard: ensure a valid checkpoint was saved
    if not (hasattr(trainer, "best_user_emb") and trainer.best_user_emb is not None):
        print("WARNING: no valid checkpoint saved (all validation metrics non-finite)")
        _atomic_write(metrics_path, {
            "error": "no_valid_checkpoint",
            "model": cli.model, "dataset": cli.dataset,
            "config_idx": cli.config_idx, "seed": cli.seed, "hp": hp,
        })
        return

    # --- Extract scores
    nq_te = sizes["nq_te"]
    nl_te = sizes["nl_te"]
    nq_va = sizes["nq_va"]
    nl_va = sizes["nl_va"]

    c_te = c_te_full[:nl_te]
    c_va = c_va_full[:nl_va]

    S_hat_te = extract_scores(trainer, builder, sizes, "te")
    S_hat_va = extract_scores(trainer, builder, sizes, "va")

    S_true_te = ds["scores"]["test"]["test"][:nq_te, :nl_te].astype(np.float32)
    S_true_va = ds["scores"]["val"]["val"][:nq_va, :nl_va].astype(np.float32)
    S_train   = ds["scores"]["train"]["train"][:sizes["nq_tr"], :sizes["nl_tr"]].astype(np.float32)

    # --- Compute routing metrics (exp_16/17 pattern)
    rank  = M.ranking_metrics(S_hat_te, S_true_te)
    cal   = M.calibration_metrics(S_hat_te, S_true_te)
    cost_out = M.cost_metrics(S_hat_te, S_true_te, c_te)
    reliab   = M.reliability_curve(S_hat_te, S_true_te)
    val_cap  = M.per_query_cap_curve(S_hat_va, S_true_va, c_va)
    cost_out["curves"]["router_envelope"] = M.cap_router_envelope(
        cost_out["curves"]["router"], val_cap["router"], c_te
    )

    gm = np.full_like(S_hat_te, float(S_train.mean()))
    gm_rank = M.ranking_metrics(gm, S_true_te)

    row = {
        "model":      cli.model,
        "dataset":    cli.dataset,
        "config_idx": cli.config_idx,
        "seed":       cli.seed,
        "hp":         hp,
        "group":      "pooled",
        **{f"test_{k}": round(v, 6) for k, v in rank.items()},
        **{f"test_{k}": round(v, 6) for k, v in cal.items()},
        **{f"test_{k}": round(v, 6) for k, v in cost_out["scalars"].items()},
        **{f"gm_{k}":   round(v, 6) for k, v in gm_rank.items()},
    }

    # --- Per-benchmark metrics (embedllm only; other datasets have no task_ids)
    bench_rows: list[dict] = []
    all_curves = {"pooled": cost_out["curves"]}
    all_rel    = {"pooled": reliab}

    task_ids_te = ds["queries"]["test"].get("task_ids")
    task_ids_va = ds["queries"]["val"].get("task_ids")
    if task_ids_te is not None:
        task_ids_te = np.asarray(task_ids_te[:nq_te])
        task_ids_va = np.asarray(task_ids_va[:nq_va]) if task_ids_va is not None else None
        gm_scalar = float(S_train.mean())

        for bname, pred in BENCHMARKS.items():
            mask_te = np.array([pred(t) for t in task_ids_te], dtype=bool)
            n_te_b  = int(mask_te.sum())
            if n_te_b == 0:
                continue

            S_hat_b  = S_hat_te[mask_te]
            S_true_b = S_true_te[mask_te]

            rank_b = M.ranking_metrics(S_hat_b, S_true_b)
            cal_b  = M.calibration_metrics(S_hat_b, S_true_b)
            cost_b = M.cost_metrics(S_hat_b, S_true_b, c_te)
            rel_b  = M.reliability_curve(S_hat_b, S_true_b)

            if task_ids_va is not None:
                mask_va = np.array([pred(t) for t in task_ids_va], dtype=bool)
                if mask_va.sum() > 0:
                    val_cap_b = M.per_query_cap_curve(
                        S_hat_va[mask_va], S_true_va[mask_va], c_va
                    )
                    cost_b["curves"]["router_envelope"] = M.cap_router_envelope(
                        cost_b["curves"]["router"], val_cap_b["router"], c_te
                    )

            gm_b      = np.full_like(S_hat_b, gm_scalar)
            gm_rank_b = M.ranking_metrics(gm_b, S_true_b)

            bench_rows.append({
                "model":      cli.model,
                "dataset":    cli.dataset,
                "config_idx": cli.config_idx,
                "seed":       cli.seed,
                "group":      bname,
                "Nq_test":    n_te_b,
                **{f"test_{k}": round(v, 6) for k, v in rank_b.items()},
                **{f"test_{k}": round(v, 6) for k, v in cal_b.items()},
                **{f"test_{k}": round(v, 6) for k, v in cost_b["scalars"].items()},
                **{f"gm_{k}":   round(v, 6) for k, v in gm_rank_b.items()},
            })
            all_curves[bname] = cost_b["curves"]
            all_rel[bname]    = rel_b

    _atomic_write(metrics_path, row)
    _atomic_write(out_dir / "cost_curves.json", all_curves)
    _atomic_write(out_dir / "reliability.json", all_rel)
    if bench_rows:
        _atomic_write(out_dir / "per_benchmark_metrics.json", bench_rows)

    print(f"  pq_auc={row.get('test_pq_auc', float('nan')):.4f}  "
          f"top1={row.get('test_top1_acc', float('nan')):.4f}  "
          f"cost_auc={row.get('test_cost_auc', float('nan')):.4f}")
    print(f"  written → {metrics_path}")


if __name__ == "__main__":
    main()

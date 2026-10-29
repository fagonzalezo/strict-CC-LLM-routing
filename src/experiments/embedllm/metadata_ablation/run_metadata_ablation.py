"""
Metadata-representation ablation (EmbedLLM): one (arm, seed) fit per task.

Answers Part E objection 3 -- *"you claim fingerprints beat metadata and never test metadata."*
The LLM-side representation is swapped at the dataset-bundle level, upstream of everything
else, so no router code changes and nothing below the encoder differs between arms:

    <variant>/llm_perf_dataset.pkl
      +-> encode_embeddings_mlpdot.py --source ...   MLPDot two-tower encoder, seeds {0,1,2}
            +-> <variant>/mlpdot_encoded_seed{S}/llm_perf_dataset_mlpdot_encoded.pkl
                  +-> this script: BF-DKR + kNN + MLPDot, all unchanged

Arms: `fp` (2048-d behavioral fingerprint, the reference), `meta_struct`, `meta_name`,
`concat`. Three routers per arm, because one metadata row invites "you only crippled your own
method" and a metadata *block* does not:

  BF-DKR     k swept over {128, 512, 2048, 4096} and val-selected per (arm, seed).
             **k is selected per arm, not inherited from `fp`** -- `fp`'s k was chosen for
             fingerprint inputs and reusing it would bias the comparison toward `fp`.
  kNN        `knn_enc` on the encoded variant (the `bfdkr-ablation` arm's protocol) is the
             reported row; `knn_raw` on the raw variant bundle is a sub-row. Raw is kept
             because the request asks for it, and flagged: `KNNRouter` uses a cosine metric,
             which is a poor geometry for a 23-d mostly-indicator `meta_struct` -- see
             DESIGN.md section 9.
  MLPDot     free: `encode_embeddings_mlpdot.py` already persisted this variant's
             `mlpdot_test_predictions.npy`, so it is scored, not refit.

The `fp` arm loads `experiments/embedllm/static-policy-baseline/results/bfdkr_ref/seed{S}/
P_test.npy` when present so BF-DKR is fit once across both experiments, and refits at the same
val-selected config when it is absent (that matrix is gitignored, so a fresh checkout has
none). Either way it is checked against the committed reproduction cells.

Bootstraps and the hypothesis verdicts are `build_comparison.py`'s job -- it reads the
per-arm `P_test.npy` matrices this script writes, so pairing needs only a shared test-query
order (which the bundle guarantees) rather than fitting every arm in one process.

Usage
-----
  python run_metadata_ablation.py --config-idx 0 --smoke --device cpu
  python run_metadata_ablation.py --config-idx 0 --device cuda
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_path, dataset_processed, results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc
from src.routers.dmkdmr import DMKDMRRouter
from src.routers.knn_baseline import predict_from_neighbors, predict_sweep

DATASET = "embedllm"
FAMILY = "embedllm"
EXPERIMENT = "metadata-ablation"

RESULTS_DIR = results_dir(FAMILY, EXPERIMENT)
VARIANT_ROOT = dataset_processed(DATASET) / "metadata_variants"
BFDKR_REF_DIR = results_dir(FAMILY, "static-policy-baseline") / "bfdkr_ref"
SWEEP_RESULTS = results_dir(FAMILY, "bfdkr-sweep")
SEED_SUMMARY = results_dir(FAMILY, "all-methods-comparison") / "seed_summary.json"

ARMS = ["fp", "meta_struct", "meta_name", "concat"]
SEEDS = [0, 1, 2]
OBJECTIVE = "pq_auc"
K_CANDIDATES = [128, 512, 2048, 4096]
CONFIGS = [(arm, seed) for arm in ARMS for seed in SEEDS]

# kNN grid, trimmed from the knn-baseline's full sweep: this experiment compares
# representations, not kNN hyperparameters, and a coarse grid selected on val is enough to
# place the row. The full grid lives in src/experiments/*/knn_baseline/.
K_Q_CANDIDATES = [1, 4, 16, 64, 256, 1024, 4096]
K_L_CANDIDATES = [1, 2, 4, 8, 16, 32, 70]

GATE_LABEL = "BF-DKR (pq_auc)"
GATE_PROTOCOL = "per-seed val-selected k"
TIE_BAND_COST_AUC = 0.0035


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pooled(rows):
    return rows if isinstance(rows, dict) else next(r for r in rows if r["group"] == "pooled")


def _bundle_paths(arm: str, seed: int) -> tuple[Path, Path]:
    """(raw bundle, encoded bundle) for this arm."""
    if arm == "fp":
        raw = dataset_path(DATASET)
        enc = dataset_processed(DATASET) / f"mlpdot_encoded_seed{seed}" / "llm_perf_dataset_mlpdot_encoded.pkl"
    else:
        raw = VARIANT_ROOT / arm / "llm_perf_dataset.pkl"
        enc = VARIANT_ROOT / arm / f"mlpdot_encoded_seed{seed}" / "llm_perf_dataset_mlpdot_encoded.pkl"
    return raw, enc


def _fp_gate(seed: int) -> dict:
    """The committed cell the `fp` arm must reproduce (per-seed val-selected k, not fixed k)."""
    summary = json.loads(SEED_SUMMARY.read_text())
    print(f"read: {SEED_SUMMARY.relative_to(ROOT)}")
    method = next(m for m in summary["methods"] if m["label"] == GATE_LABEL)
    if method["protocol"] != GATE_PROTOCOL:
        sys.exit(f"{GATE_LABEL} protocol is {method['protocol']!r}, expected {GATE_PROTOCOL!r}")
    row = next(r for r in method["per_seed"] if r["seed"] == seed)
    return {"source": f"seed_summary.json -> methods[{GATE_LABEL!r}] -> per_seed[{seed}]",
            "expected_config": row["config"],
            "expected_test_cost_auc": row["test_cost_auc"],
            "floor_cost_auc": method["floor_cost_auc"],
            "oracle_cost_auc": method["oracle_cost_auc"]}


def _sweep_k(Q_tr, L_tr, S_tr, Q_va, L_va, S_va, c_val, seed, device, k_grid):
    """Fit DMKDMR at each k and select on val `cost_auc` (ties by min val `brier`).

    The Table III protocol, applied *per arm*: `select_by_cost_auc` over this arm's own grid.
    """
    fits = []
    for k in k_grid:
        t0 = time.time()
        r = DMKDMRRouter(k=k, objective=OBJECTIVE, seed=seed, device=device,
                          verbose=False).fit(Q_tr, L_tr, S_tr, Q_va, L_va, S_va)
        P_val = r.predict(Q_va, L_va)
        val_ca = float(M.cost_metrics(P_val, S_va, c_val)["scalars"]["cost_auc"])
        val_br = float(M.brier_score(P_val, S_va))
        fits.append({"k": k, "router": r, "val_cost_auc": val_ca, "val_brier": val_br,
                      "val_pq_auc": float(M.per_query_auc(P_val, S_va)),
                      "fit_elapsed_s": round(time.time() - t0, 2)})
        print(f"    k={k:5d}  val_cost_auc={val_ca:.6f}  val_brier={val_br:.6f}  "
              f"({fits[-1]['fit_elapsed_s']:.0f}s)")
    idx = select_by_cost_auc([f["val_cost_auc"] for f in fits], [f["val_brier"] for f in fits])
    return fits[idx], [{kk: vv for kk, vv in f.items() if kk != "router"} for f in fits]


def _knn_row(Q_tr, L_tr, S_tr, Q_va, L_va, S_va, c_val, Q_te, L_te):
    """kNN with (k_q, k_l) selected on val `cost_auc`; returns (P_test, audit)."""
    k_q_grid = [k for k in K_Q_CANDIDATES if k <= Q_tr.shape[0]] or [Q_tr.shape[0]]
    k_l_grid = [k for k in K_L_CANDIDATES if k <= L_tr.shape[0]] or [L_tr.shape[0]]
    nn_q = NearestNeighbors(n_neighbors=max(k_q_grid), metric="cosine", algorithm="brute").fit(Q_tr)
    nn_l = NearestNeighbors(n_neighbors=max(k_l_grid), metric="cosine", algorithm="brute").fit(L_tr)
    va_q, va_l = nn_q.kneighbors(Q_va, return_distance=False), nn_l.kneighbors(L_va, return_distance=False)
    val_ca, val_br, combos = [], [], []
    for k_q, k_l, S_hat in predict_sweep(S_tr, va_q, va_l, k_q_grid, k_l_grid):
        val_ca.append(float(M.cost_metrics(S_hat, S_va, c_val)["scalars"]["cost_auc"]))
        val_br.append(float(M.brier_score(S_hat, S_va)))
        combos.append((k_q, k_l))
    best = combos[select_by_cost_auc(val_ca, val_br)]
    te_q, te_l = nn_q.kneighbors(Q_te, return_distance=False), nn_l.kneighbors(L_te, return_distance=False)
    P = predict_from_neighbors(S_tr, te_q[:, :best[0]], te_l[:, :best[1]])
    audit = {"k_q": int(best[0]), "k_l": int(best[1]),
             "val_cost_auc": round(val_ca[combos.index(best)], 6),
             "grid_size": len(combos)}
    return P.astype(np.float32), audit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-idx", type=int, required=True, choices=range(len(CONFIGS)))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    arm, seed = CONFIGS[args.config_idx]
    out_dir = Path(args.out_dir) / f"seed{seed}" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Metadata ablation ({DATASET}) arm={arm} seed={seed} | smoke={args.smoke} | "
          f"device={args.device} | out={out_dir}")

    raw_path, enc_path = _bundle_paths(arm, seed)
    for path, what in ((raw_path, "raw variant bundle"), (enc_path, "encoded variant bundle")):
        if not path.exists():
            sys.exit(f"Missing {what} {path} -- run "
                      "src/data_pipeline/metadata/build_metadata_bundle.py and then "
                      "encode_embeddings_mlpdot.py --source <that bundle> first.")

    base_spec = get_spec(DATASET)
    enc_spec = dataclasses.replace(base_spec, name=f"{DATASET}-{arm}-enc-seed{seed}",
                                    dataset_path=enc_path)
    raw_spec = dataclasses.replace(base_spec, name=f"{DATASET}-{arm}-raw", dataset_path=raw_path)

    sp = load_splits(enc_spec)
    Q_tr, L_tr, S_tr = sp.Q_train, sp.L_train, sp.S_train.astype(np.float32)
    Q_va, L_va, S_va = sp.Q_val, sp.L_val, sp.S_val.astype(np.float32)
    Q_te, L_te, S_te = sp.Q_test, sp.L_test, sp.S_test.astype(np.float32)
    c_val, c_test = sp.c_val, sp.c_test
    print(f"  encoded dims: train_q {Q_tr.shape} train_l {L_tr.shape} test_l {L_te.shape}")

    k_grid = [k for k in K_CANDIDATES if k <= Q_tr.shape[0]] or [Q_tr.shape[0]]
    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Q_tr.shape[0], size=min(300, Q_tr.shape[0]), replace=False)
        Q_tr, S_tr = Q_tr[qi], S_tr[qi]
        qv = rng.choice(Q_va.shape[0], size=min(200, Q_va.shape[0]), replace=False)
        Q_va, S_va = Q_va[qv], S_va[qv]
        qt = rng.choice(Q_te.shape[0], size=min(200, Q_te.shape[0]), replace=False)
        Q_te, S_te = Q_te[qt], S_te[qt]
        k_grid = [min(k_grid[0], Q_tr.shape[0])]
        print(f"  [smoke] train={Q_tr.shape[0]}q test={Q_te.shape[0]}q k_grid={k_grid}")

    # ── BF-DKR ────────────────────────────────────────────────────────────────
    reused = False
    ref_pred = BFDKR_REF_DIR / f"seed{seed}" / "P_test.npy"
    if arm == "fp" and not args.smoke and ref_pred.exists():
        print(f"read: {ref_pred.relative_to(ROOT)}  (reusing stage-1 BF-DKR fit)")
        P_bfdkr = np.load(ref_pred).astype(np.float32)
        audit_path = BFDKR_REF_DIR / f"seed{seed}" / "config_audit.json"
        ref_audit = json.loads(audit_path.read_text())
        best = {"k": ref_audit["config"]["k"],
                 "val_cost_auc": ref_audit["config"]["val_cost_auc"],
                 "val_brier": ref_audit["config"]["val_brier"],
                 "val_pq_auc": ref_audit["config"]["val_pq_auc"],
                 "fit_elapsed_s": ref_audit["config"]["fit_elapsed_s"]}
        k_audit = ref_audit["k_selection"]["grid"]
        sigmas = {"sigma_x": ref_audit["config"]["sigma_x"], "sigma_y": ref_audit["config"]["sigma_y"]}
        reused = True
    else:
        print(f"  BF-DKR k sweep ({len(k_grid)} fits), val-selected per arm:")
        best, k_audit = _sweep_k(Q_tr, L_tr, S_tr, Q_va, L_va, S_va, c_val, seed,
                                  args.device, k_grid)
        P_bfdkr = best["router"].predict(Q_te, L_te).astype(np.float32)
        sigmas = {"sigma_x": round(float(best["router"].sigma_x_), 6),
                   "sigma_y": round(float(best["router"].sigma_y_), 6)}
        best = {kk: vv for kk, vv in best.items() if kk != "router"}
    bf_cost = M.cost_metrics(P_bfdkr, S_te, c_test)["scalars"]
    print(f"  BF-DKR  k={best['k']}  test_cost_auc={bf_cost['cost_auc']:.6f}  "
          f"aiq={bf_cost['aiq']:+.6f}" + ("  [reused stage-1 fit]" if reused else ""))

    # `fp` must reproduce the committed cell -- the gate that makes every other arm's gap mean
    # something. Checked whether the matrix was reused or refit.
    gate = None
    if arm == "fp" and not args.smoke:
        gate = _fp_gate(seed)
        drift = float(bf_cost["cost_auc"] - gate["expected_test_cost_auc"])
        gate.update(seed=seed, observed_test_cost_auc=round(float(bf_cost["cost_auc"]), 6),
                     derived_k=best["k"], reused_stage1_fit=reused,
                     drift_cost_auc=round(drift, 6), tie_band_cost_auc=TIE_BAND_COST_AUC,
                     passed=bool(abs(drift) <= TIE_BAND_COST_AUC))
        print(f"  GATE fp/seed{seed}: drift={drift:+.6f} -> {'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            sys.exit(f"fp reproduction gate FAILED (|drift| {abs(drift):.6f} > {TIE_BAND_COST_AUC})")

    # ── MLPDot: already persisted by the encode step, scored not refit ────────
    mlp_pred = enc_path.parent / "mlpdot_test_predictions.npy"
    P_mlpdot = np.load(mlp_pred).astype(np.float32)
    print(f"read: {mlp_pred.relative_to(ROOT)}")
    if args.smoke:
        P_mlpdot = P_mlpdot[qt]
    mlp_cost = M.cost_metrics(P_mlpdot, S_te, c_test)["scalars"]
    print(f"  MLPDot  test_cost_auc={mlp_cost['cost_auc']:.6f}")

    # ── kNN: encoded (reported row) + raw (sub-row) ───────────────────────────
    P_knn_enc, knn_enc_audit = _knn_row(Q_tr, L_tr, S_tr, Q_va, L_va, S_va, c_val, Q_te, L_te)
    knn_enc_cost = M.cost_metrics(P_knn_enc, S_te, c_test)["scalars"]
    print(f"  kNN-enc test_cost_auc={knn_enc_cost['cost_auc']:.6f}  "
          f"(k_q={knn_enc_audit['k_q']}, k_l={knn_enc_audit['k_l']})")

    raw_sp = load_splits(raw_spec)
    rQ_tr, rL_tr, rS_tr = raw_sp.Q_train, raw_sp.L_train, raw_sp.S_train.astype(np.float32)
    rQ_va, rL_va, rS_va = raw_sp.Q_val, raw_sp.L_val, raw_sp.S_val.astype(np.float32)
    rQ_te, rL_te = raw_sp.Q_test, raw_sp.L_test
    if args.smoke:
        rQ_tr, rS_tr, rQ_va, rS_va, rQ_te = rQ_tr[qi], rS_tr[qi], rQ_va[qv], rS_va[qv], rQ_te[qt]
    P_knn_raw, knn_raw_audit = _knn_row(rQ_tr, rL_tr, rS_tr, rQ_va, rL_va, rS_va,
                                         raw_sp.c_val, rQ_te, rL_te)
    knn_raw_cost = M.cost_metrics(P_knn_raw, S_te, c_test)["scalars"]
    knn_raw_audit["caveat"] = (
        "cosine metric on the raw variant representation; for meta_struct that is a 23-d "
        "mostly-indicator vector, a geometry cosine kNN is not well suited to -- reported as "
        "a sub-row, not the kNN row (see DESIGN.md section 9)")
    print(f"  kNN-raw test_cost_auc={knn_raw_cost['cost_auc']:.6f}  "
          f"(k_q={knn_raw_audit['k_q']}, k_l={knn_raw_audit['k_l']}) [sub-row]")

    if args.smoke:
        print(f"[smoke] done (arm={arm}, seed={seed}); no artifacts written.")
        return

    # ── persist ───────────────────────────────────────────────────────────────
    preds = {"bfdkr": P_bfdkr, "mlpdot": P_mlpdot, "knn_enc": P_knn_enc, "knn_raw": P_knn_raw}
    pred_audit = {}
    for name, P in preds.items():
        path = out_dir / f"P_test_{name}.npy"
        np.save(path, P.astype(np.float32))
        pred_audit[name] = {"path": str(path.relative_to(ROOT)), "shape": list(P.shape),
                             "dtype": "float32", "sha256": _sha256(path), "committed": False}

    gm_const = float(S_tr.mean())
    bundle = evaluate_predictions(P_bfdkr, enc_spec, S_test=sp.S_test, c_test=sp.c_test,
                                   test_task_ids=sp.test_task_ids, gm_const=gm_const, out_dir=None)
    for r in bundle.rows:
        if r["group"] == "pooled":
            r.update(condition="metadata_ablation", arm=arm, seed=seed, router="BF-DKR",
                      objective=OBJECTIVE, k=best["k"], k_protocol="per_arm_val_selected",
                      val_cost_auc=best["val_cost_auc"], val_brier=best["val_brier"],
                      llm_repr_dim=int(L_tr.shape[1]), **sigmas)
    io.write_metrics(bundle.rows, out_dir / "metrics.json")
    io.write_curves(bundle.curves, out_dir / "cost_curve.json")
    io.write_reliability(bundle.reliability, out_dir / "reliability.json")
    io.write_cost_report(sp.cost_report_test, out_dir / "cost_coverage.json")

    other = {}
    for name, P, extra in (("MLPDot", P_mlpdot, {}),
                            ("kNN-enc", P_knn_enc, knn_enc_audit),
                            ("kNN-raw", P_knn_raw, knn_raw_audit)):
        b = evaluate_predictions(P, enc_spec, S_test=sp.S_test, c_test=sp.c_test,
                                  test_task_ids=sp.test_task_ids, gm_const=gm_const, out_dir=None)
        row = _pooled(b.rows)
        row.update(condition="metadata_ablation", arm=arm, seed=seed, router=name, **extra)
        other[name] = row
    _atomic_write(out_dir / "other_routers_metrics.json", other)

    _atomic_write(out_dir / "config_audit.json", {
        "condition": "metadata_ablation", "dataset": DATASET, "arm": arm, "seed": seed,
        "objective": OBJECTIVE,
        "k_protocol": "per_arm_val_selected",
        "k_protocol_note": ("k is swept and val-selected within each arm; it is NOT inherited "
                             "from the fp arm, whose k was selected for fingerprint inputs."),
        "k_selection": {"grid": k_audit, "selected_k": best["k"]},
        "llm_representation": {"arm": arm, "raw_dim": int(rL_tr.shape[1]),
                                "encoded_dim": int(L_tr.shape[1])},
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "encoded_bundle": str(enc_path.relative_to(ROOT)),
        "bfdkr_reused_stage1_fit": reused,
        "knn_enc": knn_enc_audit, "knn_raw": knn_raw_audit,
        "predictions": pred_audit,
        "reproduction_gate": gate,
    })
    if gate is not None:
        _atomic_write(Path(args.out_dir) / f"reproduction_gate_seed{seed}.json", gate)

    pooled = _pooled(bundle.rows)
    print(f"\n=== [{arm}/seed{seed}] BF-DKR cost_auc={pooled['test_cost_auc']:.6f} "
          f"aiq={pooled['test_aiq']:.6f} pq_auc={pooled['test_pq_auc']:.6f} ===")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

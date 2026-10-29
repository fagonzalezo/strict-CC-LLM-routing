"""
Static-policy baseline (EmbedLLM), stage 1: replay BF-DKR at its Table-III configuration
and persist the per-query test prediction matrix.

Everything else in this experiment -- S2b (`w = P_test.mean(0)`) and every paired bootstrap
against S1/S2/S3 -- needs BF-DKR's per-query test predictions, and no prior experiment in
this repo persisted one for DMKDMR (`experiments/embedllm/dropoutnet-clcrec-baselines/
IMPLEMENTATION.md`: "No other prior experiment persisted a raw per-query `P_test.npy`"). So
this stage fits it once per encoder seed and saves `P_test.npy` beside the usual A/B/C suite.
`experiments/embedllm/metadata-ablation/`'s `fp` arm reads the same matrix rather than
refitting, so BF-DKR is fit once for both experiments.

`P_test.npy` is deliberately **not** committed: `.gitignore`'s `*.npy` rule covers it (see
commit b63439e, "Untrack regenerable *_test_predictions.npy") and `src/tests/test_layout.py`
asserts git's view. The committed contract is the derived JSON -- `gap_bootstrap.json`,
`s2b_query_marginalized.json`'s `w` vector, and each matrix's sha256 + shape recorded in
`config_audit.json` -- so every number survives without shipping ~100 MB of matrices.

**k protocol.** Both experiment requests lock the *per-seed validation-selected k* protocol:
argmax val `cost_auc`, ties by min val `brier`, via `src/eval/selection.py::select_by_cost_auc`
over the committed sweep grid. That is the protocol behind the abstract's 0.7365 +/- 0.0056,
and it is what `full_reproduction_gate.json` checks against. It is *not* the fixed-k protocol
Table III's 3-seed BF-DKR rows use (`src/paper/gen_tab_main.py` reads a fixed k=2048 for
EmbedLLM) -- see DESIGN.md section 4. Recomputed from the committed artifacts the two protocols
agree well inside the tie-band here (val-selected 0.498651 +/- 0.014621 vs fixed-k 0.499683 +/-
0.017096), so the choice does not change a verdict; it is disclosed because the rows will sit
beside each other in Table III. k is derived here, never hardcoded.

Usage
-----
  python run_bfdkr_reference.py --seed-idx 0 --smoke --device cpu
  python run_bfdkr_reference.py --seed-idx 0 --device cuda
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

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import ROOT, dataset_processed, results_dir  # noqa: E402

from src import metrics as M
from src.eval import evaluate_predictions, get_spec, io, load_splits
from src.eval.selection import select_by_cost_auc
from src.routers.dmkdmr import DMKDMRRouter

DATASET = "embedllm"
FAMILY = "embedllm"
EXPERIMENT = "static-policy-baseline"

RESULTS_DIR = results_dir(FAMILY, EXPERIMENT)
SWEEP_RESULTS = results_dir(FAMILY, "bfdkr-sweep")
SEED_SUMMARY = results_dir(FAMILY, "all-methods-comparison") / "seed_summary.json"
ENCODED_ROOT = dataset_processed(DATASET)

CONDITION = "bfdkr_ref"
OBJECTIVE = "pq_auc"
K_CANDIDATES = [128, 512, 2048, 4096]
SEEDS = [0, 1, 2]

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
    """Normalise a metrics.json (list of group rows) down to its pooled row."""
    if isinstance(rows, dict):
        return rows
    return next(r for r in rows if r["group"] == "pooled")


def _select_k(seed: int) -> tuple[int, dict]:
    """Per-seed val-selected k over the committed sweep grid. Never hardcoded."""
    val_cost_aucs, val_briers, audit = [], [], []
    for k in K_CANDIDATES:
        path = SWEEP_RESULTS / "dmkdmr_encoded" / f"seed{seed}" / OBJECTIVE / f"k{k}" / "metrics.json"
        if not path.exists():
            sys.exit(f"Missing {path} -- the committed bfdkr-sweep grid is the k-selection input.")
        print(f"read: {path.relative_to(ROOT)}")
        pooled = _pooled(json.loads(path.read_text()))
        val_cost_aucs.append(pooled["val_cost_auc"])
        val_briers.append(pooled["val_brier"])
        audit.append({"k": k, "val_cost_auc": pooled["val_cost_auc"],
                      "val_brier": pooled["val_brier"],
                      "committed_test_cost_auc": pooled["test_cost_auc"]})
    idx = select_by_cost_auc(val_cost_aucs, val_briers)
    return K_CANDIDATES[idx], {"protocol": "per_seed_val_selected",
                                "rule": "argmax val_cost_auc, ties by min val_brier",
                                "grid": audit, "selected_k": K_CANDIDATES[idx]}


def _gate_reference(seed: int) -> dict:
    """The exact committed cell this seed must reproduce.

    Named explicitly so the gate cannot silently pass against the fixed-k cells of the same
    arm: `seed_summary.json -> methods[label == "BF-DKR (pq_auc)"] (protocol
    "per-seed val-selected k") -> per_seed[seed]`.
    """
    if not SEED_SUMMARY.exists():
        sys.exit(f"Missing {SEED_SUMMARY} -- the reproduction gate has no reference.")
    print(f"read: {SEED_SUMMARY.relative_to(ROOT)}")
    summary = json.loads(SEED_SUMMARY.read_text())
    method = next((m for m in summary["methods"] if m["label"] == GATE_LABEL), None)
    if method is None:
        sys.exit(f"{SEED_SUMMARY} has no method labelled {GATE_LABEL!r}")
    if method["protocol"] != GATE_PROTOCOL:
        sys.exit(f"{GATE_LABEL} protocol is {method['protocol']!r}, expected {GATE_PROTOCOL!r} "
                  "-- the gate would be checking the wrong k protocol.")
    row = next((r for r in method["per_seed"] if r["seed"] == seed), None)
    if row is None:
        sys.exit(f"{SEED_SUMMARY}: {GATE_LABEL} has no per_seed entry for seed {seed}")
    return {
        "source": f"{SEED_SUMMARY.relative_to(ROOT)} -> methods[label=={GATE_LABEL!r}] "
                   f"(protocol {GATE_PROTOCOL!r}) -> per_seed[seed={seed}]",
        "expected_config": row["config"],
        "expected_test_cost_auc": row["test_cost_auc"],
        "expected_test_pq_auc": row["test_pq_auc"],
        "expected_test_ece": row["test_ece"],
        "floor_cost_auc": method["floor_cost_auc"],
        "oracle_cost_auc": method["oracle_cost_auc"],
        "three_seed_mean_cost_auc": method["mean"]["cost_auc"],
        "three_seed_std_cost_auc": method["std"]["cost_auc"],
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-idx", type=int, required=True, choices=SEEDS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--encoded-root", default=str(ENCODED_ROOT))
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    seed = args.seed_idx
    out_dir = Path(args.out_dir) / CONDITION / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"BF-DKR reference replay ({DATASET}), seed={seed} | smoke={args.smoke} | "
          f"device={args.device} | out={out_dir}")

    k, k_audit = _select_k(seed)
    gate = _gate_reference(seed)
    print(f"  val-selected k = {k}  (committed cell says {gate['expected_config']}, "
          f"test_cost_auc={gate['expected_test_cost_auc']:.6f})")
    if not args.smoke and gate["expected_config"] != f"k={k}":
        sys.exit(f"k-selection disagrees with the committed cell: derived k={k}, "
                  f"committed {gate['expected_config']}. Refusing to run against a "
                  "different protocol.")

    base_spec = get_spec(DATASET)
    bundle_path = (Path(args.encoded_root) / f"mlpdot_encoded_seed{seed}"
                   / "llm_perf_dataset_mlpdot_encoded.pkl")
    if not bundle_path.exists():
        sys.exit(f"Missing encoded bundle {bundle_path} -- run "
                  f"src/data_pipeline/{DATASET.replace('-', '_')}/encode_embeddings_mlpdot.py "
                  f"--seed {seed} first.")
    enc_spec = dataclasses.replace(
        base_spec, name=f"{DATASET}-mlpdot-enc-seed{seed}", dataset_path=bundle_path,
    )

    print("\nLoading encoded splits via src.eval.load_splits ...")
    sp = load_splits(enc_spec)
    Q_tr, L_tr, S_tr = sp.Q_train, sp.L_train, sp.S_train.astype(np.float32)
    Q_va, L_va, S_va = sp.Q_val, sp.L_val, sp.S_val.astype(np.float32)
    Q_te, L_te, S_te = sp.Q_test, sp.L_test, sp.S_test.astype(np.float32)
    print(f"encoded train_q {Q_tr.shape}  train_l {L_tr.shape}  "
          f"val_q {Q_va.shape}  test_q {Q_te.shape}")

    if args.smoke:
        rng = np.random.default_rng(0)
        qi = rng.choice(Q_tr.shape[0], size=min(300, Q_tr.shape[0]), replace=False)
        li = rng.choice(L_tr.shape[0], size=min(100, L_tr.shape[0]), replace=False)
        Q_tr, L_tr, S_tr = Q_tr[qi], L_tr[li], S_tr[np.ix_(qi, li)]
        qv = rng.choice(Q_va.shape[0], size=min(200, Q_va.shape[0]), replace=False)
        lv = rng.choice(L_va.shape[0], size=min(60, L_va.shape[0]), replace=False)
        Q_va, L_va, S_va = Q_va[qv], L_va[lv], S_va[np.ix_(qv, lv)]
        c_val = sp.c_val[lv]
        qt = rng.choice(Q_te.shape[0], size=min(200, Q_te.shape[0]), replace=False)
        lt = rng.choice(L_te.shape[0], size=min(60, L_te.shape[0]), replace=False)
        Q_te, L_te, S_te = Q_te[qt], L_te[lt], S_te[np.ix_(qt, lt)]
        c_test = sp.c_test[lt]
        k = min(k, Q_tr.shape[0])
        print(f"  [smoke] train={Q_tr.shape[0]}q x {L_tr.shape[0]}l  "
              f"test={Q_te.shape[0]}q x {L_te.shape[0]}l  (k capped to {k})")
    else:
        c_val, c_test = sp.c_val, sp.c_test

    t0 = time.time()
    router = DMKDMRRouter(
        k=k, objective=OBJECTIVE, seed=seed, device=args.device, verbose=False,
    ).fit(Q_tr, L_tr, S_tr, Q_va, L_va, S_va)
    fit_s = time.time() - t0

    P_val = router.predict(Q_va, L_va)
    P_test = router.predict(Q_te, L_te)
    val_cost_auc = M.cost_metrics(P_val, S_va, c_val)["scalars"]["cost_auc"]
    val_pq = M.per_query_auc(P_val, S_va)
    val_brier = M.brier_score(P_val, S_va)
    t_cost = M.cost_metrics(P_test, S_te, c_test)["scalars"]
    t_pq = M.per_query_auc(P_test, S_te)
    t_ece = M.expected_calibration_error(P_test, S_te)
    print(f"  fit={fit_s:.1f}s  sigma_x={router.sigma_x_:.5f} sigma_y={router.sigma_y_:.5f}  "
          f"val_cost_auc={val_cost_auc:.4f}  test_cost_auc={t_cost['cost_auc']:.6f}  "
          f"test_pq_auc={t_pq:.4f}  test_ece={t_ece:.4f}")

    if args.smoke:
        print("[smoke] done -- no artifacts written.")
        return

    # --- persist the prediction matrix (untracked; sha256 recorded below) ---
    pred_path = out_dir / "P_test.npy"
    np.save(pred_path, P_test.astype(np.float32))
    pred_sha = _sha256(pred_path)
    print(f"  wrote {pred_path.relative_to(ROOT)}  shape={P_test.shape}  sha256={pred_sha[:16]}...")

    # --- reproduction gate ---
    drift = float(t_cost["cost_auc"] - gate["expected_test_cost_auc"])
    passed = abs(drift) <= TIE_BAND_COST_AUC
    gate_out = dict(gate, seed=seed, derived_k=k, objective=OBJECTIVE,
                     observed_test_cost_auc=round(float(t_cost["cost_auc"]), 6),
                     observed_test_pq_auc=round(float(t_pq), 6),
                     observed_test_ece=round(float(t_ece), 6),
                     drift_cost_auc=round(drift, 6),
                     tie_band_cost_auc=TIE_BAND_COST_AUC,
                     passed=bool(passed))
    _atomic_write(out_dir / "reproduction_gate.json", gate_out)
    print(f"  GATE seed{seed}: observed={t_cost['cost_auc']:.6f} "
          f"expected={gate['expected_test_cost_auc']:.6f} drift={drift:+.6f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(f"Reproduction gate FAILED for seed {seed}: |drift| {abs(drift):.6f} "
                  f"> tie-band {TIE_BAND_COST_AUC}. Refusing to emit a reference this "
                  "experiment's every bootstrap is paired against.")

    # --- pooled + per-benchmark + macro A/B/C suite ---
    gm_const = float(S_tr.mean())
    bundle = evaluate_predictions(
        P_test, enc_spec,
        S_test=sp.S_test, c_test=sp.c_test, test_task_ids=sp.test_task_ids,
        gm_const=gm_const, out_dir=None,
    )
    config_row = {
        "seed": seed, "objective": OBJECTIVE, "k": k,
        "sigma_x": round(float(router.sigma_x_), 6), "sigma_y": round(float(router.sigma_y_), 6),
        "sigma_x_med": round(float(router._sigma_x_med), 6),
        "sigma_y_med": round(float(router._sigma_y_med), 6),
        "val_cost_auc": round(float(val_cost_auc), 6),
        "val_brier": round(float(val_brier), 6),
        "val_pq_auc": round(float(val_pq), 6),
        "fit_elapsed_s": round(fit_s, 2),
    }
    for r in bundle.rows:
        if r["group"] == "pooled":
            r.update(condition=CONDITION, seed=seed, objective=OBJECTIVE, k=k,
                      k_protocol="per_seed_val_selected", **{
                          key: config_row[key] for key in
                          ("sigma_x", "sigma_y", "val_cost_auc", "val_brier", "val_pq_auc")})

    io.write_metrics(bundle.rows, out_dir / "metrics.json")
    io.write_curves(bundle.curves, out_dir / "cost_curve.json")
    io.write_reliability(bundle.reliability, out_dir / "reliability.json")
    io.write_cost_report(sp.cost_report_test, out_dir / "cost_coverage.json")

    _atomic_write(out_dir / "config_audit.json", {
        "condition": CONDITION,
        "dataset": DATASET,
        "seed": seed,
        "k_protocol": "per_seed_val_selected",
        "k_selection": k_audit,
        "encoded_bundle": str(bundle_path.relative_to(ROOT)),
        "config": config_row,
        "P_test": {"path": str(pred_path.relative_to(ROOT)),
                    "shape": list(P_test.shape),
                    "dtype": "float32",
                    "sha256": pred_sha,
                    "committed": False,
                    "note": "gitignored by the *.npy rule; the committed contract is the "
                            "derived JSON in this experiment's results/."},
    })

    pooled = _pooled(bundle.rows)
    print(f"\n=== POOLED [seed{seed}] cost_auc={pooled['test_cost_auc']:.6f} "
          f"aiq={pooled['test_aiq']:.6f} pq_auc={pooled['test_pq_auc']:.6f} "
          f"ece={pooled['test_ece']:.6f} ===")
    print(f"Saved {len(bundle.rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()

"""Derive Table III (`tab:main`) and the Section VI-A prose numbers from committed artifacts.

Every router row of Table III, on both datasets, is read from
`experiments/{router-eval,embedllm}/all-methods-comparison/results/seed_summary.json`, written by
`src/experiments/*/all_methods_comparison/build_seed_summary.py`. BF-DKR is the per-seed
validation-selected-k protocol (the headline 0.7365 / 0.4987), not the fixed-k ablation arm.

Also printed: floor/oracle, floor-to-oracle gap recovered, BF-DKR's margins over the strongest
baselines, the BF-DKR-vs-MLPDot paired bootstrap (fixed-k protocol, per seed), and the
single-seed validation-selected RouterEval-CC configuration (seed 1, k=2048) behind Fig. 3.

Exits non-zero if the AIQ = cost_auc - floor identity fails for any row.

Run: `python src/paper/gen_tab_main.py`.
"""

import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

DATASETS = [("router-eval", "RouterEval-CC"), ("embedllm", "EmbedLLM-CC")]
# Table III row order -> label in seed_summary.json
ROWS = [
    ("kNN", "kNN"),
    ("MLPDot", "MLPDot"),
    ("AGNN", "AGNN"),
    ("FSGNN", "FSGNN"),
    ("CLCRec", "CLCRec"),
    ("DropoutNet", "DropoutNet"),
    ("BF-DKR", "BF-DKR (pq_auc)"),
]


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def pm(mean, std):
    return f"{mean:.4f}" if std is None else f"{mean:.4f}±{std:.4f}"


gate_failures = []
summaries = {}
for family, name in DATASETS:
    print("=" * 78)
    print(f"Table III — {name} (mean ± population s.d. over encoder seeds)")
    print("=" * 78)
    s = load(results_dir(family, "all-methods-comparison") / "seed_summary.json")
    summaries[family] = s
    floor, oracle = s["floor_cost_auc"], s["oracle_cost_auc"]
    by_label = {m["label"]: m for m in s["methods"]}
    print(f"\n  {'Router':12s} {'cost_auc':>15s} {'AIQ':>8s} {'pq_auc':>15s} {'ECE':>15s}  seeds")
    for row, label in ROWS:
        m = by_label[label]
        mean, std = m["mean"], m["std"]
        print(f"  {row:12s} {pm(mean['cost_auc'], std['cost_auc']):>15s} {mean['aiq']:8.4f} "
              f"{pm(mean['pq_auc'], std['pq_auc']):>15s} {pm(mean['ece'], std['ece']):>15s}  "
              f"{m['n_seeds']}")
        if abs((mean["cost_auc"] - floor) - mean["aiq"]) > 1.5e-3:
            gate_failures.append((name, row, mean["cost_auc"] - floor, mean["aiq"]))
    print(f"\n  floor cost_auc = {floor:.4f}   oracle cost_auc = {oracle:.4f}")

    bf = by_label["BF-DKR (pq_auc)"]["mean"]["cost_auc"]
    rec = (bf - floor) / (oracle - floor)
    print(f"  floor-to-oracle gap recovered by BF-DKR: ({bf:.4f}-{floor:.4f})/"
          f"({oracle:.4f}-{floor:.4f}) = {rec:.3f}")
    for other in ("DropoutNet", "MLPDot", "FSGNN", "AGNN"):
        o = by_label[other]["mean"]["cost_auc"]
        print(f"  BF-DKR - {other:10s} = {bf - o:+.4f}")
    print()

print("=" * 78)
print("BF-DKR vs MLPDot, paired bootstrap per seed (RouterEval-CC, fixed-k protocol k=4096)")
print("=" * 78)
for seed in (0, 1, 2):
    g = load(results_dir("router-eval", "bfdkr-ablation") / "full" / f"seed{seed}" / "gap_bootstrap.json")
    lo, hi = g["cost_auc_gap_ci95"]
    print(f"    seed {seed}: Δcost_auc={g['cost_auc_gap_point']:+.4f}  CI95=[{lo:.4f}, {hi:.4f}]  "
          f"excludes zero: {g['cost_auc_gap_excludes_zero']}")

print()
print("=" * 78)
print("Single-seed validation-selected configuration (RouterEval-CC, seed 1, k=2048) — Fig. 3")
print("=" * 78)
cell = results_dir("router-eval", "bfdkr-sweep") / "dmkdmr_encoded" / "seed1" / "pq_auc" / "k2048"
p = next(r for r in load(cell / "metrics.json") if r["group"] == "pooled")
print(f"  cost_auc={p['test_cost_auc']:.4f}  AIQ={p['test_aiq']:.4f}  ECE={p['test_ece']:.4f}  "
      f"APGR={p['test_apgr']:.3f}")
for cap in ("3B", "7B", "30B"):
    print(f"  cap {cap:>3s}: router {p[f'test_quality_at_cap_{cap}_router']:.3f}  "
          f"floor {p[f'test_quality_at_cap_{cap}_floor']:.3f}")
g = load(cell / "gap_bootstrap.json")
print(f"  vs same-seed MLPDot: Δcost_auc={g['cost_auc_gap_point']:+.4f} "
      f"CI95=[{g['cost_auc_gap_ci95'][0]:.4f}, {g['cost_auc_gap_ci95'][1]:.4f}];  "
      f"Δpq_auc={g['pq_auc_gap_point']:+.4f} "
      f"CI95=[{g['pq_auc_gap_ci95'][0]:.4f}, {g['pq_auc_gap_ci95'][1]:.4f}]")

print()
if gate_failures:
    print(f"GATING CHECK FAILED for {len(gate_failures)} row(s):")
    for f in gate_failures:
        print(f"  {f}")
    sys.exit(1)
print("All AIQ = cost_auc - floor checks passed.")

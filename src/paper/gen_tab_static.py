"""Derive the static-policy comparison: Table V (RouterEval-CC) and Table S3 (EmbedLLM-CC).

Arms: S1 largest affordable, S2 learned static quality ranking (ridge on raw fingerprints),
S2b BF-DKR query-marginalized, S3 hindsight-best static policy. S3 uses test labels: it is an
ANCHOR reported beside floor and oracle, never a baseline. The BF-DKR reference is the per-seed
validation-selected k protocol (Table III's headline), refit by this experiment and gated against
`all-methods-comparison/results/seed_summary.json`.

Δ is arm minus BF-DKR, the mean of three same-seed paired-bootstrap gaps; (n/3) counts the seeds
whose 95% CI excludes zero.

Run: `python src/paper/gen_tab_static.py [--dataset {router-eval,embedllm,both}]`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

LABELS = {
    "S1": "S1 largest affordable",
    "S2": "S2 learned quality ranking",
    "S2b": "S2b BF-DKR, query-marginalized",
    "S3": "S3 hindsight-best static (ANCHOR)",
    "BFDKR": "BF-DKR (val-selected k)",
}
TABLE = {"router-eval": "Table V (RouterEval-CC)", "embedllm": "Table S3 (EmbedLLM-CC)"}


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def report(family):
    results = results_dir(family, "static-policy-baseline")
    print("=" * 78)
    print(f"{TABLE[family]}: static-policy comparison, pooled test cost_auc")
    print("=" * 78)
    rows = {r["arm"]: r for r in load(results / "comparison_metrics.json")}
    boot = load(results / "gap_bootstrap.json")["per_seed"]
    seeds = sorted(boot)
    bf = rows["BFDKR"]
    floor, oracle = bf["test_floor_cost_auc"], bf["test_oracle_cost_auc"]

    def gap(arm):
        # stored as BF-DKR minus arm; the table reports arm minus BF-DKR
        gs = [next(v for k, v in boot[s]["gaps"].items() if k.endswith(f"_minus_{arm}")) for s in seeds]
        return (-float(np.mean([g["cost_auc_gap_point"] for g in gs])),
                sum(bool(g["cost_auc_gap_excludes_zero"]) for g in gs),
                [-g["cost_auc_gap_point"] for g in gs])

    def cost(r):
        std = r.get("test_cost_auc_std")
        return f"{r['test_cost_auc']:.4f}" + (f" ± {std:.4f}" if std is not None else "")

    order = sorted(["S1", "S2", "S2b", "S3", "BFDKR"], key=lambda a: rows[a]["test_cost_auc"])
    print(f"\n  {'Arm':34s} {'cost_auc':>17s} {'Δcost_auc':>10s}  excl0   per-seed Δ")
    print(f"  {'floor (random affordable)':34s} {floor:17.4f} {floor - bf['test_cost_auc']:+10.4f}")
    for arm in order:
        if arm == "BFDKR":
            print(f"  {LABELS[arm]:34s} {cost(bf):>17s} {'---':>10s}")
            continue
        d, n, per = gap(arm)
        print(f"  {LABELS[arm]:34s} {cost(rows[arm]):>17s} {d:+10.4f}  {n}/{len(seeds)}   "
              + ", ".join(f"{x:+.4f}" for x in per))
    print(f"  {'oracle (per-query max)':34s} {oracle:17.4f} {oracle - bf['test_cost_auc']:+10.4f}")

    print(f"\n  S2 - S1 = {rows['S2']['test_cost_auc'] - rows['S1']['test_cost_auc']:+.4f}")
    for arm in rows:
        assert abs((rows[arm]["test_cost_auc"] - floor) - rows[arm]["test_aiq"]) < 2e-6, arm
    print("  aiq == cost_auc - floor for every arm: OK")

    print("\n  BF-DKR reference refit vs Table III (reproduction gate, per seed):")
    for s in (0, 1, 2):
        g = load(results / "bfdkr_ref" / f"seed{s}" / "reproduction_gate.json")
        print(f"    seed {s}: expected {g['expected_test_cost_auc']:.6f} observed "
              f"{g['observed_test_cost_auc']:.6f} drift {g['drift_cost_auc']:+.4f} passed={g['passed']}")
    print()


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", choices=["router-eval", "embedllm", "both"], default="both")
args = parser.parse_args()
for family in (["router-eval", "embedllm"] if args.dataset == "both" else [args.dataset]):
    report(family)

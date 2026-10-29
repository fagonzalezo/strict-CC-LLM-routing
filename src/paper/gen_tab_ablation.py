"""Derive the BF-DKR component ablation: Table IV (RouterEval-CC) and Table S4 (EmbedLLM-CC).

Both use the fixed-k protocol (k=4096, pq_auc objective). Δ is the mean of three same-seed
paired-bootstrap deltas against the full router (1000 resamples, shared resample-index matrix per
seed), so it can differ by rounding from the difference of the column means.

Run: `python src/paper/gen_tab_ablation.py [--dataset {router-eval,embedllm,both}]`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

ARMS = {
    "qk_off": ("h1_query_kernel.json", "No query kernel"),
    "lk_off": ("h2_llm_kernel.json", "Hard nearest LLM"),
    "bw_off": ("h3_brent.json", "No bandwidth search"),
    "knn_enc": ("h5_knn.json", "Encoded kNN"),
}
TABLE = {"router-eval": "Table IV (RouterEval-CC)", "embedllm": "Table S4 (EmbedLLM-CC)"}


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def report(family):
    results = results_dir(family, "bfdkr-ablation")
    print("=" * 78)
    print(f"{TABLE[family]}: component ablation, fixed-k protocol, three encoder seeds")
    print("=" * 78)
    att = load(results / "attribution_table.json")
    hyp = {arm: load(results / fname) for arm, (fname, _) in ARMS.items()}

    def cell(arm, key):
        return f"{att[arm][key]['mean']:.4f} ± {att[arm][key]['std']:.4f}"

    print(f"\n  {'Condition':22s} {'cost_auc':>17s} {'Δcost_auc':>10s} {'ECE':>17s} "
          f"{'Δpq_auc':>9s} {'ΔECE':>9s}  excl0  sign-consistent")
    print(f"  {'Full BF-DKR':22s} {cell('full', 'test_cost_auc'):>17s} {'---':>10s} "
          f"{cell('full', 'test_ece'):>17s}")
    for arm, (_, label) in ARMS.items():
        h = hyp[arm]
        deltas = [r["delta_cost_auc"] for r in h["per_seed"]]
        n_excl = sum(bool(r["cost_auc_gap_excludes_zero"]) for r in h["per_seed"])
        consistent = len({np.sign(d) for d in deltas}) == 1
        print(f"  {label:22s} {cell(arm, 'test_cost_auc'):>17s} {np.mean(deltas):+10.4f} "
              f"{cell(arm, 'test_ece'):>17s} {h['mean_delta_pq_auc']:+9.4f} "
              f"{h['mean_delta_ece']:+9.4f}  {n_excl}/{len(deltas)}    {consistent}")

    print("\n  Per-seed paired Δcost_auc (positive = the knockout improved on that seed):")
    print(f"  {'Arm':22s} {'seed 0':>9s} {'seed 1':>9s} {'seed 2':>9s} {'mean':>9s}")
    for arm, (_, label) in ARMS.items():
        deltas = [r["delta_cost_auc"] for r in hyp[arm]["per_seed"]]
        print(f"  {label:22s} " + " ".join(f"{d:+9.4f}" for d in deltas) + f" {np.mean(deltas):+9.4f}")

    full_mean = att["full"]["test_cost_auc"]["mean"]
    bw_mean = att["bw_off"]["test_cost_auc"]["mean"]
    print(f"\n  Rounding note: difference of column means Full - No bandwidth search = "
          f"{full_mean - bw_mean:.4f}; mean of paired Δ = "
          f"{-np.mean([r['delta_cost_auc'] for r in hyp['bw_off']['per_seed']]):.4f}")
    print()


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", choices=["router-eval", "embedllm", "both"], default="both")
args = parser.parse_args()
for family in (["router-eval", "embedllm"] if args.dataset == "both" else [args.dataset]):
    report(family)

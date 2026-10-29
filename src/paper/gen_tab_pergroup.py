"""Derive the appendix's per-group (per-benchmark) results tables and the GSM8K-advantage
check, from the same result files that back Table 2's headline rows — so the appendix cashes
§5's "retained as supplementary material" promise without any new computation.

RouterEval: 3-seed mean per group, from the `full` ablation arm
(router-eval/bfdkr-ablation) — the same arm as
Table 2's RouterEval headline row.
EmbedLLM: 3-seed mean per group, from the `pq_auc`/k=2048 arm
(embedllm/bfdkr-sweep) — the same arm as
Table 2's EmbedLLM headline row (see gen_tab_main.py).

Run: `python src/paper/gen_tab_pergroup.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

RE_ABLATION = results_dir("router-eval", "bfdkr-ablation")
EL_SWEEP = results_dir("embedllm", "bfdkr-sweep")


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def by_group(rows):
    return {r["group"]: r for r in rows}


def agg(rows_per_seed, metric):
    # ddof=0 (population std), matching attribution_table.json's convention
    # (see gen_tab_main.py) so appendix std values are consistent with Table 2/3.
    vals = [r[metric] for r in rows_per_seed]
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


print("=" * 78)
print("RouterEval per-group results — full arm (k=4096, pq_auc), 3 seeds")
print("=" * 78)
re_seeds = [
    by_group(load(RE_ABLATION / f"full/seed{s}/metrics.json"))
    for s in (0, 1, 2)
]
re_groups = [g for g in re_seeds[0].keys() if g not in ("pooled", "macro")]
print(f"{'group':12s} {'Nq':>6s} {'cost_auc':>16s} {'pq_auc':>16s} {'AIQ':>16s}")
for g in ["pooled"] + sorted(re_groups) + ["macro"]:
    rows = [seed[g] for seed in re_seeds]
    ca_m, ca_s = agg(rows, "test_cost_auc")
    pq_m, pq_s = agg(rows, "test_pq_auc")
    aiq_m, aiq_s = agg(rows, "test_aiq")
    nq = rows[0].get("Nq_test", "-")
    print(f"{g:12s} {str(nq):>6s} {ca_m:6.4f}±{ca_s:.4f}   {pq_m:6.4f}±{pq_s:.4f}   {aiq_m:+6.4f}±{aiq_s:.4f}")

print()
print("=" * 78)
print("EmbedLLM per-group results — pq_auc/k=2048 arm, 3 seeds")
print("=" * 78)
el_seeds = [
    by_group(
        load(EL_SWEEP / f"dmkdmr_encoded/seed{s}/pq_auc/k2048/metrics.json")
    )
    for s in (0, 1, 2)
]
el_groups = [g for g in el_seeds[0].keys() if g not in ("pooled", "macro")]
print(f"{'group':12s} {'Nq':>6s} {'cost_auc':>16s} {'pq_auc':>16s} {'AIQ':>16s}")
for g in ["pooled"] + sorted(el_groups) + ["macro"]:
    rows = [seed[g] for seed in el_seeds]
    ca_m, ca_s = agg(rows, "test_cost_auc")
    pq_m, pq_s = agg(rows, "test_pq_auc")
    aiq_m, aiq_s = agg(rows, "test_aiq")
    nq = rows[0].get("Nq_test", "-")
    print(f"{g:12s} {str(nq):>6s} {ca_m:6.4f}±{ca_s:.4f}   {pq_m:6.4f}±{pq_s:.4f}   {aiq_m:+6.4f}±{aiq_s:.4f}")

print()
print("=" * 78)
print("GSM8K-advantage check: RouterEval fingerprints are GSM8K-only (docs/dataset_construction.md §7.3)")
print("Does the GSM8K group show disproportionately higher AIQ than the pooled/other-group average?")
print("=" * 78)
gsm8k_rows = [seed["gsm8k"] for seed in re_seeds]
gsm8k_aiq, _ = agg(gsm8k_rows, "test_aiq")
other_groups = [g for g in re_groups if g != "gsm8k"]
other_aiqs = []
for g in other_groups:
    m, _ = agg([seed[g] for seed in re_seeds], "test_aiq")
    other_aiqs.append(m)
pooled_aiq, _ = agg([seed["pooled"] for seed in re_seeds], "test_aiq")
print(f"  GSM8K AIQ:              {gsm8k_aiq:+.4f}")
print(f"  Other groups AIQ:       {['%+.4f' % x for x in other_aiqs]}")
print(f"  Other groups mean AIQ:  {np.mean(other_aiqs):+.4f}")
print(f"  Pooled AIQ:             {pooled_aiq:+.4f}")
rank = sorted(
    [("gsm8k", gsm8k_aiq)] + list(zip(other_groups, other_aiqs)),
    key=lambda t: -t[1],
)
print(f"  AIQ rank (highest first): {rank}")

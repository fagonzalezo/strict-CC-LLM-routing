"""Derive the LLM-side representation ablation table (`tab:metadata`) from the metadata ablation,
on both datasets: each LLM-side representation crossed with each router tower.

The reference arm `fp` is the per-seed validation-selected k protocol, so its RouterEval BF-DKR
cell equals Table III's headline exactly (reproduction gate drift +0.000000 on all three seeds). On
EmbedLLM the `fp` cell reads 0.4984 against Table III's 0.4987: EmbedLLM's Brent bandwidth search is
not bit-reproducible across runs, and the drift is inside the +/-0.0035 tie band. Neither is the
fixed k=4096 arm of the ablation table.

Note on bolding: on RouterEval `concat` scores numerically higher than `fp` for every tower, so
`fp` is marked as the *reference* arm rather than the best one. On EmbedLLM this does NOT hold:
`concat` is below `fp` for BF-DKR and kNN-enc, so "concat scores highest for every tower" is a
RouterEval-only statement -- see the table caption in the main paper, which bolds nothing.

Run: `python src/paper/gen_tab_metadata.py`.
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

ARMS = ["fp", "meta_struct", "meta_name", "concat"]
TOWERS = ["BF-DKR", "MLPDot", "kNN-enc"]

DATASETS = [("router-eval", "RouterEval-CC"), ("embedllm", "EmbedLLM-CC")]


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def report(family, display_name):
    results = results_dir(family, "metadata-ablation")
    static = load(results_dir(family, "static-policy-baseline") / "comparison_metrics.json")
    s1_anchor = next(r for r in static if r["arm"] == "S1")["test_cost_auc"]

    print()
    print("#" * 78)
    print(f"# {display_name}")
    print("#" * 78)

    print("=" * 78)
    print("Representation ablation: pooled test cost_auc by arm x tower (3-seed mean +/- std)")
    print("=" * 78)

    cells = {(r["base_router"], r["arm"]): r for r in load(results / "comparison_metrics.json")}
    ref = cells[("BF-DKR", "fp")]
    floor, oracle = ref["test_floor_cost_auc"], ref["test_oracle_cost_auc"]

    hdr = f"\n  {'Arm':13s} {'raw dim':>8s} " + " ".join(f"{t:>18s}" for t in TOWERS)
    print(hdr)
    for arm in ARMS:
        row = f"  {arm:13s} {cells[('BF-DKR', arm)]['llm_raw_dim']:>8d} "
        for t in TOWERS:
            c = cells.get((t, arm))
            row += f" {c['test_cost_auc']:.4f} +/- {c['test_cost_auc_std']:.4f}" if c else f" {'n/a':>18s}"
        print(row + ("   <- reference" if arm == "fp" else ""))
    print(f"\n  anchors: floor {floor:.6f}  oracle {oracle:.6f}  "
          f"S1 (largest affordable, from static-policy-baseline) {s1_anchor:.6f}")

    print()
    print("=" * 78)
    print("BF-DKR gaps vs fp (paired bootstrap, positive = fp ahead)")
    print("=" * 78)
    boot = load(results / "gap_bootstrap.json")
    per_seed = boot["per_seed"]
    seeds = sorted(per_seed)
    band = per_seed[seeds[0]]["tie_band_cost_auc"]
    print(f"\n  tie-band +/-{band}; {per_seed[seeds[0]]['n_boot']} resamples; "
          f"n_queries={per_seed[seeds[0]]['n_queries']}; boot_seed={per_seed[seeds[0]]['boot_seed']}")
    for arm in ["meta_struct", "meta_name", "concat"]:
        deltas, signs, n_excl = [], set(), 0
        print(f"\n  fp - {arm}:")
        for s in seeds:
            g = per_seed[s]["gaps"][arm]
            lo, hi = g["cost_auc_gap_ci95"]
            deltas.append(g["cost_auc_gap_point"])
            signs.add(np.sign(g["cost_auc_gap_point"]))
            n_excl += bool(g["cost_auc_gap_excludes_zero"])
            print(f"    {s}: d={g['cost_auc_gap_point']:+.6f}  CI95=[{lo:+.4f}, {hi:+.4f}]  "
                  f"excl0={g['cost_auc_gap_excludes_zero']}  outside_band={g['outside_tie_band']}")
        consistent = len(signs) == 1
        print(f"    mean d={float(np.mean(deltas)):+.6f}  |  excludes zero {n_excl}/{len(seeds)}  |  "
              f"per-seed sign consistent: {consistent}"
              + ("" if consistent else "   <- NO consistent direction"))

    print()
    print("=" * 78)
    print("MLPDot per-seed cost_auc by arm (point estimates only -- no paired bootstrap exists)")
    print("=" * 78)
    mlp_by_arm = {arm: [] for arm in ARMS}
    for s in seeds:
        for arm in ARMS:
            m = load(results / s / arm / "other_routers_metrics.json")
            mlp_by_arm[arm].append(m["MLPDot"]["test_cost_auc"])
    for arm in ["meta_struct", "meta_name", "concat"]:
        deltas = [mlp_by_arm[arm][i] - mlp_by_arm["fp"][i] for i in range(len(seeds))]
        signs = {np.sign(d) for d in deltas}
        print(f"  MLPDot {arm} - fp: " + ", ".join(f"{d:+.4f}" for d in deltas)
              + f"   mean={float(np.mean(deltas)):+.4f}  sign-consistent: {len(signs) == 1}")

    print()
    print("=" * 78)
    print("Why metadata fails: two different mechanisms (distinct LLM vectors per bundle)")
    print("=" * 78)
    deg = load(results / "representation_degeneracy.json")
    print(f"\n  {'Arm':13s} {'train':>16s} {'val':>14s} {'test':>14s}")
    for arm in ARMS:
        d = deg["bundle_distinct_llm_vectors"][arm]
        cols = "".join(
            f" {d[sp]['n_distinct']:>7d}/{d[sp]['n_llms']:<6d} ({d[sp]['distinct_frac']:.0%})".rjust(16)
            if sp == "train" else
            f" {d[sp]['n_distinct']:>5d}/{d[sp]['n_llms']:<4d} ({d[sp]['distinct_frac']:.0%})".rjust(14)
            for sp in ["train", "val", "test"]
        )
        print(f"  {arm:13s}{cols}")
    print("\n  meta_struct cannot DISTINGUISH models; meta_name distinguishes every model and still")
    print("  cannot CHARACTERIZE one. Same outcome, opposite causes on RouterEval; on EmbedLLM")
    print("  meta_struct matches or beats fp for BF-DKR/MLPDot despite the same degeneracy pattern.")

    print()
    print("=" * 78)
    print(f"Column-order robustness ({deg['n_perm']} joint column permutations, seed {deg['perm_seed']})")
    print("=" * 78)
    print()
    for arm in ARMS:
        for tower, per_seed_rows in deg["arms"][arm].items():
            perms = [r["cost_auc_under_permutation"] for r in per_seed_rows]
            if any(x is None for x in perms):
                print(f"  {arm:13s} / {tower:9s}: unique prediction columns -- no tie to break")
                continue
            spreads = [x["spread"] for x in perms]
            tie_avg = max(abs(r["tie_averaged_delta"] or 0.0) for r in per_seed_rows)
            over = any(x["spread_exceeds_tie_band"] for x in perms)
            rng = f"{min(spreads):.4f} - {max(spreads):.4f}"
            flag = "OVER tie-band -> READ AS RANGE" if over else "under tie-band"
            print(f"  {arm:13s} / {tower:9s}: spread {rng:>17s}  "
                  f"max|tie-avg delta| {tie_avg:.4f}   {flag}")

    print()
    print("=" * 78)
    print("One-hot coverage under cold-cold splits (meta_struct)")
    print("=" * 78)
    cov = load(results / "metadata_coverage.json")["meta_struct"]
    print(f"\n  dim={cov['dim']}  n_llms={cov['n_llms']}")
    for field, c in cov["coverage"].items():
        sat = c.get("saturated_vocab", {})
        print(f"  {field:8s}: {c['n_distinct_train']:>5d} distinct in train, top-{c['top_k']} kept "
              f"-> test in-vocab {c['test_in_vocab_rate']:.3f}, test UNK {c['test_unk_rate']:.3f}; "
              f"saturated-vocab test UNK {sat.get('test_unk_rate', float('nan')):.3f}")

    print()
    print("=" * 78)
    print("H3 cross-check: is metadata's usable content essentially size?")
    print("=" * 78)
    ms = cells[("BF-DKR", "meta_struct")]["test_cost_auc"]
    verdict = "refuted" if abs(ms - s1_anchor) > band else "consistent with size alone"
    print(f"  meta_struct {ms:.6f} - S1 {s1_anchor:.6f} = {ms - s1_anchor:+.6f}  "
          f"({abs(ms - s1_anchor) / band:.0f}x the tie-band -> {verdict})")


for family, display_name in DATASETS:
    report(family, display_name)

print()
print("=" * 78)
print("Cross-dataset summary")
print("=" * 78)
print("  RouterEval (2,999 training LLMs): fp beats both metadata arms by ~0.059, 3/3 seeds each.")
print("  EmbedLLM (70 training LLMs): fp ties meta_struct for BF-DKR (mean +0.0039, signs split")
print("  2-1) and LOSES to it for MLPDot (mean -0.0255, 3/3 seeds); meta_name fails on both.")

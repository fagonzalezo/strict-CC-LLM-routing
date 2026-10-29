"""Derive Table VII (`tab:selection`): validation-to-test agreement of cost-aligned selection.

rho, p, n: Spearman correlation of validation and test cost_auc over each baseline's grid
(`selection_audit.json` -> `h2_val_test_agreement`).

Δ: change in pooled test cost_auc from selecting by validation cost_auc (tie-break validation
Brier) rather than by the pre-registered anchor configuration:
  * RouterEval-CC kNN / MLPDot: the anchor's stored test cost_auc (same roster, same splits).
  * EmbedLLM-CC MLPDot: the stored anchor predates the 107-model roster fix, so the anchor
    architecture is looked up in this run's own grid instead.
  * EmbedLLM-CC kNN: n/a -- its stored anchor predates the roster fix and its grid records no
    test scores, so no valid Δ exists.

Run: `python src/paper/gen_tab_selection.py`.
"""

import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import relative, results_dir  # noqa: E402

MLP_KEY = ("target_dim", "hidden_dim", "n_layers", "seed")


def load(path):
    print(f"  read: {relative(path)}")
    with Path(path).open() as fh:
        return json.load(fh)


def anchor_of(audit):
    return next(v for k, v in audit.items() if k.startswith("anchor_"))


def delta(family, router, audit, results):
    anchor = anchor_of(audit)
    if router == "kNN":
        if family == "embedllm":
            return None
        sel = next(r for r in load(results / "selected_metrics.json") if r["group"] == "pooled")
        return sel["test_cost_auc"] - anchor["test_cost_auc"]
    sel = next(g for g in audit["grid"] if all(g[k] == audit["selected"][k] for k in MLP_KEY))
    if family == "embedllm":
        anc = next(g for g in audit["grid"] if all(g[k] == anchor[k] for k in MLP_KEY))
        return sel["test_cost_auc"] - anc["test_cost_auc"]
    return sel["test_cost_auc"] - anchor["test_cost_auc"]


print("=" * 78)
print("Table VII: validation-to-test agreement of cost-aligned selection")
print("=" * 78)
rows = []
for router, exp in (("kNN", "knn-baseline"), ("MLPDot", "mlpdot-baseline")):
    for family, name in (("router-eval", "RouterEval-CC"), ("embedllm", "EmbedLLM-CC")):
        results = results_dir(family, exp)
        audit = load(results / "selection_audit.json")
        h2 = audit["h2_val_test_agreement"]
        rows.append((name, router, h2["n"], h2["spearman"], h2["p_value"],
                     delta(family, router, audit, results)))

print(f"\n  {'Dataset':14s} {'Router':7s} {'n':>4s} {'rho':>8s} {'p':>10s} {'Δ':>9s}")
for name, router, n, rho, p, d in rows:
    ds = "n/a" if d is None else f"{d:+.4f}"
    print(f"  {name:14s} {router:7s} {n:4d} {rho:+8.3f} {p:10.2g} {ds:>9s}")

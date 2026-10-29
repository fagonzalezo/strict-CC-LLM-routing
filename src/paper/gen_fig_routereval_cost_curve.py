"""Generate the RouterEval per-query budget-cap figure for the BF-DKR paper.

Source: the validation-selected BF-DKR cell (seed 1, pq_auc, k=2048) of
`experiments/router-eval/bfdkr-sweep`, reported in Section VI-A.
Prints `read:` / `wrote:` lines so the figure is auditable against the manuscript,
matching the `gen_tab_*.py` convention.

Run: `python src/paper/gen_fig_routereval_cost_curve.py`.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src.paths import PAPER_FIGURES, relative, results_dir  # noqa: E402

SOURCE = results_dir("router-eval", "bfdkr-sweep") / "dmkdmr_encoded/seed1/pq_auc/k2048/cost_curve.json"
OUT = PAPER_FIGURES


def points(rows):
    array = np.asarray(rows, dtype=float)
    return array[:, 0], array[:, 1]


print(f"  read: {relative(SOURCE)}")
with SOURCE.open() as handle:
    curve = json.load(handle)["pooled"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(3.3, 1.7))
colors = {"router": "#EB6834", "floor": "#6D6D6D", "oracle": "#2A78D6"}
labels = {
    "router": "BF-DKR",
    "floor": "Affordable-random floor",
    "oracle": "Affordable oracle",
}
styles = {"router": "-", "floor": "--", "oracle": ":"}
for key in ("oracle", "floor", "router"):
    x, y = points(curve[key])
    ax.step(x, y, where="post", color=colors[key], linestyle=styles[key], label=labels[key])

for budget in (3, 7, 30):
    ax.axvline(budget, color="#AAAAAA", linewidth=0.55, zorder=0)

ax.set_xlim(0, 86)
ax.set_xticks([3, 7, 30, 60])
ax.set_ylim(0.25, 1.0)
ax.set_xlabel("Per-query parameter-count cap (billions)")
ax.set_ylabel("Realized correctness")
ax.legend(loc="lower right", frameon=False, handlelength=2.1)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    dest = OUT / f"fig_routereval_cost_curve.{ext}"
    fig.savefig(dest)
    print(f"  wrote: {relative(dest)}")

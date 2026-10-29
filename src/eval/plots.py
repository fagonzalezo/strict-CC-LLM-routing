"""
Dataset-independent plotting primitives for `src/eval`.

Generalizes the three per-experiment plot scripts
(`experiments/router-eval/exp_28/scripts/plot_eval.py`,
`experiments/router-eval/exp_28/scripts/plot_method_comparison.py`,
`experiments/embedllm/exp_17/scripts/plot_method_comparison.py`) into
reusable functions: every axis label, title, and method style is a parameter
instead of a hard-coded per-experiment constant.

The val-tuned "router (val-tuned)" envelope line (drawn from a
`router_envelope` curve) has been removed project-wide — `plot_cost_quality`
only ever draws the raw router / floor / oracle / single_llm series.

The EmbedKit hyperparameter-sweep heatmap is intentionally NOT included here:
it is specific to the EmbedKit refinement library, not to routing evaluation,
and stays in each experiment's own script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 130


def plot_cost_quality(
    curves: dict,
    ax=None,
    *,
    title: str = "",
    xlabel: str = "Per-query budget cap (B params)",
    ylabel: str = "Expected quality (P(correct))",
    legend: bool = True,
):
    """Single-LLM scatter + oracle + floor + router lines on one axes.

    `curves` : {"single_llm", "oracle", "floor", "router"} -> [[cost, quality], ...]
    (the shape returned by `src.eval.report.report_pooled`'s `curves[group]`).
    No `router_envelope` is drawn even if present in `curves`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    single = np.array(curves["single_llm"])
    ax.scatter(single[:, 0], single[:, 1], s=8, c="0.75", alpha=0.4,
               label="single LLMs", zorder=1)
    orc = np.array(curves["oracle"])
    ax.plot(orc[:, 0], orc[:, 1], "-", c="tab:green", lw=1.5,
             label="oracle P(solvable≤b)", zorder=4)
    fl = np.array(curves["floor"])
    ax.plot(fl[:, 0], fl[:, 1], ":", c="tab:red", lw=1.5,
             label="floor (random within cap)", zorder=2)
    rt = np.array(curves["router"])
    ax.plot(rt[:, 0], rt[:, 1], "-o", ms=2.5, c="tab:blue", lw=1.5,
             label="router", zorder=3)

    ax.set_xscale("log")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_reliability(
    reliability: dict,
    ax=None,
    *,
    title: str = "",
    label: str = "router",
):
    """Reliability diagram: diagonal ideal line + (bin_conf, bin_acc), masked
    to bins with bin_count > 0.

    `reliability` : {"bin_conf", "bin_acc", "bin_count", ...} (the shape
    returned by `src.metrics.reliability_curve`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    conf = np.array(reliability["bin_conf"], dtype=float)
    acc = np.array(reliability["bin_acc"], dtype=float)
    cnt = np.array(reliability["bin_count"], dtype=float)
    ax.plot([0, 1], [0, 1], ":", c="0.6", label="ideal")
    m = cnt > 0
    ax.plot(conf[m], acc[m], "-o", ms=3, c="tab:purple", label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel("confidence (q̂)")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    return ax


def grid_figure(
    data: dict,
    plot_fn,
    fig_path,
    *,
    suptitle: str = "",
    order: list[str] | None = None,
    plot_kwargs: dict | None = None,
):
    """Panel grid over the top-level keys of `data` (e.g. per-benchmark + pooled).

    `plot_fn(value, ax=..., title=key, **plot_kwargs)` is called per panel —
    compatible with `plot_cost_quality` and `plot_reliability`.

    `order` : explicit key order (defaults to dict order, "pooled" last if present).
    """
    plot_kwargs = plot_kwargs or {}
    if order is None:
        keys = [k for k in data if k != "pooled"] + (["pooled"] if "pooled" in data else [])
    else:
        keys = order

    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, key in enumerate(keys):
        plot_fn(data[key], ax=axes.ravel()[i], title=key, **plot_kwargs)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if suptitle else None)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    return fig_path


def plot_method_comparison(
    method_curves: dict[str, dict],
    fig_path,
    *,
    styles: dict[str, dict] | None = None,
    title: str = "Pooled cost-quality: method comparison",
    xlabel: str = "Per-query budget cap (B params)",
    ylabel: str = "Expected quality (P(correct))",
):
    """Overlay several methods' pooled cost-quality curves with a single
    shared oracle/floor/single_llm background (asserted identical across
    methods) — no router_envelope.

    `method_curves` : {method_label: curves_dict} where each curves_dict has
    the same shape as `plot_cost_quality`'s `curves` argument. All methods
    must share the same floor curve (same evaluation grid); this is asserted.
    `styles` : {method_label: dict(color=..., marker=..., ls=..., lw=..., ms=...)}.
    Defaults to matplotlib's color cycle with circle markers if omitted.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    labels = list(method_curves.keys())
    first = method_curves[labels[0]]
    for other in labels[1:]:
        _assert_shared_grid(first["floor"], method_curves[other]["floor"], labels[0], other)

    single = np.array(first["single_llm"])
    ax.scatter(single[:, 0], single[:, 1], s=9, c="0.78", alpha=0.35, zorder=1,
               label="single LLMs")

    orc = np.array(first["oracle"])
    ax.plot(orc[:, 0], orc[:, 1], "-", color="tab:green", lw=1.4, zorder=4,
             label="oracle P(solvable ≤ b)")

    fl = np.array(first["floor"])
    ax.plot(fl[:, 0], fl[:, 1], ":", color="tab:red", lw=1.6, zorder=2,
             label="floor (random within cap)")

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    default_markers = ["o", "s", "^", "D", "v", "P"]
    for i, label in enumerate(labels):
        rt = np.array(method_curves[label]["router"])
        style = (styles or {}).get(label) or dict(
            color=default_cycle[i % len(default_cycle)],
            marker=default_markers[i % len(default_markers)],
            ls="-", lw=1.8, ms=3.5,
        )
        ax.plot(rt[:, 0], rt[:, 1], linestyle=style["ls"], marker=style["marker"],
                 color=style["color"], lw=style["lw"], ms=style["ms"], zorder=5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    return fig_path


def _assert_shared_grid(curve_a, curve_b, label_a, label_b, tol: float = 1e-6):
    xa = [p[0] for p in curve_a]
    xb = [p[0] for p in curve_b]
    assert len(xa) == len(xb), (
        f"Cap grid length mismatch: {label_a} ({len(xa)}) vs {label_b} ({len(xb)})"
    )
    max_diff = max(abs(a - b) for a, b in zip(xa, xb))
    assert max_diff <= tol, (
        f"Cap grid mismatch between {label_a} and {label_b}: max |Δ| = {max_diff:.2e} > {tol}"
    )

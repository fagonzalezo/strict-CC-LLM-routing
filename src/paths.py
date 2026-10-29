"""
Single source of truth for every repo-relative path in this project.

Why this module exists
----------------------
All code lives under `src/`, while the artifacts it reads and writes live at the repo
root (`experiments/`, `data/`, `paper/figures/`). Two asymmetries make it impossible to
derive an artifact path from a script's own `__file__`:

  1. Artifact directories keep hyphens (`experiments/router-eval/bfdkr-ablation/`) because
     that is how the paper and the source repo name them; the mirroring *code* directories
     must use Python-legal underscores (`src/experiments/router_eval/bfdkr_ablation/`).
  2. A runner at `src/experiments/<family>/<name>/run_x.py` is nowhere near its own
     `experiments/<family>/<name>/results/` — the old
     `RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"` idiom would resolve
     inside `src/`.

So every path goes through the helpers below rather than through `parents[N]` arithmetic.
`ROOT` itself is found by walking up for this file, which makes it independent of how deep
any caller sits; `BFDKR_ROOT` overrides it for out-of-tree checkouts.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ROOT", "SRC", "DATA", "EXPERIMENTS", "PAPER_FIGURES", "DOCS", "FINGERPRINTS",
    "experiment_dir", "results_dir", "reports_dir", "figures_dir",
    "dataset_raw", "dataset_processed", "dataset_path", "relative",
    "probe_set_path", "answers_dir", "fingerprints_path",
]


def _find_root() -> Path:
    env = os.environ.get("BFDKR_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, *here.parents):
        if (candidate / "src" / "paths.py").is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate the repo root above {here}. "
        "Set BFDKR_ROOT to the directory containing src/ and experiments/."
    )


ROOT: Path = _find_root()

SRC: Path = ROOT / "src"
DATA: Path = ROOT / "data"
EXPERIMENTS: Path = ROOT / "experiments"
PAPER_FIGURES: Path = ROOT / "paper" / "figures"
DOCS: Path = ROOT / "docs"
FINGERPRINTS: Path = DATA / "fingerprints"


# ── experiment artifacts ─────────────────────────────────────────────────────
#
# `family` is a dataset family directory ("router-eval", "embedllm") or, for an
# experiment spanning both datasets, the experiment name itself with `name=None`
# (that is how "coldrec-baselines" sits at the top level).

def experiment_dir(family: str, name: str | None = None) -> Path:
    return EXPERIMENTS / family if name is None else EXPERIMENTS / family / name


def results_dir(family: str, name: str | None = None) -> Path:
    return experiment_dir(family, name) / "results"


def reports_dir(family: str, name: str | None = None) -> Path:
    return experiment_dir(family, name) / "reports"


def figures_dir(family: str, name: str | None = None) -> Path:
    return reports_dir(family, name) / "figures"


# ── dataset artifacts ────────────────────────────────────────────────────────

def dataset_raw(dataset: str) -> Path:
    return DATA / dataset / "raw"


def dataset_processed(dataset: str) -> Path:
    return DATA / dataset / "processed"


def dataset_path(dataset: str) -> Path:
    """The `llm_perf_dataset.pkl` bundle a DatasetSpec loads (tier B; gitignored)."""
    return dataset_processed(dataset) / "llm_perf_dataset.pkl"


# ── fingerprints (data/fingerprints/ — not under data/<dataset>/, see src/fingerprints/) ──
#
# `data/fingerprints/` holds the probe set shared by both datasets plus each dataset's raw
# probe answers and encoded fingerprint array. It sits outside `data/<dataset>/` because the
# probe set itself is dataset-independent; see src/fingerprints/ and
# docs/dataset_construction.md.

def probe_set_path(name: str) -> Path:
    """CSV of the committed probe set `name` (e.g. "gsm8k_seed42_n64"). Tracked in git."""
    return FINGERPRINTS / "probes" / f"{name}.csv"


def answers_dir(dataset: str) -> Path:
    """Raw per-model probe-answer texts for `dataset` (gitignored; regenerable)."""
    return FINGERPRINTS / "answers" / dataset


def fingerprints_path(dataset: str) -> Path:
    """The encoded (N, n_probe * embed_dim) fingerprint bundle for `dataset` (gitignored)."""
    return FINGERPRINTS / dataset / "fingerprints.pkl"


def relative(path: Path | str) -> str:
    """Render `path` relative to ROOT for provenance printing (`read: <path>` lines)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)

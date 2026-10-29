"""
Smoke tests for the repo's layout and path wiring.

These are not tests of the method — they guard the things a reorganization silently breaks:
`src.paths` resolving the root from any working directory, every experiment the paper reads
actually existing, the library importing without the dropped modules, the tier-A generators
still running and still printing their provenance, and no private path leaking into the
public tree.

Run: `python -m pytest src/tests -q`  (from anywhere).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "paths.py").is_file())
sys.path.insert(0, str(_ROOT))

from src import paths  # noqa: E402

# (family, name) for every experiment the paper reads.
EXPERIMENTS = [
    ("router-eval", "bfdkr-ablation"),
    ("router-eval", "bfdkr-sweep"),
    ("router-eval", "mlpdot-baseline"),
    ("router-eval", "knn-baseline"),
    ("embedllm", "bfdkr-sweep"),
    ("embedllm", "mlpdot-baseline"),
    ("embedllm", "knn-baseline"),
    ("coldrec-baselines", None),
    ("router-eval", "dropoutnet-clcrec-baselines"),
    ("embedllm", "dropoutnet-clcrec-baselines"),
    ("embedllm", "all-methods-comparison"),
    ("router-eval", "all-methods-comparison"),
    ("router-eval", "static-policy-baseline"),
    ("embedllm", "static-policy-baseline"),
    ("router-eval", "metadata-ablation"),
    ("embedllm", "metadata-ablation"),
]

# Every tier-A generator; gen_tab_data.py is excluded because it needs the tier-B bundles.
TABLE_GENERATORS = ["gen_tab_main.py", "gen_tab_ablation.py", "gen_tab_static.py",
                    "gen_tab_metadata.py", "gen_tab_selection.py", "gen_tab_pergroup.py"]

# Strings that would mean a private path, cluster detail or internal note slipped in.
FORBIDDEN = ("/Users/", "/projects/", "/apps/containerCollections", "iris/fagonzal", "[[feedback",
             "[[project")


# ── src.paths ────────────────────────────────────────────────────────────────

def test_root_is_the_repo_root():
    assert (paths.ROOT / "src" / "paths.py").is_file()
    assert (paths.ROOT / "experiments").is_dir()


@pytest.mark.parametrize("cwd", [".", "src", "experiments", "src/paper"])
def test_root_resolves_from_any_cwd(cwd):
    """ROOT must come from __file__, never from the working directory."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(_ROOT)!r}); "
         "from src.paths import ROOT; print(ROOT)"],
        cwd=paths.ROOT / cwd, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(paths.ROOT)


def test_bfdkr_root_env_override(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "paths.py").touch()
    monkeypatch.setenv("BFDKR_ROOT", str(tmp_path))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.ROOT == tmp_path.resolve()
    finally:
        monkeypatch.delenv("BFDKR_ROOT")
        importlib.reload(paths)


# ── artifacts ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("family,name", EXPERIMENTS)
def test_experiment_dir_exists(family, name):
    assert paths.experiment_dir(family, name).is_dir()


@pytest.mark.parametrize("family,name", [e for e in EXPERIMENTS if e[0] != "coldrec-baselines"])
def test_results_dir_has_json(family, name):
    """Every experiment except coldrec-baselines ships committed result JSONs."""
    assert list(paths.results_dir(family, name).rglob("*.json"))


@pytest.mark.parametrize("family,name", EXPERIMENTS)
def test_experiment_has_readme(family, name):
    """Each experiment documents what it tests and how to rerun it."""
    assert (paths.experiment_dir(family, name) / "README.md").is_file()


def test_no_code_under_experiments_or_data():
    """All code lives in src/; experiments/ and data/ hold artifacts only."""
    for tree in (paths.EXPERIMENTS, paths.DATA):
        assert not list(tree.rglob("*.py")), f"stray .py under {tree}"
        assert not list(tree.rglob("*.slurm")), f"stray .slurm under {tree}"


def test_dataset_bundles_are_untrackable():
    """The multi-GB bundles are regenerable and must never become committable.

    Asserts git's view, not the filesystem's: a tier-B run legitimately fills data/ with
    bundles, and the invariant that matters is that .gitignore covers them when it does.
    A filesystem glob would pass vacuously on a fresh clone and start failing after a
    correct tier-B run — exactly backwards.
    """
    if not (paths.ROOT / ".git").exists():
        pytest.skip("not a git repo")
    probes = [paths.dataset_processed(ds) / name
              for ds in ("router-eval", "embedllm")
              for name in ("llm_perf_dataset.pkl", "embeddings.npz", "model.pt")]
    probes += [paths.fingerprints_path(ds) for ds in ("router-eval", "embedllm")]
    probes += [paths.answers_dir(ds) / "0.json" for ds in ("router-eval", "embedllm")]
    # The MLPDot-encoded bundle directory mixes already-ignored files (*.pkl, *.pt) with
    # loose .npy/.json siblings that no extension rule covers -- must be caught by the
    # path-scoped data/*/processed/mlpdot_encoded_seed*/ rule instead (see .gitignore).
    probes += [paths.dataset_processed(ds) / "mlpdot_encoded_seed0" / name
              for ds in ("router-eval", "embedllm")
              for name in ("mlpdot_test_predictions.npy", "mlpdot_reference.json")]
    # Metadata-representation variant bundles nest their own mlpdot_encoded_seed*/ output one
    # level deeper, where the rule above no longer reaches -- covered by the path-scoped
    # data/*/processed/metadata_variants/ rule instead.
    probes += [paths.dataset_processed(ds) / "metadata_variants" / "meta_struct" / name
              for ds in ("router-eval", "embedllm")
              for name in ("llm_perf_dataset.pkl", "metadata_fields.json",
                           "mlpdot_encoded_seed0/mlpdot_reference.json")]
    out = subprocess.run(["git", "check-ignore", "--no-index", *map(str, probes)],
                         cwd=paths.ROOT, capture_output=True, text=True)
    ignored = set(out.stdout.split())
    missed = [str(p) for p in probes if str(p) not in ignored]
    assert not missed, f".gitignore does not cover: {missed}"


def test_probe_set_is_trackable():
    """The opposite invariant: the committed probe set must NOT be gitignored, or a fresh
    clone would silently lose the one thing every fingerprint traces back to."""
    if not (paths.ROOT / ".git").exists():
        pytest.skip("not a git repo")
    csv_path = paths.probe_set_path("gsm8k_seed42_n64")
    assert csv_path.is_file(), f"missing committed probe set: {csv_path}"
    out = subprocess.run(["git", "check-ignore", "--no-index", str(csv_path)],
                         cwd=paths.ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"{csv_path} must not be gitignored"


def test_code_and_new_data_are_trackable():
    """No ignore rule may swallow code or the committed new data (e.g. a bare `paper/` rule
    would also match `src/paper/`)."""
    if not (paths.ROOT / ".git").exists():
        pytest.skip("not a git repo")
    must_track = [p for p in paths.SRC.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    must_track += [paths.dataset_raw("embedllm") / "model_list.csv",
                   *sorted((paths.dataset_raw("embedllm") / "model_outputs").glob("*.json"))]
    out = subprocess.run(["git", "check-ignore", "--no-index", *map(str, must_track)],
                         cwd=paths.ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"gitignored but must be tracked:\n{out.stdout}"


# ── library ──────────────────────────────────────────────────────────────────

def test_eval_and_spec_import():
    from src.eval import get_spec, load_splits  # noqa: F401
    for name in ("router-eval", "embedllm"):
        spec = get_spec(name)
        assert spec.dataset_path == paths.dataset_path(name)
        assert spec.groups


def test_cross_benchmark_spec_is_gone():
    from src.eval import get_spec
    with pytest.raises(KeyError):
        get_spec("cross-benchmark")


def test_dropped_modules_are_absent():
    for gone in ("src/routers/kdm_router.py", "src/routers/_smoke_kdm_router.py",
                 "src/pooled_eval.py", "src/routers/_smoke_clcrec_router.py",
                 "src/routers/_smoke_dropoutnet_router.py"):
        assert not (paths.ROOT / gone).exists()
    # Read the source rather than importing: src.routers pulls in torch, and tier A
    # must stay importable without it.
    # (careful: "DMKDMRRouter" contains "KDMRouter" as a substring — match on the
    # module path and the import statement, not the bare class name)
    init = (paths.SRC / "routers" / "__init__.py").read_text()
    assert "kdm_router" not in init
    assert '"KDMRouter"' not in init


# ── tier A ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", TABLE_GENERATORS)
def test_table_generator_runs_and_prints_provenance(script):
    out = subprocess.run([sys.executable, str(paths.SRC / "paper" / script)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    reads = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("read:")]
    assert reads, f"{script} printed no provenance lines"
    for line in reads:
        rel = line.split("read:", 1)[1].strip()
        assert (paths.ROOT / rel).is_file(), f"{script} claims to read a missing file: {rel}"


@pytest.mark.parametrize("dataset_dir", ["embedllm", "router_eval"])
def test_build_seed_summary_runs_and_prints_provenance(dataset_dir):
    """Each dataset's all-methods seed summary is its own self-checking generator (10+
    checks against {router_eval,embedllm}_finals_summary.json, seed_spread.json,
    best_config.json, selection_audit.json and attribution_table.json) -- a non-zero exit
    means one broke."""
    script = (paths.SRC / "experiments" / dataset_dir / "all_methods_comparison"
              / "build_seed_summary.py")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    reads = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("read:")]
    assert reads, f"{script.name} printed no provenance lines"
    for line in reads:
        rel = line.split("read:", 1)[1].strip()
        assert (paths.ROOT / rel).is_file(), \
            f"{script.name} claims to read a missing file: {rel}"


def test_aiq_identity_holds_for_agnn_fsgnn():
    """gen_tab_main.py's gate: aiq == cost_auc - floor for every AGNN/FSGNN row, on both
    datasets. Both rows are now computed from committed results/ (see
    build_{router_eval,embedllm}_finals_summary.py), not transcribed -- read here from
    their own finals_summary.json rather than hardcoded literals, so this test cannot go
    stale the way the old transcribed-literal version did when RouterEval's AGNN winner
    changed from config12 to config18.

    Recomputed here from the same artifacts so the guard is covered by the suite, not only
    by the generator asserting against itself.
    """
    for dataset in ("router-eval", "embedllm"):
        fname = "router_eval_finals_summary.json" if dataset == "router-eval" else "embedllm_finals_summary.json"
        finals = json.loads((paths.results_dir("coldrec-baselines") / fname).read_text())
        for model in ("AGNN", "FSGNN"):
            row = finals[model]
            implied_aiq = row["cost_auc"]["mean"] - row["floor_cost_auc"]
            assert abs(implied_aiq - row["aiq"]["mean"]) < 1e-6, \
                f"{dataset} {model}: AIQ identity broke"


# ── public release hygiene ───────────────────────────────────────────────────

def test_no_private_paths_in_tracked_files():
    if not (paths.ROOT / ".git").exists():
        pytest.skip("not a git repo")
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=paths.ROOT, capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for rel in files:
        if Path(rel).name == "test_layout.py":
            continue
        text = (paths.ROOT / rel).read_text(errors="ignore")
        hits += [f"{rel}: {bad}" for bad in FORBIDDEN if bad in text]
    assert not hits, "private strings found:\n" + "\n".join(hits)

#!/usr/bin/env bash
# Tier A: re-derive every table and figure of the paper from the committed results JSON.
# Writes one text file per table to paper/tables/ and the figure to paper/figures/.
# Needs only numpy and matplotlib; runs in seconds. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p paper/tables paper/figures

run() {  # run <output-name> <script> [args...]
    local out="paper/tables/$1.txt"; shift
    echo "== $* -> $out"
    python "$@" > "$out"
}

run tab_main_III            src/paper/gen_tab_main.py
run tab_ablation_IV_S4      src/paper/gen_tab_ablation.py
run tab_static_V_S3         src/paper/gen_tab_static.py
run tab_metadata_VI         src/paper/gen_tab_metadata.py
run tab_selection_VII       src/paper/gen_tab_selection.py
run tab_pergroup_S1_S2      src/paper/gen_tab_pergroup.py
python src/paper/gen_fig_routereval_cost_curve.py

# Table II needs the tier-B dataset bundles; skipped when they are absent.
if [[ -f data/router-eval/processed/llm_perf_dataset.pkl && -f data/embedllm/processed/llm_perf_dataset.pkl ]]; then
    run tab_data_II src/paper/gen_tab_data.py
else
    echo "== skipping Table II (gen_tab_data.py): dataset bundles not built (tier B)"
fi

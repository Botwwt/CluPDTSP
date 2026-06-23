# Experiment Status

Last updated: 2026-06-24.

## Scope

This repository records the completed Wentao Wang task scope from `D:\TSP\??.md`: Heter rerun, core CAADRL ablations, Heter Li&Lim-PDP-relaxed benchmark, Heter runtime/memory profiling, route-level mechanism metrics, manuscript updates, response letter updates, and repository result preservation.

CAADRL/NCS reruns assigned to Lifeng Han are out of scope. Comparisons that require missing Lifeng Han CAADRL/NCS raw outputs remain explicit reproducibility interfaces rather than fabricated numbers.

## Environment

AutoDL hardware and software audit:

- GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB.
- Driver: 580.105.08.
- CPU: Intel Xeon Platinum 8470Q, 208 logical CPUs.
- RAM: 754 GiB total.
- Conda base Python: 3.10.8.
- PyTorch: 2.1.2+cu118.
- CUDA availability from PyTorch: true.

Remote credentials are not stored in this repository.

## Seed and Dataset Decision

The unified test sets use base seed `10000` with instance order `seed + instance_id`, matching the current CluPDTSP README convention. Heter checkpoints use the upstream/default training seed `1234`. Heter's original uniform test seed `6666` remains documented as upstream provenance but is not used for shared CAADRL/Heter/NCS evaluation because it would create a different instance order.

## Completed Artifacts

- `baselines/heter/`: Heter source preserved under a baseline namespace.
- `data/pdp/unified/`: eight shared 10,000-instance test sets for PDP10/PDP20/PDP40/PDP80 under clustered and uniform distributions.
- `checkpoints/heter/`: one completed size-matched Heter checkpoint family for each required size; reused rather than retrained.
- `results/raw/heter/`: 24 validated Heter raw CSV files covering all required sizes, distributions, and Greedy/Sampling-1280/Sampling-12800 decoding.
- `results/summary/heter_main_results.csv` and `.tex`: Heter mean, standard deviation, runtime, and row-count summaries.
- `checkpoints/ablation/`: checkpoint-800 files for all six variants across clustered PDP20/PDP40/PDP80.
- `results/raw/ablation/`: 36 validated CAADRL ablation raw CSV files, each with 10,000 rows.
- `results/summary/ablation_results.csv` and `.tex`: six-variant ablation summary.
- `figures/ablation_barplot.png` and `.pdf`: ablation figure generated from the summary table.
- `results/stats/paired_caadrl_vs_heter_summary.csv` and `.tex`: aligned CAADRL-vs-Heter paired statistics for clustered PDP20/PDP40/PDP80 under Greedy and Sampling-1280.
- `results/stats/heter_route_metrics/` and `results/stats/heter_route_metrics_summary.csv/.tex`: Heter route-level mechanism metrics for 24 completed Heter files.
- `figures/route_visualization_caadr_vs_heter.png` and `.pdf`: representative aligned PDP40-clustered CAADRL-vs-Heter route visualization.
- `data/li_lim_relaxed/`, `results/raw/heter_li_lim_relaxed/`, and `results/summary/heter_li_lim_relaxed.csv/.tex`: Li&Lim-PDP-relaxed conversion and Heter evaluation for the available relaxed 100-task and 200-task groups.
- `results/profiling/heter_runtime_memory.csv` and `.tex`: synchronized Heter runtime/memory profiling for the required 24 combinations.

## Implemented Code Interfaces

- `scripts/build_unified_test_sets.py`
- `scripts/evaluate_heter.py`
- `scripts/summarize_raw_results.py`
- `scripts/profile_heter.py`
- `scripts/convert_li_lim_to_pdp_relaxed.py`
- `scripts/compute_route_metrics.py`
- `scripts/run_heter_mechanism_analysis.sh`
- `scripts/merge_paired_results.py`
- `scripts/paired_statistics.py`
- `scripts/run_heter_train.sh`, `scripts/run_heter_eval.sh`, `scripts/profile_heter.sh`
- `scripts/run_ablation_grid.sh` and `scripts/run_ablation_grid.ps1`

## Ablation Configuration

Implemented config-controlled variants:

1. `full`: full CAADRL with global + cluster encoder and learned-gate dual decoder.
2. `no_enc_cluster`: global-only encoder.
3. `cluster_only`: cluster-only encoder.
4. `no_dec_cluster`: single decoder.
5. `avg_fusion`: dual decoder with fixed average fusion instead of learned gate.
6. `no_pomo`: full model trained with `pomo_size=1` / single rollout.

No extra model copies were created for these variants; they are controlled through `experiment.py` and model parameters.

## Validation Completed

- All 24 Heter raw files validate for required columns, 10,000 rows, instance IDs `0..9999`, metadata, finite objectives, finite runtimes, non-empty provenance, and JSON routes.
- All 36 ablation raw files validate for required columns, 10,000 rows, clustered metadata, Greedy/Sampling-1280 decode labels, seed `10000`, finite objectives, finite runtimes, and JSON routes.
- Heter route metrics and CAADRL-vs-Heter route visualization were regenerated after the full-CAADRL ablation raw files became available.
- Manuscript and `responseletter.tex` now reference the completed Heter rerun, paired statistics where reproducible, route diagnostics and visualization, Li&Lim-PDP-relaxed benchmark, profiling results, and completed ablation table/figure.

## Remaining Interfaces

- Lifeng Han's CAADRL/NCS reruns are not executed here. NCS comparisons, uniform CAADRL paired comparisons, and Sampling-12800 CAADRL paired comparisons remain unfilled until matching aligned raw outputs are available.
- Heter does not expose gate probabilities or decoder logits in the saved outputs, so gate entropy, stay probability, and logit-margin metrics are intentionally not reported for Heter.

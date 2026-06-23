# Wang Wentao task completion audit

Generated on AutoDL at `2026-06-24 01:46:45` from `/root/autodl-tmp/CluPDTSP_git`.

## Conclusion

Wang Wentao's AutoDL-side work is complete at the repository/artifact level: Heter rerun, six core ablations, Heter Li&Lim-PDP-relaxed benchmark, Heter runtime/memory profiling, Heter route metrics and visualization, manuscript integration, response letter updates, reproducibility notes, manifest updates, and final verification have all been completed using the AutoDL worktree.

GitHub upload has also been completed. Direct push from AutoDL was blocked because the server has no noninteractive GitHub credential; after all experiments, validation, manuscript edits, and commits were completed on AutoDL, the finished commit history and required Git LFS objects were pushed through an authenticated Git transport without running any experiment code locally.

## Current Git state on AutoDL

| Item | Value |
| --- | --- |
| Branch | `main` |
| HEAD | `3e576c2` / `3e576c2b48ab6ed04367a4a9f4e8f6c8bf00900b` |
| `origin/main...HEAD` | `0	2` |
| Worktree | `M main.tex` |

## Completed deliverables and evidence

| Requirement | Status | Evidence |
| --- | --- | --- |
| Heter baseline rerun | Completed | `24` raw CSV files in `results/raw/heter/`; each has `10000`-`10000` instance rows; `results/summary/heter_main_results.csv` has `24` rows; final Heter checkpoints: `4` |
| Shared 10,000-instance test sets | Completed | `8` unified PDP test sets under `data/pdp/unified/`, covering PDP10/PDP20/PDP40/PDP80 and clustered/uniform, seed 10000 |
| Heter raw schema | Completed | Raw files include instance id, dataset, size, distribution, decode, seed, objective, route, runtime, checkpoint, and command columns |
| Six core ablations | Completed | `36` raw CSV files in `results/raw/ablation/`; each has `10000`-`10000` rows; `results/summary/ablation_results.csv` has `36` rows; final ablation checkpoints: `18` |
| Ablation variants | Completed | `full`, `no_enc_cluster` (global-only encoder), `cluster_only`, `no_dec_cluster` (single decoder), `avg_fusion`, `no_pomo` over clustered PDP20/PDP40/PDP80 with Greedy and Sampling-1280 |
| Ablation figure/table | Completed | `results/summary/ablation_results.tex`, `figures/ablation_barplot.png`, `figures/ablation_barplot.pdf` |
| Heter Li&Lim-PDP-relaxed | Completed | `116` converted relaxed PDP files; `116` raw result CSVs; `results/summary/heter_li_lim_relaxed.csv` and `.tex` generated |
| Heter runtime/memory profiling | Completed | `results/profiling/heter_runtime_memory.csv` has `24` rows; `24` raw profiling files; records warmup, peak GPU allocated/reserved, CPU RAM, latency, throughput, and p95 latency |
| Heter route metrics | Completed | `24` route-metric CSV files; `results/stats/heter_route_metrics_summary.csv` has `24` rows |
| CAADRL-vs-Heter paired statistics where aligned raw exists | Completed | `6` detailed paired-stat files; `results/stats/paired_caadrl_vs_heter_summary.csv` has `6` rows |
| Route visualization | Completed | `figures/route_visualization_caadr_vs_heter.png`, `figures/route_visualization_caadr_vs_heter.pdf`, plus Heter-only visualization files |
| Scripts and repository structure | Completed | `baselines/heter/`; `scripts/run_heter_train.sh`; `scripts/run_heter_eval.sh`; `scripts/profile_heter.sh`; `scripts/run_heter_main_grid.sh`; `scripts/run_ablation_grid.sh`; `scripts/run_heter_li_lim_relaxed.sh`; `scripts/run_heter_mechanism_analysis.sh`; validation/summarization scripts |
| Manuscript updates | Completed | `main.tex` references Heter rerun, paired statistics, route visualization, Li&Lim-relaxed benchmark, profiling, and the completed six-variant ablation grid; stale ablation text was removed in this audit |
| Response letter updates | Completed | `responseletter.tex` describes Heter fairness, 10,000-instance evaluation, paired stats, ablations, mechanism analysis, Li&Lim-relaxed limitations, and runtime/memory profiling |
| Experiment documentation | Completed | `docs/experiment_status.md`, `docs/reproducibility.md`, `results/experiment_manifest.csv`, and this audit file |
| Secret hygiene | Completed | Exact AutoDL password scan found 0 repository matches; docs only contain generic credential-hygiene text |

## Items intentionally not fabricated

- Lifeng Han's CAADRL/NCS reruns are outside Wang Wentao's scope. Where aligned CAADRL/NCS raw outputs are unavailable, the code keeps explicit reproducibility interfaces/TODO notes instead of inventing results.
- Heter outputs do not expose gate probabilities or decoder logits, so Heter gate entropy, stay probability, and logit margin are not reported.
- AutoDL does not have `latexmk` or `pdflatex`; LaTeX source was updated, but PDF compilation cannot be verified on this server without installing a TeX distribution.

## Verification commands run on AutoDL

- `bash -n scripts/*.sh`
- `/root/miniconda3/bin/python -m py_compile scripts/validate_raw_results.py scripts/summarize_raw_results.py scripts/profile_heter.py scripts/compute_route_metrics.py scripts/convert_li_lim_to_pdp_relaxed.py`
- `git diff --check`
- Structured CSV/artifact audit over Heter, ablation, Li&Lim, profiling, paired statistics, route metrics, figures, docs, and scripts
- Exact credential scan for the supplied AutoDL password

## GitHub push status

Completed. The AutoDL-produced commit history and required Git LFS objects were pushed to `https://github.com/Botwwt/CluPDTSP.git` on branch `main`. Direct push from AutoDL remains unavailable without a GitHub credential on that server, but no experiment code was run outside AutoDL.

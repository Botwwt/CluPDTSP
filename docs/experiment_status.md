# Experiment Status

Last updated: 2026-06-23.

## Scope

This repository is being organized for the Wentao Wang task list in `D:\TSP\待做.md`.
The active scope is Heter rerun, core ablations, Heter Li&Lim-PDP-relaxed benchmark,
Heter runtime/memory profiling, route-level mechanism metrics, manuscript updates,
response letter updates, and repository result preservation.

CAADRL/NCS reruns assigned to Lifeng Han are out of scope here. If CAADRL/NCS raw
results are absent, paired comparisons and per-instance improvement columns must
remain TODO rather than using fabricated numbers.

## Audit Summary

- Local target repository: `D:\TSP\CluPDTSP`, branch `main`, remote
  `https://github.com/Botwwt/CluPDTSP.git`.
- Heter original source: `D:\TSP\Heterogeneous-Attentions-PDP-DRL-main`.
- Existing PDP code reference: `D:\TSP\TSP`.
- The initial local audit found no reusable experiment outputs in the original
  repositories. After the subsequent remote run/synchronization, this working
  tree now contains the Heter artifacts listed below:
  - one completed Heter checkpoint family for each required size
    (PDP10/PDP20/PDP40/PDP80);
  - eight unified 10,000-instance test sets under `data/pdp/unified/`;
  - 24 validated Heter raw CSV files under `results/raw/heter/`, covering all
    required sizes, distributions, and decode strategies;
  - Heter summary tables, route-level mechanism metrics, Li--Lim-PDP-relaxed
    outputs, route-visualization interfaces, and synchronized profiling tables.
- The six requested CAADRL ablation variants are implemented as config-controlled
  code paths. A local audit found provisional Windows-local artifacts for
  `full`/PDP20 and a partial `full`/PDP40 checkpoint series, but those are not
  AutoDL-confirmed final evidence. The required AutoDL ablation grid is still
  incomplete and must be rerun or resumed on the server before manuscript use.
- AutoDL hardware audit:
  - GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB.
  - Driver: 580.105.08.
  - CPU: Intel Xeon Platinum 8470Q, 208 logical CPUs.
  - RAM: 754 GiB total.
  - Conda base Python: 3.10.8.
  - PyTorch: 2.1.2+cu118.
  - CUDA availability from PyTorch: true.
- Earlier AutoDL execution used `/root/autodl-tmp/CluPDTSP` paths recorded in the
  result provenance. In this Codex session, key/agent login to the supplied SSH
  endpoint failed and no password was forwarded through automated commands. A
  key-based login or interactive user-authenticated session is still required to
  recover any missing remote-only artifacts, especially the CAADRL ablation raw
  files referenced by provisional paired-statistics outputs.

## Seed and Dataset Decision

The current `CluPDTSP` and `D:\TSP\TSP` generators both expose default training
seed `1234`. The current `CluPDTSP` README documents unified clustered test-set
generation with base seed `10000`, using one single-instance file per seed.
The Heter upstream README documents validation seed `1234` and original uniform
test seed `6666`.

For fair cross-method evaluation in this repository, the unified test sets use
base seed `10000` with instance order `seed + instance_id`, matching the current
CluPDTSP README convention. Heter's original `6666` remains documented as the
upstream baseline seed, but it is not used for the shared CAADRL/Heter/NCS test
sets because it would create a different instance order.

## Heter Checkpoint Policy

The server audit found one completed Heter checkpoint for each required graph
size (PDP10/PDP20/PDP40/PDP80), each trained once with the upstream/default seed
`1234`, together with matching 10,000-instance evaluations. These size-matched
checkpoints are reused and must not be retrained. `scripts/run_heter_main_grid.sh`
only trains the affected size if its compatible final checkpoint is genuinely
absent; otherwise it reuses both the checkpoint and the raw result file.

The original Heter paper's graph-size-specific training convention is documented
in `baselines/heter/README.md`. The shared test seed remains `10000`, with the
same instance order for all methods.
The Heter grid otherwise retains the upstream option defaults: 800 epochs,
1,280,000 instances per epoch, training batch size 512, validation size 10,000,
and evaluation batch size 1,024. It does not substitute the CAADRL training
episode count for the Heter baseline.

## Repository Structure

Created or populated:

- `baselines/heter/`: copied from the original Heter implementation.
- `checkpoints/`
- `results/raw/heter/`
- `results/raw/ablation/`
- `results/raw/heter_li_lim_relaxed/`
- `results/summary/`
- `results/stats/`
- `results/profiling/`
- `figures/`
- `logs/`
- `scripts/`
- `docs/`

## Implemented Code Interfaces

- `scripts/build_unified_test_sets.py`: creates aggregate 10,000-instance PDP
  pickles for PDP10/PDP20/PDP40/PDP80 under clustered and uniform distributions.
- `scripts/evaluate_heter.py`: evaluates Heter checkpoints on unified test sets
  and writes instance-level raw CSV with objective, route, runtime, checkpoint,
  command, dataset, size, distribution, decode, and seed. It synchronizes CUDA
  around each decoding batch and writes the batch time divided by the exact
  number of instances in that batch, avoiding the upstream evaluator's
  batch-duration double counting in instance-level summaries.
- `scripts/summarize_raw_results.py`: summarizes raw CSV files into CSV and
  LaTeX tables with mean, standard deviation, average runtime, total runtime,
  and row counts.
- `scripts/profile_heter.py`: measures Heter latency, throughput, p95 latency,
  peak allocated/reserved CUDA memory, and CPU RAM after warm-up.
- `scripts/convert_li_lim_to_pdp_relaxed.py`: converts Li & Lim PDPTW-style
  files to single-vehicle distance-only relaxed PDP pickles.
- `scripts/compute_route_metrics.py`: computes inter-cluster switch count,
  intra-cluster edge ratio, and cluster separability index from raw routes.
- `scripts/run_heter_mechanism_analysis.sh`: computes those metrics for every
  completed Heter main-grid file, writes `results/stats/heter_route_metrics_*`,
  creates a Heter-only route visualization for a fixed PDP40 clustered instance,
  and only creates a CAADRL--Heter paired-improvement CSV plus common-instance
  comparison figure when a matching CAADRL raw file exists. Otherwise it emits
  an explicit TODO rather than fabricating a comparison.
- `scripts/merge_paired_results.py`: validates dataset, decode, seed, and
  instance-ID alignment before reporting per-instance CAADRL relative
  improvement versus Heter.
- `scripts/run_heter_train.sh`, `scripts/run_heter_eval.sh`,
  `scripts/profile_heter.sh`: Linux entry points for AutoDL.
- `scripts/run_ablation_grid.sh`: Linux/AutoDL ablation-grid entry point with
  checkpoint reuse, raw-result reuse, resume support, per-command logs, and
  train-batch fallback retries for CUDA OOM recovery.
- `scripts/run_ablation_grid.ps1`: Windows/PowerShell ablation-grid entry point
  with checkpoint reuse, raw-result reuse, resume support, and per-command logs.

## Ablation Configuration

Implemented config-controlled variants:

1. `full`: full CAADRL with global + cluster encoder and learned-gate dual decoder.
2. `no_enc_cluster`: global-only encoder.
3. `cluster_only`: cluster-only encoder.
4. `no_dec_cluster`: single decoder.
5. `avg_fusion`: dual decoder with fixed average fusion instead of learned gate.
6. `no_pomo`: full model trained with `pomo_size=1` / single rollout.

No extra model copies were created for these variants; they are controlled through
`experiment.py` and model parameters.

Current local ablation artifact audit, not final manuscript evidence:

- Two provisional `full`/PDP20 10,000-row CSVs were generated locally with
  Windows checkpoint provenance.
- A provisional `full`/PDP20 `checkpoint-800.pt` and a partial `full`/PDP40
  `checkpoint-160.pt` were found locally. Local Windows training then failed
  with CUDA OOM at epoch 163.
- These provisional raw/checkpoint/log artifacts are intentionally excluded from
  the final reproducibility set unless rerun or explicitly synchronized from
  AutoDL.
- No completed AutoDL-confirmed raw outputs are present for PDP40/PDP80 or the
  five other ablation variants.

## Execution Status

Completed and locally verified in this repository state:

1. Unified 10,000-instance PDP test sets exist for PDP10/PDP20/PDP40/PDP80 under
   clustered and uniform distributions with seed `10000`.
2. Heter checkpoints exist for PDP10/PDP20/PDP40/PDP80 and are reused rather than
   retrained.
3. Heter main evaluation is complete for 24 combinations. On 2026-06-23,
   `scripts/validate_raw_results.py` validated every Heter raw file for required
   columns, 10,000 rows, instance IDs `0..9999`, metadata, finite objectives,
   finite runtimes, non-empty provenance, and JSON routes.
4. `results/summary/heter_main_results.csv` and `.tex` summarize the validated
   Heter raw files with mean, standard deviation, average runtime, total runtime,
   and row count.
5. Heter Li--Lim-PDP-relaxed artifacts are present for the locally available
   100-task and 200-task relaxed instance groups. The benchmark is explicitly
   single-vehicle, distance-only, and not comparable with original PDPTW
   best-known solutions.
6. Heter profiling artifacts are present for the required 24 combinations with
   warm-up count, peak allocated/reserved GPU memory, CPU RAM, latency,
   throughput, p95 latency, and total runtime.
7. Heter route-level mechanism metrics are present for the required 24
   combinations: inter-cluster switch count, intra-cluster edge ratio, and
   cluster separability index.
8. `figures/route_visualization_heter.{png,pdf}` provides a reproducible
   Heter-only route visualization for PDP40-clustered instance 0. The
   CAADRL-vs-Heter comparison figure is intentionally not committed because the
   aligned CAADRL raw CSV needed to reproduce it is missing from the local
   artifact set.
9. Local smoke checks completed for Python syntax/import-light scripts, unified
   pickle loading, Heter raw validation, Li--Lim conversion logic, paired-result
   scripts, route metrics, and route plotting interfaces.

Incomplete or blocked:

1. Six-variant CAADRL ablation results are not complete. The scripts and config
   switches exist, but the requested AutoDL grid for all six variants across
   clustered PDP20/PDP40/PDP80 with Greedy and Sampling-1280 remains unfinished.
2. Provisional CAADRL--Heter paired-statistics files reference CAADRL raw paths
   under `/root/autodl-tmp/CluPDTSP/results/raw/ablation/`, but those raw files
   are not present locally. These numbers should not be treated as fully
   reproducible until the missing raw CSVs are synchronized or the ablation grid
   is rerun.
3. Remote AutoDL access is currently blocked in this non-interactive Codex
   session because SSH BatchMode key login failed and the password cannot be
   safely forwarded through shell commands. Use a key-based login or an
   interactive user-authenticated session, then run `bash
   scripts/run_ablation_grid.sh` from the AutoDL repository checkout.
4. CAADRL/NCS reruns assigned to Lifeng Han remain out of scope. Any comparison
   requiring missing CAADRL/NCS raw outputs must remain a TODO interface rather
   than being filled with surrogate numbers.

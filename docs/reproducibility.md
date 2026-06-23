# Reproducibility Notes

## Environment

The AutoDL server audit on 2026-06-22 reported:

- GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB.
- Driver: 580.105.08.
- CPU: Intel Xeon Platinum 8470Q, 208 logical CPUs.
- RAM: 754 GiB total.
- Conda base Python: 3.10.8.
- PyTorch: 2.1.2+cu118.

Activate conda before running scripts on AutoDL:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
```

Install repository requirements if needed:

```bash
python -m pip install -r requirements.txt
```

Remote access credentials must stay outside the repository. Do not place passwords, tokens, or private keys in shell scripts, environment files, manifests, logs, or committed documentation. The AutoDL artifacts recorded here were produced without writing the supplied access credential into the worktree.

Before committing model checkpoints, unified test sets, raw CSVs, or training
logs, initialize the repository's configured large-file storage and verify it:

```bash
git lfs install
git lfs track
```

`.gitattributes` routes checkpoints, test-set pickles, instance-level raw CSVs,
and logs through Git LFS. Summary tables, figures, code, and documentation are
kept as ordinary Git files.

## Shared Test Sets

Generate the shared 10,000-instance test sets:

```bash
python scripts/build_unified_test_sets.py \
  --sizes 10 20 40 80 \
  --distributions clustered uniform \
  --num-instances 10000 \
  --seed 10000 \
  --cluster-std 0.1
```

The generator uses `seed + instance_id` for each instance to preserve the
single-instance order documented in the original repository README.

## Heter Evaluation

`Heter` uses one checkpoint per required graph size, each trained once only.
When compatible PDP10/PDP20/PDP40/PDP80 checkpoints already exist, the main grid
reuses them and does not train or test again. Each raw record retains its exact
checkpoint path. The PDP80 checkpoint is used for the variable-size Li--Lim
relaxed benchmark because it is the largest completed Heter training size.

```bash
bash scripts/run_heter_main_grid.sh
bash scripts/validate_heter_main_grid.sh
bash scripts/run_heter_profile_grid.sh
```

Example greedy evaluation:

```bash
DATASET=data/pdp/unified/pdp40_test_clustered_std0.1_seed10000_n10000.pkl \
MODEL=checkpoints/heter/pdp_40/example/epoch-799.pt \
SIZE=40 \
DISTRIBUTION=clustered \
DECODE=greedy \
WIDTH=0 \
bash scripts/run_heter_eval.sh
```

Example summary:

```bash
python scripts/summarize_raw_results.py \
  "results/raw/heter/*.csv" \
  --csv-output results/summary/heter_main_results.csv \
  --tex-output results/summary/heter_main_results.tex
```

## CAADRL Ablation Grid

The six-variant ablation grid is complete for clustered PDP20/PDP40/PDP80 with Greedy and Sampling-1280 decoding. The completed artifact set contains one `checkpoint-800.pt` for each variant/size pair, 36 validated raw CSV files under `results/raw/ablation/`, summary tables under `results/summary/`, and `figures/ablation_barplot.{png,pdf}`.

To reproduce or resume the grid on AutoDL after checking existing artifacts:

```bash
bash scripts/run_ablation_grid.sh
```

The Linux/AutoDL entry point reuses existing raw CSVs, resumes from the latest `checkpoint-*.pt` below each variant/size folder, logs each command under `logs/ablation/`, and retries training with smaller batch sizes if a CUDA OOM interrupts a run.

On Windows/PowerShell, use the equivalent entry point only for smoke or local reproduction, not for the submitted AutoDL artifact set:

```powershell
.\scripts\run_ablation_grid.ps1
```

Expected outputs:

```text
checkpoints/ablation/
results/raw/ablation/
results/summary/ablation_results.csv
results/summary/ablation_results.tex
figures/ablation_barplot.png
figures/ablation_barplot.pdf
```

Paired CAADRL--Heter statistics are reported only for raw files that are aligned and present: clustered PDP20/PDP40/PDP80 under Greedy and Sampling-1280. Uniform, Sampling-12800, and NCS comparisons remain reproducibility interfaces until the matching raw outputs are available.

## Secret Hygiene

Do not store SSH passwords, access tokens, or private keys in this repository.
Before committing, run a secret scan for the provided AutoDL password and any
other local credentials.

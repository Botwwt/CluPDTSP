Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Setting {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Split-Setting {
    param([string]$Value)
    return @($Value -split '\s+' | Where-Object { $_ -ne "" })
}

function Invoke-Logged {
    param(
        [string]$LogPath,
        [string]$Executable,
        [string[]]$Arguments
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $commandLine = "$Executable $($Arguments -join ' ')"
    "`n[$(Get-Date -Format o)] $commandLine" | Tee-Object -FilePath $LogPath -Append
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments *>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: ${commandLine}"
    }
}

function Get-CheckpointEpoch {
    param([System.IO.FileInfo]$Checkpoint)
    if ($Checkpoint.BaseName -match '^checkpoint-(\d+)$') {
        return [int]$Matches[1]
    }
    return -1
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$pythonBin = Get-Setting "PYTHON_BIN" "python"
$sizes = Split-Setting (Get-Setting "SIZES" "20 40 80")
$variants = Split-Setting (Get-Setting "VARIANTS" "full no_enc_cluster cluster_only no_dec_cluster avg_fusion no_pomo")
$epochs = [int](Get-Setting "EPOCHS" "800")
$checkpointInterval = [int](Get-Setting "CHECKPOINT_INTERVAL" "20")
$trainEpisodes = [int](Get-Setting "TRAIN_EPISODES" "2816")
$trainBatchSize = [int](Get-Setting "TRAIN_BATCH_SIZE" "256")
$evalBatchSize = [int](Get-Setting "EVAL_BATCH_SIZE" "16")
$testSize = [int](Get-Setting "TEST_SIZE" "10000")
$trainSeed = [int](Get-Setting "TRAIN_SEED" "1234")
$testSeed = [int](Get-Setting "TEST_SEED" "10000")
$clusterStd = Get-Setting "CLUSTER_STD" "0.1"
$testOutDir = Get-Setting "TEST_OUT_DIR" "data/pdp/unified"
$checkpointRoot = Get-Setting "CHECKPOINT_ROOT" "checkpoints/ablation"
$rawRoot = Get-Setting "RAW_ROOT" "results/raw/ablation"
$summaryCsv = Get-Setting "SUMMARY_CSV" "results/summary/ablation_results.csv"
$summaryTex = Get-Setting "SUMMARY_TEX" "results/summary/ablation_results.tex"
$figurePng = Get-Setting "FIGURE_PNG" "figures/ablation_barplot.png"
$figurePdf = Get-Setting "FIGURE_PDF" "figures/ablation_barplot.pdf"
$logRoot = Get-Setting "LOG_ROOT" "logs/ablation"

Set-Location $repoRoot

$buildArgs = @("scripts/build_unified_test_sets.py", "--sizes") + $sizes + @(
    "--distributions", "clustered",
    "--num-instances", [string]$testSize,
    "--seed", [string]$testSeed,
    "--cluster-std", $clusterStd,
    "--name", "test",
    "--out-dir", $testOutDir
)
Invoke-Logged `
    -LogPath (Join-Path $logRoot "build_unified_test_sets.log") `
    -Executable $pythonBin `
    -Arguments $buildArgs

foreach ($variant in $variants) {
    foreach ($size in $sizes) {
        $resultRoot = Join-Path $checkpointRoot "pdp${size}_${variant}"
        $checkpoint = Get-ChildItem -Path $resultRoot -Recurse -Filter "checkpoint-${epochs}.pt" -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            Select-Object -Last 1

        if ($null -eq $checkpoint) {
            $resumeCheckpoint = Get-ChildItem -Path $resultRoot -Recurse -Filter "checkpoint-*.pt" -ErrorAction SilentlyContinue |
                Where-Object { (Get-CheckpointEpoch $_) -gt 0 -and (Get-CheckpointEpoch $_) -lt $epochs } |
                Sort-Object @{ Expression = { Get-CheckpointEpoch $_ } }, FullName |
                Select-Object -Last 1

            $trainArgs = @(
                "experiment.py",
                "--task", "train",
                "--ablation", $variant,
                "--problem_size", [string]([int]$size),
                "--distribution", "clustered",
                "--seed", [string]$trainSeed,
                "--epochs", [string]$epochs,
                "--checkpoint-interval", [string]$checkpointInterval,
                "--train_episodes", [string]$trainEpisodes,
                "--train_batch_size", [string]$trainBatchSize,
                "--result_dir", $resultRoot
            )
            if ($null -ne $resumeCheckpoint) {
                $resumeEpoch = Get-CheckpointEpoch $resumeCheckpoint
                Write-Host "resume $variant PDP$size from $($resumeCheckpoint.FullName) epoch $resumeEpoch"
                $trainArgs += @(
                    "--resume_path", $resumeCheckpoint.DirectoryName,
                    "--resume_epoch", [string]$resumeEpoch
                )
            }
            Invoke-Logged `
                -LogPath (Join-Path $logRoot "train_${variant}_pdp${size}.log") `
                -Executable $pythonBin `
                -Arguments $trainArgs

            $checkpoint = Get-ChildItem -Path $resultRoot -Recurse -Filter "checkpoint-${epochs}.pt" |
                Sort-Object FullName |
                Select-Object -Last 1
            if ($null -eq $checkpoint) {
                throw "Missing checkpoint-${epochs}.pt under $resultRoot after training"
            }
        }

        $dataset = Join-Path $testOutDir "pdp${size}_test_clustered_std${clusterStd}_seed${testSeed}_n${testSize}.pkl"
        foreach ($decode in @("greedy", "sampling")) {
            if ($decode -eq "greedy") {
                $width = 0
                $label = "greedy"
            } else {
                $width = 1280
                $label = "sampling_1280"
            }
            $output = Join-Path $rawRoot "ablation_${variant}_pdp${size}_clustered_${label}.csv"
            if (Test-Path -LiteralPath $output) {
                Write-Host "reuse existing $output"
                continue
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $output) | Out-Null
            $evalArgs = @(
                "scripts/evaluate_caadrl.py",
                "--dataset", $dataset,
                "--checkpoint", $checkpoint.FullName,
                "--output", $output,
                "--ablation", $variant,
                "--variant-label", $variant,
                "--size", [string]([int]$size),
                "--distribution", "clustered",
                "--decode", $decode,
                "--width", [string]$width,
                "--seed", [string]$testSeed,
                "--num-instances", [string]$testSize,
                "--batch-size", [string]$evalBatchSize,
                "--force"
            )
            Invoke-Logged `
                -LogPath (Join-Path $logRoot "eval_${variant}_pdp${size}_${label}.log") `
                -Executable $pythonBin `
                -Arguments $evalArgs
        }
    }
}

$summaryArgs = @(
    "scripts/summarize_raw_results.py",
    "$rawRoot/*.csv",
    "--csv-output", $summaryCsv,
    "--tex-output", $summaryTex
)
Invoke-Logged `
    -LogPath (Join-Path $logRoot "summarize_ablation.log") `
    -Executable $pythonBin `
    -Arguments $summaryArgs

$plotArgs = @(
    "scripts/plot_ablation.py",
    "--summary", $summaryCsv,
    "--decode", "Greedy",
    "--png", $figurePng,
    "--pdf", $figurePdf
)
Invoke-Logged `
    -LogPath (Join-Path $logRoot "plot_ablation.log") `
    -Executable $pythonBin `
    -Arguments $plotArgs

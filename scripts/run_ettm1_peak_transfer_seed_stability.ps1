param(
    [int[]]$PredLens = @(96, 192, 336),
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$RootPath = "./data/",
    [int]$SeqLen = 336,
    [int]$LabelLen = 48,
    [int]$BatchSize = 8,
    [double]$BaselineLearningRate = 0.0001,
    [double]$DefaultFinetuneLearningRate = 0.00001,
    [int]$TrainEpochs = 10,
    [int]$Patience = 3,
    [double]$OverallMseTolerance = 0.03,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$DefaultMyFramPython = "C:\Users\33932\.conda\envs\my_fram\python.exe"
if ($PythonExe -eq "python" -and (Test-Path $DefaultMyFramPython)) {
    $PythonExe = $DefaultMyFramPython
}

function Invoke-Checked {
    param([string]$Name, [string[]]$Command)
    Write-Host "==> $Name"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Name"
    }
}

function Get-FinetuneConfig {
    param([int]$PredLen)
    if ($PredLen -eq 336) {
        return @{
            LearningRate = 0.000001
            ExtraArgs = @("--use_event_weighted_loss", "1", "--event_weight", "1.0", "--use_peak_shape_loss", "0", "--peak_weight", "0.0")
        }
    }
    return @{
        LearningRate = $DefaultFinetuneLearningRate
        ExtraArgs = @()
    }
}

foreach ($PredLen in $PredLens) {
    foreach ($Seed in $Seeds) {
        $BaselineDes = "ettm1_peak_transfer_seed$Seed`_p$PredLen`_baseline"
        $EventDes = "ettm1_peak_transfer_seed$Seed`_p$PredLen`_finetune_loss"
        $GatedDes = "ettm1_peak_transfer_seed$Seed`_p$PredLen`_gated_alpha_1p0"
        $BaselineSetting = "long_term_forecast_DLinear_ETTm1_ftM_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$BaselineDes`_0"
        $EventSetting = "long_term_forecast_DLinear_ETTm1_ftM_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$EventDes`_0"
        $GatedSetting = "long_term_forecast_DLinear_ETTm1_ftM_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$GatedDes`_0"
        $FinetuneConfig = Get-FinetuneConfig -PredLen $PredLen

        $CommonArgs = @(
            "main.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model", "DLinear",
            "--data", "ETTm1",
            "--root_path", $RootPath,
            "--data_path", "ETTm1.csv",
            "--features", "M",
            "--target", "OT",
            "--freq", "t",
            "--seq_len", "$SeqLen",
            "--label_len", "$LabelLen",
            "--pred_len", "$PredLen",
            "--enc_in", "7",
            "--c_out", "7",
            "--batch_size", "$BatchSize",
            "--train_epochs", "$TrainEpochs",
            "--patience", "$Patience",
            "--seed", "$Seed",
            "--dlinear_init_avg", "0",
            "--use_zscore", "1",
            "--use_revin", "0",
            "--use_standard_time_features", "0",
            "--use_oracle_features", "0",
            "--use_rule_adapter", "0",
            "--use_intervention_layer", "0",
            "--use_hard_intervention", "0",
            "--inverse", "0",
            "--llm_rule_path", "./llm_rules/generated_rules/ETTm1_peak_rules.json"
        )

        $BaselineCommand = @($PythonExe) + $CommonArgs + @(
            "--learning_rate", "$BaselineLearningRate",
            "--early_stop_metric", "base_mse",
            "--selection_metric", "base_mse",
            "--use_llm_features", "0",
            "--use_llm_rule_features", "0",
            "--use_dataset_aware_loss", "0",
            "--use_event_weighted_loss", "0",
            "--use_zero_consistency_loss", "0",
            "--use_peak_shape_loss", "0",
            "--use_diff_loss", "0",
            "--use_freq_loss", "0",
            "--des", $BaselineDes
        )
        Invoke-Checked -Name "ETTm1 pred_len=$PredLen seed=$Seed baseline" -Command $BaselineCommand

        $EventCommand = @($PythonExe) + $CommonArgs + @(
            "--learning_rate", "$($FinetuneConfig.LearningRate)",
            "--load_pretrained_checkpoint", "./checkpoints/$BaselineSetting/checkpoint.pth",
            "--early_stop_metric", "base_mse",
            "--selection_metric", "guarded_event_mse",
            "--overall_mse_tolerance", "$OverallMseTolerance",
            "--baseline_metric_path", "./checkpoints/$BaselineSetting/validation_history.json",
            "--use_llm_features", "0",
            "--use_llm_rule_features", "0",
            "--use_dataset_aware_loss", "1",
            "--des", $EventDes
        ) + $FinetuneConfig.ExtraArgs
        Invoke-Checked -Name "ETTm1 pred_len=$PredLen seed=$Seed loss expert" -Command $EventCommand

        $GatedCommand = @(
            $PythonExe,
            "analysis/evaluate_rule_gated_ensemble.py",
            "--baseline_result_dir", "./results/$BaselineSetting",
            "--event_result_dir", "./results/$EventSetting",
            "--output_dir", "./results/$GatedSetting",
            "--alpha", "1.0"
        )
        Invoke-Checked -Name "ETTm1 pred_len=$PredLen seed=$Seed gated evaluation" -Command $GatedCommand
    }
}

$PredLensCsv = ($PredLens -join ",")
$SeedsCsv = ($Seeds -join ",")
$SummaryCommand = @(
    $PythonExe,
    "analysis/summarize_peak_transfer_seed_stability.py",
    "--pred_lens", $PredLensCsv,
    "--seeds", $SeedsCsv,
    "--output_csv", "artifacts/core_results/ettm1_peak_transfer_seed_stability.csv",
    "--output_json", "artifacts/core_results/ettm1_peak_transfer_seed_stability.json"
)
Invoke-Checked -Name "ETTm1 seed stability summary" -Command $SummaryCommand

param(
    [string[]]$Datasets = @("ETTm1"),
    [int[]]$PredLens = @(192, 336, 720),
    [string]$RootPath = "./data/",
    [string]$Features = "M",
    [string]$Target = "OT",
    [int]$SeqLen = 336,
    [int]$LabelLen = 48,
    [int]$BatchSize = 8,
    [double]$BaselineLearningRate = 0.0001,
    [double]$FinetuneLearningRate = 0.00001,
    [int]$TrainEpochs = 10,
    [int]$Patience = 3,
    [double]$OverallMseTolerance = 0.03,
    [double]$GateAlpha = 1.0,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$DefaultMyFramPython = "C:\Users\33932\.conda\envs\my_fram\python.exe"
if ($PythonExe -eq "python" -and (Test-Path $DefaultMyFramPython)) {
    $PythonExe = $DefaultMyFramPython
}

function Get-RulePath {
    param([string]$Data)
    if ($Data -eq "ETTm1") {
        return "./llm_rules/generated_rules/ETTm1_peak_rules.json"
    }
    return "./llm_rules/generated_rules/$Data`_peak_transfer_rules.json"
}

function Invoke-Checked {
    param([string]$Name, [string[]]$Command)
    Write-Host "==> $Name"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Name"
    }
}

foreach ($Data in $Datasets) {
    $Lower = $Data.ToLowerInvariant()
    $DataPath = "$Data.csv"
    $Freq = if ($Data.StartsWith("ETTm")) { "t" } else { "h" }
    $RulePath = Get-RulePath -Data $Data

    foreach ($PredLen in $PredLens) {
        $BaselineDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_baseline"
        $EventDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_finetune_loss"
        $GatedDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_gated_alpha_1p0"
        $BaselineSetting = "long_term_forecast_DLinear_$Data`_ft$Features`_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$BaselineDes`_0"
        $EventSetting = "long_term_forecast_DLinear_$Data`_ft$Features`_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$EventDes`_0"
        $GatedSetting = "long_term_forecast_DLinear_$Data`_ft$Features`_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$GatedDes`_0"

        $CommonArgs = @(
            "main.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model", "DLinear",
            "--data", $Data,
            "--root_path", $RootPath,
            "--data_path", $DataPath,
            "--features", $Features,
            "--target", $Target,
            "--freq", $Freq,
            "--seq_len", "$SeqLen",
            "--label_len", "$LabelLen",
            "--pred_len", "$PredLen",
            "--enc_in", "7",
            "--c_out", "7",
            "--batch_size", "$BatchSize",
            "--train_epochs", "$TrainEpochs",
            "--patience", "$Patience",
            "--dlinear_init_avg", "0",
            "--use_zscore", "1",
            "--use_revin", "0",
            "--use_standard_time_features", "0",
            "--use_oracle_features", "0",
            "--use_rule_adapter", "0",
            "--use_intervention_layer", "0",
            "--use_hard_intervention", "0",
            "--inverse", "0",
            "--llm_rule_path", $RulePath
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
        Invoke-Checked -Name "$Data pred_len=$PredLen baseline" -Command $BaselineCommand

        $EventCommand = @($PythonExe) + $CommonArgs + @(
            "--learning_rate", "$FinetuneLearningRate",
            "--load_pretrained_checkpoint", "./checkpoints/$BaselineSetting/checkpoint.pth",
            "--early_stop_metric", "base_mse",
            "--selection_metric", "guarded_event_mse",
            "--overall_mse_tolerance", "$OverallMseTolerance",
            "--baseline_metric_path", "./checkpoints/$BaselineSetting/validation_history.json",
            "--use_llm_features", "0",
            "--use_llm_rule_features", "0",
            "--use_dataset_aware_loss", "1",
            "--des", $EventDes
        )
        Invoke-Checked -Name "$Data pred_len=$PredLen loss expert" -Command $EventCommand

        $GatedCommand = @(
            $PythonExe,
            "analysis/evaluate_rule_gated_ensemble.py",
            "--baseline_result_dir", "./results/$BaselineSetting",
            "--event_result_dir", "./results/$EventSetting",
            "--output_dir", "./results/$GatedSetting",
            "--alpha", "$GateAlpha"
        )
        Invoke-Checked -Name "$Data pred_len=$PredLen gated evaluation" -Command $GatedCommand
    }
}

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
    [int]$GenerateRules = 1,
    [int]$RuleMiningPredLen = 96,
    [string]$RuleModel = "gpt-5.5",
    [string]$RuleBaseUrl = "",
    [string]$RuleApiKeyEnv = "OPENAI_API_KEY",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$DefaultMyFramPython = "C:\Users\33932\.conda\envs\my_fram\python.exe"
if ($PythonExe -eq "python" -and (Test-Path $DefaultMyFramPython)) {
    $PythonExe = $DefaultMyFramPython
}

function Get-RulePath {
    param([string]$Data)
    $Candidate = "./llm_rules/generated_rules/$Data`_peak_transfer_rules.json"
    if (Test-Path $Candidate) {
        return $Candidate
    }
    if ($Data -eq "ETTm1") {
        return "./llm_rules/generated_rules/ETTm1_peak_rules.json"
    }
    return $Candidate
}

function Get-RuleReportPath {
    param([string]$Data)
    $Lower = $Data.ToLowerInvariant()
    return "./artifacts/core_results/$Lower`_peak_transfer_llm_rule_generation_report.json"
}

function Invoke-Checked {
    param([string]$Name, [string[]]$Command)
    Write-Host "==> $Name"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Name"
    }
}

function Ensure-DatasetRule {
    param(
        [string]$Data,
        [string]$DataPath,
        [string]$RulePath,
        [string]$RuleReportPath
    )
    if ($GenerateRules -eq 0) {
        if (-not (Test-Path $RulePath)) {
            throw "Missing dataset-level event rule file: $RulePath. Re-run with -GenerateRules 1 or create this file first."
        }
        return
    }

    $GenerateArgs = @(
        "analysis/generate_dataset_llm_rules.py",
        "--data", $Data,
        "--root_path", $RootPath,
        "--data_path", $DataPath,
        "--features", $Features,
        "--target", $Target,
        "--seq_len", "$SeqLen",
        "--pred_len", "$RuleMiningPredLen",
        "--model", $RuleModel,
        "--api_key_env", $RuleApiKeyEnv,
        "--output_rule_path", $RulePath,
        "--output_report_path", $RuleReportPath
    )
    if ($RuleBaseUrl -ne "") {
        $GenerateArgs += @("--base_url", $RuleBaseUrl)
    }

    Invoke-Checked -Name "$Data generate dataset-level event rules" -Command (@($PythonExe) + $GenerateArgs)
}

function Get-FinetuneConfig {
    param([string]$Data, [int]$PredLen, [double]$DefaultLearningRate)
    $Lower = $Data.ToLowerInvariant()
    if ($Data -eq "ETTm1" -and $PredLen -eq 336) {
        return @{
            Des = "$Lower`_gpt55_peak_transfer_p$PredLen`_strict_ew1_nopk_lr1e6"
            GatedDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_strict_gated_ew1_nopk_lr1e6"
            LearningRate = 0.000001
            ExtraArgs = @("--use_event_weighted_loss", "1", "--event_weight", "1.0", "--use_peak_shape_loss", "0", "--peak_weight", "0.0")
        }
    }
    if ($Data -eq "ETTm1" -and $PredLen -eq 720) {
        return @{
            Des = "$Lower`_gpt55_peak_transfer_p$PredLen`_strict_ew1_nopk_lr5e7"
            GatedDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_strict_gated_ew1_nopk_lr5e7"
            LearningRate = 0.0000005
            ExtraArgs = @("--use_event_weighted_loss", "1", "--event_weight", "1.0", "--use_peak_shape_loss", "0", "--peak_weight", "0.0")
        }
    }
    return @{
        Des = "$Lower`_gpt55_peak_transfer_p$PredLen`_finetune_loss"
        GatedDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_gated_alpha_1p0"
        LearningRate = $DefaultLearningRate
        ExtraArgs = @()
    }
}

foreach ($Data in $Datasets) {
    $Lower = $Data.ToLowerInvariant()
    $DataPath = "$Data.csv"
    $Freq = if ($Data.StartsWith("ETTm")) { "t" } else { "h" }
    $RulePath = Get-RulePath -Data $Data
    $RuleReportPath = Get-RuleReportPath -Data $Data

    Ensure-DatasetRule -Data $Data -DataPath $DataPath -RulePath $RulePath -RuleReportPath $RuleReportPath

    foreach ($PredLen in $PredLens) {
        $BaselineDes = "$Lower`_gpt55_peak_transfer_p$PredLen`_baseline"
        $FinetuneConfig = Get-FinetuneConfig -Data $Data -PredLen $PredLen -DefaultLearningRate $FinetuneLearningRate
        $EventDes = $FinetuneConfig.Des
        $GatedDes = $FinetuneConfig.GatedDes
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

param(
    [string[]]$Datasets = @("ETTm1", "ETTh1", "ETTh2", "ETTm2"),
    [string]$RootPath = "./data/",
    [string]$Features = "M",
    [string]$Target = "OT",
    [int]$SeqLen = 336,
    [int]$LabelLen = 48,
    [int]$PredLen = 96,
    [int]$BatchSize = 8,
    [double]$LearningRate = 0.0001,
    [int]$TrainEpochs = 10,
    [int]$Patience = 3,
    [string]$OpenAIBaseUrl = $(if ($env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL } else { "https://api.ruikon.com/v1" }),
    [string]$OpenAIModel = $(if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-5.5" }),
    [string]$OpenAIApiKeyEnv = $(if ($env:OPENAI_API_KEY_ENV) { $env:OPENAI_API_KEY_ENV } else { "OPENAI_API_KEY" })
)

$ErrorActionPreference = "Stop"

foreach ($Data in $Datasets) {
    $Lower = $Data.ToLowerInvariant()
    $DataPath = "$Data.csv"
    $RulePath = "./llm_rules/generated_rules/$Data`_rules.json"
    $RuleReport = "./artifacts/core_results/$Lower`_llm_rule_generation_report.json"
    $Freq = if ($Data.StartsWith("ETTm")) { "t" } else { "h" }

    & python analysis/generate_dataset_llm_rules.py `
        --data $Data `
        --root_path $RootPath `
        --data_path $DataPath `
        --features $Features `
        --target $Target `
        --seq_len $SeqLen `
        --profile_split train `
        --base_url $OpenAIBaseUrl `
        --model $OpenAIModel `
        --api_key_env $OpenAIApiKeyEnv `
        --output_rule_path $RulePath `
        --output_report_path $RuleReport
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $CommonArgs = @(
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
        "--learning_rate", "$LearningRate",
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

    $BaselineDes = "$Lower`_rulegate_baseline"
    $EventDes = "$Lower`_rulegate_generated_loss"
    $BaselineSetting = "long_term_forecast_DLinear_$Data`_ft$Features`_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$BaselineDes`_0"
    $EventSetting = "long_term_forecast_DLinear_$Data`_ft$Features`_sl$SeqLen`_ll$LabelLen`_pl$PredLen`_$EventDes`_0"

    & python main.py @CommonArgs `
        --early_stop_metric base_mse `
        --use_llm_features 0 `
        --use_llm_rule_features 0 `
        --use_dataset_aware_loss 0 `
        --use_event_weighted_loss 0 `
        --use_zero_consistency_loss 0 `
        --use_peak_shape_loss 0 `
        --use_diff_loss 0 `
        --use_freq_loss 0 `
        --des $BaselineDes
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Use the LLM-generated loss config from the rule JSON.
    & python main.py @CommonArgs `
        --early_stop_metric base_mse `
        --selection_metric guarded_event_mse `
        --overall_mse_tolerance 0.03 `
        --baseline_metric_path "./checkpoints/$BaselineSetting/validation_history.json" `
        --use_llm_features 0 `
        --use_llm_rule_features 0 `
        --use_dataset_aware_loss 1 `
        --des $EventDes
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

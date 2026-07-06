param(
    [string]$PythonExe = "python",
    [int]$ShiftSteps = 96,
    [int]$RandomSeed = 2024,
    [string]$OutputCsv = "artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.csv",
    [string]$OutputJson = "artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.json"
)

$ErrorActionPreference = "Stop"

$DefaultMyFramPython = "C:\Users\33932\.conda\envs\my_fram\python.exe"
if ($PythonExe -eq "python" -and (Test-Path $DefaultMyFramPython)) {
    $PythonExe = $DefaultMyFramPython
}

& $PythonExe "analysis/evaluate_peak_transfer_mask_ablation.py" `
    --output_csv $OutputCsv `
    --output_json $OutputJson `
    --shift_steps $ShiftSteps `
    --random_seed $RandomSeed

if ($LASTEXITCODE -ne 0) {
    throw "Mask ablation evaluation failed with exit code $LASTEXITCODE"
}

# Core Innovation Plan

## Method Boundary

This project is not an LLM-as-forecaster system. The LLM does not predict point values, does not run inside model
`forward`, does not run at batch-level training, and does not run during inference.

The LLM is a train-only dataset characteristic miner. It consumes train-split evidence and produces structured rule
JSON. The training and inference code only reads this JSON to build masks, auxiliary features, and dataset-aware loss
configuration.

## Train-Only Rule Mining

Run:

```bash
python -m llm_miner.run_miner --data ETTm1 --root_path ./data/ --data_path ETTm1.csv --features M --target OT --seq_len 336 --pred_len 96 --output_dir ./artifacts/llm_miner/ETTm1
```

Outputs:

- `dataset_summary.json`
- `candidate_rules.json`
- `figures/*.png`
- `llm_prompt.md`

No validation or test rows are summarized.

## Rule JSON To Training Signals

The rule JSON is converted into:

- event masks for event-window metrics and event-weighted losses;
- zero masks for scaled zero-consistency loss;
- peak masks for peak-shape loss;
- LLM rule auxiliary features such as event masks, days-to-event, and rule confidence.

Calendar-periodic rules must include an `anchor`. ETTh1 must not use `ETTm1_rules.json`.

## ETTm1 Core Ablation

Run:

```bash
bash scripts/run_ettm1_core_innovation.sh
```

The script compares pure DLinear, standard time features, LLM rule features, dataset-aware loss, feature/loss
combinations, and an oracle-like hard intervention. The hard intervention is not a deployable method result.

## Metrics

Report both overall and event-window metrics:

- normalized and original MSE/MAE;
- event-window MSE/MAE;
- zero-event MSE/MAE;
- rule consistency score;
- event and zero-event point counts.

Overall MSE should not materially degrade while event-window metrics improve.


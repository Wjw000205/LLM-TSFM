# Train-Only LLM Mining Protocol

The LLM is used once before model training. It is not called during model forward passes, batch training, validation, testing, or inference.

Workflow:

1. Build a train-only summary:

```bash
python -m llm_miner.build_dataset_summary --root_path ./data/ --data_path ETTm1.csv --data ETTm1 --target OT --seq_len 96 --output_path llm_miner/outputs/ETTm1_summary.json
```

2. Optionally build a train-only visualization:

```bash
python -m llm_miner.build_visualization --root_path ./data/ --data_path ETTm1.csv --data ETTm1 --target OT --seq_len 96 --output_path llm_miner/outputs/ETTm1_train.png
```

3. Build the prompt:

```bash
python -m llm_miner.build_llm_prompt --summary_path llm_miner/outputs/ETTm1_summary.json --output_path llm_miner/outputs/ETTm1_prompt.txt
```

4. Paste the prompt into an LLM outside the training loop and save its response.

5. Parse the response into rules:

```bash
python -m llm_miner.parse_llm_response --response_path llm_miner/outputs/ETTm1_response.json --output_path llm_rules/example_rules/ETTm1_rules.json
```

Only train split data is summarized. Validation and test data must not be used to mine rules.

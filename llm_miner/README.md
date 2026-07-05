# Train-Only LLM Mining Protocol

The LLM is used once before model training as a dataset-level characteristic miner. It is not called during model
forward passes, batch training, validation, testing, or inference.

Run the full offline miner without calling an API:

```bash
python -m llm_miner.run_miner \
  --data ETTm1 \
  --root_path ./data/ \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 336 \
  --pred_len 96 \
  --output_dir ./artifacts/llm_miner/ETTm1
```

This writes:

- `dataset_summary.json`
- `candidate_rules.json`
- `llm_prompt.md`
- `figures/*.png`

After manually saving an LLM response JSON, parse and validate it:

```bash
python -m llm_miner.parse_llm_response \
  --response_path ./artifacts/llm_miner/ETTm1/llm_response.json \
  --output_rule_path ./llm_rules/generated_rules/ETTm1_rules.json

python -m llm_miner.validate_rules \
  --rule_path ./llm_rules/generated_rules/ETTm1_rules.json \
  --data ETTm1 \
  --root_path ./data/ \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 336 \
  --output_dir ./artifacts/llm_miner/ETTm1
```

Only train split data is summarized. Validation and test rows must not be used to mine rules.

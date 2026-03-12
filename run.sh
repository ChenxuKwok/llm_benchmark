python your_script.py \
  --input_jsonl data/en_data.jsonl \
  --output_jsonl data/en_processed.jsonl \
  --error_jsonl data/en_processed.errors.jsonl \
  --gpus 0,1 \
  --pr_batch_size 16
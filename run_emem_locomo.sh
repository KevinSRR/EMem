#!/bin/bash

# LoCoMo EMem Evaluation Script
# This script runs the EMem evaluation on the LoCoMo dataset
#
# Requires OPENAI_API_KEY environment variable to be set
export SUPPORT_JSON_SCHEMA=true
export OUTPUT_INPUT_TOKEN_RATIO=1.2
export PYTHONUNBUFFERED=1
export LOG_LEVEL=INFO

python -u run_emem_locomo_cached.py --llm_model 'gpt-4o-mini' --embedding_model "text-embedding-3-small" --save_frequency 100 --save_dir 'outputs/emem_locomo_eval' --dataset_path "./data/locomo10.json"
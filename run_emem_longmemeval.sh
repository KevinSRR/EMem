#!/bin/bash

# LongMemEval EMem Evaluation Script
# This script runs the EMem evaluation on the LongMemEval dataset
#
# Usage:
#   ./run_emem_longmemeval.sh                 # Run with pre-caching + evaluation (default)
#   ./run_emem_longmemeval.sh --precache-only # Only pre-cache API calls
#   ./run_emem_longmemeval.sh --skip-precache # Skip pre-caching (use existing cache)


# Set environment variables
# Requires OPENAI_API_KEY environment variable to be set
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export SUPPORT_JSON_SCHEMA=true
export LOG_LEVEL=INFO
export OUTPUT_INPUT_TOKEN_RATIO=1.3
export PYTHONUNBUFFERED=1



# Configuration parameters
DATASET_PATH="data/longmemeval/preprocess/longmemeval_s_cleaned.json"

# Model configuration
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"

# Output configuration
SAVE_DIR="outputs/emem_longmemeval_eval"
mkdir -p "$SAVE_DIR"

# Parse command-line arguments
PRECACHE_ARGS=""
if [[ "$1" == "--precache-only" ]]; then
    PRECACHE_ARGS="--precache_only"
    echo "Running in PRE-CACHE ONLY mode"
elif [[ "$1" == "--skip-precache" ]]; then
    PRECACHE_ARGS="--skip_precache"
    echo "Running in SKIP PRE-CACHE mode (using existing cache)"
else
    echo "Running in DEFAULT mode (pre-cache + evaluation)"
fi


# Run the evaluation
python -u run_emem_longmemeval_cached.py \
    --dataset_path "$DATASET_PATH" \
    --llm_model "$LLM_MODEL" \
    --embedding_model "$EMBEDDING_MODEL" \
    --save_dir "$SAVE_DIR" \
    --save_frequency 50 \
    $PRECACHE_ARGS
    
echo ""
echo "Process completed. Results saved to: $SAVE_DIR"

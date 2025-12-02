# EMem Makefile
# Provides commands for running evaluation workflows on LoCoMo and LongMemEval datasets

.PHONY: help install eval-locomo eval-locomo-emem eval-longmemeval eval-longmemeval-emem \
        eval-longmemeval-precache eval-longmemeval-skip-precache \
        final-eval-locomo final-eval-longmemeval clean

# Default target
help:
	@echo "EMem: A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents"
	@echo ""
	@echo "Setup commands:"
	@echo "  make install                    - Install dependencies using uv"
	@echo ""
	@echo "Runtime Evaluation (EMem-G with PPR - default):"
	@echo "  make eval-locomo                - Run EMem-G on LoCoMo dataset"
	@echo "  make eval-longmemeval           - Run EMem-G on LongMemEval dataset"
	@echo ""
	@echo "Runtime Evaluation (EMem without PPR):"
	@echo "  make eval-locomo-emem           - Run EMem (no PPR) on LoCoMo dataset"
	@echo "  make eval-longmemeval-emem      - Run EMem (no PPR) on LongMemEval dataset"
	@echo ""
	@echo "Final Evaluation for Paper (standardized metrics):"
	@echo "  make final-eval-locomo RESULT_FILE=<path>      - Final eval on LoCoMo results"
	@echo "  make final-eval-longmemeval RESULT_FILE=<path> - Final eval on LongMemEval results"
	@echo ""
	@echo "Advanced LongMemEval commands:"
	@echo "  make eval-longmemeval-precache      - Only pre-cache API calls"
	@echo "  make eval-longmemeval-skip-precache - Skip pre-caching step"
	@echo ""
	@echo "Utility commands:"
	@echo "  make clean                      - Clean output directories"
	@echo ""
	@echo "Environment variables:"
	@echo "  OPENAI_API_KEY      - OpenAI API key (required)"
	@echo "  LLM_MODEL           - LLM model name (default: gpt-4o-mini)"
	@echo "  EMBEDDING_MODEL     - Embedding model name (default: text-embedding-3-small)"
	@echo "  SAVE_FREQUENCY      - Save frequency for intermediate results"
	@echo "  NUM_RUNS            - Number of LLM judge runs for final eval (default: 3)"
	@echo ""
	@echo "Examples:"
	@echo "  make eval-locomo LLM_MODEL=gpt-4o"
	@echo "  make eval-locomo-emem                              # EMem without PPR"
	@echo "  make final-eval-locomo RESULT_FILE=results/emem_locomo_results.pkl NUM_RUNS=3"

# Default configuration
LLM_MODEL ?= gpt-4o-mini
EMBEDDING_MODEL ?= text-embedding-3-small
LOCOMO_SAVE_FREQUENCY ?= 100
LONGMEMEVAL_SAVE_FREQUENCY ?= 50
NUM_RUNS ?= 3
MAX_CONCURRENT ?= 50

# Dataset paths
LOCOMO_DATASET ?= ./data/locomo10.json
LONGMEMEVAL_DATASET ?= data/longmemeval/preprocess/longmemeval_s_cleaned.json

# Output directories
LOCOMO_SAVE_DIR ?= outputs/emem_locomo_eval
LONGMEMEVAL_SAVE_DIR ?= outputs/emem_longmemeval_eval

# Environment setup for all commands
define setup_env
	export SUPPORT_JSON_SCHEMA=true && \
	export PYTHONUNBUFFERED=1 && \
	export LOG_LEVEL=INFO
endef

# =============================================================================
# SETUP
# =============================================================================

# Install dependencies using uv
install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Downloading required NLTK data..."
	uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
	@echo "Dependencies installed successfully."

# =============================================================================
# LOCOMO EVALUATION
# =============================================================================

# Run EMem-G (with PPR) evaluation on LoCoMo dataset (default)
eval-locomo:
	@echo "Running EMem-G (with PPR) evaluation on LoCoMo dataset..."
	@echo "  Model Variant: EMem-G (with PPR)"
	@echo "  LLM Model: $(LLM_MODEL)"
	@echo "  Embedding Model: $(EMBEDDING_MODEL)"
	@echo "  Dataset: $(LOCOMO_DATASET)"
	@echo "  Save Directory: $(LOCOMO_SAVE_DIR)"
	@echo ""
	$(call setup_env) && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.2 && \
	uv run python -u run_emem_locomo_cached.py \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_frequency $(LOCOMO_SAVE_FREQUENCY) \
		--save_dir '$(LOCOMO_SAVE_DIR)' \
		--dataset_path '$(LOCOMO_DATASET)' \
		--ratio_or_count 1

# Run EMem (without PPR) evaluation on LoCoMo dataset
eval-locomo-emem:
	@echo "Running EMem (without PPR) evaluation on LoCoMo dataset..."
	@echo "  Model Variant: EMem (without PPR)"
	@echo "  LLM Model: $(LLM_MODEL)"
	@echo "  Embedding Model: $(EMBEDDING_MODEL)"
	@echo "  Dataset: $(LOCOMO_DATASET)"
	@echo "  Save Directory: $(LOCOMO_SAVE_DIR)"
	@echo ""
	$(call setup_env) && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.2 && \
	uv run python -u run_emem_locomo_cached.py \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_frequency $(LOCOMO_SAVE_FREQUENCY) \
		--save_dir '$(LOCOMO_SAVE_DIR)' \
		--dataset_path '$(LOCOMO_DATASET)' \
		--skip_retrieval_ppr

# =============================================================================
# LONGMEMEVAL EVALUATION
# =============================================================================

# Run EMem-G (with PPR) evaluation on LongMemEval dataset (default)
eval-longmemeval:
	@echo "Running EMem-G (with PPR) evaluation on LongMemEval dataset..."
	@echo "  Model Variant: EMem-G (with PPR)"
	@echo "  LLM Model: $(LLM_MODEL)"
	@echo "  Embedding Model: $(EMBEDDING_MODEL)"
	@echo "  Dataset: $(LONGMEMEVAL_DATASET)"
	@echo "  Save Directory: $(LONGMEMEVAL_SAVE_DIR)"
	@echo ""
	@mkdir -p $(LONGMEMEVAL_SAVE_DIR)
	$(call setup_env) && \
	export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
	export TOKENIZERS_PARALLELISM=false && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.3 && \
	uv run python -u run_emem_longmemeval_cached.py \
		--dataset_path '$(LONGMEMEVAL_DATASET)' \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_dir '$(LONGMEMEVAL_SAVE_DIR)' \
		--save_frequency $(LONGMEMEVAL_SAVE_FREQUENCY)

# Run EMem (without PPR) evaluation on LongMemEval dataset
eval-longmemeval-emem:
	@echo "Running EMem (without PPR) evaluation on LongMemEval dataset..."
	@echo "  Model Variant: EMem (without PPR)"
	@echo "  LLM Model: $(LLM_MODEL)"
	@echo "  Embedding Model: $(EMBEDDING_MODEL)"
	@echo "  Dataset: $(LONGMEMEVAL_DATASET)"
	@echo "  Save Directory: $(LONGMEMEVAL_SAVE_DIR)"
	@echo ""
	@mkdir -p $(LONGMEMEVAL_SAVE_DIR)
	$(call setup_env) && \
	export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
	export TOKENIZERS_PARALLELISM=false && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.3 && \
	uv run python -u run_emem_longmemeval_cached.py \
		--dataset_path '$(LONGMEMEVAL_DATASET)' \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_dir '$(LONGMEMEVAL_SAVE_DIR)' \
		--save_frequency $(LONGMEMEVAL_SAVE_FREQUENCY) \
		--skip_retrieval_ppr

# Pre-cache only for LongMemEval
eval-longmemeval-precache:
	@echo "Pre-caching API calls for LongMemEval dataset..."
	@mkdir -p $(LONGMEMEVAL_SAVE_DIR)
	$(call setup_env) && \
	export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
	export TOKENIZERS_PARALLELISM=false && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.3 && \
	uv run python -u run_emem_longmemeval_cached.py \
		--dataset_path '$(LONGMEMEVAL_DATASET)' \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_dir '$(LONGMEMEVAL_SAVE_DIR)' \
		--precache_only

# Run LongMemEval evaluation without pre-caching (assumes cache exists)
eval-longmemeval-skip-precache:
	@echo "Running EMem-G evaluation on LongMemEval dataset (using existing cache)..."
	@mkdir -p $(LONGMEMEVAL_SAVE_DIR)
	$(call setup_env) && \
	export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
	export TOKENIZERS_PARALLELISM=false && \
	export OUTPUT_INPUT_TOKEN_RATIO=1.3 && \
	uv run python -u run_emem_longmemeval_cached.py \
		--dataset_path '$(LONGMEMEVAL_DATASET)' \
		--llm_model '$(LLM_MODEL)' \
		--embedding_model '$(EMBEDDING_MODEL)' \
		--save_dir '$(LONGMEMEVAL_SAVE_DIR)' \
		--save_frequency $(LONGMEMEVAL_SAVE_FREQUENCY) \
		--skip_precache

# =============================================================================
# FINAL EVALUATION (for paper reporting)
# =============================================================================

# Final evaluation for LoCoMo results (LLM judge + BLEU/F1/EM)
final-eval-locomo:
ifndef RESULT_FILE
	$(error RESULT_FILE is required. Usage: make final-eval-locomo RESULT_FILE=path/to/results.pkl)
endif
	@echo "Running final evaluation on LoCoMo results..."
	@echo "  Result file: $(RESULT_FILE)"
	@echo "  Number of LLM judge runs: $(NUM_RUNS)"
	@echo "  Max concurrent requests: $(MAX_CONCURRENT)"
	@echo ""
	$(call setup_env) && \
	uv run python -u analysis/final_eval.py \
		--result_file '$(RESULT_FILE)' \
		--dataset_type locomo \
		--num_runs $(NUM_RUNS) \
		--max_concurrent $(MAX_CONCURRENT)

# Final evaluation for LongMemEval results (LLM judge + BLEU/F1/EM)
final-eval-longmemeval:
ifndef RESULT_FILE
	$(error RESULT_FILE is required. Usage: make final-eval-longmemeval RESULT_FILE=path/to/results.pkl)
endif
	@echo "Running final evaluation on LongMemEval results..."
	@echo "  Result file: $(RESULT_FILE)"
	@echo "  Number of LLM judge runs: $(NUM_RUNS)"
	@echo "  Max concurrent requests: $(MAX_CONCURRENT)"
	@echo "  LongMemEval data path: $(LONGMEMEVAL_DATASET)"
	@echo ""
	$(call setup_env) && \
	uv run python -u analysis/final_eval.py \
		--result_file '$(RESULT_FILE)' \
		--dataset_type longmemeval \
		--longmemeval_path '$(LONGMEMEVAL_DATASET)' \
		--num_runs $(NUM_RUNS) \
		--max_concurrent $(MAX_CONCURRENT)

# =============================================================================
# UTILITIES
# =============================================================================

# Clean output directories
clean:
	@echo "Cleaning output directories..."
	rm -rf outputs/
	rm -rf results/
	rm -rf src/logs/
	@echo "Clean complete."

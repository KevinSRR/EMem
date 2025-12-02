"""
Evaluation on LongMemEval Dataset with Checkpoint/Resume Support

This script evaluates EMem on the LongMemEval dataset with advanced caching and
checkpoint/resume capabilities.

Key Features:
=============

1. PRE-CACHING (--precache_only, --skip_precache):
   - Pre-caches all LLM API calls for session structuring before evaluation
   - Dramatically speeds up evaluation by caching OpenIE/EDU extraction results
   - Can be run separately (--precache_only) or as part of full evaluation

2. CHECKPOINT/RESUME (--resume_from_result):
   - Resume evaluation from an existing result file
   - Automatically skips questions that have already been evaluated
   - Maintains correct result order even when dataset is enriched with new questions
   - Result file format is fully backward compatible

3. INTERMEDIATE SAVES (--save_frequency):
   - Periodically save results during evaluation (e.g., every N questions)
   - Protects against data loss from crashes or interruptions
   - Saved files are in the same format as final results
   - Can be used with --resume_from_result to continue interrupted runs

Usage Examples:
===============

# Full evaluation with intermediate saves every 50 questions
python run_emem_longmemeval_cached.py \
    --dataset_path data/longmemeval.json \
    --save_frequency 50

# Resume from existing result file (skip already-evaluated questions)
python run_emem_longmemeval_cached.py \
    --dataset_path data/longmemeval.json \
    --resume_from_result results/emem_longmemeval_results_2024-01-15-10-30.pkl \
    --save_frequency 50

# Pre-cache only (prepare for fast evaluation later)
python run_emem_longmemeval_cached.py \
    --dataset_path data/longmemeval.json \
    --precache_only

# Evaluate with pre-existing cache (skip pre-caching step)
python run_emem_longmemeval_cached.py \
    --dataset_path data/longmemeval.json \
    --skip_precache \
    --save_frequency 50

Dataset Enrichment Scenario:
============================
If you've already evaluated the dataset and later enrich it with new questions:

1. Original evaluation:
   python run_emem_longmemeval_cached.py --dataset_path data/v1.json

2. Dataset enriched with new questions → data/v2.json

3. Resume with enriched dataset (only evaluates new questions):
   python run_emem_longmemeval_cached.py \
       --dataset_path data/v2.json \
       --resume_from_result results/emem_results_v1.pkl \
       --save_frequency 50

The script intelligently detects which questions are new and only evaluates those,
while maintaining the correct order of all results (cached + new).

Technical Details:
==================
- Questions are identified by (sample_id, question) tuple
- sample_id uses the actual sample.sample_id from the dataset (not loop index)
- This ensures robustness when dataset order changes or is enriched
- Results are always in the same order as they appear in the dataset
- Intermediate saves are atomic (writes to .tmp then renames)
"""

import os
import json

from src.emem import EMem
from src.emem.utils.misc_utils import string_to_bool, compute_mdhash_id
from src.emem.utils.config_utils import BaseConfig
from src.emem.evaluation.locomo_eval import LoCoMoEvaluationNew
from src.emem.llm.openai_gpt_batch import CacheOpenAI
from src.emem.utils.conversation_data_utils import load_locomo_dataset, QA, Turn, Session, Conversation, LoCoMoSample
# Import the conversion script
from data.longmemeval.preprocess.longmemeval_to_locomo_converter import convert_longmemeval_to_locomo

from dataclasses import asdict
import argparse

import logging
import sys

# Configure logging to capture all logger messages
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)

logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs go to stdout
    ],
    force=True  # Force reconfiguration
)

# Set the root logger level to capture all sub-loggers
logging.getLogger().setLevel(numeric_level)

# Also set specific logger levels for EMem components
logging.getLogger('src.emem').setLevel(numeric_level)
logging.getLogger('__main__').setLevel(numeric_level)

print(f"Logging configured with level: {log_level}")
print("="*50)

from pathlib import Path
import numpy as np

import nltk
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import pickle
import random
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime




def _generate_final_aggregate_from_tracker(rolling_metrics_tracker):
    """Generate final aggregate results from the rolling tracker without re-evaluation."""
    import statistics
    
    aggregate_results = {}
    
    # Generate overall statistics
    overall_metrics = rolling_metrics_tracker['overall']
    if overall_metrics['exact_match']:
        aggregate_results['overall'] = {}
        for metric_name, values in overall_metrics.items():
            if values:
                aggregate_results['overall'][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    # Generate per-category statistics
    category_metrics = rolling_metrics_tracker['categories']
    for category, cat_data in category_metrics.items():
        if cat_data['exact_match']:
            aggregate_results[f'category_{category}'] = {}
            for metric_name, values in cat_data.items():
                if values:
                    aggregate_results[f'category_{category}'][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
    
    return aggregate_results


def _display_rolling_results(logger, rolling_metrics_tracker, samples_processed, total_samples):
    """Display rolling evaluation results efficiently without re-evaluation."""
    import statistics
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ROLLING EVALUATION RESULTS - After Sample {samples_processed}/{total_samples}")
    logger.info(f"{'='*60}")
    
    # Calculate overall rolling metrics
    overall_metrics = rolling_metrics_tracker['overall']
    total_processed = len(overall_metrics['exact_match'])
    logger.info(f"Total Questions Processed: {total_processed}")
    logger.info(f"Samples Processed: {samples_processed}/{total_samples}")
    
    if total_processed > 0:
        logger.info(f"\nOverall Rolling Metrics:")
        for metric_name, values in overall_metrics.items():
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                logger.info(f"  {metric_name.replace('_', ' ').title()}: {mean_val:.4f} (±{std_val:.4f})")
        
        # Calculate per-category rolling metrics
        category_metrics = rolling_metrics_tracker['categories']
        if category_metrics:
            logger.info(f"\nPer-Category Rolling Metrics:")
            for category in sorted(category_metrics.keys()):
                cat_data = category_metrics[category]
                count = len(cat_data['exact_match'])
                if count > 0:
                    logger.info(f"  Category {category} ({count} questions):")
                    em_mean = statistics.mean(cat_data['exact_match'])
                    f1_mean = statistics.mean(cat_data['f1_score'])
                    bleu_mean = statistics.mean(cat_data['bleu_score'])
                    llm_mean = statistics.mean(cat_data['llm_score'])
                    logger.info(f"    EM: {em_mean:.4f}, F1: {f1_mean:.4f}, BLEU: {bleu_mean:.4f}, LLM: {llm_mean:.4f}")
    
    logger.info(f"{'='*60}\n")


def load_existing_results(result_file_path: str, logger: logging.Logger) -> tuple:
    """
    Load existing results from a checkpoint file.
    
    Args:
        result_file_path: Path to existing result pickle file
        logger: Logger instance
        
    Returns:
        Tuple of (existing_results_dict, result_lookup, loaded_final_results)
        - existing_results_dict: Dict with all final_results structure
        - result_lookup: Dict mapping (sample_id, question) -> result for fast lookup
        - loaded_final_results: The complete loaded results structure
    """
    if not os.path.exists(result_file_path):
        logger.info(f"No existing result file found at {result_file_path}")
        return None, {}, None
    
    try:
        logger.info(f"Loading existing results from {result_file_path}")
        with open(result_file_path, 'rb') as f:
            loaded_results = pickle.load(f)
        
        # Build lookup dict for fast checking: (sample_id, question) -> result
        # Note: sample_id here refers to the actual sample.sample_id from the dataset,
        # not the loop index, to ensure robustness across dataset modifications
        result_lookup = {}
        if 'individual_results' in loaded_results:
            for result in loaded_results['individual_results']:
                # Use the sample_id stored in the result (which should be sample.sample_id)
                key = (result['sample_id'], result['question'])
                result_lookup[key] = result
        
        logger.info(f"Loaded {len(result_lookup)} existing question results")
        logger.info(f"  Total samples in checkpoint: {loaded_results.get('total_samples', 'unknown')}")
        logger.info(f"  Total questions in checkpoint: {loaded_results.get('total_questions', 'unknown')}")
        
        return loaded_results, result_lookup, loaded_results
        
    except Exception as e:
        logger.warning(f"Error loading existing results: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None, {}, None


def save_intermediate_results(output_path: str, final_results: dict, logger: logging.Logger):
    """
    Save intermediate results in the same format as final results.
    
    Args:
        output_path: Path to save the results
        final_results: Dictionary containing all results
        logger: Logger instance
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to temporary file first, then rename for atomicity
        temp_path = output_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(final_results, f)
        
        # Atomic rename
        os.replace(temp_path, output_path)
        
        logger.info(f"Intermediate results saved to {output_path}")
        logger.info(f"  Questions saved: {final_results.get('total_questions', 0)}")
        
    except Exception as e:
        logger.warning(f"Error saving intermediate results: {e}")
        import traceback
        logger.warning(traceback.format_exc())


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('longmemeval_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_longmemeval_dataset(file_path: str, converted_file_path: str = None, max_samples: int = None) -> List[LoCoMoSample]:
    """
    Load LongMemEval dataset and convert to LoCoMo format if needed.
    
    Args:
        file_path: Path to original LongMemEval JSON file
        converted_file_path: Path to converted LoCoMo format file (will be created if doesn't exist)
        max_samples: Maximum number of samples to load
        
    Returns:
        List of LoCoMoSample objects
    """
    if converted_file_path is None:
        converted_file_path = file_path.replace('.json', '_locomo_format.json')
    
    # Check if converted file exists and is newer than original
    if (os.path.exists(converted_file_path) and 
        os.path.getmtime(converted_file_path) > os.path.getmtime(file_path)):
        logging.info(f"Loading pre-converted LoCoMo format from {converted_file_path}")
        samples = load_locomo_dataset(converted_file_path)
    else:
        logging.info(f"Converting LongMemEval to LoCoMo format...")
        convert_longmemeval_to_locomo(file_path, converted_file_path, max_samples=max_samples)
        samples = load_locomo_dataset(converted_file_path)
    
    # Apply max_samples limit if specified
    if max_samples is not None:
        if max_samples >= 1.0:
            samples = samples[:int(max_samples)]
        elif max_samples > 0:
            samples = samples[:int(max_samples * len(samples))]
        logging.info(f"Limited to {len(samples)} samples (max_samples={max_samples})")
    
    return samples


def precache_all_sessions(args, mem_model_config: BaseConfig, llm_model_name: str, embedding_model_name: str):
    """
    Pre-cache all session structuring API calls for LongMemEval dataset.
    
    This function collects all unique sessions from all test samples, batches them,
    and runs batch openie to cache all API calls.
    This dramatically speeds up the actual evaluation since all API responses are cached.
    
    Args:
        args: Command line arguments
        mem_model_config: memory model configuration
        llm_model_name: LLM model name
        embedding_model_name: Embedding model name
    """
    logger = logging.getLogger('longmemeval_eval')
    logger.info("="*70)
    logger.info("STARTING PRE-CACHING OF ALL SESSION STRUCTURING")
    logger.info("="*70)
    
    # Load all test samples
    logger.info(f"Loading dataset from {args.dataset_path}")
    test_samples: List[LoCoMoSample] = load_longmemeval_dataset(args.dataset_path, max_samples=args.ratio_or_count)
    logger.info(f"Loaded {len(test_samples)} samples")
    
    # Verify speaker names are consistent across all samples
    logger.info("Verifying speaker names consistency...")
    for sample_idx, sample in enumerate(test_samples):
        speaker_a = sample.conversation.speaker_a
        speaker_b = sample.conversation.speaker_b
        assert speaker_a == "User", f"Sample {sample_idx} has unexpected speaker_a: {speaker_a}, expected 'User'"
        assert speaker_b == "Assistant", f"Sample {sample_idx} has unexpected speaker_b: {speaker_b}, expected 'Assistant'"
    logger.info("✓ All samples have consistent speaker names: User and Assistant")
    
    # Collect all unique sessions across all samples
    # Use session hash as key to deduplicate (some samples may share sessions)
    unique_sessions = {}  # session_hash -> Session object
    session_to_samples = defaultdict(list)  # session_hash -> list of sample_ids that use this session
    
    logger.info("Collecting all unique sessions across samples...")
    
    for sample_idx, sample in enumerate(test_samples):
        for session_id, session in sample.conversation.sessions.items():
            # Create a hash for the session based on its content
            # Using the same hashing mechanism as ContentStore
            session_dict = asdict(session)
            session_str = json.dumps(session_dict, sort_keys=True, default=str)
            session_hash = compute_mdhash_id(session_str, prefix="session-")
            
            if session_hash not in unique_sessions:
                unique_sessions[session_hash] = session
            session_to_samples[session_hash].append(sample.sample_id)
    
    logger.info(f"Found {len(unique_sessions)} unique sessions across {len(test_samples)} samples")
    logger.info(f"Average sessions per sample: {sum(len(s.conversation.sessions) for s in test_samples) / len(test_samples):.2f}")
    logger.info(f"Session reuse statistics:")
    reuse_counts = defaultdict(int)
    for session_hash, sample_ids in session_to_samples.items():
        reuse_counts[len(sample_ids)] += 1
    for num_samples, count in sorted(reuse_counts.items()):
        logger.info(f"  {count} sessions used in {num_samples} sample(s)")
    
    # Initialize with a temporary save directory for pre-caching
    # IMPORTANT: Use the same base save_dir as evaluation to ensure cache is shared
    # The cache location is determined by os.path.dirname(global_config.save_dir)
    # Both pre-cache and eval should use subdirectories under args.save_dir so they share the cache
    precache_save_dir = os.path.join(args.save_dir, "precache_temp")
    expected_cache_dir = os.path.join(args.save_dir, "llm_cache")
    
    logger.info(f"Initializing for pre-caching")
    logger.info(f"  Pre-cache save_dir: {precache_save_dir}")
    logger.info(f"  Expected cache location: {expected_cache_dir}")
    logger.info(f"  Eval will use save_dir: {args.save_dir}/sample_{{id}}")
    logger.info(f"  Eval cache location: {expected_cache_dir} (same as pre-cache)")
    
    # Create a copy of config to avoid modifying the original
    from copy import deepcopy
    precache_config = deepcopy(mem_model_config)
    
    mem_model = EMem(
        save_dir=precache_save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        global_config=precache_config
    )
    
    # Verify the cache directory is correct
    actual_cache_dir = mem_model.llm_model.cache_dir
    logger.info(f"  Actual cache directory: {actual_cache_dir}")
    if actual_cache_dir != expected_cache_dir:
        logger.warning(f"  WARNING: Cache directory mismatch! Expected {expected_cache_dir}, got {actual_cache_dir}")
        logger.warning(f"  This means pre-cache and evaluation will use different caches!")
    else:
        logger.info(f"  ✓ Cache directory verified - pre-cache and eval will share the same cache")
    
    # Prepare sessions dict for batch processing
    sessions_dict = {session_hash: session for session_hash, session in unique_sessions.items()}
    speaker_names = ["User", "Assistant"]
    
    # Process sessions in batches to avoid memory issues
    BATCH_SIZE = 400
    session_items = list(sessions_dict.items())
    total_batches = (len(session_items) + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Starting batch structuring of {len(sessions_dict)} unique sessions...")
    logger.info(f"Processing in {total_batches} batches of up to {BATCH_SIZE} sessions each")
    logger.info("This will cache all API calls for later reuse during evaluation")
    
    # Run batch structuring in chunks - this will cache all API calls
    try:
        total_sessions_processed = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(session_items))
            batch_sessions = dict(session_items[start_idx:end_idx])
            
            logger.info(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_sessions)} sessions)...")
            
            # Run batch structuring for this batch - results are cached, we don't need to store them
            batch_results = mem_model.openie.batch_conversation_openie_w_indep_summary_uniee_separate_edus_v2_wo_session_summary(
                sessions=batch_sessions,
                speaker_names=speaker_names,
                skip_edu_context_gen=mem_model_config.skip_edu_context_gen,
            )
            
            total_sessions_processed += len(batch_results)
            logger.info(f"✓ Batch {batch_idx + 1} completed: {len(batch_results)} sessions processed")
            
            # Log some statistics for this batch
            batch_edus = sum(len(result['edus']) for result in batch_results.values())
            logger.info(f"  Batch EDUs extracted: {batch_edus}")
            logger.info(f"  Total sessions processed so far: {total_sessions_processed}/{len(sessions_dict)}")
        
        logger.info(f"\n✓ Successfully pre-cached structuring for all {total_sessions_processed} sessions")
        
    except Exception as e:
        logger.error(f"Error during pre-caching: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("="*70)
    logger.info("PRE-CACHING COMPLETED SUCCESSFULLY")
    logger.info("All API calls are now cached and will be reused during evaluation")
    logger.info("="*70)


def evaluate_dataset(args, mem_model_config: BaseConfig, llm_model_name: str, embedding_model_name: str):
    """Evaluate EMem on the LongMemEval dataset using conversation retrieval and QA."""

    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_emem_longmemeval_{llm_model_name.replace('/', '_')}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "src", "logs", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {args.dataset_path}")

    # Load and convert LongMemEval dataset 
    test_samples: List[LoCoMoSample] = load_longmemeval_dataset(args.dataset_path, max_samples=args.ratio_or_count)
    logger.info(f"Loaded {len(test_samples)} samples")
    
    # Determine output path (use resume file if provided, otherwise generate new)
    if args.resume_from_result:
        output_path = args.resume_from_result
        logger.info(f"Using existing result file: {output_path}")
    else:
        output_path = os.path.join(os.path.dirname(__file__), "results", f"emem_longmemeval_results_{timestamp}.pkl")
        logger.info(f"Will create new result file: {output_path}")
    
    # Load existing results if resuming
    existing_results, result_lookup, loaded_checkpoint = load_existing_results(
        output_path if args.resume_from_result else "", logger
    )
    
    if result_lookup:
        logger.info(f"Resume mode: Will skip {len(result_lookup)} already-evaluated questions")
        logger.info(f"Save frequency: Every {args.save_frequency} questions" if args.save_frequency else "Only at the end")

    # Initialize evaluation model for LLM judge
    evaluation_llm = CacheOpenAI.from_experiment_config(mem_model_config)
    evaluator = LoCoMoEvaluationNew(llm_model=evaluation_llm, global_config=mem_model_config)
    
    # Store results for batch evaluation
    all_questions = []
    all_gold_answers = []
    all_predictions = []
    all_categories = []
    sample_qa_mapping = []  # Track which sample each QA belongs to
    
    # Rolling evaluation tracking (incremental)
    rolling_metrics_tracker = {
        'overall': {'exact_match': [], 'f1_score': [], 'bleu_score': [], 'llm_score': []},
        'categories': defaultdict(lambda: {'exact_match': [], 'f1_score': [], 'bleu_score': [], 'llm_score': []})
    }
    
    # Store results
    results = []
    total_questions = 0
    questions_processed = 0  # Track for save_frequency
    questions_skipped = 0  # Track skipped questions
    category_counts = defaultdict(int)

    # Evaluate each sample
    error_num = 0
    allow_categories = [1,2,3,4]

    # Log expected cache location for verification
    expected_cache_dir = os.path.join(args.save_dir, "llm_cache")
    logger.info(f"\nExpected cache directory for all samples: {expected_cache_dir}")
    
    # Track if we need to initialize model for this sample (avoid if all questions cached)
    mem_model = None
    
    for sample_idx, sample in enumerate(test_samples):
        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(test_samples)}")
        
        # Collect all QA pairs for this sample and mark which are cached vs new
        sample_qa_list: List[QA] = []
        new_qa_list: List[QA] = []
        qa_is_cached = []  # Track which questions are cached (in order)
        
        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                sample_qa_list.append(qa)
                
                # Check if this question already has cached results
                # Use sample.sample_id (not loop index) for robustness
                cache_key = (sample.sample_id, qa.question)
                if cache_key in result_lookup:
                    qa_is_cached.append(True)
                else:
                    qa_is_cached.append(False)
                    new_qa_list.append(qa)

        if not sample_qa_list:
            logger.info(f"No valid QA pairs for sample {sample_idx}")
            continue
        
        num_cached = sum(qa_is_cached)
        num_new = len(new_qa_list)
        logger.info(f"Sample has {len(sample_qa_list)} questions: {num_cached} cached, {num_new} new")
        
        # Skip initialization if all questions were cached
        if not new_qa_list:
            logger.info(f"All questions cached for sample {sample_idx}, skipping RAG")
            
            # Add cached results in order and update metrics
            for qa in sample_qa_list:
                cache_key = (sample.sample_id, qa.question)
                cached_result = result_lookup[cache_key]
                results.append(cached_result)
                questions_skipped += 1
                
                # Update rolling metrics with cached result
                if 'exact_match' in cached_result:
                    category = cached_result['category']
                    rolling_metrics_tracker['overall']['exact_match'].append(cached_result['exact_match'])
                    rolling_metrics_tracker['overall']['f1_score'].append(cached_result['f1_score'])
                    rolling_metrics_tracker['overall']['bleu_score'].append(cached_result['bleu_score'])
                    rolling_metrics_tracker['overall']['llm_score'].append(cached_result['llm_score'])
                    
                    rolling_metrics_tracker['categories'][category]['exact_match'].append(cached_result['exact_match'])
                    rolling_metrics_tracker['categories'][category]['f1_score'].append(cached_result['f1_score'])
                    rolling_metrics_tracker['categories'][category]['bleu_score'].append(cached_result['bleu_score'])
                    rolling_metrics_tracker['categories'][category]['llm_score'].append(cached_result['llm_score'])
            
            # Display rolling results for cached-only sample (same as if it was freshly evaluated)
            _display_rolling_results(logger, rolling_metrics_tracker, sample_idx + 1, len(test_samples))
            
            # No need to check for periodic save here - only cached questions (no new work done)
            continue
        
        # Initialize EMem for this sample (only if we have new questions)
        sample_save_dir = os.path.join(args.save_dir, f"sample_{sample.sample_id}")
        mem_model = EMem(
            save_dir=sample_save_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            global_config=mem_model_config
        )
        
        # Verify cache directory on first sample with new questions
        if sample_idx == 0 or (sample_idx > 0 and questions_processed == 0):
            actual_cache_dir = mem_model.llm_model.cache_dir
            logger.info(f"First sample actual cache directory: {actual_cache_dir}")
            if actual_cache_dir != expected_cache_dir:
                logger.error(f"ERROR: Cache directory mismatch!")
                logger.error(f"  Expected: {expected_cache_dir}")
                logger.error(f"  Actual: {actual_cache_dir}")
                logger.error(f"  Pre-cached data will not be used!")
            else:
                logger.info(f"✓ Cache directory verified - using shared cache with pre-cached data")

        # Index the conversation for this sample
        logger.info("Indexing conversation sessions...")
        mem_model.index_conversation(sample)
        logger.info("Conversation indexed successfully")

        # Perform retrieval and QA for new questions only
        logger.info(f"Running RAG QA for {len(new_qa_list)} new questions...")
        
        try:
            # Use the new conversation RAG QA method
            qa_results = mem_model.rag_qa_conversation(
                queries=new_qa_list,  # Only process new questions
                gold_docs=None,  # No gold docs available in LongMemEval
                gold_answers=None  # Will be extracted from QA objects
            )
            

            if len(qa_results) == 5:
                queries_solutions, all_response_message, all_metadata, query_retrieval_traces, overall_qa_results = qa_results
                logger.info(f"Sample {sample_idx} QA results: {overall_qa_results}")
            elif len(qa_results) == 4:
                queries_solutions, all_response_message, all_metadata, query_retrieval_traces = qa_results
                overall_qa_results = None
            else:
                # Fallback for unexpected return format
                queries_solutions, all_response_message, all_metadata = qa_results[:3]
                query_retrieval_traces = []
                overall_qa_results = None
            
            # Build a mapping from question to query solution for new questions
            new_qa_results = {}
            for qa_idx, (qa, query_solution) in enumerate(zip(new_qa_list, queries_solutions)):
                # Extract token counts from metadata if available
                prompt_tokens = 0
                completion_tokens = 0
                if qa_idx < len(all_metadata):
                    metadata = all_metadata[qa_idx]
                    prompt_tokens = metadata.get('prompt_tokens', 0)
                    completion_tokens = metadata.get('completion_tokens', 0)
                
                # Extract query retrieval trace for this question
                query_trace = query_retrieval_traces[qa_idx] if qa_idx < len(query_retrieval_traces) else None
                
                new_qa_results[qa.question] = {
                    "prediction": query_solution.answer,
                    "retrieved_sessions": len(query_solution.docs) if query_solution.docs else 0,
                    "retrieved_edus": len(query_solution.edus) if query_solution.edus else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "query_retrieval_trace": query_trace
                }
            
            # Collect data for evaluation (new questions only, to be evaluated)
            sample_questions = []
            sample_gold_answers = []
            sample_predictions = []
            sample_categories = []
            
            # Process all questions in original order (mixing cached and new)
            for qa_idx, (qa, is_cached) in enumerate(zip(sample_qa_list, qa_is_cached)):
                if is_cached:
                    # Use cached result
                    cache_key = (sample.sample_id, qa.question)
                    cached_result = result_lookup[cache_key]
                    results.append(cached_result)
                    questions_skipped += 1
                    
                    # Update rolling metrics with cached result
                    if 'exact_match' in cached_result:
                        category = cached_result['category']
                        rolling_metrics_tracker['overall']['exact_match'].append(cached_result['exact_match'])
                        rolling_metrics_tracker['overall']['f1_score'].append(cached_result['f1_score'])
                        rolling_metrics_tracker['overall']['bleu_score'].append(cached_result['bleu_score'])
                        rolling_metrics_tracker['overall']['llm_score'].append(cached_result['llm_score'])
                        
                        rolling_metrics_tracker['categories'][category]['exact_match'].append(cached_result['exact_match'])
                        rolling_metrics_tracker['categories'][category]['f1_score'].append(cached_result['f1_score'])
                        rolling_metrics_tracker['categories'][category]['bleu_score'].append(cached_result['bleu_score'])
                        rolling_metrics_tracker['categories'][category]['llm_score'].append(cached_result['llm_score'])
                else:
                    # Use newly computed result
                    qa_result = new_qa_results[qa.question]
                    prediction = qa_result["prediction"]
                    
                    # Store for evaluation
                    sample_questions.append(qa.question)
                    sample_gold_answers.append([qa.final_answer] if qa.final_answer else [""])
                    sample_predictions.append(prediction)
                    sample_categories.append(qa.category)
                    
                    # Store individual result for final output (will be updated with metrics later)
                    result = {
                        "sample_id": sample.sample_id,  # Use actual sample ID, not loop index
                        "question": qa.question,
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "retrieved_sessions": qa_result["retrieved_sessions"],
                        "retrieved_edus": qa_result["retrieved_edus"],
                        "prompt_tokens": qa_result["prompt_tokens"],
                        "completion_tokens": qa_result["completion_tokens"],
                        "query_retrieval_trace": qa_result["query_retrieval_trace"]
                    }
                    results.append(result)
                    questions_processed += 1
            
            # Evaluate current sample's new questions only (efficient, one-time evaluation)
            if sample_questions:
                logger.info(f"\nEvaluating {len(sample_questions)} new questions from sample {sample_idx + 1}...")
                sample_aggregate_results, sample_example_results = evaluator.calculate_metric_scores(
                    gold_answers=sample_gold_answers,
                    predicted_answers=sample_predictions,
                    questions=sample_questions,
                    categories=sample_categories,
                    batch_size=len(sample_questions),  # Process all questions in this sample at once
                    real_time_logging=False  # Don't show detailed logging for individual sample
                )
                
                # Update results with sample evaluation metrics
                # Need to find the indices of new questions in results (skip cached ones)
                eval_result_idx = 0
                for result_idx in range(len(results) - len(sample_qa_list), len(results)):
                    result = results[result_idx]
                    # Check if this result has metrics (cached) or not (new)
                    if 'exact_match' not in result:
                        # This is a new result, update it with evaluation metrics
                        eval_result = sample_example_results[eval_result_idx]
                        result.update({
                            "exact_match": eval_result["exact_match"],
                            "f1_score": eval_result["f1"],
                            "bleu_score": eval_result["bleu1"],
                            "llm_score": eval_result["llm_score"]
                        })
                        
                        # Update rolling metrics incrementally (efficient)
                        category = sample_categories[eval_result_idx]
                        rolling_metrics_tracker['overall']['exact_match'].append(eval_result["exact_match"])
                        rolling_metrics_tracker['overall']['f1_score'].append(eval_result["f1"])
                        rolling_metrics_tracker['overall']['bleu_score'].append(eval_result["bleu1"])
                        rolling_metrics_tracker['overall']['llm_score'].append(eval_result["llm_score"])
                        
                        rolling_metrics_tracker['categories'][category]['exact_match'].append(eval_result["exact_match"])
                        rolling_metrics_tracker['categories'][category]['f1_score'].append(eval_result["f1"])
                        rolling_metrics_tracker['categories'][category]['bleu_score'].append(eval_result["bleu1"])
                        rolling_metrics_tracker['categories'][category]['llm_score'].append(eval_result["llm_score"])
                        
                        eval_result_idx += 1
                
                # Display rolling results efficiently (no re-evaluation)
                _display_rolling_results(logger, rolling_metrics_tracker, sample_idx + 1, len(test_samples))
                
                # Periodic save based on save_frequency (based on new questions processed)
                if args.save_frequency and questions_processed > 0 and questions_processed % args.save_frequency == 0:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Saving intermediate results (every {args.save_frequency} new questions)")
                    logger.info(f"Questions processed: {questions_processed}, Questions skipped: {questions_skipped}")
                    logger.info(f"{'='*50}")
                    
                    # Generate intermediate aggregate results
                    if rolling_metrics_tracker['overall']['exact_match']:
                        intermediate_aggregate = _generate_final_aggregate_from_tracker(rolling_metrics_tracker)
                    else:
                        intermediate_aggregate = {}
                    
                    # Prepare intermediate results
                    intermediate_results = {
                        "model": f"EMem-{llm_model_name}",
                        "embedding_model": embedding_model_name,
                        "dataset": args.dataset_path,
                        "total_questions": len(results),
                        "total_samples": len(test_samples),
                        "errors": error_num,
                        "category_distribution": {
                            str(cat): count for cat, count in category_counts.items()
                        },
                        "aggregate_metrics": intermediate_aggregate,
                        "individual_results": results,
                        "checkpoint_info": {
                            "questions_processed": questions_processed,
                            "questions_skipped": questions_skipped,
                            "samples_completed": sample_idx + 1,
                            "is_complete": False
                        }
                    }
                    
                    save_intermediate_results(output_path, intermediate_results, logger)

        except Exception as e:
            import traceback
            logger.warning(f"Error processing sample {sample_idx}: {e}")
            logger.warning(traceback.format_exc())
            error_num += 1
            continue


    # Generate final aggregate results from rolling tracker (no re-evaluation needed)
    logger.info("\nGenerating final aggregate results from accumulated data...")
    if rolling_metrics_tracker['overall']['exact_match']:
        aggregate_results = _generate_final_aggregate_from_tracker(rolling_metrics_tracker)
    else:
        logger.warning("No questions to evaluate")
        aggregate_results = {}
        
    # Prepare final results
    final_results = {
        "model": f"EMem-{llm_model_name}",
        "embedding_model": embedding_model_name,
        "dataset": args.dataset_path,
        "total_questions": total_questions,
        "total_samples": len(test_samples),
        "errors": error_num,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
        "checkpoint_info": {
            "questions_processed": questions_processed,
            "questions_skipped": questions_skipped,
            "samples_completed": len(test_samples),
            "is_complete": True
        }
    }
    
    logger.info(f"Error number: {error_num}")
    logger.info(f"Questions processed (new): {questions_processed}")
    logger.info(f"Questions skipped (cached): {questions_skipped}")
    
    # Save final results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_results, f)
    logger.info(f"Final results saved to {output_path}")
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info(f"Total samples processed: {len(test_samples)}")
    logger.info(f"Errors encountered: {error_num}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
    
    if aggregate_results:
        logger.info("\nFinal Aggregate Metrics:")
        for split_name, metrics in aggregate_results.items():
            logger.info(f"\n{split_name.replace('_', ' ').title()}:")
            for metric_name, stats in metrics.items():
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EMem on LongMemEval dataset")
    parser.add_argument("--dataset_path", type=str, default="./data/longmemeval/preprocess/longmemeval_s_cleaned.json", 
                       help="Path to the LongMemEval dataset file")
    parser.add_argument("--save_dir", type=str, default=None, 
                       help="Path to save evaluation results")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                       help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-small-3",
                       help="Embedding model name")
    parser.add_argument("--ratio_or_count", type=float, default=None,
                       help="Ratio of dataset to evaluate (0.0 to 1.0) or absolute count")
    parser.add_argument("--force_index_from_scratch", action="store_true",
                       help="Force indexing from scratch")
    parser.add_argument("--force_openie_from_scratch", action="store_true",
                       help="Force OpenIE from scratch")
    parser.add_argument("--precache_only", action="store_true",
                       help="Only run pre-caching of API calls without evaluation")
    parser.add_argument("--skip_precache", action="store_true",
                       help="Skip pre-caching step and run evaluation directly (assumes cache exists)")
    parser.add_argument("--save_frequency", type=int, default=None,
                       help="Save intermediate results after every N questions (None = only save at end)")
    parser.add_argument("--resume_from_result", type=str, default=None,
                       help="Path to existing result file to resume from (skip already-evaluated questions)")
    parser.add_argument("--skip_retrieval_ppr", action="store_true",
                       help="Skip PPR retrieval step (EMem mode). Default is EMem-G mode with PPR.")
    args = parser.parse_args()

    # Set default save directory if not provided
    if args.save_dir is None:
        args.save_dir = 'outputs/emem_longmemeval_eval'

    # Determine model variant name
    model_variant = "EMem" if args.skip_retrieval_ppr else "EMem-G"

    # EMem configuration for conversation evaluation
    mem_model_config = BaseConfig(
        max_new_tokens=2048 * 4,
        seed=42,
        temperature=0,
        force_openie_from_scratch=False,
        force_index_from_scratch=False,
        save_openie=True,
        openie_mode="edu_based_contextual_ee_online",
        embedding_batch_size=256,
        embedding_return_as_normalized=True,
        # embedding_max_seq_len=512,
        synonymy_edge_sim_threshold=0.9,
        linking_top_k=30,
        qa_top_k=10,
        date_format_type="longmemeval",  # Set the date format for LongMemEval
        save_dir=args.save_dir,  # Set save_dir to use the same base directory
        skip_edu_context_gen=True, # Skip EDU context generation
        skip_retrieval_ppr=args.skip_retrieval_ppr,
    )
    
    print(f"Starting {model_variant} evaluation on LongMemEval dataset")
    print(f"Model Variant: {model_variant} ({'without' if args.skip_retrieval_ppr else 'with'} PPR)")
    print(f"Dataset: {args.dataset_path}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Evaluation Ratio: {args.ratio_or_count or 'Full Eval Set'}")
    print(f"Pre-cache Only: {args.precache_only}")
    print(f"Skip Pre-cache: {args.skip_precache}")
    print(f"Save Frequency: {args.save_frequency or 'Only at end'}")
    print(f"Resume from Result: {args.resume_from_result or 'None (fresh start)'}")
    print("="*50)

    # Pre-caching step (runs first unless explicitly skipped)
    if not args.skip_precache:
        try:
            print("\n" + "="*70)
            print("STEP 1: PRE-CACHING API CALLS")
            print("="*70)
            precache_all_sessions(
                args=args,
                mem_model_config=mem_model_config,
                llm_model_name=args.llm_model,
                embedding_model_name=args.embedding_model
            )
            print("\n✓ Pre-caching completed successfully!")
            
            if args.precache_only:
                print("\nPre-cache only mode: Exiting without running evaluation")
                return None
                
        except Exception as e:
            print(f"\nError during pre-caching: {e}")
            if args.precache_only:
                raise
            else:
                print("Continuing with evaluation despite pre-caching error...")
    
    # Run evaluation
    if not args.precache_only:
        try:
            print("\n" + "="*70)
            print("STEP 2: RUNNING EVALUATION")
            print("="*70)
            final_results = evaluate_dataset(
                args=args, 
                mem_model_config=mem_model_config, 
                llm_model_name=args.llm_model, 
                embedding_model_name=args.embedding_model,
            )
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: results/emem_longmemeval_results_*.pkl")
            return final_results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise

if __name__ == "__main__":
    main()

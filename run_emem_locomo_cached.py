"""
Evaluation on LoCoMo Dataset with Checkpoint/Resume Support

This script evaluates EMem on the LoCoMo dataset with checkpoint/resume capabilities.

Key Features:
=============

1. CHECKPOINT/RESUME (--resume_from_result):
   - Resume evaluation from an existing result file
   - Automatically skips questions that have already been evaluated
   - Maintains correct result order even when dataset is enriched with new questions
   - Result file format is fully backward compatible

2. INTERMEDIATE SAVES (--save_frequency):
   - Periodically save results during evaluation (e.g., every N questions)
   - Protects against data loss from crashes or interruptions
   - Saved files are in the same format as final results
   - Can be used with --resume_from_result to continue interrupted runs

Usage Examples:
===============

# Full evaluation with intermediate saves every 25 questions
python run_emem_locomo_cached.py \
    --dataset_path data/locomo.json \
    --save_frequency 25

# Resume from existing result file (skip already-evaluated questions)
python run_emem_locomo_cached.py \
    --dataset_path data/locomo.json \
    --resume_from_result results/emem_locomo_results_2024-01-15-10-30.pkl \
    --save_frequency 25

Dataset Enrichment Scenario:
============================
If you've already evaluated the dataset and later enrich it with new questions:

1. Original evaluation:
   python run_emem_locomo_cached.py --dataset_path data/v1.json

2. Dataset enriched with new questions → data/v2.json

3. Resume with enriched dataset (only evaluates new questions):
   python run_emem_locomo_cached.py \
       --dataset_path data/v2.json \
       --resume_from_result results/emem_results_v1.pkl \
       --save_frequency 25

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
from src.emem.utils.misc_utils import string_to_bool
from src.emem.utils.config_utils import BaseConfig
from src.emem.evaluation.locomo_eval import LoCoMoEvaluationNew
from src.emem.llm.openai_gpt_batch import CacheOpenAI
from src.emem.utils.conversation_data_utils import load_locomo_dataset, QA, Turn, Session, Conversation, LoCoMoSample

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

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import nltk
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import pickle
import random
from tqdm import tqdm
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
    logger = logging.getLogger('locomo_eval')
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




def evaluate_dataset(args, mem_model_config: BaseConfig, llm_model_name: str, embedding_model_name: str):
    """Evaluate EMem on the LoCoMo dataset using conversation retrieval and QA with checkpoint/resume support."""

    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_emem_locomo_{llm_model_name.replace('/', '_')}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "src", "logs", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {args.dataset_path}")

    # Load dataset 
    test_samples: List[LoCoMoSample] = load_locomo_dataset(args.dataset_path)
    logger.info(f"Loaded {len(test_samples)} samples")
    
    # Select subset of samples based on ratio
    if args.ratio_or_count is not None:
        if args.ratio_or_count < 1.0:
            num_samples = max(1, int(len(test_samples) * args.ratio_or_count))
            test_samples = test_samples[:num_samples]
            logger.info(f"Using {num_samples} test samples ({args.ratio_or_count*100:.1f}% of dataset)")
        else:
            test_samples = test_samples[:int(args.ratio_or_count)]
            logger.info(f"Using {int(args.ratio_or_count)} test samples of dataset)")

    # Determine output path (use resume file if provided, otherwise generate new)
    if args.resume_from_result:
        output_path = args.resume_from_result
        logger.info(f"Using existing result file: {output_path}")
    else:
        output_path = os.path.join(os.path.dirname(__file__), "results", f"emem_locomo_results_{timestamp}.pkl")
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
    questions_processed = 0  # Track for save_frequency (only new questions)
    questions_skipped = 0  # Track skipped questions (cached)
    category_counts = defaultdict(int)

    # Evaluate each sample
    error_num = 0
    allow_categories = [1,2,3,4]
    
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
                gold_docs=None,  # No gold docs available in LoCoMo
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
    parser = argparse.ArgumentParser(description="Evaluate EMem on LoCoMo dataset with checkpoint/resume support")
    parser.add_argument("--dataset_path", type=str, default="./data/locomo10.json", 
                       help="Path to the dataset file")
    parser.add_argument("--save_dir", type=str, default=None, 
                       help="Path to save evaluation results")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                       help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-small-3",
                       help="Embedding model name")
    parser.add_argument("--ratio_or_count", type=float, default=None,
                       help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--force_index_from_scratch", action="store_true",
                       help="Force indexing from scratch")
    parser.add_argument("--force_openie_from_scratch", action="store_true",
                       help="Force OpenIE from scratch")
    parser.add_argument("--save_frequency", type=int, default=None,
                       help="Save intermediate results after every N questions (None = only save at end)")
    parser.add_argument("--resume_from_result", type=str, default=None,
                       help="Path to existing result file to resume from (skip already-evaluated questions)")
    parser.add_argument("--skip_retrieval_ppr", action="store_true",
                       help="Skip PPR retrieval step (EMem mode). Default is EMem-G mode with PPR.")
    args = parser.parse_args()

    # Set default save directory if not provided
    if args.save_dir is None:
        args.save_dir = 'outputs/emem_locomo_eval'

    # Determine model variant name
    model_variant = "EMem" if args.skip_retrieval_ppr else "EMem-G"

    # Configuration for conversation evaluation
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
        date_format_type="locomo",  # Explicitly set date format for LoCoMo dataset
        save_dir=args.save_dir,  # Set save_dir to use the same base directory
        skip_retrieval_ppr=args.skip_retrieval_ppr,
    )
    
    print(f"Starting {model_variant} evaluation on LoCoMo dataset")
    print(f"Model Variant: {model_variant} ({'without' if args.skip_retrieval_ppr else 'with'} PPR)")
    print(f"Dataset: {args.dataset_path}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Evaluation Ratio: {args.ratio_or_count or 'Full Eval Set'}")
    print(f"Save Frequency: {args.save_frequency or 'Only at end'}")
    print(f"Resume from Result: {args.resume_from_result or 'None (fresh start)'}")
    print("="*50)

    # Run evaluation
    try:
        final_results = evaluate_dataset(
            args=args, 
            mem_model_config=mem_model_config, 
            llm_model_name=args.llm_model, 
            embedding_model_name=args.embedding_model,
        )
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: results/emem_locomo_results_*.json")
        return final_results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
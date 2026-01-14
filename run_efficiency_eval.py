"""
Efficiency Evaluation Script for EMem and EMem-G

This script measures:
1. Average LLM input token consumption during the answering stage
2. Average retrieval time per query
3. Total answering stage time per sample

Aligns with Table 2 metrics in the Nemori paper (https://arxiv.org/pdf/2508.03341).
"""

import os
import json
import argparse
import logging
import sys
import time
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Configure logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)

logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger('efficiency_eval')


@dataclass
class EfficiencyMetrics:
    """Stores efficiency metrics for a single query."""
    query_idx: int = 0
    sample_id: str = ""
    question: str = ""
    
    # Token consumption (input tokens)
    retrieval_prompt_tokens: int = 0
    retrieval_completion_tokens: int = 0
    qa_prompt_tokens: int = 0
    qa_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    
    # Timing (seconds)
    retrieval_time: float = 0.0
    qa_time: float = 0.0
    total_time: float = 0.0
    
    # LLM call counts
    retrieval_llm_calls: int = 0
    qa_llm_calls: int = 0
    total_llm_calls: int = 0
    
    # Cache hit info
    cache_hits: int = 0
    api_calls: int = 0


@dataclass
class SampleEfficiencyMetrics:
    """Stores efficiency metrics aggregated for a sample."""
    sample_id: str = ""
    num_queries: int = 0
    
    # Aggregate token consumption
    total_retrieval_prompt_tokens: int = 0
    total_retrieval_completion_tokens: int = 0
    total_qa_prompt_tokens: int = 0
    total_qa_completion_tokens: int = 0
    
    # Aggregate timing
    indexing_time: float = 0.0
    total_retrieval_time: float = 0.0
    total_qa_time: float = 0.0
    total_answering_time: float = 0.0
    
    # Per-query averages
    avg_retrieval_prompt_tokens: float = 0.0
    avg_qa_prompt_tokens: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_qa_time: float = 0.0
    avg_answering_time: float = 0.0


class TokenTracker:
    """Tracks token consumption across LLM calls."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        self.cache_hits = 0
        self.api_calls = 0
    
    def add(self, prompt_tokens: int, completion_tokens: int, cache_hit: bool = False):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.call_count += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.api_calls += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'call_count': self.call_count,
            'cache_hits': self.cache_hits,
            'api_calls': self.api_calls
        }


def run_efficiency_evaluation(
    args,
    mem_model_config,
    llm_model_name: str,
    embedding_model_name: str
) -> Dict[str, Any]:
    """
    Run efficiency evaluation on the LoCoMo dataset.
    
    Measures token consumption and timing for both retrieval and QA stages.
    """
    from src.emem import EMem
    from src.emem.utils.config_utils import BaseConfig
    from src.emem.utils.conversation_data_utils import load_locomo_dataset, QA, LoCoMoSample
    
    logger.info(f"Loading dataset from {args.dataset_path}")
    test_samples: List[LoCoMoSample] = load_locomo_dataset(args.dataset_path)
    logger.info(f"Loaded {len(test_samples)} samples")
    
    # Select subset if specified
    if args.num_samples is not None:
        test_samples = test_samples[:args.num_samples]
        logger.info(f"Using {len(test_samples)} samples for evaluation")
    
    model_variant = "EMem" if args.skip_retrieval_ppr else "EMem-G"
    
    all_sample_metrics = []
    all_query_metrics = []
    allow_categories = [1, 2, 3, 4]
    
    total_start_time = time.time()
    
    for sample_idx, sample in enumerate(tqdm(test_samples, desc=f"Evaluating {model_variant}")):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {sample_idx + 1}/{len(test_samples)} (ID: {sample.sample_id})")
        
        sample_metrics = SampleEfficiencyMetrics(sample_id=sample.sample_id)
        
        # Filter valid QA pairs
        sample_qa_list = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        if not sample_qa_list:
            logger.info(f"No valid QA pairs for sample {sample_idx}")
            continue
        
        sample_metrics.num_queries = len(sample_qa_list)
        
        # Initialize EMem for this sample (fresh index per sample)
        sample_save_dir = os.path.join(args.save_dir, f"efficiency_sample_{sample.sample_id}")
        
        indexing_start = time.time()
        
        mem_model = EMem(
            save_dir=sample_save_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            global_config=mem_model_config
        )
        
        # Index the conversation
        logger.info("Indexing conversation sessions...")
        mem_model.index_conversation(sample)
        
        indexing_end = time.time()
        sample_metrics.indexing_time = indexing_end - indexing_start
        logger.info(f"Indexing completed in {sample_metrics.indexing_time:.2f}s")
        
        # Track token consumption during retrieval and QA
        retrieval_tracker = TokenTracker()
        qa_tracker = TokenTracker()
        
        # Wrap the LLM model to track token consumption
        original_batch_infer = mem_model.llm_model.batch_infer
        original_infer = mem_model.llm_model.infer
        
        current_tracker = retrieval_tracker
        
        def tracked_batch_infer(messages, **kwargs):
            nonlocal current_tracker
            results = original_batch_infer(messages, **kwargs)
            for result in results:
                if result is not None and len(result) >= 3:
                    _, metadata, cache_hit = result
                    prompt_tokens = metadata.get('prompt_tokens', 0)
                    completion_tokens = metadata.get('completion_tokens', 0)
                    current_tracker.add(prompt_tokens, completion_tokens, cache_hit)
            return results
        
        def tracked_infer(messages, **kwargs):
            nonlocal current_tracker
            result = original_infer(messages, **kwargs)
            if result is not None and len(result) >= 3:
                _, metadata, cache_hit = result
                prompt_tokens = metadata.get('prompt_tokens', 0)
                completion_tokens = metadata.get('completion_tokens', 0)
                current_tracker.add(prompt_tokens, completion_tokens, cache_hit)
            return result
        
        # Patch the LLM model
        mem_model.llm_model.batch_infer = tracked_batch_infer
        mem_model.llm_model.infer = tracked_infer
        
        # Also patch the openie module's LLM
        mem_model.openie.llm_model.batch_infer = tracked_batch_infer
        mem_model.openie.llm_model.infer = tracked_infer
        
        try:
            # === RETRIEVAL PHASE ===
            current_tracker = retrieval_tracker
            retrieval_start = time.time()
            
            # Perform retrieval
            retrieval_results, query_traces = mem_model.retrieve_structuring_conversation_enhanced_v2(
                queries=[qa.question for qa in sample_qa_list],
                gold_docs=None,
                no_query_trace_saving=True
            )
            
            retrieval_end = time.time()
            sample_metrics.total_retrieval_time = retrieval_end - retrieval_start
            
            retrieval_stats = retrieval_tracker.get_stats()
            sample_metrics.total_retrieval_prompt_tokens = retrieval_stats['prompt_tokens']
            sample_metrics.total_retrieval_completion_tokens = retrieval_stats['completion_tokens']
            
            logger.info(f"Retrieval: {sample_metrics.total_retrieval_time:.2f}s, "
                       f"Prompt tokens: {sample_metrics.total_retrieval_prompt_tokens}, "
                       f"Completion tokens: {sample_metrics.total_retrieval_completion_tokens}")
            
            # === QA PHASE ===
            current_tracker = qa_tracker
            qa_start = time.time()
            
            # Prepare QuerySolution objects with category info
            for i, query_solution in enumerate(retrieval_results):
                query_solution.category = sample_qa_list[i].category
                query_solution.expected_answer = sample_qa_list[i].final_answer
            
            # Perform QA
            queries_solutions, _, qa_metadata, _ = mem_model.qa_conversation(
                retrieval_results, 
                save_qa_traces=False
            )
            
            qa_end = time.time()
            sample_metrics.total_qa_time = qa_end - qa_start
            
            qa_stats = qa_tracker.get_stats()
            sample_metrics.total_qa_prompt_tokens = qa_stats['prompt_tokens']
            sample_metrics.total_qa_completion_tokens = qa_stats['completion_tokens']
            
            logger.info(f"QA: {sample_metrics.total_qa_time:.2f}s, "
                       f"Prompt tokens: {sample_metrics.total_qa_prompt_tokens}, "
                       f"Completion tokens: {sample_metrics.total_qa_completion_tokens}")
            
            # Calculate totals and averages
            sample_metrics.total_answering_time = (
                sample_metrics.total_retrieval_time + sample_metrics.total_qa_time
            )
            
            if sample_metrics.num_queries > 0:
                sample_metrics.avg_retrieval_prompt_tokens = (
                    sample_metrics.total_retrieval_prompt_tokens / sample_metrics.num_queries
                )
                sample_metrics.avg_qa_prompt_tokens = (
                    sample_metrics.total_qa_prompt_tokens / sample_metrics.num_queries
                )
                sample_metrics.avg_retrieval_time = (
                    sample_metrics.total_retrieval_time / sample_metrics.num_queries
                )
                sample_metrics.avg_qa_time = (
                    sample_metrics.total_qa_time / sample_metrics.num_queries
                )
                sample_metrics.avg_answering_time = (
                    sample_metrics.total_answering_time / sample_metrics.num_queries
                )
            
            # Store per-query metrics
            for q_idx, (qa, query_solution) in enumerate(zip(sample_qa_list, queries_solutions)):
                query_metrics = EfficiencyMetrics(
                    query_idx=q_idx,
                    sample_id=sample.sample_id,
                    question=qa.question,
                    retrieval_prompt_tokens=int(sample_metrics.total_retrieval_prompt_tokens / sample_metrics.num_queries),
                    qa_prompt_tokens=int(sample_metrics.total_qa_prompt_tokens / sample_metrics.num_queries),
                    total_prompt_tokens=int((sample_metrics.total_retrieval_prompt_tokens + sample_metrics.total_qa_prompt_tokens) / sample_metrics.num_queries),
                    retrieval_time=sample_metrics.avg_retrieval_time,
                    qa_time=sample_metrics.avg_qa_time,
                    total_time=sample_metrics.avg_answering_time
                )
                all_query_metrics.append(query_metrics)
            
            all_sample_metrics.append(sample_metrics)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        finally:
            # Restore original methods
            mem_model.llm_model.batch_infer = original_batch_infer
            mem_model.llm_model.infer = original_infer
    
    total_end_time = time.time()
    total_evaluation_time = total_end_time - total_start_time
    
    # Calculate aggregate statistics
    aggregate_results = calculate_aggregate_statistics(
        all_sample_metrics, 
        all_query_metrics,
        model_variant,
        total_evaluation_time
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_path = os.path.join(
        args.save_dir, 
        f"efficiency_results_{model_variant}_{timestamp}.json"
    )
    
    results = {
        "model_variant": model_variant,
        "llm_model": llm_model_name,
        "embedding_model": embedding_model_name,
        "dataset": args.dataset_path,
        "num_samples": len(all_sample_metrics),
        "total_queries": len(all_query_metrics),
        "total_evaluation_time": total_evaluation_time,
        "aggregate_results": aggregate_results,
        "sample_metrics": [asdict(m) for m in all_sample_metrics],
        "query_metrics": [asdict(m) for m in all_query_metrics]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print_efficiency_summary(aggregate_results, model_variant)
    
    return results


def calculate_aggregate_statistics(
    sample_metrics: List[SampleEfficiencyMetrics],
    query_metrics: List[EfficiencyMetrics],
    model_variant: str,
    total_time: float
) -> Dict[str, Any]:
    """Calculate aggregate statistics across all samples and queries."""
    
    if not sample_metrics:
        return {}
    
    # Per-sample aggregates
    indexing_times = [m.indexing_time for m in sample_metrics]
    retrieval_times = [m.total_retrieval_time for m in sample_metrics]
    qa_times = [m.total_qa_time for m in sample_metrics]
    answering_times = [m.total_answering_time for m in sample_metrics]
    
    retrieval_prompt_tokens = [m.total_retrieval_prompt_tokens for m in sample_metrics]
    qa_prompt_tokens = [m.total_qa_prompt_tokens for m in sample_metrics]
    
    # Per-query aggregates
    avg_retrieval_times = [m.avg_retrieval_time for m in sample_metrics]
    avg_qa_times = [m.avg_qa_time for m in sample_metrics]
    avg_answering_times = [m.avg_answering_time for m in sample_metrics]
    
    avg_retrieval_prompt = [m.avg_retrieval_prompt_tokens for m in sample_metrics]
    avg_qa_prompt = [m.avg_qa_prompt_tokens for m in sample_metrics]
    
    total_queries = sum(m.num_queries for m in sample_metrics)
    
    return {
        "model_variant": model_variant,
        "num_samples": len(sample_metrics),
        "total_queries": total_queries,
        
        # Token consumption (averaged per query)
        "avg_retrieval_input_tokens_per_query": np.mean(avg_retrieval_prompt),
        "avg_qa_input_tokens_per_query": np.mean(avg_qa_prompt),
        "avg_total_input_tokens_per_query": np.mean(avg_retrieval_prompt) + np.mean(avg_qa_prompt),
        
        "std_retrieval_input_tokens_per_query": np.std(avg_retrieval_prompt),
        "std_qa_input_tokens_per_query": np.std(avg_qa_prompt),
        
        # Timing (averaged per query)
        "avg_retrieval_time_per_query": np.mean(avg_retrieval_times),
        "avg_qa_time_per_query": np.mean(avg_qa_times),
        "avg_answering_time_per_query": np.mean(avg_answering_times),
        
        "std_retrieval_time_per_query": np.std(avg_retrieval_times),
        "std_qa_time_per_query": np.std(avg_qa_times),
        "std_answering_time_per_query": np.std(avg_answering_times),
        
        # Per-sample timing
        "avg_indexing_time_per_sample": np.mean(indexing_times),
        "avg_retrieval_time_per_sample": np.mean(retrieval_times),
        "avg_qa_time_per_sample": np.mean(qa_times),
        "avg_answering_time_per_sample": np.mean(answering_times),
        
        # Totals
        "total_indexing_time": sum(indexing_times),
        "total_retrieval_time": sum(retrieval_times),
        "total_qa_time": sum(qa_times),
        "total_answering_time": sum(answering_times),
        "total_evaluation_time": total_time,
        
        "total_retrieval_prompt_tokens": sum(retrieval_prompt_tokens),
        "total_qa_prompt_tokens": sum(qa_prompt_tokens),
        "total_prompt_tokens": sum(retrieval_prompt_tokens) + sum(qa_prompt_tokens),
    }


def print_efficiency_summary(aggregate: Dict[str, Any], model_variant: str):
    """Print a formatted summary of efficiency metrics."""
    
    print("\n" + "="*70)
    print(f"EFFICIENCY EVALUATION SUMMARY - {model_variant}")
    print("="*70)
    
    print(f"\nDataset Statistics:")
    print(f"  Samples evaluated: {aggregate.get('num_samples', 0)}")
    print(f"  Total queries: {aggregate.get('total_queries', 0)}")
    
    print(f"\nToken Consumption (Per Query Average):")
    print(f"  Retrieval stage: {aggregate.get('avg_retrieval_input_tokens_per_query', 0):.1f} tokens "
          f"(±{aggregate.get('std_retrieval_input_tokens_per_query', 0):.1f})")
    print(f"  QA stage: {aggregate.get('avg_qa_input_tokens_per_query', 0):.1f} tokens "
          f"(±{aggregate.get('std_qa_input_tokens_per_query', 0):.1f})")
    print(f"  Total: {aggregate.get('avg_total_input_tokens_per_query', 0):.1f} tokens")
    
    print(f"\nTiming (Per Query Average):")
    print(f"  Retrieval time: {aggregate.get('avg_retrieval_time_per_query', 0):.3f}s "
          f"(±{aggregate.get('std_retrieval_time_per_query', 0):.3f})")
    print(f"  QA time: {aggregate.get('avg_qa_time_per_query', 0):.3f}s "
          f"(±{aggregate.get('std_qa_time_per_query', 0):.3f})")
    print(f"  Total answering time: {aggregate.get('avg_answering_time_per_query', 0):.3f}s "
          f"(±{aggregate.get('std_answering_time_per_query', 0):.3f})")
    
    print(f"\nTiming (Per Sample Average):")
    print(f"  Indexing time: {aggregate.get('avg_indexing_time_per_sample', 0):.2f}s")
    print(f"  Retrieval time: {aggregate.get('avg_retrieval_time_per_sample', 0):.2f}s")
    print(f"  QA time: {aggregate.get('avg_qa_time_per_sample', 0):.2f}s")
    
    print(f"\nTotal Evaluation Time: {aggregate.get('total_evaluation_time', 0):.2f}s")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EMem Efficiency Evaluation")
    parser.add_argument("--dataset_path", type=str, default="./data/locomo10.json",
                       help="Path to the dataset file")
    parser.add_argument("--save_dir", type=str, default="outputs/efficiency_eval",
                       help="Directory to save results")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                       help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                       help="Embedding model name")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--skip_retrieval_ppr", action="store_true",
                       help="Skip PPR retrieval (EMem mode). Default is EMem-G with PPR.")
    parser.add_argument("--force_fresh_index", action="store_true",
                       help="Force fresh indexing for each sample (ignore cached indices)")
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from src.emem.utils.config_utils import BaseConfig
    
    # Configuration
    mem_model_config = BaseConfig(
        max_new_tokens=2048 * 4,
        seed=42,
        temperature=0,
        force_openie_from_scratch=args.force_fresh_index,
        force_index_from_scratch=args.force_fresh_index,
        save_openie=True,
        openie_mode="edu_based_contextual_ee_online",
        embedding_batch_size=256,
        embedding_return_as_normalized=True,
        synonymy_edge_sim_threshold=0.9,
        linking_top_k=30,
        qa_top_k=10,
        date_format_type="locomo",
        save_dir=args.save_dir,
        skip_retrieval_ppr=args.skip_retrieval_ppr,
    )
    
    model_variant = "EMem" if args.skip_retrieval_ppr else "EMem-G"
    
    print(f"\n{'='*60}")
    print(f"EMem Efficiency Evaluation")
    print(f"{'='*60}")
    print(f"Model Variant: {model_variant}")
    print(f"Dataset: {args.dataset_path}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Num Samples: {args.num_samples or 'All'}")
    print(f"Save Directory: {args.save_dir}")
    print(f"{'='*60}\n")
    
    results = run_efficiency_evaluation(
        args=args,
        mem_model_config=mem_model_config,
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model
    )
    
    return results


if __name__ == "__main__":
    main()




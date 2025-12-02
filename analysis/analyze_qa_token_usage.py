"""
Analyze QA Token Usage from EMem LongMemEval Results

This script analyzes the min and max input tokens for QA prompts (including retrieved memories)
by reading token counts directly from the result pickle file.

Usage:
    python analyze_qa_token_usage.py --result_file <path_to_pkl_file>
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

# Add the memory_conversation directory to sys.path to load custom objects from pickle
script_dir = Path(__file__).parent.parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

print(f"Added to sys.path: {script_dir}")


def load_result_file(result_path: str) -> dict:
    """Load the pickled result file."""
    print(f"Loading result file: {result_path}")
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_token_stats_from_results(results: dict) -> Dict[str, List[int]]:
    """
    Extract token statistics directly from individual results in the pickle file.
    
    Args:
        results: The loaded results dictionary
    
    Returns:
        Dictionary with lists of token counts
    """
    token_stats = defaultdict(list)
    individual_results = results.get('individual_results', [])
    
    if not individual_results:
        print("No individual results found in file")
        return {}
    
    print(f"Processing {len(individual_results)} individual results...")
    
    results_with_tokens = 0
    results_without_tokens = 0
    
    for result in individual_results:
        # Check if this result has token information
        if 'prompt_tokens' in result and 'completion_tokens' in result:
            prompt_tokens = result['prompt_tokens']
            completion_tokens = result['completion_tokens']
            
            if prompt_tokens > 0:  # Ensure valid token count
                token_stats['qa_input_tokens'].append(prompt_tokens)
                token_stats['qa_output_tokens'].append(completion_tokens)
                token_stats['qa_total_tokens'].append(prompt_tokens + completion_tokens)
                results_with_tokens += 1
        else:
            results_without_tokens += 1
    
    print(f"\n{'='*80}")
    print(f"Token Extraction Summary:")
    print(f"  Results with token counts: {results_with_tokens}")
    print(f"  Results without token counts: {results_without_tokens}")
    
    if results_without_tokens > 0:
        print(f"\n  ℹ Note: {results_without_tokens} results don't have token counts.")
        print(f"     This is normal for results from older runs or cached questions.")
        print(f"     To get token counts for all questions, re-run the evaluation.")
    
    return dict(token_stats)


def analyze_token_statistics(token_stats: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for token usage.
    
    Returns:
        Dictionary with min, max, mean, median, and percentiles
    """
    summary = {}
    
    for key, values in token_stats.items():
        if not values:
            continue
        
        summary[key] = {
            'count': len(values),
            'min': int(np.min(values)),
            'max': int(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
            'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)),
            'std': float(np.std(values)),
            'total': int(np.sum(values))
        }
    
    return summary


def print_summary(summary: Dict[str, Dict[str, float]], result_info: Optional[dict] = None):
    """Print a formatted summary of token statistics."""
    print("\n" + "="*80)
    print("QA TOKEN USAGE ANALYSIS")
    print("="*80)
    
    if result_info:
        print(f"\nResult File Information:")
        print(f"  Model: {result_info.get('model', 'Unknown')}")
        print(f"  Embedding Model: {result_info.get('embedding_model', 'Unknown')}")
        print(f"  Total Questions: {result_info.get('total_questions', 'Unknown')}")
        print(f"  Total Samples: {result_info.get('total_samples', 'Unknown')}")
    
    print("\n" + "-"*80)
    print("TOKEN STATISTICS FOR QA PROMPTS (including retrieved memories)")
    print("-"*80)
    
    for key, stats in sorted(summary.items()):
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  Count:      {stats['count']:,}")
        print(f"  Min:        {stats['min']:,} tokens")
        print(f"  Max:        {stats['max']:,} tokens")
        print(f"  Mean:       {stats['mean']:,.1f} tokens")
        print(f"  Median:     {stats['median']:,.1f} tokens")
        print(f"  Std Dev:    {stats['std']:,.1f} tokens")
        print(f"  25th %ile:  {stats['p25']:,.1f} tokens")
        print(f"  75th %ile:  {stats['p75']:,.1f} tokens")
        print(f"  90th %ile:  {stats['p90']:,.1f} tokens")
        print(f"  95th %ile:  {stats['p95']:,.1f} tokens")
        print(f"  99th %ile:  {stats['p99']:,.1f} tokens")
        print(f"  Total:      {stats['total']:,} tokens")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("-"*80)
    if 'qa_input_tokens' in summary:
        stats = summary['qa_input_tokens']
        print(f"✓ MIN QA Input Tokens:     {stats['min']:,} tokens")
        print(f"✓ MAX QA Input Tokens:     {stats['max']:,} tokens")
        print(f"✓ Average QA Input Tokens: {stats['mean']:,.1f} tokens")
        print(f"✓ Median QA Input Tokens:  {stats['median']:,.1f} tokens")
    if 'qa_output_tokens' in summary:
        stats = summary['qa_output_tokens']
        print(f"\n✓ MIN QA Output Tokens:     {stats['min']:,} tokens")
        print(f"✓ MAX QA Output Tokens:     {stats['max']:,} tokens")
        print(f"✓ Average QA Output Tokens: {stats['mean']:,.1f} tokens")
    if 'qa_total_tokens' in summary:
        stats = summary['qa_total_tokens']
        print(f"\n✓ MIN QA Total Tokens:     {stats['min']:,} tokens")
        print(f"✓ MAX QA Total Tokens:     {stats['max']:,} tokens")
        print(f"✓ Average QA Total Tokens: {stats['mean']:,.1f} tokens")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze QA token usage from EMem results")
    parser.add_argument("--result_file", type=str, required=True,
                       help="Path to the result pickle file")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Path to save JSON summary (optional)")
    
    args = parser.parse_args()
    
    # Load result file
    try:
        results = load_result_file(args.result_file)
        print(f"✓ Loaded results with {len(results.get('individual_results', []))} individual results")
    except Exception as e:
        print(f"✗ Error loading result file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract token statistics from results
    token_stats = extract_token_stats_from_results(results)
    
    if not token_stats or not any(token_stats.values()):
        print("\n✗ Error: No token statistics found in result file.")
        print("This means the result file was created before token tracking was added.")
        print("Please re-run the evaluation with the updated code to get token counts.")
        return
    
    # Analyze statistics
    summary = analyze_token_statistics(token_stats)
    
    # Print summary
    print_summary(summary, results)
    
    # Save to JSON if requested
    if args.output_json:
        output_data = {
            'result_file': args.result_file,
            'result_info': {
                'model': results.get('model'),
                'embedding_model': results.get('embedding_model'),
                'total_questions': results.get('total_questions'),
                'total_samples': results.get('total_samples'),
            },
            'token_statistics': summary
        }
        
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Summary saved to: {args.output_json}")


if __name__ == "__main__":
    main()

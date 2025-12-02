#!/usr/bin/env python3
"""
Final Evaluation Script for EMem

Computes standardized evaluation metrics for paper reporting:
1. LLM Judge scores (with multiple runs for mean/std)
2. BLEU-1, F1, and Exact Match scores

Supports both LoCoMo and LongMemEval datasets with dataset-specific LLM judge prompts.

Usage:
    python final_eval.py --result_file path/to/results.pkl --dataset_type locomo --num_runs 3
    python final_eval.py --result_file path/to/results.pkl --dataset_type longmemeval --num_runs 3
"""

import os
import sys
import json
import pickle
import argparse
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
from tqdm.asyncio import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
os.environ['SUPPORT_JSON_SCHEMA'] = 'true'

from openai import AsyncOpenAI
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: Error downloading NLTK data: {e}")


class Grade(BaseModel):
    """Pydantic model for LongMemEval LLM judge response."""
    is_correct: str = Field(description='yes or no')


@dataclass
class LLMJudgeRequest:
    """Stores a single LLM judge request."""
    task_id: int
    question: str
    gold_answer: str
    generated_answer: str
    category: str
    sample_id: int
    run_id: int
    dataset_type: str


# =============================================================================
# BLEU, F1, EM Calculation (exact copy from locomo_eval.py)
# =============================================================================

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function."""
    text = str(text)
    return text.lower().replace('.', ' ').replace(",", ' ').replace('!', ' ').replace('?', ' ').split()


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception as e:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores


def calculate_metrics_mem0_style(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics following mem0's approach."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
        }
    
    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate BLEU scores
    bleu_scores = calculate_bleu_scores(prediction, reference)
    
    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }
    
    return metrics


# =============================================================================
# Data Loading and Extraction
# =============================================================================

def load_results_from_pickle(result_file: str) -> Dict[str, Any]:
    """Load results from pickle file."""
    with open(result_file, 'rb') as f:
        results = pickle.load(f)
    return results


def load_longmemeval_original(data_path: str) -> Dict[str, str]:
    """Load original LongMemEval data and create mapping from sample to question_type."""
    with open(data_path, 'r') as f:
        original_data = json.load(f)
    
    sample_to_type = {}
    for idx, sample in enumerate(original_data):
        sample_to_type[idx] = sample.get('question_type', 'unknown')
    
    return sample_to_type


def extract_predictions_and_references(results: Dict[str, Any], 
                                       dataset_type: str,
                                       longmemeval_path: str = None) -> List[Dict[str, Any]]:
    """Extract predictions, references, and categories from results."""
    individual_results = results.get('individual_results', [])
    
    sample_to_type = {}
    if dataset_type == 'longmemeval':
        if longmemeval_path is None:
            raise ValueError("longmemeval_path is required for LongMemEval dataset")
        sample_to_type = load_longmemeval_original(longmemeval_path)
    
    extracted = []
    for item in individual_results:
        if dataset_type == 'locomo':
            category = str(item['category'])
        else:
            sample_id = item['sample_id']
            sample_id_int = int(sample_id) if isinstance(sample_id, str) else sample_id
            category = sample_to_type.get(sample_id_int, 'unknown')
        
        # Skip category 5 (adversarial questions)
        if category == '5' or category == 5:
            continue
        
        # For longmemeval, extract question from the formatted string
        if dataset_type == 'longmemeval':
            original_question = item['question'].split('\n')[1].lstrip("User: ") if '\n' in item['question'] else item['question']
        else:
            original_question = item['question']
        
        extracted.append({
            'question': original_question,
            'prediction': item['prediction'],
            'reference': item['reference'],
            'category': category,
            'sample_id': item['sample_id']
        })
    
    return extracted


# =============================================================================
# LLM Judge Evaluation
# =============================================================================

# LoCoMo prompt - exact prompt from llm_judege.py
LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

# LongMemEval prompts - exact prompts from longmemeval_llm_judge.py
TEMPORAL_REASONING_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""

KNOWLEDGE_UPDATE_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""

SINGLE_SESSION_PREFERENCE_PROMPT = """
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>
"""

LONGMEMEVAL_DEFAULT_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""


async def evaluate_llm_judge_async(client: AsyncOpenAI, 
                                   question: str, 
                                   gold_answer: str, 
                                   generated_answer: str,
                                   dataset_type: str = 'locomo',
                                   category: str = None) -> Tuple[int, str]:
    """Async LLM judge evaluation with dataset-specific prompts."""
    try:
        if dataset_type == 'longmemeval':
            # Select prompt based on question type/category
            if category == 'temporal-reasoning':
                user_prompt = TEMPORAL_REASONING_PROMPT.format(
                    question=question, gold_answer=gold_answer, response=generated_answer
                )
            elif category == 'knowledge-update':
                user_prompt = KNOWLEDGE_UPDATE_PROMPT.format(
                    question=question, gold_answer=gold_answer, response=generated_answer
                )
            elif category == 'single-session-preference':
                user_prompt = SINGLE_SESSION_PREFERENCE_PROMPT.format(
                    question=question, gold_answer=gold_answer, response=generated_answer
                )
            else:
                user_prompt = LONGMEMEVAL_DEFAULT_PROMPT.format(
                    question=question, gold_answer=gold_answer, response=generated_answer
                )
            
            response_obj = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert grader that determines if answers to questions match a gold standard answer"},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=Grade,
                temperature=0,
            )
            
            result = response_obj.choices[0].message.parsed
            is_correct = result.is_correct.strip().lower()
            score = 1 if is_correct == "yes" else 0
            response_content = json.dumps({"is_correct": result.is_correct})
            return score, response_content
            
        else:  # locomo
            user_prompt = LOCOMO_ACCURACY_PROMPT.format(
                question=question, gold_answer=gold_answer, generated_answer=generated_answer
            )
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert grader that determines if answers to questions match a gold standard answer."},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            response_content = response.choices[0].message.content
            label = json.loads(response_content)["label"]
            score = 1 if label == "CORRECT" else 0
            return score, response_content
        
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0, f"ERROR: {str(e)}"


async def batch_evaluate_llm_judge(data_items: List[Dict[str, Any]], 
                                   num_runs: int = 3,
                                   max_concurrent: int = 50,
                                   dataset_type: str = 'locomo') -> Dict[str, Any]:
    """Batch evaluate LLM judge with multiple runs."""
    client = AsyncOpenAI()
    
    all_requests = []
    task_id = 0
    for run_id in range(num_runs):
        for idx, item in enumerate(data_items):
            all_requests.append(LLMJudgeRequest(
                task_id=task_id,
                question=item['question'],
                gold_answer=item['reference'],
                generated_answer=item['prediction'],
                category=item['category'],
                sample_id=item['sample_id'],
                run_id=run_id,
                dataset_type=dataset_type
            ))
            task_id += 1
    
    total_requests = len(all_requests)
    print(f"\nTotal LLM judge requests: {total_requests} ({len(data_items)} items × {num_runs} runs)")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results_by_item_and_run = defaultdict(lambda: defaultdict(dict))
    
    async def process_request(req: LLMJudgeRequest, pbar: tqdm):
        async with semaphore:
            score, raw_response = await evaluate_llm_judge_async(
                client, req.question, req.gold_answer, req.generated_answer,
                dataset_type=req.dataset_type, category=req.category
            )
            
            item_key = (req.sample_id, req.question)
            results_by_item_and_run[item_key][req.run_id] = {
                'score': score,
                'raw_response': raw_response,
                'category': req.category,
                'question': req.question,
                'gold_answer': req.gold_answer,
                'generated_answer': req.generated_answer,
                'sample_id': req.sample_id
            }
            
            pbar.update(1)
            return score
    
    pbar = tqdm(total=total_requests, desc="LLM Judge Evaluation", unit="req")
    await asyncio.gather(*[process_request(req, pbar) for req in all_requests])
    pbar.close()
    
    return results_by_item_and_run


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_llm_judge_statistics(results_by_item_and_run: Dict, num_runs: int) -> Dict[str, Any]:
    """Compute mean and std statistics from multiple LLM judge runs."""
    scores_by_item = {}
    categories = set()
    
    for item_key, runs in results_by_item_and_run.items():
        scores = [runs[run_id]['score'] for run_id in range(num_runs)]
        category = runs[0]['category']
        categories.add(category)
        
        scores_by_item[item_key] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'category': category,
            'question': runs[0]['question'],
            'gold_answer': runs[0]['gold_answer'],
            'generated_answer': runs[0]['generated_answer'],
            'sample_id': runs[0]['sample_id']
        }
    
    # Compute aggregate statistics across runs
    overall_scores_by_run = []
    for run_id in range(num_runs):
        run_scores = [runs[run_id]['score'] for item_key, runs in results_by_item_and_run.items()]
        overall_scores_by_run.append(np.mean(run_scores))
    
    overall_mean = np.mean(overall_scores_by_run)
    overall_std = np.std(overall_scores_by_run)
    
    # Per-category statistics
    category_stats = {}
    for category in sorted(categories):
        category_scores_by_run = []
        for run_id in range(num_runs):
            run_category_scores = [
                runs[run_id]['score'] 
                for item_key, runs in results_by_item_and_run.items()
                if runs[0]['category'] == category
            ]
            if run_category_scores:
                category_scores_by_run.append(np.mean(run_category_scores))
        
        if category_scores_by_run:
            category_stats[category] = {
                'mean': np.mean(category_scores_by_run),
                'std': np.std(category_scores_by_run),
                'count': len([item for item in scores_by_item.values() if item['category'] == category]),
                'scores_by_run': category_scores_by_run
            }
    
    return {
        'overall': {
            'mean': overall_mean,
            'std': overall_std,
            'count': len(scores_by_item),
            'scores_by_run': overall_scores_by_run
        },
        'per_category': category_stats,
        'per_item': scores_by_item
    }


def compute_text_metrics(data_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute BLEU-1, F1, and EM scores."""
    print("\nComputing BLEU-1, F1, and Exact Match scores...")
    
    all_metrics = []
    category_metrics = defaultdict(list)
    
    for item in tqdm(data_items, desc="Computing text metrics"):
        metrics = calculate_metrics_mem0_style(item['prediction'], item['reference'])
        metrics['category'] = item['category']
        all_metrics.append(metrics)
        category_metrics[item['category']].append(metrics)
    
    # Compute overall statistics
    overall_stats = {
        'exact_match': {
            'mean': np.mean([m['exact_match'] for m in all_metrics]),
            'std': np.std([m['exact_match'] for m in all_metrics]),
            'count': len(all_metrics)
        },
        'f1': {
            'mean': np.mean([m['f1'] for m in all_metrics]),
            'std': np.std([m['f1'] for m in all_metrics]),
            'count': len(all_metrics)
        },
        'bleu1': {
            'mean': np.mean([m['bleu1'] for m in all_metrics]),
            'std': np.std([m['bleu1'] for m in all_metrics]),
            'count': len(all_metrics)
        }
    }
    
    # Per-category statistics
    per_category = {}
    for category, metrics_list in category_metrics.items():
        per_category[category] = {
            'exact_match': {
                'mean': np.mean([m['exact_match'] for m in metrics_list]),
                'std': np.std([m['exact_match'] for m in metrics_list]),
                'count': len(metrics_list)
            },
            'f1': {
                'mean': np.mean([m['f1'] for m in metrics_list]),
                'std': np.std([m['f1'] for m in metrics_list]),
                'count': len(metrics_list)
            },
            'bleu1': {
                'mean': np.mean([m['bleu1'] for m in metrics_list]),
                'std': np.std([m['bleu1'] for m in metrics_list]),
                'count': len(metrics_list)
            }
        }
    
    return {
        'overall': overall_stats,
        'per_category': per_category,
        'per_item': all_metrics
    }


# =============================================================================
# Output
# =============================================================================

def save_results(llm_judge_stats: Dict[str, Any],
                text_metrics: Dict[str, Any],
                output_path: str,
                metadata: Dict[str, Any]):
    """Save evaluation results to JSON file."""
    output_data = {
        'metadata': metadata,
        'llm_judge': {
            'overall': llm_judge_stats['overall'],
            'per_category': llm_judge_stats['per_category']
        },
        'text_metrics': {
            'overall': text_metrics['overall'],
            'per_category': text_metrics['per_category']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(llm_judge_stats: Dict[str, Any], 
                 text_metrics: Dict[str, Any],
                 num_runs: int):
    """Print comprehensive summary of evaluation results."""
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nNumber of LLM judge runs: {num_runs}")
    print(f"Total items evaluated: {llm_judge_stats['overall']['count']}")
    
    # Overall metrics
    print("\n" + "-"*40)
    print("OVERALL METRICS")
    print("-"*40)
    
    print(f"\n  LLM Judge Score:")
    print(f"    Mean: {llm_judge_stats['overall']['mean']:.4f}")
    print(f"    Std:  {llm_judge_stats['overall']['std']:.4f}")
    
    print(f"\n  Exact Match:")
    print(f"    Mean: {text_metrics['overall']['exact_match']['mean']:.4f}")
    print(f"    Std:  {text_metrics['overall']['exact_match']['std']:.4f}")
    
    print(f"\n  F1 Score:")
    print(f"    Mean: {text_metrics['overall']['f1']['mean']:.4f}")
    print(f"    Std:  {text_metrics['overall']['f1']['std']:.4f}")
    
    print(f"\n  BLEU-1:")
    print(f"    Mean: {text_metrics['overall']['bleu1']['mean']:.4f}")
    print(f"    Std:  {text_metrics['overall']['bleu1']['std']:.4f}")
    
    # Per-category metrics
    print("\n" + "-"*40)
    print("PER-CATEGORY METRICS")
    print("-"*40)
    
    all_categories = sorted(set(llm_judge_stats['per_category'].keys()) | 
                           set(text_metrics['per_category'].keys()))
    
    for category in all_categories:
        llm_cat = llm_judge_stats['per_category'].get(category, {})
        text_cat = text_metrics['per_category'].get(category, {})
        count = llm_cat.get('count', text_cat.get('exact_match', {}).get('count', 0))
        
        print(f"\n  Category {category} (n={count}):")
        
        if llm_cat:
            print(f"    LLM Judge:  {llm_cat['mean']:.4f} (±{llm_cat['std']:.4f})")
        if text_cat:
            print(f"    EM:         {text_cat['exact_match']['mean']:.4f} (±{text_cat['exact_match']['std']:.4f})")
            print(f"    F1:         {text_cat['f1']['mean']:.4f} (±{text_cat['f1']['std']:.4f})")
            print(f"    BLEU-1:     {text_cat['bleu1']['mean']:.4f} (±{text_cat['bleu1']['std']:.4f})")
    
    print("\n" + "="*80)
    
    # Print LaTeX-friendly table
    print("\nLaTeX-friendly results (Overall):")
    print(f"LLM Judge & {llm_judge_stats['overall']['mean']*100:.2f} $\\pm$ {llm_judge_stats['overall']['std']*100:.2f} \\\\")
    print(f"EM & {text_metrics['overall']['exact_match']['mean']*100:.2f} $\\pm$ {text_metrics['overall']['exact_match']['std']*100:.2f} \\\\")
    print(f"F1 & {text_metrics['overall']['f1']['mean']*100:.2f} $\\pm$ {text_metrics['overall']['f1']['std']*100:.2f} \\\\")
    print(f"BLEU-1 & {text_metrics['overall']['bleu1']['mean']*100:.2f} $\\pm$ {text_metrics['overall']['bleu1']['std']*100:.2f} \\\\")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation for EMem - computes LLM judge, BLEU-1, F1, and EM scores"
    )
    parser.add_argument(
        "--result_file", 
        type=str, 
        required=True,
        help="Path to results pickle file"
    )
    parser.add_argument(
        "--dataset_type", 
        type=str, 
        required=True,
        choices=['locomo', 'longmemeval'],
        help="Dataset type: 'locomo' or 'longmemeval'"
    )
    parser.add_argument(
        "--longmemeval_path", 
        type=str,
        default="data/longmemeval/preprocess/longmemeval_s_cleaned.json",
        help="Path to original LongMemEval JSON file (for longmemeval dataset)"
    )
    parser.add_argument(
        "--num_runs", 
        type=int, 
        default=3,
        help="Number of times to run LLM judge (default: 3)"
    )
    parser.add_argument(
        "--max_concurrent", 
        type=int, 
        default=50,
        help="Maximum concurrent API requests (default: 50)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="Directory to save results (default: same as result_file)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.result_file):
        raise FileNotFoundError(f"Result file not found: {args.result_file}")
    
    if args.dataset_type == 'longmemeval' and not os.path.exists(args.longmemeval_path):
        raise FileNotFoundError(f"LongMemEval data file not found: {args.longmemeval_path}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.result_file) or "."
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    result_basename = Path(args.result_file).stem
    output_filename = f"final_eval_{result_basename}_{args.num_runs}runs_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print("="*80)
    print("FINAL EVALUATION FOR PAPER REPORTING")
    print("="*80)
    print(f"Result file: {args.result_file}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Number of LLM judge runs: {args.num_runs}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Output path: {output_path}")
    if args.dataset_type == 'longmemeval':
        print(f"LongMemEval original data: {args.longmemeval_path}")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results = load_results_from_pickle(args.result_file)
    
    # Extract predictions and references
    print("Extracting predictions and references...")
    data_items = extract_predictions_and_references(
        results, 
        args.dataset_type,
        args.longmemeval_path if args.dataset_type == 'longmemeval' else None
    )
    
    print(f"Found {len(data_items)} items to evaluate")
    
    # Category distribution
    category_dist = defaultdict(int)
    for item in data_items:
        category_dist[item['category']] += 1
    print("\nCategory distribution:")
    for category in sorted(category_dist.keys()):
        print(f"  Category {category}: {category_dist[category]} items")
    
    # Compute text metrics (BLEU, F1, EM)
    text_metrics = compute_text_metrics(data_items)
    
    # Run LLM judge evaluation
    print(f"\nRunning LLM judge evaluation ({args.num_runs} runs per item)...")
    results_by_item_and_run = asyncio.run(
        batch_evaluate_llm_judge(
            data_items, 
            num_runs=args.num_runs,
            max_concurrent=args.max_concurrent,
            dataset_type=args.dataset_type
        )
    )
    
    # Compute LLM judge statistics
    print("\nComputing LLM judge statistics...")
    llm_judge_stats = compute_llm_judge_statistics(results_by_item_and_run, args.num_runs)
    
    # Prepare metadata
    metadata = {
        'result_file': args.result_file,
        'dataset_type': args.dataset_type,
        'num_runs': args.num_runs,
        'max_concurrent': args.max_concurrent,
        'timestamp': timestamp,
        'model_info': {
            'model': results.get('model', 'unknown'),
            'embedding_model': results.get('embedding_model', 'unknown'),
            'dataset': results.get('dataset', 'unknown')
        }
    }
    
    # Save results
    save_results(llm_judge_stats, text_metrics, output_path, metadata)
    
    # Print summary
    print_summary(llm_judge_stats, text_metrics, args.num_runs)
    
    print(f"✓ Final evaluation completed successfully!")


if __name__ == "__main__":
    main()


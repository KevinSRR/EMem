"""LoCoMo dataset evaluation metrics adapted from mem0 evaluation approach."""

from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter, defaultdict
import numpy as np
import statistics
import re
import string
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import logging
import concurrent.futures
import threading
from tqdm import tqdm

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer
from ..llm.openai_gpt_batch import CacheOpenAI
from ..prompts.prompt_template_manager import PromptTemplateManager

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")

# Try to import SentenceTransformer for similarity computation
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.warning(f"Could not load SentenceTransformer model: {e}")
    sentence_model = None


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function."""
    text = str(text)
    return text.lower().replace('.', ' ').replace(",", ' ').replace('!', ' ').replace('?', ' ').split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }


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
            logger.warning(f"Error calculating BLEU score: {e}")
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        logger.warning(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }


def calculate_meteor_score_safe(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        logger.warning(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        logger.warning(f"Error calculating sentence similarity: {e}")
        return 0.0


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
    
    # Combine all metrics (following mem0's approach - only EM, F1, and BLEU)
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }
    
    return metrics


class LoCoMoEvaluationNew(BaseMetric):
    """New LoCoMo evaluation metric following mem0's evaluation approach with LLM judge and batch processing."""
    
    metric_name: str = "locomo_evaluation_new"

    def __init__(self, llm_model: Optional[CacheOpenAI] = None, global_config: Optional[BaseConfig] = None, max_workers: int = 10):
        super().__init__(global_config)
        self.llm_model = llm_model
        self.max_workers = max_workers
        
        # Initialize prompt template manager if LLM model is provided
        if self.llm_model:
            self.prompt_template_manager = PromptTemplateManager(
                role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
            )
        
        # Thread-safe result storage
        self.results_lock = threading.Lock()
        
    def _evaluate_llm_judge_single(self, question: str, gold_answer: str, generated_answer: str) -> int:
        """Evaluate a single question using LLM judge."""
        if not self.llm_model:
            logger.warning("LLM model not provided, skipping LLM judge evaluation")
            return 0
            
        try:
            # Import the response model
            from ..prompts.templates.llm_judge_accuracy import LLMJudgeAccuracy
            
            # Render the prompt
            messages = self.prompt_template_manager.render(
                name='llm_judge_accuracy',
                question=question,
                gold_answer=gold_answer,
                generated_answer=generated_answer
            )
            
            # Call LLM with structured output
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                response_format=LLMJudgeAccuracy
            )
            
            # Parse the response
            llm_response = LLMJudgeAccuracy.model_validate_json(raw_response)
            return 1 if llm_response.label == "CORRECT" else 0
            
        except Exception as e:
            logger.warning(f"Error in LLM judge evaluation: {e}")
            return 0
    
    def _evaluate_llm_judge_batch(self, questions: List[str], gold_answers: List[str], 
                                generated_answers: List[str]) -> List[int]:
        """Evaluate multiple questions using LLM judge in batch."""
        if not self.llm_model:
            raise ValueError("LLM model not provided, skipping LLM judge evaluation")
            logger.warning("LLM model not provided, skipping LLM judge evaluation")
            return [0] * len(questions)
            
        try:
            # Import the response model
            from ..prompts.templates.llm_judge_accuracy import LLMJudgeAccuracy
            
            # Prepare batch messages
            batch_messages = []
            for question, gold_answer, generated_answer in zip(questions, gold_answers, generated_answers):
                messages = self.prompt_template_manager.render(
                    name='llm_judge_accuracy',
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer
                )
                batch_messages.append(messages)
            
            # Batch inference
            batch_results = self.llm_model.batch_infer(
                messages=batch_messages,
                response_format=LLMJudgeAccuracy
            )
            
            # Process results
            llm_scores = []
            for result in batch_results:
                try:
                    raw_response, metadata, cache_hit = result
                    llm_response = LLMJudgeAccuracy.model_validate_json(raw_response)
                    llm_scores.append(1 if llm_response.label == "CORRECT" else 0)
                except Exception as e:
                    logger.warning(f"Error parsing LLM judge result: {e}")
                    llm_scores.append(0)
            
            return llm_scores
            
        except Exception as e:
            raise e
            logger.warning(f"Error in batch LLM judge evaluation: {e}")
            return [0] * len(questions)

    def process_item_batch(self, questions: List[str], gold_answers: List[str], 
                          predicted_answers: List[str], categories: List[str]) -> List[Dict]:
        """Process a batch of evaluation items, following mem0's evaluation approach."""
        local_results = []
        
        # Calculate traditional metrics for all items
        traditional_metrics = []
        for pred_answer, gold_answer in zip(predicted_answers, gold_answers):
            # Skip category 5 (following mem0's approach)
            metrics = calculate_metrics_mem0_style(pred_answer, gold_answer)
            traditional_metrics.append(metrics)
        
        # Calculate LLM judge scores in batch
        llm_scores = self._evaluate_llm_judge_batch(questions, gold_answers, predicted_answers)
        
        # Combine results
        for i, (question, gold_answer, pred_answer, category, metrics, llm_score) in enumerate(
            zip(questions, gold_answers, predicted_answers, categories, traditional_metrics, llm_scores)
        ):
            local_results.append({
                "question": question,
                "answer": gold_answer,
                "response": pred_answer,
                "category": category,
                "exact_match": metrics["exact_match"],
                "f1_score": metrics["f1"],
                "bleu_score": metrics["bleu1"],  # Use BLEU-1 as primary BLEU score
                "llm_score": llm_score,
                "all_metrics": metrics
            })
        
        return local_results

    def calculate_metric_scores(self, 
                              gold_answers: List[List[str]], 
                              predicted_answers: List[str],
                              questions: List[str],
                              categories: Optional[List[str]] = None,
                              batch_size: int = 50,
                              real_time_logging: bool = True,
                              aggregation_fn: Callable = np.max) -> Tuple[Dict[str, Union[float, Dict]], List[Dict[str, float]]]:
        """
        Calculate comprehensive metrics for LoCoMo evaluation following mem0's approach.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            questions (List[str]): List of questions.
            categories (Optional[List[str]]): List of question categories for per-category evaluation.
            batch_size (int): Size of batches for processing.
            real_time_logging (bool): Whether to log real-time results.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, Union[float, Dict]], List[Dict[str, float]]]: 

                - A dictionary with overall and per-category averaged metrics containing:

                  * "overall": Dict with aggregated metrics across all examples, where each metric contains: 'mean', 'std', 'median', 'min', 'max', 'count'

                  * "category_{category_name}": Dict with same structure as overall but for each category

                  * Metrics include: 'exact_match', 'f1_score', 'bleu_score', 'llm_score'

                - A list of dictionaries with metrics for each example containing:
                
                  * "exact_match": Binary exact match score (0.0 or 1.0)

                  * "f1": F1 score between 0.0 and 1.0

                  * "bleu1": BLEU-1 score between 0.0 and 1.0

                  * "llm_score": LLM judge score between 0.0 and 1.0
        """
        assert len(gold_answers) == len(predicted_answers) == len(questions), \
            "Length of gold answers, predicted answers, and questions should be the same."
        
        if categories is not None:
            assert len(categories) == len(gold_answers), "Length of categories should match length of answers."
        else:
            categories = ["unknown"] * len(gold_answers)

        all_results = [] # list of {
        #     "question": question,
        #     "answer": gold_answer,
        #     "response": pred_answer,
        #     "category": category,
        #     "exact_match": metrics["exact_match"],
        #     "f1_score": metrics["f1"],
        #     "bleu_score": metrics["bleu1"],  # Use BLEU-1 as primary BLEU score
        #     "llm_score": llm_score,
        #     "all_metrics": metrics
        # }
        all_metrics = []
        
        # Initialize real-time tracking
        category_trackers = defaultdict(lambda: {
            'exact_match': [], 'f1_score': [], 'bleu_score': [], 'llm_score': []
        })
        overall_tracker = {
            'exact_match': [], 'f1_score': [], 'bleu_score': [], 'llm_score': []
        }
        
        # Process in batches with real-time logging
        total_batches = (len(predicted_answers) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(predicted_answers))
            
            # Prepare batch data - handle multiple gold answers by taking the first one for LLM judge
            batch_questions = questions[start_idx:end_idx]
            batch_gold_answers = [gold_list[0] for gold_list in gold_answers[start_idx:end_idx]]
            batch_predicted_answers = predicted_answers[start_idx:end_idx]
            batch_categories = categories[start_idx:end_idx]
            
            # Process batch
            batch_results = self.process_item_batch(
                batch_questions, batch_gold_answers, batch_predicted_answers, batch_categories
            )
            
            # Process results for each item in batch (handle multiple gold answers)
            for local_idx, result in enumerate(batch_results):
                global_idx = start_idx + local_idx
                gold_list = gold_answers[global_idx]
                category = batch_categories[local_idx]
                
                # Skip category 5 (following mem0's approach)
                if str(category) == "5":
                    continue
                
                # Handle multiple gold answers by taking the best score
                if len(gold_list) > 1:
                    # Calculate metrics against all gold answers and take the best
                    all_metrics_for_gold = []
                    for gold in gold_list:
                        metrics = calculate_metrics_mem0_style(batch_predicted_answers[local_idx], gold)
                        all_metrics_for_gold.append(metrics)
                    
                    # Aggregate using provided function (default: max)
                    aggregated_metrics = {}
                    for metric_name in all_metrics_for_gold[0].keys():
                        metric_values = [m[metric_name] for m in all_metrics_for_gold]
                        aggregated_metrics[metric_name] = aggregation_fn(metric_values)
                    
                    # Update result with aggregated metrics
                    result["exact_match"] = aggregated_metrics["exact_match"]
                    result["f1_score"] = aggregated_metrics["f1"]
                    result["bleu_score"] = aggregated_metrics["bleu1"]
                    result["all_metrics"] = aggregated_metrics
                
                # Store results
                all_results.append(result)
                all_metrics.append(result["all_metrics"])
                
                # Real-time tracking
                category_trackers[category]['exact_match'].append(result["exact_match"])
                category_trackers[category]['f1_score'].append(result["f1_score"])
                category_trackers[category]['bleu_score'].append(result["bleu_score"])
                category_trackers[category]['llm_score'].append(result["llm_score"])
                
                overall_tracker['exact_match'].append(result["exact_match"])
                overall_tracker['f1_score'].append(result["f1_score"])
                overall_tracker['bleu_score'].append(result["bleu_score"])
                overall_tracker['llm_score'].append(result["llm_score"])
            
            # Real-time logging
            if real_time_logging:
                processed_count = len(overall_tracker['exact_match'])
                logger.info(f"Processed {processed_count} items. Current running averages:")
                logger.info(f"  Overall - EM: {np.mean(overall_tracker['exact_match']):.4f}, "
                           f"F1: {np.mean(overall_tracker['f1_score']):.4f}, "
                           f"BLEU: {np.mean(overall_tracker['bleu_score']):.4f}, "
                           f"LLM: {np.mean(overall_tracker['llm_score']):.4f}")
                
                for cat, tracker in category_trackers.items():
                    if tracker['exact_match']:  # Only show categories with data
                        logger.info(f"  Category {cat} - EM: {np.mean(tracker['exact_match']):.4f}, "
                                   f"F1: {np.mean(tracker['f1_score']):.4f}, "
                                   f"BLEU: {np.mean(tracker['bleu_score']):.4f}, "
                                   f"LLM: {np.mean(tracker['llm_score']):.4f} "
                                   f"({len(tracker['exact_match'])} items)")
        
        # Calculate final aggregate statistics
        pooled_results = self._aggregate_results(all_results)
        
        # Convert to expected format
        example_results = [
            {
                "exact_match": result["exact_match"],
                "f1": result["f1_score"], 
                "bleu1": result["bleu_score"],
                "llm_score": result["llm_score"]
            }
            for result in all_results
        ]
        
        return pooled_results, example_results

    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Union[float, Dict]]:
        """Calculate aggregate statistics for all metrics, split by category.
        
        Returns:
            {
                "overall": dict from metric name to aggregated values (mean, std, median, min, max, count),
                "category_{category}": dict from metric name to category data aggregated values (mean, std, median, min, max, count)
            }
        """
        if not all_results:
            return {}
        
        # Initialize aggregates
        aggregates = defaultdict(list)
        category_aggregates = defaultdict(lambda: defaultdict(list))
        
        # Collect all values
        for result in all_results:
            category = result["category"]
            
            for metric_name in ["exact_match", "f1_score", "bleu_score", "llm_score"]:
                value = result[metric_name]
                aggregates[metric_name].append(value)
                category_aggregates[category][metric_name].append(value)
        
        # Calculate overall statistics
        results = {"overall": {}}
        for metric_name, values in aggregates.items():
            results["overall"][metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        # Calculate per-category statistics
        for category in sorted(category_aggregates.keys()):
            results[f"category_{category}"] = {}
            for metric_name, values in category_aggregates[category].items():
                if values:
                    results[f"category_{category}"][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        return results


# For backward compatibility, also provide old class names
class LoCoMoComprehensiveEval(BaseMetric):
    """Legacy comprehensive evaluation - now redirects to new implementation."""
    metric_name: str = "locomo_comprehensive"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        self._new_evaluator = LoCoMoEvaluationNew(llm_model=None, global_config=global_config)
    
    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str],
                               categories: Optional[List[int]] = None, aggregation_fn: Callable = np.max):
        """Legacy interface redirecting to new implementation."""
        questions = [f"Question {i}" for i in range(len(predicted_answers))]  # Dummy questions
        categories_str = [str(c) if c is not None else "unknown" for c in (categories or [None] * len(predicted_answers))]
        return self._new_evaluator.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            questions=questions,
            categories=categories_str,
            real_time_logging=False
        )


class LoCoMoCategoryEval(BaseMetric):
    """Legacy category evaluation - now redirects to new implementation."""
    metric_name: str = "locomo_category"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        self._new_evaluator = LoCoMoEvaluationNew(llm_model=None, global_config=global_config)
    
    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str],
                               categories: List[int], aggregation_fn: Callable = np.max):
        """Legacy interface redirecting to new implementation."""
        questions = [f"Question {i}" for i in range(len(predicted_answers))]  # Dummy questions
        categories_str = [str(c) for c in categories]
        aggregate_results, example_results = self._new_evaluator.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            questions=questions,
            categories=categories_str,
            real_time_logging=False
        )
        
        # Convert to legacy format
        pooled_results = {}
        example_eval_results = []
        
        for category in sorted(set(categories)):
            cat_key = f"category_{category}"
            if cat_key in aggregate_results:
                pooled_results[f"category_{category}_accuracy"] = aggregate_results[cat_key].get("exact_match", {}).get("mean", 0.0)
                pooled_results[f"category_{category}_count"] = aggregate_results[cat_key].get("exact_match", {}).get("count", 0)
        
        # Overall accuracy
        if "overall" in aggregate_results:
            pooled_results["overall_accuracy"] = aggregate_results["overall"].get("exact_match", {}).get("mean", 0.0)
        
        for result in example_results:
            example_eval_results.append({
                "accuracy": result["exact_match"],
                f"category_{categories[len(example_eval_results)]}_accuracy": result["exact_match"]
            })
        
        return pooled_results, example_eval_results

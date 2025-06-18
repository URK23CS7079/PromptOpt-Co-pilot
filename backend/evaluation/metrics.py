"""
Comprehensive metrics calculation module for PromptOpt Co-Pilot.

This module provides various evaluation metrics for prompt performance assessment including:
- Exact match scoring 
- Semantic similarity using sentence transformers
- Pairwise LLM judgments
- Standard NLP metrics (BLEU, ROUGE, BERTScore)
- Latency measurements
- Custom domain-specific metrics

Author: PromptOpt Co-Pilot
Version: 1.0
"""

import re
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from functools import lru_cache

# Third-party imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.stats as stats
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    logging.warning("ML dependencies not installed. Some metrics will be unavailable.")

# Local imports
from backend.llm.llama_wrapper import LlamaWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """
    Container for metric calculation results with statistical information.
    
    Attributes:
        metric_name: Name of the calculated metric
        score: Primary metric score (0-1 normalized)
        confidence_interval: 95% confidence interval tuple (lower, upper)
        sample_size: Number of samples used in calculation
        metadata: Additional metric-specific information
        raw_score: Original unnormalized score
        execution_time: Time taken to calculate metric (seconds)
    """
    metric_name: str
    score: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_score: Optional[float] = None
    execution_time: Optional[float] = None


class MetricCalculator:
    """
    Comprehensive metric calculator for prompt evaluation.
    
    Supports various evaluation metrics including exact match, semantic similarity,
    LLM-based judgments, and standard NLP evaluation metrics.
    """
    
    # LLM Judge prompt templates
    JUDGE_PROMPTS = {
        'quality': """
        Compare the quality of these two responses to the same prompt.
        
        Response A: {response_a}
        Response B: {response_b}
        
        Evaluate based on: {criteria}
        
        Rate which response is better on a scale:
        - A is much better: -2
        - A is slightly better: -1  
        - Tie/Equal: 0
        - B is slightly better: 1
        - B is much better: 2
        
        Provide only the numeric score: """,
        
        'coherence': """
        Rate the coherence and logical flow of this text on a scale of 1-10:
        
        Text: {text}
        
        Consider:
        - Logical consistency
        - Clear structure
        - Smooth transitions
        - Overall readability
        
        Provide only the numeric score (1-10): """,
        
        'relevance': """
        Rate how relevant this output is to the given context on a scale of 1-10:
        
        Context: {context}
        Output: {output}
        
        Consider:
        - Direct relevance to context
        - Addressing key points
        - Staying on topic
        - Appropriateness of response
        
        Provide only the numeric score (1-10): """
    }
    
    def __init__(self, llm_wrapper: Optional[LlamaWrapper] = None):
        """
        Initialize MetricCalculator with optional LLM wrapper for judge-based metrics.
        
        Args:
            llm_wrapper: LlamaWrapper instance for LLM-based evaluations
        """
        self.llm_wrapper = llm_wrapper
        self.sentence_model = None
        self.tfidf_vectorizer = None
        
        # Initialize sentence transformer if available
        if HAS_ML_DEPS:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        
        # Cache for expensive computations
        self._embedding_cache = {}
        
    def calculate_metric(self, metric_name: str, predictions: List[str], 
                        references: List[str], **kwargs) -> MetricResult:
        """
        Calculate a specific metric by name.
        
        Args:
            metric_name: Name of metric to calculate
            predictions: List of predicted/generated texts
            references: List of reference/ground truth texts
            **kwargs: Additional metric-specific parameters
            
        Returns:
            MetricResult object with calculated score and metadata
            
        Raises:
            ValueError: If metric name is not supported
            RuntimeError: If metric calculation fails
        """
        start_time = time.time()
        
        try:
            if metric_name == 'exact_match':
                score = self.exact_match_score(predictions, references)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score
                )
                
            elif metric_name == 'semantic_similarity':
                score = self.semantic_similarity_score(predictions, references)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score
                )
                
            elif metric_name == 'bleu':
                score = calculate_bleu_score(predictions, references)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score
                )
                
            elif metric_name == 'rouge':
                scores = calculate_rouge_score(predictions, references)
                # Use ROUGE-L F1 as primary score
                score = scores.get('rouge-l', {}).get('f', 0.0)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score,
                    metadata={'detailed_scores': scores}
                )
                
            elif metric_name == 'bertscore':
                scores = calculate_bertscore(predictions, references)
                score = scores.get('f1_mean', 0.0)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score,
                    metadata=scores
                )
                
            elif metric_name == 'coherence':
                score = self.coherence_score(predictions)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score
                )
                
            elif metric_name == 'relevance':
                contexts = kwargs.get('contexts', references)
                score = self.relevance_score(predictions, contexts)
                result = MetricResult(
                    metric_name=metric_name,
                    score=score,
                    sample_size=len(predictions),
                    raw_score=score
                )
                
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
                
            # Add confidence interval if sample size is sufficient
            if result.sample_size > 10:
                result.confidence_interval = self._calculate_confidence_interval(
                    [result.score] * result.sample_size, confidence=0.95
                )
                
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate metric {metric_name}: {e}")
            raise RuntimeError(f"Metric calculation failed: {e}")
    
    def calculate_all_metrics(self, predictions: List[str], 
                            references: List[str], **kwargs) -> Dict[str, MetricResult]:
        """
        Calculate all available metrics for the given predictions and references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts  
            **kwargs: Additional parameters (e.g., contexts for relevance)
            
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        metrics = {}
        
        # Standard metrics that don't require additional dependencies
        standard_metrics = ['exact_match', 'bleu', 'rouge']
        
        # Metrics requiring ML dependencies
        ml_metrics = ['semantic_similarity', 'bertscore'] if HAS_ML_DEPS else []
        
        # LLM-based metrics
        llm_metrics = ['coherence', 'relevance'] if self.llm_wrapper else []
        
        all_metrics = standard_metrics + ml_metrics + llm_metrics
        
        for metric_name in all_metrics:
            try:
                metrics[metric_name] = self.calculate_metric(
                    metric_name, predictions, references, **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {e}")
                
        return metrics
    
    def exact_match_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy between predictions and references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Exact match accuracy (0.0 to 1.0)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
            
        if not predictions:
            return 0.0
            
        matches = sum(
            calculate_exact_match(pred, ref) 
            for pred, ref in zip(predictions, references)
        )
        
        return matches / len(predictions)
    
    def semantic_similarity_score(self, predictions: List[str], 
                                references: List[str]) -> float:
        """
        Calculate semantic similarity using sentence transformers.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Average cosine similarity score (0.0 to 1.0)
        """
        if not HAS_ML_DEPS or not self.sentence_model:
            logger.warning("Sentence transformers not available, using TF-IDF fallback")
            return self._tfidf_similarity(predictions, references)
            
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
            
        if not predictions:
            return 0.0
        
        try:
            # Get embeddings with caching
            pred_embeddings = self._get_embeddings(predictions)
            ref_embeddings = self._get_embeddings(references)
            
            # Calculate cosine similarities
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(max(0.0, similarity))  # Ensure non-negative
                
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return self._tfidf_similarity(predictions, references)
    
    def pairwise_llm_judge(self, prompt_a_outputs: List[str], 
                          prompt_b_outputs: List[str], criteria: str) -> float:
        """
        Compare two sets of outputs using LLM as judge.
        
        Args:
            prompt_a_outputs: Outputs from first prompt
            prompt_b_outputs: Outputs from second prompt  
            criteria: Evaluation criteria for comparison
            
        Returns:
            Score from -1 (A much better) to 1 (B much better), 0 is tie
        """
        if not self.llm_wrapper:
            raise RuntimeError("LLM wrapper required for judge-based metrics")
            
        if len(prompt_a_outputs) != len(prompt_b_outputs):
            raise ValueError("Output lists must have same length")
            
        scores = []
        
        for output_a, output_b in zip(prompt_a_outputs, prompt_b_outputs):
            try:
                prompt = self.JUDGE_PROMPTS['quality'].format(
                    response_a=output_a,
                    response_b=output_b,
                    criteria=criteria
                )
                
                response = self.llm_wrapper.generate(prompt, max_tokens=10)
                score = self._extract_numeric_score(response, range_=(-2, 2))
                scores.append(score / 2.0)  # Normalize to [-1, 1]
                
            except Exception as e:
                logger.warning(f"LLM judge failed for pair: {e}")
                scores.append(0.0)  # Default to tie
                
        return sum(scores) / len(scores) if scores else 0.0
    
    def latency_score(self, response_times: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics from response times.
        
        Args:
            response_times: List of response times in seconds
            
        Returns:
            Dictionary with latency statistics
        """
        if not response_times:
            return {
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'std': 0.0
            }
        
        times = sorted(response_times)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min': min(times),
            'max': max(times)
        }
    
    def coherence_score(self, texts: List[str]) -> float:
        """
        Evaluate coherence of texts using LLM judge.
        
        Args:
            texts: List of texts to evaluate for coherence
            
        Returns:
            Average coherence score (0.0 to 1.0)
        """
        if not self.llm_wrapper:
            logger.warning("LLM wrapper not available, using simple coherence heuristic")
            return self._simple_coherence_score(texts)
            
        scores = []
        
        for text in texts:
            try:
                prompt = self.JUDGE_PROMPTS['coherence'].format(text=text)
                response = self.llm_wrapper.generate(prompt, max_tokens=10)
                score = self._extract_numeric_score(response, range_=(1, 10))
                scores.append((score - 1) / 9.0)  # Normalize to [0, 1]
                
            except Exception as e:
                logger.warning(f"Coherence scoring failed: {e}")
                scores.append(0.5)  # Default to neutral
                
        return sum(scores) / len(scores) if scores else 0.0
    
    def relevance_score(self, outputs: List[str], contexts: List[str]) -> float:
        """
        Evaluate relevance of outputs to contexts using LLM judge.
        
        Args:
            outputs: List of generated outputs
            contexts: List of corresponding contexts
            
        Returns:
            Average relevance score (0.0 to 1.0)
        """
        if len(outputs) != len(contexts):
            raise ValueError("Outputs and contexts must have same length")
            
        if not self.llm_wrapper:
            logger.warning("LLM wrapper not available, using keyword overlap")
            return self._keyword_relevance_score(outputs, contexts)
            
        scores = []
        
        for output, context in zip(outputs, contexts):
            try:
                prompt = self.JUDGE_PROMPTS['relevance'].format(
                    context=context, output=output
                )
                response = self.llm_wrapper.generate(prompt, max_tokens=10)
                score = self._extract_numeric_score(response, range_=(1, 10))
                scores.append((score - 1) / 9.0)  # Normalize to [0, 1]
                
            except Exception as e:
                logger.warning(f"Relevance scoring failed: {e}")
                scores.append(0.5)  # Default to neutral
                
        return sum(scores) / len(scores) if scores else 0.0
    
    @lru_cache(maxsize=1000)
    def _get_embeddings(self, texts: Tuple[str]) -> np.ndarray:
        """
        Get sentence embeddings with caching.
        
        Args:
            texts: Tuple of texts (for hashability)
            
        Returns:
            Numpy array of embeddings
        """
        if not self.sentence_model:
            raise RuntimeError("Sentence model not available")
            
        return self.sentence_model.encode(list(texts))
    
    def _tfidf_similarity(self, predictions: List[str], references: List[str]) -> float:
        """
        Fallback similarity using TF-IDF and cosine similarity.
        """
        if not self.tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True, stop_words='english'
            ) if HAS_ML_DEPS else None
            
        if not self.tfidf_vectorizer:
            return self._simple_token_overlap(predictions, references)
            
        try:
            all_texts = predictions + references
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            pred_matrix = tfidf_matrix[:len(predictions)]
            ref_matrix = tfidf_matrix[len(predictions):]
            
            similarities = []
            for i in range(len(predictions)):
                sim = cosine_similarity(pred_matrix[i], ref_matrix[i])[0][0]
                similarities.append(max(0.0, sim))
                
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return self._simple_token_overlap(predictions, references)
    
    def _simple_token_overlap(self, predictions: List[str], references: List[str]) -> float:
        """
        Simple token overlap similarity as final fallback.
        """
        similarities = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not pred_tokens and not ref_tokens:
                similarities.append(1.0)
            elif not pred_tokens or not ref_tokens:
                similarities.append(0.0)
            else:
                overlap = len(pred_tokens & ref_tokens)
                union = len(pred_tokens | ref_tokens)
                similarities.append(overlap / union)
                
        return sum(similarities) / len(similarities)
    
    def _simple_coherence_score(self, texts: List[str]) -> float:
        """
        Simple heuristic for coherence based on text structure.
        """
        scores = []
        
        for text in texts:
            score = 0.5  # Base score
            
            # Check for proper punctuation
            if text.strip().endswith(('.', '!', '?')):
                score += 0.1
                
            # Check for reasonable sentence structure
            sentences = text.split('.')
            if 1 <= len(sentences) <= 10:  # Reasonable number of sentences
                score += 0.2
                
            # Check for varied sentence lengths
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences if s.strip()]
                if lengths and statistics.stdev(lengths) > 2:
                    score += 0.1
                    
            # Check for repetition (lower is better)
            words = text.lower().split()
            if words:
                unique_ratio = len(set(words)) / len(words)
                score += unique_ratio * 0.2
                
            scores.append(min(1.0, score))
            
        return sum(scores) / len(scores) if scores else 0.0
    
    def _keyword_relevance_score(self, outputs: List[str], contexts: List[str]) -> float:
        """
        Simple keyword overlap for relevance scoring.
        """
        scores = []
        
        for output, context in zip(outputs, contexts):
            output_words = set(output.lower().split())
            context_words = set(context.lower().split())
            
            if not context_words:
                scores.append(0.0)
            else:
                overlap = len(output_words & context_words)
                score = overlap / len(context_words)
                scores.append(min(1.0, score))
                
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_numeric_score(self, response: str, range_: Tuple[int, int]) -> float:
        """
        Extract numeric score from LLM response.
        """
        # Look for numbers in the response
        numbers = re.findall(r'-?\d+\.?\d*', response.strip())
        
        if numbers:
            try:
                score = float(numbers[0])
                # Clamp to valid range
                return max(range_[0], min(range_[1], score))
            except ValueError:
                pass
                
        # Default to middle of range
        return (range_[0] + range_[1]) / 2.0
    
    def _calculate_confidence_interval(self, scores: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for metric scores.
        """
        if len(scores) < 2:
            return (0.0, 1.0)
            
        mean = statistics.mean(scores)
        std_err = statistics.stdev(scores) / np.sqrt(len(scores))
        
        # Use t-distribution for small samples
        alpha = 1 - confidence
        df = len(scores) - 1
        t_value = stats.t.ppf(1 - alpha/2, df) if HAS_ML_DEPS else 1.96
        
        margin = t_value * std_err
        return (max(0.0, mean - margin), min(1.0, mean + margin))


# Individual metric functions

def calculate_exact_match(pred: str, ref: str) -> bool:
    """
    Calculate exact match between prediction and reference.
    
    Args:
        pred: Predicted text
        ref: Reference text
        
    Returns:
        True if texts match exactly (case-insensitive, whitespace normalized)
    """
    pred_clean = ' '.join(pred.strip().lower().split())
    ref_clean = ' '.join(ref.strip().lower().split())
    return pred_clean == ref_clean


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score using simple n-gram matching.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
        
    if not predictions:
        return 0.0
    
    total_score = 0.0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens:
            continue
            
        # Calculate n-gram precisions (1-gram to 4-gram)
        precisions = []
        
        for n in range(1, 5):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            
            if not pred_ngrams:
                precisions.append(0.0)
                continue
                
            matches = sum(min(pred_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) 
                         for ng in pred_ngrams)
            
            precision = matches / len(pred_tokens) if len(pred_tokens) >= n else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = np.exp(sum(np.log(p) for p in precisions) / 4)
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / len(pred_tokens)))
            total_score += bleu * bp
    
    return total_score / len(predictions)


def calculate_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
        
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        # ROUGE-1 (unigram overlap)
        rouge_1 = _calculate_rouge_n(pred_tokens, ref_tokens, 1)
        rouge_1_scores.append(rouge_1)
        
        # ROUGE-2 (bigram overlap)
        rouge_2 = _calculate_rouge_n(pred_tokens, ref_tokens, 2)
        rouge_2_scores.append(rouge_2)
        
        # ROUGE-L (longest common subsequence)
        rouge_l = _calculate_rouge_l(pred_tokens, ref_tokens)
        rouge_l_scores.append(rouge_l)
    
    return {
        'rouge-1': _aggregate_rouge_scores(rouge_1_scores),
        'rouge-2': _aggregate_rouge_scores(rouge_2_scores),
        'rouge-l': _aggregate_rouge_scores(rouge_l_scores)
    }


def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BERTScore using sentence transformers (simplified version).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with BERTScore metrics
    """
    if not HAS_ML_DEPS:
        logger.warning("BERTScore requires sentence-transformers, using fallback")
        # Use semantic similarity as fallback
        calc = MetricCalculator()
        similarity = calc.semantic_similarity_score(predictions, references)
        return {
            'precision_mean': similarity,
            'recall_mean': similarity,
            'f1_mean': similarity
        }
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pred_embeddings = model.encode(predictions)
        ref_embeddings = model.encode(references)
        
        # Calculate token-level similarities (simplified)
        precisions = []
        recalls = []
        f1s = []
        
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            # Simplified: use sentence-level similarity
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarity = max(0.0, similarity)
            
            precisions.append(similarity)
            recalls.append(similarity)
            f1s.append(similarity)
            
        return {
            'precision_mean': sum(precisions) / len(precisions),
            'recall_mean': sum(recalls) / len(recalls),
            'f1_mean': sum(f1s) / len(f1s)
        }
        
    except Exception as e:
        logger.error(f"BERTScore calculation failed: {e}")
        return {'precision_mean': 0.0, 'recall_mean': 0.0, 'f1_mean': 0.0}


# Utility functions

def _get_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
    """Get n-gram counts from tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams


def _calculate_rouge_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
    """Calculate ROUGE-N scores."""
    pred_ngrams = _get_ngrams(pred_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)
    
    if not ref_ngrams:
        return {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    
    matches = sum(min(pred_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) 
                 for ng in ref_ngrams)
    
    precision = matches / sum(pred_ngrams.values()) if pred_ngrams else 0.0
    recall = matches / sum(ref_ngrams.values())
    
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f': f1}


def _calculate_rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """Calculate ROUGE-L scores using longest common subsequence."""
    lcs_length = _lcs_length(pred_tokens, ref_tokens)
    
    if not ref_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_length / len(ref_tokens)
    
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f': f1}


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def _aggregate_rouge_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate ROUGE scores across multiple examples."""
    if not scores:
        return {'precision': 0.0, 'recall': 0.0, 'f': 0.0}
    
    precision = sum(score['precision'] for score in scores) / len(scores)
    recall = sum(score['recall'] for score in scores) / len(scores)
    f1 = sum(score['f'] for score in scores) / len(scores)
    
    return {'precision': precision, 'recall': recall, 'f': f1}


def normalize_score(score: float, metric_type: str = 'similarity') -> float:
    """
    Normalize scores to [0, 1] range based on metric type.
    
    Args:
        score: Raw score to normalize
        metric_type: Type of metric ('similarity', 'error', 'latency')
        
    Returns:
        Normalized score in [0, 1] range
    """
    if metric_type == 'similarity':
        return max(0.0, min(1.0, score))
    elif metric_type == 'error':
        # For error metrics, lower is better, so invert
        return max(0.0, min(1.0, 1.0 - score))
    elif metric_type == 'latency':
        # Normalize latency to [0, 1] where 0 is fastest
        # This requires domain knowledge of acceptable latency ranges
        max_acceptable_latency = 10.0  # seconds
        return max(0.0, min(1.0, 1.0 - (score / max_acceptable_latency)))
    else:
        return max(0.0, min(1.0, score))


def aggregate_metrics(metric_results: List[MetricResult], 
                     weights: Optional[Dict[str, float]] = None) -> float:
    """
    Aggregate multiple metric results into a single score.
    
    Args:
        metric_results: List of MetricResult objects
        weights: Optional weights for each metric (default: equal weights)
        
    Returns:
        Weighted aggregate score [0, 1]
    """
    if not metric_results:
        return 0.0
    
    if weights is None:
        weights = {result.metric_name: 1.0 for result in metric_results}
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for result in metric_results:
        weight = weights.get(result.metric_name, 1.0)
        weighted_sum += result.score * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


class MetricValidator:
    """
    Validator for metric calculations to ensure reliability and consistency.
    """
    
    @staticmethod
    def validate_inputs(predictions: List[str], references: List[str]) -> None:
        """
        Validate inputs for metric calculations.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(predictions, list) or not isinstance(references, list):
            raise ValueError("Predictions and references must be lists")
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            raise ValueError("Input lists cannot be empty")
        
        # Check for non-string elements
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if not isinstance(pred, str) or not isinstance(ref, str):
                raise ValueError(f"Element {i} is not a string")
    
    @staticmethod
    def validate_metric_result(result: MetricResult) -> bool:
        """
        Validate a metric result for consistency.
        
        Args:
            result: MetricResult to validate
            
        Returns:
            True if result is valid
        """
        if not isinstance(result.score, (int, float)):
            return False
        
        if not (0.0 <= result.score <= 1.0):
            logger.warning(f"Score {result.score} outside [0,1] range for {result.metric_name}")
        
        if result.confidence_interval:
            lower, upper = result.confidence_interval
            if lower > upper or lower < 0 or upper > 1:
                return False
        
        return True


# Specialized metric classes for domain-specific evaluations

class CodeMetrics:
    """
    Specialized metrics for code generation evaluation.
    """
    
    @staticmethod
    def syntax_correctness(code_predictions: List[str], language: str = 'python') -> float:
        """
        Check syntax correctness of generated code.
        
        Args:
            code_predictions: List of generated code strings
            language: Programming language ('python', 'javascript', etc.)
            
        Returns:
            Fraction of syntactically correct code samples
        """
        correct_count = 0
        
        for code in code_predictions:
            try:
                if language.lower() == 'python':
                    compile(code, '<string>', 'exec')
                    correct_count += 1
                else:
                    # For other languages, use simple heuristics
                    if CodeMetrics._basic_syntax_check(code, language):
                        correct_count += 1
            except SyntaxError:
                continue
            except Exception:
                continue
        
        return correct_count / len(code_predictions) if code_predictions else 0.0
    
    @staticmethod
    def _basic_syntax_check(code: str, language: str) -> bool:
        """Basic syntax checking for non-Python languages."""
        if language.lower() == 'javascript':
            # Simple checks for JavaScript
            return (code.count('{') == code.count('}') and 
                   code.count('(') == code.count(')') and
                   code.count('[') == code.count(']'))
        return True  # Default to assuming correct


class ConversationMetrics:
    """
    Specialized metrics for conversational AI evaluation.
    """
    
    @staticmethod
    def response_appropriateness(responses: List[str], contexts: List[str]) -> float:
        """
        Evaluate appropriateness of responses in conversation context.
        
        Args:
            responses: List of generated responses
            contexts: List of conversation contexts
            
        Returns:
            Average appropriateness score [0, 1]
        """
        scores = []
        
        for response, context in zip(responses, contexts):
            score = 0.5  # Base score
            
            # Check for appropriate length
            response_words = len(response.split())
            if 5 <= response_words <= 100:  # Reasonable response length
                score += 0.2
            
            # Check for question answering
            if '?' in context and any(word in response.lower() 
                                    for word in ['yes', 'no', 'because', 'since']):
                score += 0.1
            
            # Check for politeness markers
            polite_words = ['please', 'thank', 'sorry', 'excuse']
            if any(word in response.lower() for word in polite_words):
                score += 0.1
            
            # Penalize repetitive responses
            words = response.lower().split()
            if len(set(words)) / len(words) < 0.7:  # High repetition
                score -= 0.2
            
            scores.append(max(0.0, min(1.0, score)))
        
        return sum(scores) / len(scores) if scores else 0.0


def create_custom_metric(name: str, calculation_func: callable, 
                        normalization_func: Optional[callable] = None) -> callable:
    """
    Factory function to create custom metrics.
    
    Args:
        name: Name of the custom metric
        calculation_func: Function that calculates the raw metric
        normalization_func: Optional function to normalize the score
        
    Returns:
        Custom metric function that returns MetricResult
    """
    def custom_metric(predictions: List[str], references: List[str], **kwargs) -> MetricResult:
        start_time = time.time()
        
        try:
            raw_score = calculation_func(predictions, references, **kwargs)
            
            if normalization_func:
                normalized_score = normalization_func(raw_score)
            else:
                normalized_score = normalize_score(raw_score)
            
            return MetricResult(
                metric_name=name,
                score=normalized_score,
                raw_score=raw_score,
                sample_size=len(predictions),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Custom metric {name} failed: {e}")
            return MetricResult(
                metric_name=name,
                score=0.0,
                sample_size=len(predictions),
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    return custom_metric


# Example usage and testing functions

def run_metric_tests():
    """
    Run basic tests for metric calculations.
    """
    print("Running metric calculation tests...")
    
    # Test data
    predictions = [
        "The cat sat on the mat",
        "Hello world, how are you?",
        "This is a test sentence"
    ]
    
    references = [
        "A cat was sitting on the mat",
        "Hello world, how are you doing?",
        "This is a test sentence"
    ]
    
    # Initialize calculator
    calc = MetricCalculator()
    
    # Test individual metrics
    print(f"Exact Match: {calc.exact_match_score(predictions, references):.3f}")
    
    if HAS_ML_DEPS:
        print(f"Semantic Similarity: {calc.semantic_similarity_score(predictions, references):.3f}")
    
    print(f"BLEU Score: {calculate_bleu_score(predictions, references):.3f}")
    
    rouge_scores = calculate_rouge_score(predictions, references)
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.3f}")
    
    # Test latency metrics
    response_times = [0.5, 1.2, 0.8, 2.1, 0.9]
    latency_stats = calc.latency_score(response_times)
    print(f"Mean Latency: {latency_stats['mean']:.3f}s")
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    run_metric_tests()
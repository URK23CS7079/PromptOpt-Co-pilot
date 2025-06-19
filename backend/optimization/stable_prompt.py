"""
StablePrompt System for PromptOpt Co-Pilot

This module implements stability-focused prompt variant generation and optimization
techniques that maintain consistency across different inputs while improving performance.
The system uses consistency-based optimization, robustness testing, and template
extraction to create reliable prompt variants.

Key Features:
- Consistency-based optimization
- Robustness testing with noise injection
- Cross-validation stability assessment
- Template extraction from successful prompts
- Stability scoring and metrics
"""

import logging
import statistics
import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, Counter
import json
import hashlib

# Import dependencies
from backend.llm.llama_wrapper import LlamaWrapper
from backend.core.database import DatabaseManager
from backend.evaluation.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StablePromptConfig:
    """Configuration for StablePrompt engine with stability parameters."""
    
    consistency_threshold: float = 0.85
    stability_weight: float = 0.7
    noise_tolerance: float = 0.1
    max_iterations: int = 10
    min_test_cases: int = 5
    cross_validation_folds: int = 3
    robustness_test_levels: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])
    template_similarity_threshold: float = 0.8
    max_variants_per_generation: int = 20
    consistency_smoothing_factor: float = 0.1
    stability_decay_rate: float = 0.95
    

@dataclass
class StableVariant:
    """Represents a stable prompt variant with comprehensive metrics."""
    
    content: str
    stability_score: float
    consistency_metrics: Dict[str, float]
    robustness_rating: float
    generation_method: str
    template_id: Optional[str] = None
    noise_tolerance: float = 0.0
    cross_validation_score: float = 0.0
    variant_hash: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate hash for variant identification."""
        if not self.variant_hash:
            self.variant_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]


class StabilityMetrics:
    """Utility class for calculating various stability metrics."""
    
    @staticmethod
    def calculate_consistency_score(responses: List[str]) -> float:
        """
        Calculate consistency score based on response similarity.
        
        Args:
            responses: List of responses to analyze
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(responses) < 2:
            return 1.0
            
        # Simple similarity based on common words
        word_sets = [set(response.lower().split()) for response in responses]
        
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def calculate_robustness_score(original_responses: List[str], 
                                 noisy_responses: List[str]) -> float:
        """
        Calculate robustness score by comparing original and noisy responses.
        
        Args:
            original_responses: Responses from original prompt
            noisy_responses: Responses from noisy prompt
            
        Returns:
            Robustness score between 0 and 1
        """
        if len(original_responses) != len(noisy_responses):
            return 0.0
            
        similarities = []
        for orig, noisy in zip(original_responses, noisy_responses):
            orig_words = set(orig.lower().split())
            noisy_words = set(noisy.lower().split())
            
            intersection = len(orig_words & noisy_words)
            union = len(orig_words | noisy_words)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def calculate_variance_score(numeric_values: List[float]) -> float:
        """
        Calculate variance-based stability score.
        
        Args:
            numeric_values: List of numeric values to analyze
            
        Returns:
            Variance score (lower variance = higher stability)
        """
        if len(numeric_values) < 2:
            return 1.0
            
        variance = statistics.variance(numeric_values)
        # Convert to stability score (inverse of variance, normalized)
        return 1.0 / (1.0 + variance)


class PromptStabilizer:
    """Utility class for prompt stabilization techniques."""
    
    @staticmethod
    def add_consistency_instructions(prompt: str) -> str:
        """
        Add consistency-enhancing instructions to a prompt.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Enhanced prompt with consistency instructions
        """
        consistency_instructions = [
            "Please provide a consistent and reliable response.",
            "Ensure your answer follows the same format and style.",
            "Maintain consistency in your reasoning approach.",
            "Use a systematic method to approach this task."
        ]
        
        selected_instruction = random.choice(consistency_instructions)
        return f"{prompt}\n\n{selected_instruction}"
    
    @staticmethod
    def normalize_prompt_structure(prompt: str) -> str:
        """
        Normalize prompt structure for better stability.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Normalized prompt with consistent structure
        """
        # Remove extra whitespace
        prompt = re.sub(r'\s+', ' ', prompt.strip())
        
        # Ensure consistent punctuation
        prompt = re.sub(r'([.!?])\s*', r'\1 ', prompt)
        
        # Add structure markers if missing
        if not prompt.endswith(('.', '!', '?', ':')):
            prompt += '.'
        
        return prompt
    
    @staticmethod
    def add_robustness_clauses(prompt: str) -> str:
        """
        Add robustness-enhancing clauses to a prompt.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Enhanced prompt with robustness clauses
        """
        robustness_clauses = [
            "Consider edge cases and variations in your response.",
            "Provide a robust answer that works across different scenarios.",
            "Ensure your response is stable and reliable.",
            "Account for potential variations in the input."
        ]
        
        selected_clause = random.choice(robustness_clauses)
        return f"{prompt} {selected_clause}"
    
    @staticmethod
    def extract_template_pattern(successful_prompts: List[str]) -> str:
        """
        Extract common template pattern from successful prompts.
        
        Args:
            successful_prompts: List of successful prompt examples
            
        Returns:
            Extracted template pattern
        """
        if not successful_prompts:
            return ""
        
        # Find common phrases and structures
        word_frequency = Counter()
        for prompt in successful_prompts:
            words = prompt.lower().split()
            word_frequency.update(words)
        
        # Extract most common patterns
        common_words = [word for word, count in word_frequency.most_common(10)]
        
        # Simple template extraction (could be enhanced with NLP)
        template_parts = []
        for word in common_words:
            if word in ['please', 'provide', 'explain', 'describe', 'analyze']:
                template_parts.append(word)
        
        if template_parts:
            return f"Template: {' '.join(template_parts[:3])}"
        
        return "Generic template"


class StablePromptEngine:
    """
    Main engine for stability-focused prompt variant generation and optimization.
    
    This class implements various techniques for creating stable, consistent prompt
    variants that maintain reliability across different inputs and conditions.
    """
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: StablePromptConfig):
        """
        Initialize StablePrompt engine.
        
        Args:
            llm_wrapper: LLM wrapper for prompt execution
            config: Configuration for stability parameters
        """
        self.llm = llm_wrapper
        self.config = config
        self.db = DatabaseManager()
        self.metrics_calc = MetricsCalculator()
        self.stabilizer = PromptStabilizer()
        self.stability_metrics = StabilityMetrics()
        
        # Cache for template patterns
        self.template_cache: Dict[str, str] = {}
        
        logger.info(f"StablePrompt engine initialized with config: {config}")
    
    def generate_stable_variants(self, base_prompt: str, num_variants: int) -> List[StableVariant]:
        """
        Generate stable prompt variants using multiple stabilization techniques.
        
        Args:
            base_prompt: Base prompt to generate variants from
            num_variants: Number of variants to generate
            
        Returns:
            List of stable prompt variants with metrics
        """
        logger.info(f"Generating {num_variants} stable variants for prompt: {base_prompt[:50]}...")
        
        variants = []
        generation_methods = [
            self._generate_consistency_variant,
            self._generate_robustness_variant,
            self._generate_template_variant,
            self._generate_normalized_variant,
            self._generate_hybrid_variant
        ]
        
        # Generate variants using different methods
        for i in range(num_variants):
            method = generation_methods[i % len(generation_methods)]
            try:
                variant_content = method(base_prompt)
                variant = self._create_stable_variant(
                    variant_content, 
                    method.__name__,
                    base_prompt
                )
                variants.append(variant)
                
            except Exception as e:
                logger.error(f"Error generating variant {i}: {str(e)}")
                continue
        
        # Sort by stability score
        variants.sort(key=lambda x: x.stability_score, reverse=True)
        
        logger.info(f"Generated {len(variants)} stable variants")
        return variants[:num_variants]
    
    def consistency_optimization(self, prompt: str, test_cases: List[dict]) -> str:
        """
        Optimize prompt for consistency across test cases.
        
        Args:
            prompt: Prompt to optimize
            test_cases: Test cases for consistency evaluation
            
        Returns:
            Optimized prompt with improved consistency
        """
        logger.info(f"Optimizing prompt consistency with {len(test_cases)} test cases")
        
        if len(test_cases) < self.config.min_test_cases:
            logger.warning(f"Insufficient test cases for consistency optimization")
            return prompt
        
        best_prompt = prompt
        best_consistency = 0.0
        
        for iteration in range(self.config.max_iterations):
            # Generate candidate prompts
            candidates = self._generate_consistency_candidates(best_prompt)
            
            # Evaluate consistency for each candidate
            for candidate in candidates:
                consistency_score = self._evaluate_consistency(candidate, test_cases)
                
                if consistency_score > best_consistency:
                    best_consistency = consistency_score
                    best_prompt = candidate
                    logger.info(f"Iteration {iteration}: New best consistency: {best_consistency:.3f}")
            
            # Early stopping if threshold reached
            if best_consistency >= self.config.consistency_threshold:
                logger.info(f"Consistency threshold reached: {best_consistency:.3f}")
                break
        
        logger.info(f"Consistency optimization completed. Final score: {best_consistency:.3f}")
        return best_prompt
    
    def stability_scoring(self, variants: List[str], test_cases: List[dict]) -> List[float]:
        """
        Calculate stability scores for prompt variants.
        
        Args:
            variants: List of prompt variants
            test_cases: Test cases for stability evaluation
            
        Returns:
            List of stability scores for each variant
        """
        logger.info(f"Calculating stability scores for {len(variants)} variants")
        
        scores = []
        
        for i, variant in enumerate(variants):
            try:
                # Evaluate variant on test cases
                responses = []
                for test_case in test_cases:
                    response = self.llm.generate_response(variant, test_case.get('input', ''))
                    responses.append(response)
                
                # Calculate stability metrics
                consistency_score = self.stability_metrics.calculate_consistency_score(responses)
                
                # Calculate variance in response lengths (as a stability proxy)
                response_lengths = [len(response) for response in responses]
                variance_score = self.stability_metrics.calculate_variance_score(response_lengths)
                
                # Combined stability score
                stability_score = (
                    self.config.stability_weight * consistency_score +
                    (1 - self.config.stability_weight) * variance_score
                )
                
                scores.append(stability_score)
                
            except Exception as e:
                logger.error(f"Error calculating stability for variant {i}: {str(e)}")
                scores.append(0.0)
        
        logger.info(f"Stability scoring completed. Average score: {statistics.mean(scores):.3f}")
        return scores
    
    def template_stabilization(self, prompt: str, examples: List[dict]) -> str:
        """
        Stabilize prompt using template extraction from successful examples.
        
        Args:
            prompt: Prompt to stabilize
            examples: Examples of successful prompts
            
        Returns:
            Template-stabilized prompt
        """
        logger.info(f"Applying template stabilization with {len(examples)} examples")
        
        if not examples:
            return prompt
        
        # Extract successful prompts
        successful_prompts = [ex.get('prompt', '') for ex in examples if ex.get('success', False)]
        
        if not successful_prompts:
            return prompt
        
        # Extract template pattern
        template_pattern = self.stabilizer.extract_template_pattern(successful_prompts)
        
        # Apply template to current prompt
        if template_pattern and template_pattern != "Generic template":
            stabilized_prompt = f"{template_pattern}\n\n{prompt}"
        else:
            # Apply general stabilization
            stabilized_prompt = self.stabilizer.normalize_prompt_structure(prompt)
            stabilized_prompt = self.stabilizer.add_consistency_instructions(stabilized_prompt)
        
        logger.info("Template stabilization completed")
        return stabilized_prompt
    
    def noise_injection_testing(self, prompt: str, noise_levels: List[float]) -> dict:
        """
        Test prompt robustness using noise injection at different levels.
        
        Args:
            prompt: Prompt to test
            noise_levels: List of noise levels to test (0.0 to 1.0)
            
        Returns:
            Dictionary with robustness test results
        """
        logger.info(f"Running noise injection testing with levels: {noise_levels}")
        
        results = {
            'original_prompt': prompt,
            'noise_tests': [],
            'overall_robustness': 0.0,
            'robustness_by_level': {}
        }
        
        # Generate baseline responses
        baseline_responses = self._generate_test_responses(prompt, 5)
        
        for noise_level in noise_levels:
            try:
                # Create noisy version of prompt
                noisy_prompt = self._inject_noise(prompt, noise_level)
                
                # Generate responses with noisy prompt
                noisy_responses = self._generate_test_responses(noisy_prompt, 5)
                
                # Calculate robustness score
                robustness_score = self.stability_metrics.calculate_robustness_score(
                    baseline_responses, noisy_responses
                )
                
                test_result = {
                    'noise_level': noise_level,
                    'noisy_prompt': noisy_prompt,
                    'robustness_score': robustness_score,
                    'baseline_responses': baseline_responses,
                    'noisy_responses': noisy_responses
                }
                
                results['noise_tests'].append(test_result)
                results['robustness_by_level'][noise_level] = robustness_score
                
            except Exception as e:
                logger.error(f"Error in noise injection test at level {noise_level}: {str(e)}")
                results['robustness_by_level'][noise_level] = 0.0
        
        # Calculate overall robustness
        if results['robustness_by_level']:
            results['overall_robustness'] = statistics.mean(results['robustness_by_level'].values())
        
        logger.info(f"Noise injection testing completed. Overall robustness: {results['overall_robustness']:.3f}")
        return results
    
    def cross_validation_stability(self, prompt: str, datasets: List[List[dict]]) -> float:
        """
        Evaluate prompt stability using cross-validation across multiple datasets.
        
        Args:
            prompt: Prompt to evaluate
            datasets: List of datasets for cross-validation
            
        Returns:
            Cross-validation stability score
        """
        logger.info(f"Running cross-validation stability test with {len(datasets)} datasets")
        
        if len(datasets) < 2:
            logger.warning("Insufficient datasets for cross-validation")
            return 0.0
        
        fold_scores = []
        
        for i, test_dataset in enumerate(datasets):
            try:
                # Use other datasets as training/reference
                training_datasets = [d for j, d in enumerate(datasets) if j != i]
                
                # Evaluate prompt on test dataset
                test_responses = []
                for test_case in test_dataset:
                    response = self.llm.generate_response(prompt, test_case.get('input', ''))
                    test_responses.append(response)
                
                # Calculate consistency within this fold
                fold_consistency = self.stability_metrics.calculate_consistency_score(test_responses)
                fold_scores.append(fold_consistency)
                
                logger.info(f"Fold {i+1} consistency: {fold_consistency:.3f}")
                
            except Exception as e:
                logger.error(f"Error in cross-validation fold {i}: {str(e)}")
                fold_scores.append(0.0)
        
        # Calculate overall stability
        overall_stability = statistics.mean(fold_scores) if fold_scores else 0.0
        stability_variance = statistics.variance(fold_scores) if len(fold_scores) > 1 else 0.0
        
        # Penalize high variance (instability across folds)
        stability_score = overall_stability * (1.0 - stability_variance)
        
        logger.info(f"Cross-validation completed. Stability score: {stability_score:.3f}")
        return stability_score
    
    # Private helper methods
    
    def _generate_consistency_variant(self, base_prompt: str) -> str:
        """Generate variant focused on consistency."""
        variant = self.stabilizer.add_consistency_instructions(base_prompt)
        return self.stabilizer.normalize_prompt_structure(variant)
    
    def _generate_robustness_variant(self, base_prompt: str) -> str:
        """Generate variant focused on robustness."""
        variant = self.stabilizer.add_robustness_clauses(base_prompt)
        return self.stabilizer.normalize_prompt_structure(variant)
    
    def _generate_template_variant(self, base_prompt: str) -> str:
        """Generate variant using template extraction."""
        # Use cached template if available
        template_key = hashlib.md5(base_prompt.encode()).hexdigest()[:8]
        if template_key in self.template_cache:
            template = self.template_cache[template_key]
            return f"{template}\n\n{base_prompt}"
        
        return self.stabilizer.normalize_prompt_structure(base_prompt)
    
    def _generate_normalized_variant(self, base_prompt: str) -> str:
        """Generate normalized variant."""
        return self.stabilizer.normalize_prompt_structure(base_prompt)
    
    def _generate_hybrid_variant(self, base_prompt: str) -> str:
        """Generate hybrid variant using multiple techniques."""
        variant = self.stabilizer.normalize_prompt_structure(base_prompt)
        variant = self.stabilizer.add_consistency_instructions(variant)
        variant = self.stabilizer.add_robustness_clauses(variant)
        return variant
    
    def _create_stable_variant(self, content: str, generation_method: str, base_prompt: str) -> StableVariant:
        """Create StableVariant object with metrics."""
        # Generate simple test cases for evaluation
        test_cases = self._generate_simple_test_cases(base_prompt)
        
        # Calculate basic metrics
        consistency_score = self._evaluate_consistency(content, test_cases)
        robustness_score = self._evaluate_robustness(content)
        
        # Calculate overall stability score
        stability_score = (
            self.config.stability_weight * consistency_score +
            (1 - self.config.stability_weight) * robustness_score
        )
        
        consistency_metrics = {
            'consistency_score': consistency_score,
            'structural_consistency': self._calculate_structural_consistency(content),
            'semantic_consistency': consistency_score  # Simplified
        }
        
        return StableVariant(
            content=content,
            stability_score=stability_score,
            consistency_metrics=consistency_metrics,
            robustness_rating=robustness_score,
            generation_method=generation_method,
            noise_tolerance=self.config.noise_tolerance,
            metadata={'base_prompt_hash': hashlib.md5(base_prompt.encode()).hexdigest()[:8]}
        )
    
    def _generate_consistency_candidates(self, prompt: str) -> List[str]:
        """Generate candidate prompts for consistency optimization."""
        candidates = []
        
        # Add consistency instructions
        candidates.append(self.stabilizer.add_consistency_instructions(prompt))
        
        # Normalize structure
        candidates.append(self.stabilizer.normalize_prompt_structure(prompt))
        
        # Add robustness clauses
        candidates.append(self.stabilizer.add_robustness_clauses(prompt))
        
        # Hybrid approach
        hybrid = self.stabilizer.normalize_prompt_structure(prompt)
        hybrid = self.stabilizer.add_consistency_instructions(hybrid)
        candidates.append(hybrid)
        
        return candidates
    
    def _evaluate_consistency(self, prompt: str, test_cases: List[dict]) -> float:
        """Evaluate prompt consistency across test cases."""
        if not test_cases:
            return 0.0
        
        responses = []
        for test_case in test_cases[:5]:  # Limit for efficiency
            try:
                response = self.llm.generate_response(prompt, test_case.get('input', ''))
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                responses.append("")
        
        return self.stability_metrics.calculate_consistency_score(responses)
    
    def _evaluate_robustness(self, prompt: str) -> float:
        """Evaluate prompt robustness using simple noise injection."""
        try:
            # Generate baseline response
            baseline_response = self.llm.generate_response(prompt, "test input")
            
            # Generate noisy response
            noisy_prompt = self._inject_noise(prompt, 0.1)
            noisy_response = self.llm.generate_response(noisy_prompt, "test input")
            
            # Calculate similarity
            return self.stability_metrics.calculate_robustness_score(
                [baseline_response], [noisy_response]
            )
        except Exception as e:
            logger.error(f"Error evaluating robustness: {str(e)}")
            return 0.0
    
    def _calculate_structural_consistency(self, prompt: str) -> float:
        """Calculate structural consistency of prompt."""
        # Simple structural analysis
        has_clear_structure = bool(re.search(r'[.!?]', prompt))
        has_proper_spacing = not bool(re.search(r'\s{2,}', prompt))
        has_consistent_case = prompt.strip() == prompt.strip().lower() or prompt.strip() == prompt.strip().upper()
        
        score = sum([has_clear_structure, has_proper_spacing, has_consistent_case]) / 3.0
        return score
    
    def _generate_simple_test_cases(self, base_prompt: str) -> List[dict]:
        """Generate simple test cases for evaluation."""
        test_cases = [
            {'input': 'test input 1'},
            {'input': 'test input 2'},
            {'input': 'test input 3'},
            {'input': 'sample data'},
            {'input': 'example query'}
        ]
        return test_cases
    
    def _inject_noise(self, prompt: str, noise_level: float) -> str:
        """Inject noise into prompt at specified level."""
        words = prompt.split()
        num_words_to_modify = int(len(words) * noise_level)
        
        if num_words_to_modify == 0:
            return prompt
        
        # Randomly select words to modify
        indices_to_modify = random.sample(range(len(words)), min(num_words_to_modify, len(words)))
        
        for idx in indices_to_modify:
            original_word = words[idx]
            # Simple character-level noise
            if len(original_word) > 1:
                char_idx = random.randint(0, len(original_word) - 1)
                chars = list(original_word)
                chars[char_idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                words[idx] = ''.join(chars)
        
        return ' '.join(words)
    
    def _generate_test_responses(self, prompt: str, num_responses: int) -> List[str]:
        """Generate test responses for a prompt."""
        responses = []
        for i in range(num_responses):
            try:
                response = self.llm.generate_response(prompt, f"test input {i+1}")
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating test response {i+1}: {str(e)}")
                responses.append("")
        return responses


# Utility functions for external use

def create_stable_prompt_engine(llm_wrapper: LlamaWrapper, 
                              config: Optional[StablePromptConfig] = None) -> StablePromptEngine:
    """
    Factory function to create StablePrompt engine with default configuration.
    
    Args:
        llm_wrapper: LLM wrapper instance
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Configured StablePrompt engine
    """
    if config is None:
        config = StablePromptConfig()
    
    return StablePromptEngine(llm_wrapper, config)


def evaluate_prompt_stability(prompt: str, 
                            test_cases: List[dict], 
                            llm_wrapper: LlamaWrapper) -> Dict[str, float]:
    """
    Utility function to quickly evaluate prompt stability.
    
    Args:
        prompt: Prompt to evaluate
        test_cases: Test cases for evaluation
        llm_wrapper: LLM wrapper instance
        
    Returns:
        Dictionary with stability metrics
    """
    engine = create_stable_prompt_engine(llm_wrapper)
    
    consistency_score = engine._evaluate_consistency(prompt, test_cases)
    robustness_score = engine._evaluate_robustness(prompt)
    
    return {
        'consistency_score': consistency_score,
        'robustness_score': robustness_score,
        'overall_stability': (consistency_score + robustness_score) / 2.0
    }


class AdvancedStabilityAnalyzer:
    """Advanced stability analysis with sophisticated metrics and techniques."""
    
    def __init__(self, llm_wrapper: LlamaWrapper):
        self.llm = llm_wrapper
        self.semantic_cache = {}
        
    def semantic_consistency_analysis(self, responses: List[str]) -> Dict[str, float]:
        """
        Perform semantic consistency analysis using advanced NLP techniques.
        
        Args:
            responses: List of responses to analyze
            
        Returns:
            Dictionary with semantic consistency metrics
        """
        if len(responses) < 2:
            return {'semantic_consistency': 1.0, 'semantic_variance': 0.0}
        
        # Simple semantic analysis (can be enhanced with embeddings)
        semantic_features = []
        for response in responses:
            # Extract semantic features
            features = {
                'sentiment_words': len([w for w in response.lower().split() 
                                      if w in ['good', 'bad', 'excellent', 'poor', 'great', 'terrible']]),
                'question_words': len([w for w in response.lower().split() 
                                     if w in ['what', 'how', 'why', 'when', 'where', 'who']]),
                'action_words': len([w for w in response.lower().split() 
                                   if w in ['analyze', 'evaluate', 'assess', 'determine', 'calculate']]),
                'length_category': 'short' if len(response) < 50 else 'medium' if len(response) < 200 else 'long'
            }
            semantic_features.append(features)
        
        # Calculate semantic consistency
        consistency_scores = []
        for feature_key in ['sentiment_words', 'question_words', 'action_words']:
            values = [f[feature_key] for f in semantic_features]
            if len(set(values)) == 1:
                consistency_scores.append(1.0)
            else:
                variance = statistics.variance(values) if len(values) > 1 else 0
                consistency_scores.append(1.0 / (1.0 + variance))
        
        semantic_consistency = statistics.mean(consistency_scores)
        semantic_variance = statistics.variance([f['sentiment_words'] + f['question_words'] + f['action_words'] 
                                               for f in semantic_features]) if len(semantic_features) > 1 else 0.0
        
        return {
            'semantic_consistency': semantic_consistency,
            'semantic_variance': semantic_variance,
            'feature_consistency': {k: statistics.variance([f[k] for f in semantic_features]) 
                                  for k in ['sentiment_words', 'question_words', 'action_words']}
        }
    
    def temporal_stability_analysis(self, prompt: str, time_intervals: List[int]) -> Dict[str, Any]:
        """
        Analyze prompt stability over time intervals (simulated).
        
        Args:
            prompt: Prompt to analyze
            time_intervals: List of time intervals to simulate
            
        Returns:
            Temporal stability analysis results
        """
        results = {
            'intervals': time_intervals,
            'stability_over_time': [],
            'degradation_rate': 0.0,
            'temporal_consistency': 0.0
        }
        
        baseline_responses = []
        for i in range(5):
            response = self.llm.generate_response(prompt, f"temporal_test_{i}")
            baseline_responses.append(response)
        
        baseline_consistency = self.semantic_consistency_analysis(baseline_responses)['semantic_consistency']
        
        for interval in time_intervals:
            # Simulate temporal degradation
            degraded_prompt = self._simulate_temporal_degradation(prompt, interval)
            
            temporal_responses = []
            for i in range(5):
                response = self.llm.generate_response(degraded_prompt, f"temporal_test_{i}")
                temporal_responses.append(response)
            
            temporal_consistency = self.semantic_consistency_analysis(temporal_responses)['semantic_consistency']
            
            results['stability_over_time'].append({
                'interval': interval,
                'consistency': temporal_consistency,
                'degradation': baseline_consistency - temporal_consistency
            })
        
        # Calculate degradation rate
        if len(results['stability_over_time']) > 1:
            degradations = [s['degradation'] for s in results['stability_over_time']]
            results['degradation_rate'] = statistics.mean(degradations)
        
        results['temporal_consistency'] = statistics.mean([s['consistency'] for s in results['stability_over_time']])
        
        return results
    
    def _simulate_temporal_degradation(self, prompt: str, interval: int) -> str:
        """Simulate temporal degradation of prompt (for testing purposes)."""
        # Simple simulation: add minor variations based on interval
        degradation_factor = interval * 0.01
        if degradation_factor > 0.1:
            # Add slight modifications to simulate degradation
            words = prompt.split()
            num_modifications = int(len(words) * degradation_factor)
            if num_modifications > 0:
                indices = random.sample(range(len(words)), min(num_modifications, len(words)))
                for idx in indices:
                    if random.random() < 0.5:
                        words[idx] = words[idx] + "_mod"
            return ' '.join(words)
        return prompt


class StabilityBenchmark:
    """Benchmarking suite for prompt stability evaluation."""
    
    def __init__(self, engine: StablePromptEngine):
        self.engine = engine
        self.benchmark_results = {}
    
    def run_comprehensive_benchmark(self, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive stability benchmark on multiple test prompts.
        
        Args:
            test_prompts: List of prompts to benchmark
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Running comprehensive stability benchmark on {len(test_prompts)} prompts")
        
        benchmark_results = {
            'prompt_results': [],
            'aggregate_metrics': {},
            'performance_statistics': {},
            'stability_rankings': []
        }
        
        all_stability_scores = []
        all_consistency_scores = []
        all_robustness_scores = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Benchmarking prompt {i+1}/{len(test_prompts)}")
            
            # Generate test cases
            test_cases = self._generate_benchmark_test_cases(prompt)
            
            # Run stability tests
            prompt_results = {
                'original_prompt': prompt,
                'prompt_id': f"bench_{i+1}",
                'stability_metrics': {},
                'variants_performance': [],
                'optimization_results': {}
            }
            
            # Test original prompt stability
            original_stability = self.engine.stability_scoring([prompt], test_cases)[0]
            original_consistency = self.engine._evaluate_consistency(prompt, test_cases)
            original_robustness = self.engine._evaluate_robustness(prompt)
            
            prompt_results['stability_metrics'] = {
                'original_stability': original_stability,
                'original_consistency': original_consistency,
                'original_robustness': original_robustness
            }
            
            # Generate and test variants
            variants = self.engine.generate_stable_variants(prompt, 5)
            for variant in variants:
                variant_performance = {
                    'method': variant.generation_method,
                    'stability_score': variant.stability_score,
                    'consistency_metrics': variant.consistency_metrics,
                    'robustness_rating': variant.robustness_rating,
                    'improvement_over_original': variant.stability_score - original_stability
                }
                prompt_results['variants_performance'].append(variant_performance)
            
            # Test consistency optimization
            optimized_prompt = self.engine.consistency_optimization(prompt, test_cases)
            optimized_stability = self.engine.stability_scoring([optimized_prompt], test_cases)[0]
            
            prompt_results['optimization_results'] = {
                'optimized_prompt': optimized_prompt,
                'optimized_stability': optimized_stability,
                'optimization_improvement': optimized_stability - original_stability
            }
            
            # Collect aggregate data
            all_stability_scores.append(original_stability)
            all_consistency_scores.append(original_consistency)
            all_robustness_scores.append(original_robustness)
            
            benchmark_results['prompt_results'].append(prompt_results)
        
        # Calculate aggregate metrics
        benchmark_results['aggregate_metrics'] = {
            'mean_stability': statistics.mean(all_stability_scores),
            'median_stability': statistics.median(all_stability_scores),
            'std_stability': statistics.stdev(all_stability_scores) if len(all_stability_scores) > 1 else 0.0,
            'mean_consistency': statistics.mean(all_consistency_scores),
            'mean_robustness': statistics.mean(all_robustness_scores),
            'stability_range': max(all_stability_scores) - min(all_stability_scores)
        }
        
        # Performance statistics
        benchmark_results['performance_statistics'] = {
            'best_performing_prompt': test_prompts[all_stability_scores.index(max(all_stability_scores))],
            'worst_performing_prompt': test_prompts[all_stability_scores.index(min(all_stability_scores))],
            'prompts_above_threshold': sum(1 for score in all_stability_scores 
                                         if score >= self.engine.config.consistency_threshold),
            'average_improvement_from_variants': statistics.mean([
                max([v['improvement_over_original'] for v in result['variants_performance']], default=0)
                for result in benchmark_results['prompt_results']
            ])
        }
        
        # Stability rankings
        prompt_stability_pairs = list(zip(test_prompts, all_stability_scores))
        prompt_stability_pairs.sort(key=lambda x: x[1], reverse=True)
        benchmark_results['stability_rankings'] = [
            {'rank': i+1, 'prompt': prompt[:100], 'stability_score': score}
            for i, (prompt, score) in enumerate(prompt_stability_pairs)
        ]
        
        self.benchmark_results = benchmark_results
        logger.info("Comprehensive benchmark completed")
        
        return benchmark_results
    
    def _generate_benchmark_test_cases(self, prompt: str) -> List[dict]:
        """Generate standardized test cases for benchmarking."""
        return [
            {'input': 'benchmark_input_1', 'expected_type': 'analysis'},
            {'input': 'benchmark_input_2', 'expected_type': 'explanation'},
            {'input': 'benchmark_input_3', 'expected_type': 'summary'},
            {'input': 'benchmark_input_4', 'expected_type': 'evaluation'},
            {'input': 'benchmark_input_5', 'expected_type': 'recommendation'}
        ]
    
    def generate_stability_report(self) -> str:
        """Generate comprehensive stability report."""
        if not self.benchmark_results:
            return "No benchmark results available. Run benchmark first."
        
        report = []
        report.append("STABLEPROMPT STABILITY ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Aggregate metrics
        metrics = self.benchmark_results['aggregate_metrics']
        report.append(f"\nAGGREGATE STABILITY METRICS:")
        report.append(f"Mean Stability Score: {metrics['mean_stability']:.3f}")
        report.append(f"Median Stability Score: {metrics['median_stability']:.3f}")
        report.append(f"Stability Standard Deviation: {metrics['std_stability']:.3f}")
        report.append(f"Mean Consistency Score: {metrics['mean_consistency']:.3f}")
        report.append(f"Mean Robustness Score: {metrics['mean_robustness']:.3f}")
        
        # Performance statistics
        perf = self.benchmark_results['performance_statistics']
        report.append(f"\nPERFORMANCE STATISTICS:")
        report.append(f"Prompts Above Threshold: {perf['prompts_above_threshold']}")
        report.append(f"Average Improvement from Variants: {perf['average_improvement_from_variants']:.3f}")
        report.append(f"Best Performing Prompt: {perf['best_performing_prompt'][:80]}...")
        
        # Top stability rankings
        report.append(f"\nTOP STABILITY RANKINGS:")
        for ranking in self.benchmark_results['stability_rankings'][:5]:
            report.append(f"Rank {ranking['rank']}: {ranking['stability_score']:.3f} - {ranking['prompt'][:60]}...")
        
        return "\n".join(report)


# Unit Tests
class TestStablePromptEngine:
    """Unit tests for StablePrompt engine components."""
    
    def __init__(self):
        self.mock_llm = self._create_mock_llm()
        self.config = StablePromptConfig()
        self.engine = StablePromptEngine(self.mock_llm, self.config)
    
    def _create_mock_llm(self):
        """Create mock LLM for testing."""
        class MockLlamaWrapper:
            def generate_response(self, prompt: str, input_text: str) -> str:
                return f"Mock response for: {input_text[:20]}... using prompt: {prompt[:30]}..."
        return MockLlamaWrapper()
    
    def test_stability_metrics(self):
        """Test stability metrics calculations."""
        print("Testing stability metrics...")
        
        # Test consistency score
        responses = ["The answer is yes", "The answer is yes", "The answer is yes"]
        consistency = StabilityMetrics.calculate_consistency_score(responses)
        assert consistency > 0.8, f"Expected high consistency, got {consistency}"
        
        responses = ["Yes", "No", "Maybe"]
        consistency = StabilityMetrics.calculate_consistency_score(responses)
        assert consistency < 0.5, f"Expected low consistency, got {consistency}"
        
        # Test robustness score
        original = ["The quick brown fox"]
        modified = ["The quick brown fox"]
        robustness = StabilityMetrics.calculate_robustness_score(original, modified)
        assert robustness > 0.9, f"Expected high robustness, got {robustness}"
        
        print("✓ Stability metrics tests passed")
    
    def test_prompt_stabilization(self):
        """Test prompt stabilization techniques."""
        print("Testing prompt stabilization...")
        
        stabilizer = PromptStabilizer()
        
        # Test consistency instructions
        original = "Analyze the data"
        enhanced = stabilizer.add_consistency_instructions(original)
        assert len(enhanced) > len(original), "Consistency instructions should add content"
        
        # Test normalization
        messy_prompt = "  Analyze   the  data.  "
        normalized = stabilizer.normalize_prompt_structure(messy_prompt)
        assert "  " not in normalized, "Normalization should remove extra spaces"
        
        # Test robustness clauses
        robust = stabilizer.add_robustness_clauses(original)
        assert len(robust) > len(original), "Robustness clauses should add content"
        
        print("✓ Prompt stabilization tests passed")
    
    def test_variant_generation(self):
        """Test stable variant generation."""
        print("Testing variant generation...")
        
        test_prompt = "Please analyze the following data carefully."
        variants = self.engine.generate_stable_variants(test_prompt, 3)
        
        assert len(variants) <= 3, f"Expected max 3 variants, got {len(variants)}"
        assert all(isinstance(v, StableVariant) for v in variants), "All variants should be StableVariant objects"
        assert all(v.stability_score >= 0 for v in variants), "All stability scores should be non-negative"
        
        print("✓ Variant generation tests passed")
    
    def test_consistency_optimization(self):
        """Test consistency optimization."""
        print("Testing consistency optimization...")
        
        test_prompt = "Analyze the data"
        test_cases = [{'input': f'test_{i}'} for i in range(5)]
        
        optimized = self.engine.consistency_optimization(test_prompt, test_cases)
        assert len(optimized) >= len(test_prompt), "Optimized prompt should not be shorter"
        
        print("✓ Consistency optimization tests passed")
    
    def run_all_tests(self):
        """Run all unit tests."""
        print("Running StablePrompt Engine Unit Tests")
        print("=" * 40)
        
        try:
            self.test_stability_metrics()
            self.test_prompt_stabilization()
            self.test_variant_generation()
            self.test_consistency_optimization()
            
            print("\n✅ All tests passed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage and comprehensive testing
    
    # Mock LLM wrapper for testing
    class MockLlamaWrapper:
        def generate_response(self, prompt: str, input_text: str) -> str:
            # Simulate more realistic responses
            responses = [
                f"Analysis of {input_text}: This shows positive trends with moderate confidence.",
                f"Evaluation of {input_text}: The data indicates stable performance across metrics.",
                f"Assessment of {input_text}: Key findings suggest optimization opportunities exist.",
                f"Review of {input_text}: Overall performance meets expected benchmarks."
            ]
            return random.choice(responses)
    
    # Run unit tests
    print("1. RUNNING UNIT TESTS")
    print("=" * 50)
    test_suite = TestStablePromptEngine()
    test_results = test_suite.run_all_tests()
    
    if not test_results:
        print("Unit tests failed. Exiting.")
        exit(1)
    
    # Demonstration of full system
    print("\n\n2. STABLEPROMPT SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    # Test configuration
    config = StablePromptConfig(
        consistency_threshold=0.8,
        stability_weight=0.7,
        max_iterations=5,
        noise_tolerance=0.15
    )
    
    # Create engine with mock LLM
    mock_llm = MockLlamaWrapper()
    engine = StablePromptEngine(mock_llm, config)
    
    # Test prompts for demonstration
    test_prompts = [
        "Please analyze the following data and provide insights about trends and patterns.",
        "Evaluate the performance metrics and suggest improvements for optimization.",
        "Summarize the key findings from the research study and highlight implications.",
        "Assess the risk factors and provide recommendations for mitigation strategies."
    ]
    
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    
    # Generate stable variants
    print("\n3. GENERATING STABLE VARIANTS")
    print("-" * 30)
    
    for i, prompt in enumerate(test_prompts[:2]):  # Test first 2 prompts
        print(f"\nPrompt {i+1}: {prompt}")
        variants = engine.generate_stable_variants(prompt, 3)
        
        print(f"Generated {len(variants)} variants:")
        for j, variant in enumerate(variants):
            print(f"  Variant {j+1} ({variant.generation_method}):")
            print(f"    Stability Score: {variant.stability_score:.3f}")
            print(f"    Consistency Score: {variant.consistency_metrics.get('consistency_score', 0):.3f}")
            print(f"    Content: {variant.content[:80]}...")
    
    # Demonstrate advanced analysis
    print("\n4. ADVANCED STABILITY ANALYSIS")
    print("-" * 30)
    
    analyzer = AdvancedStabilityAnalyzer(mock_llm)
    
    # Test semantic consistency
    test_responses = [
        "The data shows good performance with positive trends.",
        "Analysis reveals good results and positive indicators.",
        "The findings demonstrate good outcomes with positive signals."
    ]
    
    semantic_analysis = analyzer.semantic_consistency_analysis(test_responses)
    print(f"Semantic Consistency: {semantic_analysis['semantic_consistency']:.3f}")
    print(f"Semantic Variance: {semantic_analysis['semantic_variance']:.3f}")
    
    # Demonstrate benchmarking
    print("\n5. STABILITY BENCHMARKING")
    print("-" * 25)
    
    benchmark = StabilityBenchmark(engine)
    benchmark_results = benchmark.run_comprehensive_benchmark(test_prompts[:2])
    
    print("\nBenchmark Results:")
    print(f"Mean Stability: {benchmark_results['aggregate_metrics']['mean_stability']:.3f}")
    print(f"Mean Consistency: {benchmark_results['aggregate_metrics']['mean_consistency']:.3f}")
    print(f"Mean Robustness: {benchmark_results['aggregate_metrics']['mean_robustness']:.3f}")
    
    # Generate stability report
    report = benchmark.generate_stability_report()
    print(f"\n6. STABILITY REPORT")
    print("-" * 18)
    print(report)
    
    # Demonstrate noise injection testing
    print("\n7. NOISE INJECTION TESTING")
    print("-" * 26)
    
    test_prompt = test_prompts[0]
    noise_results = engine.noise_injection_testing(test_prompt, [0.05, 0.1, 0.15])
    
    print(f"Overall Robustness: {noise_results['overall_robustness']:.3f}")
    for level, score in noise_results['robustness_by_level'].items():
        print(f"Noise Level {level}: Robustness {score:.3f}")
    
    print("\n✅ StablePrompt system demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Stable variant generation with multiple strategies")
    print("- Consistency-based optimization")
    print("- Robustness testing with noise injection")
    print("- Advanced semantic analysis")
    print("- Comprehensive benchmarking")
    print("- Detailed stability reporting")
    
    logger.info("StablePrompt engine demonstration completed successfully")
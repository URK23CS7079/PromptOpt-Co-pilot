"""
Unified Variant Generator for PromptOpt Co-Pilot

This module orchestrates multiple prompt generation strategies including APE,
StablePrompt, and custom mutation techniques to create diverse, high-quality
prompt variants for systematic evaluation and optimization.

Author: PromptOpt Co-Pilot Development Team
Version: 1.0.0
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Union
import logging
from itertools import combinations
import json
import numpy as np
from difflib import SequenceMatcher

# Internal imports
from backend.llm.llama_wrapper import LlamaWrapper
from backend.optimization.ape_engine import APEEngine
from backend.optimization.stable_prompt import StablePromptEngine
from backend.core.database import DatabaseManager
from backend.evaluation.metrics import QualityMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """Represents a generated prompt variant with metadata."""
    id: str
    content: str
    strategy: str
    quality_score: float = 0.0
    diversity_score: float = 0.0
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"{self.strategy}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'strategy': self.strategy,
            'quality_score': self.quality_score,
            'diversity_score': self.diversity_score,
            'generation_time': self.generation_time,
            'metadata': self.metadata,
            'parent_id': self.parent_id,
            'generation_params': self.generation_params
        }


@dataclass
class VariantConfig:
    """Configuration for variant generation strategies."""
    enabled_strategies: List[str] = field(default_factory=lambda: ['ape', 'stable', 'custom'])
    quality_threshold: float = 0.6
    diversity_weight: float = 0.3
    max_variants_per_strategy: int = 10
    batch_size: int = 5
    parallel_generation: bool = True
    remove_duplicates: bool = True
    similarity_threshold: float = 0.8
    temperature: float = 0.7
    max_tokens: int = 512
    custom_mutations: List[str] = field(default_factory=lambda: [
        'paraphrase', 'expand', 'simplify', 'formalize', 'contextualize'
    ])
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        valid_strategies = {'ape', 'stable', 'custom'}
        if not all(s in valid_strategies for s in self.enabled_strategies):
            raise ValueError(f"Invalid strategies. Must be subset of {valid_strategies}")
        
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        
        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("Diversity weight must be between 0 and 1")
        
        return True


@dataclass
class GenerationStats:
    """Statistics for variant generation process."""
    total_variants: int = 0
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)
    quality_distribution: Dict[str, float] = field(default_factory=dict)
    generation_time: float = 0.0
    filtered_count: int = 0
    duplicate_count: int = 0
    average_quality: float = 0.0
    diversity_score: float = 0.0
    success_rate: float = 0.0
    
    def update_from_variants(self, variants: List[PromptVariant]):
        """Update statistics from variant list."""
        self.total_variants = len(variants)
        
        # Strategy breakdown
        self.strategy_breakdown = defaultdict(int)
        for variant in variants:
            self.strategy_breakdown[variant.strategy] += 1
        
        # Quality statistics
        if variants:
            qualities = [v.quality_score for v in variants]
            self.average_quality = np.mean(qualities)
            self.quality_distribution = {
                'min': float(np.min(qualities)),
                'max': float(np.max(qualities)),
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities))
            }
        
        # Generation time
        self.generation_time = sum(v.generation_time for v in variants)


class VariantGenerator:
    """
    Unified variant generator that orchestrates multiple prompt generation strategies.
    
    This class serves as the central coordinator for APE, StablePrompt, and custom
    mutation techniques, providing a unified interface for generating diverse,
    high-quality prompt variants.
    """
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: VariantConfig):
        """
        Initialize the variant generator.
        
        Args:
            llm_wrapper: Wrapper for LLM interactions
            config: Configuration for variant generation
        """
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.config.validate()
        
        # Initialize strategy engines
        self.ape_engine = APEEngine(llm_wrapper) if 'ape' in config.enabled_strategies else None
        self.stable_engine = StablePromptEngine(llm_wrapper) if 'stable' in config.enabled_strategies else None
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.quality_metrics = QualityMetrics()
        
        # Statistics tracking
        self.generation_stats = GenerationStats()
        self._variant_cache: Dict[str, List[PromptVariant]] = {}
        
        logger.info(f"VariantGenerator initialized with strategies: {config.enabled_strategies}")
    
    async def generate_variants(
        self, 
        base_prompt: str, 
        strategy: str, 
        num_variants: int
    ) -> List[PromptVariant]:
        """
        Generate variants using a specific strategy.
        
        Args:
            base_prompt: Base prompt to generate variants from
            strategy: Strategy to use ('ape', 'stable', 'custom')
            num_variants: Number of variants to generate
            
        Returns:
            List of generated prompt variants
        """
        start_time = time.time()
        
        try:
            if strategy not in self.config.enabled_strategies:
                raise ValueError(f"Strategy '{strategy}' not enabled in configuration")
            
            # Generate variants based on strategy
            variants = []
            if strategy == 'ape':
                variants = await self.coordinate_ape_generation(
                    base_prompt, 
                    {'num_variants': num_variants}
                )
            elif strategy == 'stable':
                variants = await self.coordinate_stable_generation(
                    base_prompt, 
                    {'num_variants': num_variants}
                )
            elif strategy == 'custom':
                variants = await self.coordinate_custom_mutations(
                    base_prompt, 
                    {'num_variants': num_variants}
                )
            
            # Calculate quality scores
            for variant in variants:
                variant.quality_score = await self.calculate_variant_quality(variant)
                variant.generation_time = time.time() - start_time
            
            # Apply quality filtering
            filtered_variants = self.quality_filtering(variants, self.config.quality_threshold)
            
            logger.info(f"Generated {len(filtered_variants)} variants using {strategy} strategy")
            return filtered_variants
            
        except Exception as e:
            logger.error(f"Error generating variants with strategy {strategy}: {str(e)}")
            return []
    
    async def hybrid_generation(
        self, 
        base_prompt: str, 
        strategies: List[str]
    ) -> List[PromptVariant]:
        """
        Generate variants using multiple strategies in combination.
        
        Args:
            base_prompt: Base prompt to generate variants from
            strategies: List of strategies to combine
            
        Returns:
            Combined list of variants from all strategies
        """
        start_time = time.time()
        all_variants = []
        
        try:
            # Generate variants from each strategy
            if self.config.parallel_generation:
                # Parallel generation
                tasks = []
                for strategy in strategies:
                    if strategy in self.config.enabled_strategies:
                        task = self.generate_variants(
                            base_prompt, 
                            strategy, 
                            self.config.max_variants_per_strategy
                        )
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        all_variants.extend(result)
                    else:
                        logger.error(f"Error in parallel generation: {result}")
            else:
                # Sequential generation
                for strategy in strategies:
                    if strategy in self.config.enabled_strategies:
                        variants = await self.generate_variants(
                            base_prompt, 
                            strategy, 
                            self.config.max_variants_per_strategy
                        )
                        all_variants.extend(variants)
            
            # Remove duplicates
            if self.config.remove_duplicates:
                all_variants = self.remove_duplicate_variants(all_variants)
            
            # Optimize for diversity
            optimized_variants = self.diversity_optimization(all_variants)
            
            # Update statistics
            self.generation_stats.generation_time = time.time() - start_time
            self.generation_stats.update_from_variants(optimized_variants)
            
            logger.info(f"Hybrid generation completed: {len(optimized_variants)} variants")
            return optimized_variants
            
        except Exception as e:
            logger.error(f"Error in hybrid generation: {str(e)}")
            return []
    
    def quality_filtering(
        self, 
        variants: List[PromptVariant], 
        threshold: float
    ) -> List[PromptVariant]:
        """
        Filter variants based on quality threshold.
        
        Args:
            variants: List of variants to filter
            threshold: Minimum quality score threshold
            
        Returns:
            Filtered list of high-quality variants
        """
        filtered = [v for v in variants if v.quality_score >= threshold]
        self.generation_stats.filtered_count = len(variants) - len(filtered)
        
        logger.info(f"Quality filtering: {len(filtered)}/{len(variants)} variants passed")
        return filtered
    
    def diversity_optimization(self, variants: List[PromptVariant]) -> List[PromptVariant]:
        """
        Optimize variant selection for diversity.
        
        Args:
            variants: List of variants to optimize
            
        Returns:
            Optimized list with improved diversity
        """
        if len(variants) <= 1:
            return variants
        
        # Calculate pairwise similarities
        similarity_matrix = self._calculate_similarity_matrix(variants)
        
        # Select diverse variants using greedy algorithm
        selected_indices = []
        remaining_indices = list(range(len(variants)))
        
        # Start with highest quality variant
        best_idx = max(remaining_indices, key=lambda i: variants[i].quality_score)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Greedily select variants that maximize diversity
        while remaining_indices and len(selected_indices) < self.config.max_variants_per_strategy:
            best_candidate = None
            best_score = -1
            
            for candidate_idx in remaining_indices:
                # Calculate diversity score
                min_similarity = min(
                    similarity_matrix[candidate_idx][selected_idx] 
                    for selected_idx in selected_indices
                )
                diversity_score = 1 - min_similarity
                
                # Combine with quality score
                combined_score = (
                    (1 - self.config.diversity_weight) * variants[candidate_idx].quality_score +
                    self.config.diversity_weight * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        # Update diversity scores
        optimized_variants = [variants[i] for i in selected_indices]
        for i, variant in enumerate(optimized_variants):
            variant.diversity_score = self.calculate_diversity_score([variant], variants)
        
        logger.info(f"Diversity optimization: selected {len(optimized_variants)} variants")
        return optimized_variants
    
    async def batch_generation(
        self, 
        prompts: List[str], 
        config: dict
    ) -> List[List[PromptVariant]]:
        """
        Generate variants for multiple prompts in batch.
        
        Args:
            prompts: List of base prompts
            config: Batch generation configuration
            
        Returns:
            List of variant lists, one for each input prompt
        """
        start_time = time.time()
        batch_results = []
        
        try:
            strategy = config.get('strategy', 'hybrid')
            strategies = config.get('strategies', self.config.enabled_strategies)
            
            if self.config.parallel_generation:
                # Parallel batch processing
                tasks = []
                for prompt in prompts:
                    if strategy == 'hybrid':
                        task = self.hybrid_generation(prompt, strategies)
                    else:
                        task = self.generate_variants(
                            prompt, 
                            strategy, 
                            config.get('num_variants', self.config.max_variants_per_strategy)
                        )
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing prompt {i}: {result}")
                        batch_results[i] = []
            else:
                # Sequential batch processing
                for prompt in prompts:
                    if strategy == 'hybrid':
                        variants = await self.hybrid_generation(prompt, strategies)
                    else:
                        variants = await self.generate_variants(
                            prompt, 
                            strategy, 
                            config.get('num_variants', self.config.max_variants_per_strategy)
                        )
                    batch_results.append(variants)
            
            processing_time = time.time() - start_time
            logger.info(f"Batch generation completed in {processing_time:.2f}s for {len(prompts)} prompts")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            return [[] for _ in prompts]
    
    def get_generation_statistics(self) -> GenerationStats:
        """
        Get comprehensive generation statistics.
        
        Returns:
            Current generation statistics
        """
        return self.generation_stats
    
    async def coordinate_ape_generation(
        self, 
        prompt: str, 
        params: dict
    ) -> List[PromptVariant]:
        """
        Coordinate APE-based variant generation.
        
        Args:
            prompt: Base prompt
            params: Generation parameters
            
        Returns:
            List of APE-generated variants
        """
        if not self.ape_engine:
            logger.warning("APE engine not initialized")
            return []
        
        try:
            # Generate variants using APE
            ape_results = await self.ape_engine.evolve_prompt(
                prompt, 
                generations=params.get('num_variants', 5),
                population_size=params.get('population_size', 10)
            )
            
            variants = []
            for i, result in enumerate(ape_results):
                variant = PromptVariant(
                    id=f"ape_{i}_{hashlib.md5(result.encode()).hexdigest()[:8]}",
                    content=result,
                    strategy='ape',
                    metadata={'generation_method': 'evolutionary'},
                    generation_params=params
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error in APE generation: {str(e)}")
            return []
    
    async def coordinate_stable_generation(
        self, 
        prompt: str, 
        params: dict
    ) -> List[PromptVariant]:
        """
        Coordinate StablePrompt-based variant generation.
        
        Args:
            prompt: Base prompt
            params: Generation parameters
            
        Returns:
            List of stable variants
        """
        if not self.stable_engine:
            logger.warning("Stable engine not initialized")
            return []
        
        try:
            # Generate stable variants
            stable_results = await self.stable_engine.generate_stable_variants(
                prompt,
                num_variants=params.get('num_variants', 5),
                consistency_threshold=params.get('consistency_threshold', 0.8)
            )
            
            variants = []
            for i, result in enumerate(stable_results):
                variant = PromptVariant(
                    id=f"stable_{i}_{hashlib.md5(result.encode()).hexdigest()[:8]}",
                    content=result,
                    strategy='stable',
                    metadata={'generation_method': 'consistency_based'},
                    generation_params=params
                )
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error in stable generation: {str(e)}")
            return []
    
    async def coordinate_custom_mutations(
        self, 
        prompt: str, 
        params: dict
    ) -> List[PromptVariant]:
        """
        Coordinate custom mutation-based variant generation.
        
        Args:
            prompt: Base prompt
            params: Generation parameters
            
        Returns:
            List of mutated variants
        """
        try:
            variants = []
            mutations = params.get('mutations', self.config.custom_mutations)
            num_variants = params.get('num_variants', len(mutations))
            
            # Apply different mutation types
            mutation_prompts = {
                'paraphrase': f"Paraphrase the following prompt while maintaining its core meaning: {prompt}",
                'expand': f"Expand and add more detail to this prompt: {prompt}",
                'simplify': f"Simplify this prompt to make it clearer and more concise: {prompt}",
                'formalize': f"Make this prompt more formal and professional: {prompt}",
                'contextualize': f"Add relevant context to make this prompt more specific: {prompt}"
            }
            
            selected_mutations = mutations[:num_variants]
            
            for i, mutation_type in enumerate(selected_mutations):
                if mutation_type in mutation_prompts:
                    try:
                        result = await self.llm_wrapper.generate(
                            mutation_prompts[mutation_type],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens
                        )
                        
                        variant = PromptVariant(
                            id=f"custom_{mutation_type}_{i}",
                            content=result.strip(),
                            strategy='custom',
                            metadata={
                                'mutation_type': mutation_type,
                                'generation_method': 'custom_mutation'
                            },
                            generation_params=params
                        )
                        variants.append(variant)
                        
                    except Exception as e:
                        logger.error(f"Error in {mutation_type} mutation: {str(e)}")
                        continue
            
            return variants
            
        except Exception as e:
            logger.error(f"Error in custom mutations: {str(e)}")
            return []
    
    async def calculate_variant_quality(self, variant: PromptVariant) -> float:
        """
        Calculate quality score for a variant.
        
        Args:
            variant: Variant to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Use quality metrics to assess variant
            quality_scores = await self.quality_metrics.evaluate_prompt(variant.content)
            
            # Combine different quality aspects
            clarity_score = quality_scores.get('clarity', 0.5)
            specificity_score = quality_scores.get('specificity', 0.5)
            coherence_score = quality_scores.get('coherence', 0.5)
            
            # Weighted average
            quality_score = (
                0.4 * clarity_score +
                0.3 * specificity_score +
                0.3 * coherence_score
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating variant quality: {str(e)}")
            return 0.5  # Default neutral score
    
    def calculate_diversity_score(
        self, 
        variants: List[PromptVariant], 
        reference_set: Optional[List[PromptVariant]] = None
    ) -> float:
        """
        Calculate diversity score for a set of variants.
        
        Args:
            variants: Variants to evaluate
            reference_set: Optional reference set for comparison
            
        Returns:
            Diversity score between 0 and 1
        """
        if len(variants) <= 1:
            return 0.0
        
        try:
            total_similarity = 0.0
            comparison_count = 0
            
            # Calculate pairwise similarities
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    similarity = self._calculate_text_similarity(
                        variants[i].content, 
                        variants[j].content
                    )
                    total_similarity += similarity
                    comparison_count += 1
            
            if comparison_count == 0:
                return 0.0
            
            average_similarity = total_similarity / comparison_count
            diversity_score = 1.0 - average_similarity
            
            return min(max(diversity_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {str(e)}")
            return 0.0
    
    def remove_duplicate_variants(self, variants: List[PromptVariant]) -> List[PromptVariant]:
        """
        Remove duplicate variants based on content similarity.
        
        Args:
            variants: List of variants to deduplicate
            
        Returns:
            List with duplicates removed
        """
        if len(variants) <= 1:
            return variants
        
        unique_variants = []
        seen_content = set()
        
        for variant in variants:
            # Check for exact duplicates first
            content_hash = hashlib.md5(variant.content.encode()).hexdigest()
            if content_hash in seen_content:
                continue
            
            # Check for similar content
            is_duplicate = False
            for unique_variant in unique_variants:
                similarity = self._calculate_text_similarity(
                    variant.content, 
                    unique_variant.content
                )
                if similarity > self.config.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_variants.append(variant)
                seen_content.add(content_hash)
        
        duplicate_count = len(variants) - len(unique_variants)
        self.generation_stats.duplicate_count = duplicate_count
        
        logger.info(f"Removed {duplicate_count} duplicate variants")
        return unique_variants
    
    def _calculate_similarity_matrix(self, variants: List[PromptVariant]) -> List[List[float]]:
        """Calculate pairwise similarity matrix for variants."""
        n = len(variants)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    similarity = self._calculate_text_similarity(
                        variants[i].content, 
                        variants[j].content
                    )
                    matrix[i][j] = similarity
        
        return matrix
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using sequence matching."""
        try:
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except Exception:
            return 0.0
    
    async def save_variants_to_db(self, variants: List[PromptVariant]) -> bool:
        """
        Save variants to database.
        
        Args:
            variants: List of variants to save
            
        Returns:
            Success status
        """
        try:
            variant_data = [variant.to_dict() for variant in variants]
            success = await self.db_manager.save_variants(variant_data)
            
            if success:
                logger.info(f"Saved {len(variants)} variants to database")
            else:
                logger.error("Failed to save variants to database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving variants to database: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear variant cache."""
        self._variant_cache.clear()
        logger.info("Variant cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._variant_cache),
            'cached_prompts': list(self._variant_cache.keys()),
            'memory_usage': sum(
                len(variants) for variants in self._variant_cache.values()
            )
        }


# Utility functions for testing and validation
def create_sample_config() -> VariantConfig:
    """Create a sample configuration for testing."""
    return VariantConfig(
        enabled_strategies=['ape', 'stable', 'custom'],
        quality_threshold=0.7,
        diversity_weight=0.4,
        max_variants_per_strategy=8,
        batch_size=3,
        parallel_generation=True,
        remove_duplicates=True,
        similarity_threshold=0.85
    )


def validate_variant_quality(variant: PromptVariant) -> bool:
    """
    Validate that a variant meets basic quality requirements.
    
    Args:
        variant: Variant to validate
        
    Returns:
        True if variant is valid
    """
    if not variant.content or len(variant.content.strip()) < 10:
        return False
    
    if not variant.strategy or variant.strategy not in ['ape', 'stable', 'custom']:
        return False
    
    if variant.quality_score < 0 or variant.quality_score > 1:
        return False
    
    return True


# Advanced utility functions
class VariantAnalyzer:
    """Advanced analysis tools for variant generation results."""
    
    @staticmethod
    def analyze_strategy_performance(variants: List[PromptVariant]) -> Dict[str, Any]:
        """
        Analyze performance of different generation strategies.
        
        Args:
            variants: List of variants to analyze
            
        Returns:
            Strategy performance analysis
        """
        strategy_stats = defaultdict(lambda: {
            'count': 0,
            'avg_quality': 0.0,
            'avg_diversity': 0.0,
            'avg_generation_time': 0.0,
            'quality_range': [1.0, 0.0]
        })
        
        for variant in variants:
            stats = strategy_stats[variant.strategy]
            stats['count'] += 1
            stats['avg_quality'] += variant.quality_score
            stats['avg_diversity'] += variant.diversity_score
            stats['avg_generation_time'] += variant.generation_time
            
            # Update quality range
            stats['quality_range'][0] = min(stats['quality_range'][0], variant.quality_score)
            stats['quality_range'][1] = max(stats['quality_range'][1], variant.quality_score)
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats['count'] > 0:
                stats['avg_quality'] /= stats['count']
                stats['avg_diversity'] /= stats['count']
                stats['avg_generation_time'] /= stats['count']
        
        return dict(strategy_stats)
    
    @staticmethod
    def identify_quality_patterns(variants: List[PromptVariant]) -> Dict[str, Any]:
        """
        Identify patterns in variant quality across different dimensions.
        
        Args:
            variants: List of variants to analyze
            
        Returns:
            Quality pattern analysis
        """
        patterns = {
            'length_quality_correlation': 0.0,
            'strategy_quality_ranking': {},
            'quality_distribution': {},
            'outliers': [],
            'best_performers': [],
            'improvement_suggestions': []
        }
        
        if not variants:
            return patterns
        
        # Length-quality correlation
        lengths = [len(v.content) for v in variants]
        qualities = [v.quality_score for v in variants]
        
        if len(lengths) > 1:
            patterns['length_quality_correlation'] = np.corrcoef(lengths, qualities)[0, 1]
        
        # Strategy quality ranking
        strategy_performance = VariantAnalyzer.analyze_strategy_performance(variants)
        patterns['strategy_quality_ranking'] = {
            strategy: stats['avg_quality'] 
            for strategy, stats in strategy_performance.items()
        }
        
        # Quality distribution
        quality_bins = np.histogram(qualities, bins=5, range=(0, 1))
        patterns['quality_distribution'] = {
            'bins': quality_bins[1].tolist(),
            'counts': quality_bins[0].tolist()
        }
        
        # Identify outliers (variants with unusually high/low quality)
        if qualities:
            q1, q3 = np.percentile(qualities, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            patterns['outliers'] = [
                v.id for v in variants 
                if v.quality_score < lower_bound or v.quality_score > upper_bound
            ]
        
        # Best performers
        patterns['best_performers'] = [
            v.id for v in sorted(variants, key=lambda x: x.quality_score, reverse=True)[:3]
        ]
        
        return patterns


class VariantOptimizer:
    """Advanced optimization techniques for variant generation."""
    
    def __init__(self, generator: VariantGenerator):
        """Initialize with variant generator reference."""
        self.generator = generator
    
    async def adaptive_generation(
        self, 
        base_prompt: str, 
        target_quality: float = 0.8,
        max_iterations: int = 5
    ) -> List[PromptVariant]:
        """
        Adaptively generate variants until target quality is reached.
        
        Args:
            base_prompt: Base prompt to optimize
            target_quality: Target average quality score
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized variants meeting quality target
        """
        best_variants = []
        current_prompt = base_prompt
        
        for iteration in range(max_iterations):
            # Generate variants with current best strategy
            variants = await self.generator.hybrid_generation(
                current_prompt, 
                self.generator.config.enabled_strategies
            )
            
            if not variants:
                break
            
            # Check if target quality is met
            avg_quality = np.mean([v.quality_score for v in variants])
            logger.info(f"Iteration {iteration + 1}: Average quality = {avg_quality:.3f}")
            
            if avg_quality >= target_quality:
                best_variants = variants
                break
            
            # Select best variant as new base for next iteration
            best_variant = max(variants, key=lambda x: x.quality_score)
            current_prompt = best_variant.content
            best_variants = variants
        
        return best_variants
    
    async def multi_objective_optimization(
        self, 
        base_prompt: str,
        quality_weight: float = 0.6,
        diversity_weight: float = 0.4
    ) -> List[PromptVariant]:
        """
        Optimize variants for both quality and diversity using Pareto optimization.
        
        Args:
            base_prompt: Base prompt to optimize
            quality_weight: Weight for quality objective
            diversity_weight: Weight for diversity objective
            
        Returns:
            Pareto-optimal variants
        """
        # Generate large pool of variants
        all_variants = await self.generator.hybrid_generation(
            base_prompt, 
            self.generator.config.enabled_strategies
        )
        
        if len(all_variants) < 2:
            return all_variants
        
        # Calculate combined scores
        diversity_scores = []
        for variant in all_variants:
            other_variants = [v for v in all_variants if v.id != variant.id]
            diversity = self.generator.calculate_diversity_score([variant], other_variants)
            diversity_scores.append(diversity)
        
        # Multi-objective scoring
        pareto_variants = []
        for i, variant in enumerate(all_variants):
            combined_score = (
                quality_weight * variant.quality_score +
                diversity_weight * diversity_scores[i]
            )
            variant.metadata['combined_score'] = combined_score
            variant.diversity_score = diversity_scores[i]
        
        # Select Pareto-optimal solutions
        sorted_variants = sorted(all_variants, key=lambda x: x.metadata['combined_score'], reverse=True)
        
        # Simple Pareto selection (non-dominated solutions)
        for candidate in sorted_variants:
            is_dominated = False
            for selected in pareto_variants:
                if (selected.quality_score >= candidate.quality_score and 
                    selected.diversity_score >= candidate.diversity_score and
                    (selected.quality_score > candidate.quality_score or 
                     selected.diversity_score > candidate.diversity_score)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_variants.append(candidate)
        
        return pareto_variants[:self.generator.config.max_variants_per_strategy]


class VariantValidator:
    """Validation utilities for generated variants."""
    
    @staticmethod
    def validate_variant_set(variants: List[PromptVariant]) -> Dict[str, Any]:
        """
        Comprehensive validation of a variant set.
        
        Args:
            variants: List of variants to validate
            
        Returns:
            Validation results and issues
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        if not variants:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No variants provided")
            return validation_results
        
        # Check individual variant validity
        invalid_variants = []
        for variant in variants:
            if not validate_variant_quality(variant):
                invalid_variants.append(variant.id)
        
        if invalid_variants:
            validation_results['issues'].append(f"Invalid variants: {invalid_variants}")
            validation_results['is_valid'] = False
        
        # Check diversity
        if len(variants) > 1:
            similarities = []
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    sim = SequenceMatcher(
                        None, 
                        variants[i].content.lower(), 
                        variants[j].content.lower()
                    ).ratio()
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            if avg_similarity > 0.9:
                validation_results['warnings'].append(
                    f"High similarity detected (avg: {avg_similarity:.2f})"
                )
        
        # Quality distribution analysis
        qualities = [v.quality_score for v in variants]
        validation_results['statistics'] = {
            'count': len(variants),
            'avg_quality': np.mean(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'quality_std': np.std(qualities)
        }
        
        # Generate recommendations
        if validation_results['statistics']['avg_quality'] < 0.6:
            validation_results['recommendations'].append(
                "Consider lowering quality threshold or improving generation strategies"
            )
        
        if validation_results['statistics']['quality_std'] < 0.1:
            validation_results['recommendations'].append(
                "Low quality variance - consider more diverse generation strategies"
            )
        
        return validation_results


# Performance benchmarking utilities
class VariantBenchmark:
    """Benchmarking tools for variant generation performance."""
    
    def __init__(self, generator: VariantGenerator):
        """Initialize with variant generator."""
        self.generator = generator
        self.benchmark_results = []
    
    async def benchmark_strategies(
        self, 
        test_prompts: List[str],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark different generation strategies.
        
        Args:
            test_prompts: List of prompts for testing
            iterations: Number of iterations per test
            
        Returns:
            Comprehensive benchmark results
        """
        results = {
            'strategy_performance': {},
            'scalability_metrics': {},
            'quality_consistency': {},
            'generation_speed': {}
        }
        
        strategies = self.generator.config.enabled_strategies
        
        for strategy in strategies:
            strategy_results = {
                'total_time': 0.0,
                'total_variants': 0,
                'avg_quality': 0.0,
                'quality_variance': 0.0,
                'success_rate': 0.0
            }
            
            successful_runs = 0
            all_qualities = []
            
            for iteration in range(iterations):
                for prompt in test_prompts:
                    start_time = time.time()
                    
                    try:
                        variants = await self.generator.generate_variants(
                            prompt, strategy, 5
                        )
                        
                        generation_time = time.time() - start_time
                        
                        if variants:
                            successful_runs += 1
                            strategy_results['total_time'] += generation_time
                            strategy_results['total_variants'] += len(variants)
                            
                            qualities = [v.quality_score for v in variants]
                            all_qualities.extend(qualities)
                        
                    except Exception as e:
                        logger.error(f"Benchmark error for {strategy}: {e}")
                        continue
            
            # Calculate final metrics
            total_runs = iterations * len(test_prompts)
            if successful_runs > 0:
                strategy_results['success_rate'] = successful_runs / total_runs
                strategy_results['avg_quality'] = np.mean(all_qualities) if all_qualities else 0.0
                strategy_results['quality_variance'] = np.var(all_qualities) if all_qualities else 0.0
                strategy_results['avg_time_per_variant'] = (
                    strategy_results['total_time'] / strategy_results['total_variants']
                    if strategy_results['total_variants'] > 0 else 0.0
                )
            
            results['strategy_performance'][strategy] = strategy_results
        
        return results
    
    def generate_performance_report(self, benchmark_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            benchmark_results: Results from benchmark_strategies
            
        Returns:
            Formatted performance report
        """
        report = ["=== Variant Generation Performance Report ===\n"]
        
        strategy_perf = benchmark_results.get('strategy_performance', {})
        
        if not strategy_perf:
            return "No benchmark data available."
        
        # Strategy comparison
        report.append("Strategy Performance Comparison:")
        report.append("-" * 40)
        
        for strategy, metrics in strategy_perf.items():
            report.append(f"\n{strategy.upper()} Strategy:")
            report.append(f"  Success Rate: {metrics['success_rate']:.1%}")
            report.append(f"  Average Quality: {metrics['avg_quality']:.3f}")
            report.append(f"  Quality Variance: {metrics['quality_variance']:.3f}")
            report.append(f"  Time per Variant: {metrics.get('avg_time_per_variant', 0):.3f}s")
            report.append(f"  Total Variants: {metrics['total_variants']}")
        
        # Recommendations
        report.append("\n\nRecommendations:")
        report.append("-" * 20)
        
        # Find best performing strategy
        best_strategy = max(
            strategy_perf.items(),
            key=lambda x: x[1]['avg_quality'] * x[1]['success_rate']
        )[0]
        
        report.append(f"• Best overall strategy: {best_strategy}")
        
        # Performance warnings
        for strategy, metrics in strategy_perf.items():
            if metrics['success_rate'] < 0.8:
                report.append(f"• {strategy} has low success rate ({metrics['success_rate']:.1%})")
            
            if metrics['quality_variance'] > 0.1:
                report.append(f"• {strategy} shows high quality variance")
        
        return "\n".join(report)


# Example usage and comprehensive testing
if __name__ == "__main__":
    async def main():
        """Comprehensive example usage of the VariantGenerator system."""
        from backend.llm.llama_wrapper import LlamaWrapper
        
        # Initialize components
        llm_wrapper = LlamaWrapper()
        config = create_sample_config()
        generator = VariantGenerator(llm_wrapper, config)
        
        # Initialize advanced utilities
        optimizer = VariantOptimizer(generator)
        benchmark = VariantBenchmark(generator)
        
        print("=== PromptOpt Co-Pilot Variant Generator Demo ===\n")
        
        # Test prompts
        test_prompts = [
            "Explain the concept of machine learning to a beginner.",
            "Write a professional email requesting a meeting.",
            "Describe the benefits of renewable energy."
        ]
        
        print("1. Basic Variant Generation:")
        print("-" * 30)
        variants = await generator.generate_variants(test_prompts[0], 'custom', 3)
        
        for i, variant in enumerate(variants, 1):
            print(f"{i}. {variant.content[:80]}...")
            print(f"   Quality: {variant.quality_score:.3f} | Strategy: {variant.strategy}")
        
        print(f"\n2. Hybrid Generation:")
        print("-" * 30)
        hybrid_variants = await generator.hybrid_generation(
            test_prompts[0], 
            ['custom', 'stable']
        )
        print(f"Generated {len(hybrid_variants)} hybrid variants")
        
        # Advanced optimization
        print(f"\n3. Adaptive Optimization:")
        print("-" * 30)
        optimized_variants = await optimizer.adaptive_generation(
            test_prompts[0], 
            target_quality=0.7,
            max_iterations=3
        )
        print(f"Adaptive optimization produced {len(optimized_variants)} variants")
        
        # Multi-objective optimization
        print(f"\n4. Multi-Objective Optimization:")
        print("-" * 30)
        pareto_variants = await optimizer.multi_objective_optimization(
            test_prompts[0],
            quality_weight=0.6,
            diversity_weight=0.4
        )
        print(f"Pareto optimization selected {len(pareto_variants)} variants")
        
        # Validation
        print(f"\n5. Variant Validation:")
        print("-" * 30)
        validation_results = VariantValidator.validate_variant_set(hybrid_variants)
        print(f"Validation passed: {validation_results['is_valid']}")
        print(f"Average quality: {validation_results['statistics']['avg_quality']:.3f}")
        
        if validation_results['warnings']:
            print(f"Warnings: {validation_results['warnings']}")
        
        # Performance analysis
        print(f"\n6. Strategy Performance Analysis:")
        print("-" * 30)
        if hybrid_variants:
            performance = VariantAnalyzer.analyze_strategy_performance(hybrid_variants)
            for strategy, stats in performance.items():
                print(f"{strategy}: Avg Quality = {stats['avg_quality']:.3f}, Count = {stats['count']}")
        
        # Statistics
        print(f"\n7. Generation Statistics:")
        print("-" * 30)
        stats = generator.get_generation_statistics()
        print(f"Total variants generated: {stats.total_variants}")
        print(f"Average quality: {stats.average_quality:.3f}")
        print(f"Total generation time: {stats.generation_time:.2f}s")
        print(f"Strategy breakdown: {dict(stats.strategy_breakdown)}")
        
        # Batch processing demo
        print(f"\n8. Batch Processing:")
        print("-" * 30)
        batch_results = await generator.batch_generation(
            test_prompts[:2],
            {'strategy': 'hybrid', 'strategies': ['custom']}
        )
        total_batch_variants = sum(len(variants) for variants in batch_results)
        print(f"Batch processing generated {total_batch_variants} total variants")
        
        print(f"\n=== Demo Complete ===")
    
    # Run the comprehensive example
    asyncio.run(main())
"""
Automatic Prompt Engineer (APE) System for PromptOpt Co-Pilot

This module implements the APE methodology for automatic prompt optimization through:
- Evolutionary search with mutation and selection
- Beam search for systematic exploration
- Instruction-following prompt generation
- Style-based paraphrasing
- Multi-strategy variant generation

Based on research from "Large Language Models Are Human-Level Prompt Engineers"
by Zhou et al. (2022) - https://arxiv.org/abs/2211.01910

The APE system uses local LLMs to generate and refine prompt variants through
iterative improvement, mimicking natural selection and systematic search.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
import asyncio
import hashlib
from collections import defaultdict

# Internal imports
from backend.llm.llama_wrapper import LlamaWrapper
from backend.core.database import get_db_connection

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class APEConfig:
    """Configuration for APE Engine operations"""
    generation_temperature: float = 0.8
    max_variants: int = 50
    beam_width: int = 5
    mutation_rate: float = 0.3
    selection_pressure: float = 0.7
    max_generations: int = 10
    population_size: int = 20
    elite_size: int = 5
    crossover_rate: float = 0.2
    diversity_threshold: float = 0.8
    max_prompt_length: int = 2000
    min_prompt_length: int = 10
    paraphrase_styles: List[str] = field(default_factory=lambda: [
        "formal", "casual", "technical", "simplified", "detailed", "concise"
    ])
    mutation_strategies: List[str] = field(default_factory=lambda: [
        "add_context", "simplify", "specify", "format", "rephrase", "restructure"
    ])


@dataclass
class PromptVariant:
    """Represents a generated prompt variant with metadata"""
    content: str
    generation_method: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    variant_id: str = field(default="")
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.variant_id:
            # Generate unique ID based on content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.variant_id = f"{self.generation_method}_{content_hash}"


class APEEngine:
    """
    Automatic Prompt Engineer - Main engine for prompt optimization
    
    Implements multiple strategies for prompt variant generation:
    1. Evolutionary Search: Mimics natural selection with mutations and crossover
    2. Beam Search: Systematic exploration of prompt space
    3. Instruction Following: Direct instruction-based modifications
    4. Paraphrasing: Style and format variations
    """
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: APEConfig):
        """
        Initialize APE Engine
        
        Args:
            llm_wrapper: Interface to local LLM for generation
            config: Configuration parameters for APE operations
        """
        self.llm = llm_wrapper
        self.config = config
        self.generation_history: List[PromptVariant] = []
        self.best_variants: List[PromptVariant] = []
        
        # Initialize mutation strategies
        self.mutation_functions = {
            "add_context": self.add_context_mutation,
            "simplify": self.simplify_mutation,
            "specify": self.specify_mutation,
            "format": self.format_mutation,
            "rephrase": self.rephrase_mutation,
            "restructure": self.restructure_mutation
        }
        
        logger.info(f"APE Engine initialized with config: {asdict(config)}")
    
    async def generate_variants(
        self, 
        base_prompt: str, 
        num_variants: int,
        test_cases: Optional[List[Dict]] = None
    ) -> List[PromptVariant]:
        """
        Main entry point for generating prompt variants using multiple strategies
        
        Args:
            base_prompt: Original prompt to optimize
            num_variants: Number of variants to generate
            test_cases: Optional test cases for immediate scoring
            
        Returns:
            List of generated prompt variants, sorted by score
        """
        logger.info(f"Generating {num_variants} variants for prompt: {base_prompt[:100]}...")
        
        try:
            # Validate input
            if not base_prompt.strip():
                raise ValueError("Base prompt cannot be empty")
            
            if len(base_prompt) > self.config.max_prompt_length:
                logger.warning(f"Base prompt truncated from {len(base_prompt)} to {self.config.max_prompt_length} chars")
                base_prompt = base_prompt[:self.config.max_prompt_length]
            
            all_variants = []
            variants_per_method = max(1, num_variants // 4)  # Distribute across 4 main methods
            
            # Strategy 1: Evolutionary Search
            logger.info("Running evolutionary search...")
            evolutionary_variants = await self.evolutionary_search(
                population=[base_prompt], 
                generations=min(5, self.config.max_generations),
                target_size=variants_per_method
            )
            all_variants.extend(evolutionary_variants)
            
            # Strategy 2: Beam Search
            logger.info("Running beam search...")
            beam_variants = await self.beam_search_variants(
                base_prompt, 
                beam_width=min(self.config.beam_width, variants_per_method)
            )
            all_variants.extend(beam_variants)
            
            # Strategy 3: Instruction-based variants
            logger.info("Generating instruction-based variants...")
            instruction_variants = await self._generate_instruction_variants(
                base_prompt, 
                variants_per_method
            )
            all_variants.extend(instruction_variants)
            
            # Strategy 4: Paraphrasing variants
            logger.info("Generating paraphrasing variants...")
            paraphrase_variants = await self._generate_paraphrase_variants(
                base_prompt, 
                variants_per_method
            )
            all_variants.extend(paraphrase_variants)
            
            # Remove duplicates and ensure diversity
            unique_variants = self._ensure_diversity(all_variants)
            
            # Limit to requested number
            if len(unique_variants) > num_variants:
                unique_variants = unique_variants[:num_variants]
            
            # Score variants if test cases provided
            if test_cases:
                logger.info("Scoring generated variants...")
                await self._score_variants_batch(unique_variants, test_cases)
                unique_variants.sort(key=lambda v: v.score, reverse=True)
            
            # Store generation history
            self.generation_history.extend(unique_variants)
            self._update_best_variants(unique_variants)
            
            logger.info(f"Generated {len(unique_variants)} unique variants")
            return unique_variants
            
        except Exception as e:
            logger.error(f"Error in generate_variants: {str(e)}")
            raise
    
    async def evolutionary_search(
        self, 
        population: List[str], 
        generations: int,
        target_size: int = None
    ) -> List[PromptVariant]:
        """
        Evolutionary search algorithm for prompt optimization
        
        Uses genetic algorithm principles:
        - Selection: Keep best performing prompts
        - Mutation: Apply random modifications
        - Crossover: Combine elements from parent prompts
        - Elitism: Preserve best candidates
        
        Args:
            population: Initial population of prompts
            generations: Number of evolutionary generations
            target_size: Target population size
            
        Returns:
            List of evolved prompt variants
        """
        logger.info(f"Starting evolutionary search: {len(population)} initial, {generations} generations")
        
        try:
            if target_size is None:
                target_size = self.config.population_size
            
            # Initialize population with variants
            current_pop = [
                PromptVariant(
                    content=prompt,
                    generation_method="evolutionary_base",
                    generation=0
                ) for prompt in population
            ]
            
            # Expand initial population through mutations
            while len(current_pop) < target_size:
                parent = random.choice(current_pop)
                mutated = await self._mutate_prompt(parent.content)
                if mutated and mutated != parent.content:
                    current_pop.append(PromptVariant(
                        content=mutated,
                        generation_method="evolutionary_init",
                        parent_id=parent.variant_id,
                        generation=0
                    ))
            
            best_variants = []
            
            for gen in range(generations):
                logger.debug(f"Generation {gen + 1}/{generations}")
                
                # Selection: Keep elite performers
                elite_size = min(self.config.elite_size, len(current_pop))
                elite = current_pop[:elite_size]  # Assume sorted by fitness
                
                new_population = elite.copy()
                
                # Generate offspring through mutation and crossover
                while len(new_population) < target_size:
                    if random.random() < self.config.crossover_rate and len(elite) >= 2:
                        # Crossover: Combine two parents
                        parent1, parent2 = random.sample(elite, 2)
                        offspring_content = await self._crossover_prompts(
                            parent1.content, parent2.content
                        )
                        if offspring_content:
                            offspring = PromptVariant(
                                content=offspring_content,
                                generation_method="evolutionary_crossover",
                                parent_id=f"{parent1.variant_id},{parent2.variant_id}",
                                generation=gen + 1
                            )
                            new_population.append(offspring)
                    else:
                        # Mutation: Modify single parent
                        parent = random.choice(elite)
                        mutated_content = await self._mutate_prompt(parent.content)
                        if mutated_content and mutated_content != parent.content:
                            mutant = PromptVariant(
                                content=mutated_content,
                                generation_method="evolutionary_mutation",
                                parent_id=parent.variant_id,
                                generation=gen + 1
                            )
                            new_population.append(mutant)
                
                current_pop = new_population
                best_variants.extend(current_pop)
            
            # Return diverse set of best variants
            final_variants = self._ensure_diversity(best_variants)
            logger.info(f"Evolutionary search completed: {len(final_variants)} variants generated")
            
            return final_variants[:target_size]
            
        except Exception as e:
            logger.error(f"Error in evolutionary_search: {str(e)}")
            return []
    
    async def beam_search_variants(self, prompt: str, beam_width: int) -> List[PromptVariant]:
        """
        Beam search for systematic prompt exploration
        
        Maintains multiple candidate paths simultaneously, expanding the most
        promising ones at each step. Less random than evolutionary search.
        
        Args:
            prompt: Base prompt to expand from
            beam_width: Number of candidates to maintain
            
        Returns:
            List of prompt variants from beam search
        """
        logger.info(f"Starting beam search with width {beam_width}")
        
        try:
            # Initialize beam with base prompt
            beam = [PromptVariant(
                content=prompt,
                generation_method="beam_base",
                generation=0
            )]
            
            all_variants = []
            max_depth = 3  # Limit search depth
            
            for depth in range(max_depth):
                new_candidates = []
                
                for current in beam:
                    # Generate expansions for current candidate
                    expansions = await self._generate_beam_expansions(
                        current.content, 
                        beam_width
                    )
                    
                    for expansion in expansions:
                        variant = PromptVariant(
                            content=expansion,
                            generation_method=f"beam_depth_{depth + 1}",
                            parent_id=current.variant_id,
                            generation=depth + 1
                        )
                        new_candidates.append(variant)
                
                # Select best candidates for next beam
                if new_candidates:
                    # Simple diversity-based selection (in production, use scoring)
                    diverse_candidates = self._ensure_diversity(new_candidates)
                    beam = diverse_candidates[:beam_width]
                    all_variants.extend(new_candidates)
                else:
                    break
            
            logger.info(f"Beam search completed: {len(all_variants)} variants generated")
            return all_variants
            
        except Exception as e:
            logger.error(f"Error in beam_search_variants: {str(e)}")
            return []
    
    async def paraphrase_prompt(self, prompt: str, style: str) -> str:
        """
        Generate paraphrased version of prompt in specified style
        
        Args:
            prompt: Original prompt
            style: Target style (formal, casual, technical, etc.)
            
        Returns:
            Paraphrased prompt
        """
        try:
            paraphrase_instruction = f"""
Paraphrase the following prompt in a {style} style while preserving its core meaning and intent:

Original prompt: {prompt}

Requirements:
- Maintain the same instructional purpose
- Adapt tone and vocabulary to {style} style
- Keep the same level of specificity
- Preserve any important constraints or requirements

Paraphrased prompt:"""

            response = await self.llm.generate_async(
                paraphrase_instruction,
                temperature=self.config.generation_temperature,
                max_tokens=len(prompt.split()) * 2  # Allow for expansion
            )
            
            paraphrased = self._extract_clean_response(response)
            return paraphrased if paraphrased != prompt else prompt
            
        except Exception as e:
            logger.error(f"Error in paraphrase_prompt: {str(e)}")
            return prompt
    
    async def instruction_following_variant(self, prompt: str, instruction: str) -> str:
        """
        Generate prompt variant by following specific instruction
        
        Args:
            prompt: Original prompt
            instruction: Modification instruction
            
        Returns:
            Modified prompt
        """
        try:
            modification_prompt = f"""
Apply the following instruction to modify this prompt:

Instruction: {instruction}
Original prompt: {prompt}

Generate a modified version that follows the instruction while maintaining the prompt's core purpose:

Modified prompt:"""

            response = await self.llm.generate_async(
                modification_prompt,
                temperature=self.config.generation_temperature
            )
            
            modified = self._extract_clean_response(response)
            return modified if modified != prompt else prompt
            
        except Exception as e:
            logger.error(f"Error in instruction_following_variant: {str(e)}")
            return prompt
    
    async def score_variants(
        self, 
        variants: List[str], 
        test_cases: List[Dict]
    ) -> List[float]:
        """
        Quick scoring of prompt variants against test cases
        
        Args:
            variants: List of prompt strings to score
            test_cases: Test cases with input/expected output
            
        Returns:
            List of scores (0.0 to 1.0) for each variant
        """
        logger.info(f"Scoring {len(variants)} variants against {len(test_cases)} test cases")
        
        try:
            scores = []
            
            for variant in variants:
                variant_score = await self._score_single_variant(variant, test_cases)
                scores.append(variant_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in score_variants: {str(e)}")
            return [0.0] * len(variants)
    
    # Mutation Strategy Implementations
    
    async def add_context_mutation(self, prompt: str) -> str:
        """Add contextual information to the prompt"""
        context_instruction = f"""
Add helpful context or background information to make this prompt more effective:

Original: {prompt}

Enhanced prompt with context:"""
        
        try:
            response = await self.llm.generate_async(context_instruction, temperature=0.7)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in add_context_mutation: {e}")
            return prompt
    
    async def simplify_mutation(self, prompt: str) -> str:
        """Simplify the prompt for better clarity"""
        simplify_instruction = f"""
Simplify this prompt to make it clearer and more direct:

Original: {prompt}

Simplified prompt:"""
        
        try:
            response = await self.llm.generate_async(simplify_instruction, temperature=0.5)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in simplify_mutation: {e}")
            return prompt
    
    async def specify_mutation(self, prompt: str) -> str:
        """Make the prompt more specific and detailed"""
        specify_instruction = f"""
Make this prompt more specific and detailed with concrete requirements:

Original: {prompt}

More specific prompt:"""
        
        try:
            response = await self.llm.generate_async(specify_instruction, temperature=0.6)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in specify_mutation: {e}")
            return prompt
    
    async def format_mutation(self, prompt: str) -> str:
        """Change the format or structure of the prompt"""
        format_instruction = f"""
Restructure this prompt with better formatting, organization, or presentation:

Original: {prompt}

Reformatted prompt:"""
        
        try:
            response = await self.llm.generate_async(format_instruction, temperature=0.7)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in format_mutation: {e}")
            return prompt
    
    async def rephrase_mutation(self, prompt: str) -> str:
        """Rephrase the prompt with different wording"""
        rephrase_instruction = f"""
Rephrase this prompt using different words while keeping the same meaning:

Original: {prompt}

Rephrased prompt:"""
        
        try:
            response = await self.llm.generate_async(rephrase_instruction, temperature=0.8)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in rephrase_mutation: {e}")
            return prompt
    
    async def restructure_mutation(self, prompt: str) -> str:
        """Restructure the logical flow of the prompt"""
        restructure_instruction = f"""
Restructure this prompt with a different logical flow or organization:

Original: {prompt}

Restructured prompt:"""
        
        try:
            response = await self.llm.generate_async(restructure_instruction, temperature=0.7)
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in restructure_mutation: {e}")
            return prompt
    
    # Helper Methods
    
    async def _generate_instruction_variants(self, prompt: str, count: int) -> List[PromptVariant]:
        """Generate variants using instruction-following approach"""
        instructions = [
            "Make it more conversational and engaging",
            "Add step-by-step structure",
            "Include examples or demonstrations",
            "Make it more concise and focused",
            "Add error handling considerations",
            "Include output format specifications",
            "Make it more creative and open-ended",
            "Add constraints and limitations"
        ]
        
        variants = []
        selected_instructions = random.sample(instructions, min(count, len(instructions)))
        
        for instruction in selected_instructions:
            try:
                variant_content = await self.instruction_following_variant(prompt, instruction)
                if variant_content and variant_content != prompt:
                    variant = PromptVariant(
                        content=variant_content,
                        generation_method="instruction_following",
                        metadata={"instruction": instruction}
                    )
                    variants.append(variant)
            except Exception as e:
                logger.error(f"Error generating instruction variant: {e}")
                continue
        
        return variants
    
    async def _generate_paraphrase_variants(self, prompt: str, count: int) -> List[PromptVariant]:
        """Generate variants using paraphrasing approach"""
        variants = []
        styles = random.sample(self.config.paraphrase_styles, min(count, len(self.config.paraphrase_styles)))
        
        for style in styles:
            try:
                variant_content = await self.paraphrase_prompt(prompt, style)
                if variant_content and variant_content != prompt:
                    variant = PromptVariant(
                        content=variant_content,
                        generation_method="paraphrase",
                        metadata={"style": style}
                    )
                    variants.append(variant)
            except Exception as e:
                logger.error(f"Error generating paraphrase variant: {e}")
                continue
        
        return variants
    
    async def _mutate_prompt(self, prompt: str) -> str:
        """Apply random mutation to prompt"""
        strategy = random.choice(list(self.mutation_functions.keys()))
        mutation_func = self.mutation_functions[strategy]
        
        try:
            return await mutation_func(prompt)
        except Exception as e:
            logger.error(f"Error in mutation strategy {strategy}: {e}")
            return prompt
    
    async def _crossover_prompts(self, parent1: str, parent2: str) -> str:
        """Combine elements from two parent prompts"""
        crossover_instruction = f"""
Combine elements from these two prompts to create a new, effective prompt:

Prompt 1: {parent1}

Prompt 2: {parent2}

Create a hybrid prompt that incorporates the best aspects of both:

Combined prompt:"""
        
        try:
            response = await self.llm.generate_async(
                crossover_instruction, 
                temperature=self.config.generation_temperature
            )
            return self._extract_clean_response(response)
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1  # Fallback to parent1
    
    async def _generate_beam_expansions(self, prompt: str, count: int) -> List[str]:
        """Generate expansions for beam search"""
        expansion_instruction = f"""
Generate {count} different variations of this prompt, each with a slight modification or improvement:

Original prompt: {prompt}

Generate {count} variations:
1."""
        
        try:
            response = await self.llm.generate_async(
                expansion_instruction,
                temperature=self.config.generation_temperature,
                max_tokens=500
            )
            
            # Parse numbered list from response
            expansions = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    expansion = re.sub(r'^\d+\.\s*', '', line).strip()
                    if expansion and expansion != prompt:
                        expansions.append(expansion)
            
            return expansions[:count]
            
        except Exception as e:
            logger.error(f"Error generating beam expansions: {e}")
            return []
    
    def _ensure_diversity(self, variants: List[PromptVariant]) -> List[PromptVariant]:
        """Remove similar variants to ensure diversity"""
        if not variants:
            return variants
        
        diverse_variants = [variants[0]]  # Always include first variant
        
        for variant in variants[1:]:
            is_diverse = True
            for existing in diverse_variants:
                similarity = self._calculate_similarity(variant.content, existing.content)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_variants.append(variant)
        
        return diverse_variants
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts (simple word overlap)"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _score_single_variant(self, variant: str, test_cases: List[Dict]) -> float:
        """Score a single variant against test cases"""
        if not test_cases:
            return 0.5  # Neutral score if no test cases
        
        total_score = 0.0
        valid_cases = 0
        
        for test_case in test_cases:
            try:
                # Use the variant as a prompt with the test input
                test_prompt = f"{variant}\n\nInput: {test_case.get('input', '')}"
                
                response = await self.llm.generate_async(
                    test_prompt,
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=200
                )
                
                # Simple scoring based on response quality
                expected = test_case.get('expected', '')
                score = self._evaluate_response_quality(response, expected)
                total_score += score
                valid_cases += 1
                
            except Exception as e:
                logger.error(f"Error scoring test case: {e}")
                continue
        
        return total_score / valid_cases if valid_cases > 0 else 0.0
    
    def _evaluate_response_quality(self, response: str, expected: str) -> float:
        """Simple response quality evaluation"""
        if not response or not response.strip():
            return 0.0
        
        # Basic quality metrics
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        length_score = min(1.0, len(response.split()) / 50)  # Prefer longer responses up to 50 words
        score += length_score * 0.3
        
        # Contains expected keywords (if provided)
        if expected:
            expected_words = set(expected.lower().split())
            response_words = set(response.lower().split())
            keyword_overlap = len(expected_words.intersection(response_words)) / len(expected_words)
            score += keyword_overlap * 0.7
        else:
            # If no expected output, just check for coherence
            score += 0.7 if len(response.split()) > 5 else 0.3
        
        return min(1.0, score)
    
    async def _score_variants_batch(self, variants: List[PromptVariant], test_cases: List[Dict]):
        """Score a batch of variants efficiently"""
        for variant in variants:
            variant.score = await self._score_single_variant(variant.content, test_cases)
    
    def _update_best_variants(self, variants: List[PromptVariant]):
        """Update the best variants list"""
        all_variants = self.best_variants + variants
        all_variants.sort(key=lambda v: v.score, reverse=True)
        self.best_variants = all_variants[:20]  # Keep top 20
    
    def _extract_clean_response(self, response: str) -> str:
        """Extract clean response from LLM output"""
        if not response:
            return ""
        
        # Remove common prefixes and suffixes
        cleaned = response.strip()
        
        # Remove common prompt-like prefixes
        prefixes_to_remove = [
            "Here's the", "Here is the", "The modified", "The enhanced",
            "Modified prompt:", "Enhanced prompt:", "Paraphrased prompt:",
            "Simplified prompt:", "Reformatted prompt:", "Combined prompt:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Ensure reasonable length
        if len(cleaned) < self.config.min_prompt_length:
            return ""
        if len(cleaned) > self.config.max_prompt_length:
            cleaned = cleaned[:self.config.max_prompt_length]
        
        return cleaned
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process"""
        if not self.generation_history:
            return {}
        
        methods = defaultdict(int)
        scores = []
        
        for variant in self.generation_history:
            methods[variant.generation_method] += 1
            scores.append(variant.score)
        
        return {
            "total_variants": len(self.generation_history),
            "methods_used": dict(methods),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "best_score": max(scores) if scores else 0,
            "best_variants": [
                {"content": v.content[:100] + "...", "score": v.score, "method": v.generation_method}
                for v in self.best_variants[:5]
            ]
        }
    
    async def save_variants_to_db(self, variants: List[PromptVariant], base_prompt_id: str):
        """Save generated variants to database"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for variant in variants:
                cursor.execute("""
                    INSERT INTO prompt_variants 
                    (variant_id, base_prompt_id, content, generation_method, score, metadata, parent_id, generation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    variant.variant_id,
                    base_prompt_id,
                    variant.content,
                    variant.generation_method,
                    variant.score,
                    json.dumps(variant.metadata),
                    variant.parent_id,
                    variant.generation,
                    variant.created_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved {len(variants)} variants to database")
            
        except Exception as e:
            logger.error(f"Error saving variants to database: {e}")


# Example usage and testing functions
async def example_usage():
    """Example of how to use the APE Engine"""
    
    # Mock LLM wrapper for demonstration
    class MockLlamaWrapper:
        async def generate_async(self, prompt: str, **kwargs):
            # In real implementation, this would call the actual LLM
            return f"Generated response for: {prompt[:50]}..."
    
    # Initialize APE Engine
    config = APEConfig(
        generation_temperature=0.7,
        max_variants=20,
        beam_width=3,
        mutation_rate=0.3,
        selection_pressure=0.8
    )
    
    llm_wrapper = MockLlamaWrapper()
    ape_engine = APEEngine(llm_wrapper, config)
    
    # Base prompt to optimize
    base_prompt = "Write a creative story about a robot learning to paint."
    
    # Test cases for evaluation
    test_cases = [
        {
            "input": "robot painting",
            "expected": "creative story robot art"
        },
        {
            "input": "learning art",
            "expected": "learning process artistic development"
        }
    ]
    
    try:
        # Generate variants
        variants = await ape_engine.generate_variants(
            base_prompt=base_prompt,
            num_variants=10,
            test_cases=test_cases
        )
        
        print(f"Generated {len(variants)} variants:")
        for i, variant in enumerate(variants[:5]):  # Show top 5
            print(f"\n{i+1}. Method: {variant.generation_method}")
            print(f"   Score: {variant.score:.3f}")
            print(f"   Content: {variant.content[:100]}...")
        
        # Show generation statistics
        stats = ape_engine.get_generation_stats()
        print(f"\nGeneration Statistics: {stats}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")


# Unit tests for APE Engine
import unittest
from unittest.mock import Mock, AsyncMock


class TestAPEEngine(unittest.TestCase):
    """Unit tests for APE Engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_llm.generate_async = AsyncMock(return_value="Mock response")
        
        self.config = APEConfig(
            generation_temperature=0.7,
            max_variants=10,
            beam_width=3,
            population_size=10
        )
        
        self.ape_engine = APEEngine(self.mock_llm, self.config)
    
    def test_config_initialization(self):
        """Test APE configuration initialization"""
        self.assertEqual(self.config.generation_temperature, 0.7)
        self.assertEqual(self.config.max_variants, 10)
        self.assertEqual(self.config.beam_width, 3)
        self.assertIn("formal", self.config.paraphrase_styles)
        self.assertIn("add_context", self.config.mutation_strategies)
    
    def test_prompt_variant_creation(self):
        """Test PromptVariant dataclass creation"""
        variant = PromptVariant(
            content="Test prompt",
            generation_method="test_method",
            score=0.8
        )
        
        self.assertEqual(variant.content, "Test prompt")
        self.assertEqual(variant.generation_method, "test_method")
        self.assertEqual(variant.score, 0.8)
        self.assertTrue(variant.variant_id)  # Should be auto-generated
        self.assertIsInstance(variant.created_at, datetime)
    
    async def test_mutation_strategies(self):
        """Test individual mutation strategies"""
        test_prompt = "Write a story about cats."
        
        # Test each mutation strategy
        for strategy_name, strategy_func in self.ape_engine.mutation_functions.items():
            result = await strategy_func(test_prompt)
            self.assertIsInstance(result, str)
            # In real implementation, result should be different from input
    
    def test_similarity_calculation(self):
        """Test prompt similarity calculation"""
        prompt1 = "Write a story about dogs and cats"
        prompt2 = "Write a tale about cats and dogs"
        prompt3 = "Generate a poem about flowers"
        
        # Similar prompts should have high similarity
        similarity_high = self.ape_engine._calculate_similarity(prompt1, prompt2)
        self.assertGreater(similarity_high, 0.5)
        
        # Different prompts should have low similarity
        similarity_low = self.ape_engine._calculate_similarity(prompt1, prompt3)
        self.assertLess(similarity_low, 0.5)
    
    def test_diversity_enforcement(self):
        """Test diversity enforcement in variant generation"""
        # Create similar variants
        variants = [
            PromptVariant("Write a story about cats", "test"),
            PromptVariant("Write a tale about cats", "test"),
            PromptVariant("Write a story about dogs", "test"),
            PromptVariant("Generate a poem about flowers", "test")
        ]
        
        diverse_variants = self.ape_engine._ensure_diversity(variants)
        
        # Should have fewer variants due to similarity filtering
        self.assertLessEqual(len(diverse_variants), len(variants))
        self.assertGreater(len(diverse_variants), 0)
    
    def test_response_cleaning(self):
        """Test response cleaning functionality"""
        # Test with prefix removal
        response_with_prefix = "Here's the modified prompt: Write a better story"
        cleaned = self.ape_engine._extract_clean_response(response_with_prefix)
        self.assertEqual(cleaned, "Write a better story")
        
        # Test with empty response
        empty_response = ""
        cleaned_empty = self.ape_engine._extract_clean_response(empty_response)
        self.assertEqual(cleaned_empty, "")
        
        # Test with too short response
        short_response = "Hi"
        cleaned_short = self.ape_engine._extract_clean_response(short_response)
        self.assertEqual(cleaned_short, "")
    
    def test_response_quality_evaluation(self):
        """Test response quality evaluation"""
        # Good response
        good_response = "This is a well-written creative story about robots learning to paint with vivid details."
        expected = "creative story robot art"
        score_good = self.ape_engine._evaluate_response_quality(good_response, expected)
        self.assertGreater(score_good, 0.5)
        
        # Poor response
        poor_response = "No."
        score_poor = self.ape_engine._evaluate_response_quality(poor_response, expected)
        self.assertLess(score_poor, 0.5)
        
        # Empty response
        empty_response = ""
        score_empty = self.ape_engine._evaluate_response_quality(empty_response, expected)
        self.assertEqual(score_empty, 0.0)
    
    async def test_paraphrase_prompt(self):
        """Test prompt paraphrasing functionality"""
        original_prompt = "Write a creative story about robots."
        style = "formal"
        
        # Mock LLM response
        self.mock_llm.generate_async.return_value = "Compose a creative narrative regarding robotic entities."
        
        paraphrased = await self.ape_engine.paraphrase_prompt(original_prompt, style)
        
        # Should call LLM with paraphrase instruction
        self.mock_llm.generate_async.assert_called()
        self.assertIsInstance(paraphrased, str)
    
    async def test_instruction_following_variant(self):
        """Test instruction-following variant generation"""
        original_prompt = "Write a story."
        instruction = "Make it more detailed and specific."
        
        # Mock LLM response
        self.mock_llm.generate_async.return_value = "Write a detailed story with specific characters and setting."
        
        variant = await self.ape_engine.instruction_following_variant(original_prompt, instruction)
        
        # Should call LLM with instruction
        self.mock_llm.generate_async.assert_called()
        self.assertIsInstance(variant, str)
    
    def test_generation_stats(self):
        """Test generation statistics collection"""
        # Add some mock variants to history
        self.ape_engine.generation_history = [
            PromptVariant("Test 1", "evolutionary", score=0.8),
            PromptVariant("Test 2", "beam_search", score=0.7),
            PromptVariant("Test 3", "evolutionary", score=0.9)
        ]
        
        stats = self.ape_engine.get_generation_stats()
        
        self.assertEqual(stats["total_variants"], 3)
        self.assertEqual(stats["methods_used"]["evolutionary"], 2)
        self.assertEqual(stats["methods_used"]["beam_search"], 1)
        self.assertEqual(stats["average_score"], 0.8)
        self.assertEqual(stats["best_score"], 0.9)


# Integration test example
async def integration_test():
    """Integration test with actual LLM wrapper"""
    try:
        from backend.llm.llama_wrapper import LlamaWrapper
        
        # Initialize with real LLM wrapper
        llm_wrapper = LlamaWrapper(model_path="path/to/model")
        config = APEConfig(max_variants=5, beam_width=2)
        ape_engine = APEEngine(llm_wrapper, config)
        
        # Test with simple prompt
        base_prompt = "Explain quantum computing in simple terms."
        variants = await ape_engine.generate_variants(base_prompt, num_variants=5)
        
        print(f"Integration test: Generated {len(variants)} variants")
        for variant in variants:
            print(f"- {variant.generation_method}: {variant.content[:50]}...")
        
    except ImportError:
        print("Integration test skipped: LlamaWrapper not available")
    except Exception as e:
        print(f"Integration test error: {e}")


# Performance testing
async def performance_test():
    """Performance test for APE Engine"""
    import time
    
    # Mock LLM with faster responses
    class FastMockLLM:
        async def generate_async(self, prompt: str, **kwargs):
            await asyncio.sleep(0.01)  # Simulate fast LLM
            return f"Fast response for: {prompt[:30]}..."
    
    llm_wrapper = FastMockLLM()
    config = APEConfig(max_variants=20, beam_width=5)
    ape_engine = APEEngine(llm_wrapper, config)
    
    base_prompt = "Create a marketing campaign for a new product."
    
    start_time = time.time()
    variants = await ape_engine.generate_variants(base_prompt, num_variants=20)
    end_time = time.time()
    
    print(f"Performance test: Generated {len(variants)} variants in {end_time - start_time:.2f} seconds")
    print(f"Average time per variant: {(end_time - start_time) / len(variants):.3f} seconds")


# Main execution
if __name__ == "__main__":
    # Run example usage
    print("=== APE Engine Example Usage ===")
    asyncio.run(example_usage())
    
    # Run unit tests
    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n=== Integration Test ===")
    asyncio.run(integration_test())
    
    # Run performance test
    print("\n=== Performance Test ===")
    asyncio.run(performance_test())
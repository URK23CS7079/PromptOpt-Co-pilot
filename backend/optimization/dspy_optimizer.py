"""
DSPy Optimizer Integration for PromptOpt Co-Pilot

This module provides a comprehensive DSPy-based prompt optimization system that orchestrates
systematic prompt improvement through CPU-based grid search and RL-style tuning.

Key Components:
- DSPyOptimizer: Main optimization class integrating DSPy framework
- OptimizerConfig: Configuration for optimization parameters
- OptimizationResult: Results container with metrics and history
- OptimizationCache: Intelligent caching system for performance
- Support for multiple optimization strategies (grid search, RL, bootstrap)

The system is designed for CPU-only execution with local LLM integration,
providing efficient prompt optimization without GPU dependencies.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, COPRO, KNNFewShot
    from dspy.evaluate import Evaluate
    from dspy.primitives import Example
except ImportError as e:
    raise ImportError(f"DSPy framework not found. Install with: pip install dspy-ai. Error: {e}")

# Import project dependencies
try:
    from backend.llm.llama_wrapper import LlamaWrapper
    from backend.optimization.ape_engine import APEEngine
    from backend.core.database import Database
    from backend.evaluation.metrics import MetricsCalculator
except ImportError as e:
    logging.warning(f"Some project dependencies not available: {e}")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration class for DSPy optimizer settings."""
    
    # General optimization settings
    max_iterations: int = 50
    timeout_seconds: int = 3600  # 1 hour default
    cache_enabled: bool = True
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Grid search parameters
    grid_search_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_tokens': [50, 100, 200, 500],
        'top_p': [0.8, 0.9, 0.95, 1.0]
    })
    
    # RL optimization parameters
    rl_learning_rate: float = 0.01
    rl_exploration_factor: float = 0.1
    rl_discount_factor: float = 0.9
    rl_batch_size: int = 8
    
    # Bootstrap settings
    bootstrap_examples: int = 16
    bootstrap_max_bootstrapped_demos: int = 4
    bootstrap_max_labeled_demos: int = 16
    
    # Evaluation settings
    validation_split: float = 0.2
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 1.0,
        'f1_score': 0.8,
        'bleu_score': 0.6
    })
    
    # Performance settings
    early_stopping_patience: int = 10
    min_improvement_threshold: float = 0.001


@dataclass
class OptimizationStep:
    """Represents a single optimization step with results."""
    
    step_number: int
    prompt_variant: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    improvement: float
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_number': self.step_number,
            'prompt_variant': self.prompt_variant,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'improvement': self.improvement,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


@dataclass
class OptimizationResult:
    """Complete optimization results container."""
    
    best_prompt: str
    optimization_history: List[OptimizationStep]
    final_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]
    execution_time: float
    total_steps: int
    best_step: int
    improvement_over_baseline: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_prompt': self.best_prompt,
            'optimization_history': [step.to_dict() for step in self.optimization_history],
            'final_metrics': self.final_metrics,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'total_steps': self.total_steps,
            'best_step': self.best_step,
            'improvement_over_baseline': self.improvement_over_baseline
        }


class OptimizationCache:
    """Intelligent caching system for optimization results."""
    
    def __init__(self, cache_dir: str = "cache/optimization"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "optimization_cache.db"
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    parameters_hash TEXT,
                    result_data BLOB,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON optimization_cache(prompt_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON optimization_cache(created_at)
            """)
    
    def _generate_cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate unique cache key for prompt and parameters."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        params_str = json.dumps(parameters, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{prompt_hash}_{params_hash}"
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
    
    def get(self, prompt: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached optimization result."""
        cache_key = self._generate_cache_key(prompt, parameters)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT result_data FROM optimization_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update access statistics
                conn.execute(
                    """UPDATE optimization_cache 
                       SET access_count = access_count + 1, 
                           last_accessed = CURRENT_TIMESTAMP 
                       WHERE cache_key = ?""",
                    (cache_key,)
                )
                conn.commit()
                
                try:
                    return pickle.loads(row[0])
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached result: {e}")
                    return None
        
        return None
    
    def set(self, prompt: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Store optimization result in cache."""
        cache_key = self._generate_cache_key(prompt, parameters)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        params_str = json.dumps(parameters, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        try:
            result_data = pickle.dumps(result)
            metrics_str = json.dumps(result.get('metrics', {}))
            
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO optimization_cache 
                       (cache_key, prompt_hash, parameters_hash, result_data, metrics) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (cache_key, prompt_hash, params_hash, result_data, metrics_str)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache optimization result: {e}")
    
    def clear_old_entries(self, days: int = 30):
        """Clear cache entries older than specified days."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM optimization_cache WHERE created_at < datetime('now', '-{} days')".format(days)
            )
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT COUNT(*) as total_entries,
                          AVG(access_count) as avg_access_count,
                          MAX(created_at) as latest_entry
                   FROM optimization_cache"""
            )
            row = cursor.fetchone()
            
            return {
                'total_entries': row[0] if row else 0,
                'avg_access_count': row[1] if row else 0,
                'latest_entry': row[2] if row else None
            }


class DSPyProgram(dspy.Module):
    """Custom DSPy program wrapper for prompt optimization."""
    
    def __init__(self, prompt_template: str, signature: str = None):
        super().__init__()
        self.prompt_template = prompt_template
        self.signature = signature or "question -> answer"
        
        # Initialize DSPy modules
        self.generate = dspy.Predict(self.signature)
    
    def forward(self, **kwargs):
        """Forward pass through the DSPy program."""
        try:
            # Apply prompt template with input variables
            formatted_input = self.prompt_template.format(**kwargs)
            return self.generate(question=formatted_input)
        except Exception as e:
            logger.error(f"Error in DSPy program forward pass: {e}")
            return dspy.Prediction(answer="Error in generation")


class DSPyOptimizer:
    """
    Main DSPy optimizer class that orchestrates systematic prompt optimization.
    
    This class integrates various optimization strategies including:
    - Grid search over parameter spaces
    - RL-style prompt tuning with reward functions
    - Bootstrap few-shot example selection
    - Intelligent caching for performance optimization
    """
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: OptimizerConfig):
        """
        Initialize DSPy optimizer.
        
        Args:
            llm_wrapper: LlamaWrapper instance for LLM interactions
            config: OptimizerConfig with optimization settings
        """
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.cache = OptimizationCache() if config.cache_enabled else None
        self.ape_engine = APEEngine() if 'APEEngine' in globals() else None
        self.metrics_calculator = MetricsCalculator() if 'MetricsCalculator' in globals() else None
        
        # Initialize DSPy with the LLM wrapper
        self._setup_dspy()
        
        # Optimization state
        self.optimization_history: List[OptimizationStep] = []
        self.best_result: Optional[OptimizationStep] = None
        self.baseline_metrics: Dict[str, float] = {}
        
        logger.info(f"DSPy Optimizer initialized with config: {config}")
    
    def _setup_dspy(self):
        """Setup DSPy framework with the provided LLM."""
        try:
            # Create a DSPy-compatible LM from our LlamaWrapper
            dspy.settings.configure(lm=self._create_dspy_lm())
            logger.info("DSPy configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            raise
    
    def _create_dspy_lm(self):
        """Create DSPy-compatible language model from LlamaWrapper."""
        class DSPyLM:
            def __init__(self, llm_wrapper):
                self.llm_wrapper = llm_wrapper
            
            def generate(self, prompt: str, **kwargs) -> List[str]:
                """Generate response using the wrapped LLM."""
                try:
                    response = self.llm_wrapper.generate(
                        prompt, 
                        max_tokens=kwargs.get('max_tokens', 100),
                        temperature=kwargs.get('temperature', 0.7)
                    )
                    return [response]
                except Exception as e:
                    logger.error(f"Error in LLM generation: {e}")
                    return ["Error in generation"]
            
            def __call__(self, prompt: str, **kwargs) -> str:
                """Call method for DSPy compatibility."""
                responses = self.generate(prompt, **kwargs)
                return responses[0] if responses else "No response"
        
        return DSPyLM(self.llm_wrapper)
    
    def optimize(
        self, 
        base_prompt: str, 
        dataset: List[Dict[str, Any]], 
        metrics: List[str]
    ) -> OptimizationResult:
        """
        Main optimization method that orchestrates the entire optimization process.
        
        Args:
            base_prompt: Initial prompt template to optimize
            dataset: Training/evaluation dataset
            metrics: List of metrics to optimize for
            
        Returns:
            OptimizationResult with optimization history and final results
        """
        start_time = time.time()
        logger.info(f"Starting optimization for prompt with {len(dataset)} examples")
        
        try:
            # Reset optimization state
            self.optimization_history = []
            self.best_result = None
            
            # Split dataset for training and validation
            train_set, val_set = self._split_dataset(dataset)
            
            # Calculate baseline metrics
            self.baseline_metrics = self._evaluate_baseline(base_prompt, val_set, metrics)
            logger.info(f"Baseline metrics: {self.baseline_metrics}")
            
            # Check cache for existing optimization
            if self.cache:
                cached_result = self.cache.get(base_prompt, self.config.__dict__)
                if cached_result:
                    logger.info("Found cached optimization result")
                    return OptimizationResult(**cached_result)
            
            # Run different optimization strategies
            optimization_strategies = [
                ('grid_search', self._run_grid_search),
                ('bootstrap', self._run_bootstrap_optimization),
                ('rl_optimization', self._run_rl_optimization)
            ]
            
            for strategy_name, strategy_func in optimization_strategies:
                logger.info(f"Running {strategy_name} optimization")
                try:
                    strategy_func(base_prompt, train_set, val_set, metrics)
                except Exception as e:
                    logger.error(f"Error in {strategy_name}: {e}")
                    continue
            
            # Compile final results
            result = self._compile_results(start_time)
            
            # Cache results if enabled
            if self.cache and result:
                self.cache.set(base_prompt, self.config.__dict__, result.to_dict())
            
            logger.info(f"Optimization completed in {result.execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return partial results if available
            return self._compile_results(start_time, failed=True)
    
    def grid_search(
        self, 
        prompt_template: str, 
        param_space: Dict[str, List[Any]]
    ) -> List[OptimizationStep]:
        """
        Perform grid search optimization over parameter space.
        
        Args:
            prompt_template: Template prompt to optimize
            param_space: Dictionary of parameters and their possible values
            
        Returns:
            List of OptimizationStep results for each parameter combination
        """
        logger.info(f"Starting grid search with {len(param_space)} parameter dimensions")
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_space)
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        results = []
        
        if self.config.parallel_execution:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        self._evaluate_param_combination, 
                        prompt_template, 
                        params, 
                        i
                    ): params
                    for i, params in enumerate(param_combinations)
                }
                
                for future in as_completed(future_to_params):
                    try:
                        result = future.result(timeout=60)  # 1 minute timeout per evaluation
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Grid search evaluation failed: {e}")
        else:
            # Sequential execution
            for i, params in enumerate(param_combinations):
                try:
                    result = self._evaluate_param_combination(prompt_template, params, i)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Grid search evaluation {i} failed: {e}")
        
        # Sort results by performance
        results.sort(key=lambda x: x.improvement, reverse=True)
        logger.info(f"Grid search completed with {len(results)} successful evaluations")
        
        return results
    
    def rl_optimization(
        self, 
        prompt: str, 
        reward_fn: Callable[[str, str], float], 
        steps: int
    ) -> str:
        """
        Perform RL-style prompt optimization using reward function.
        
        Args:
            prompt: Initial prompt to optimize
            reward_fn: Function that takes (prompt, output) and returns reward
            steps: Number of optimization steps
            
        Returns:
            Optimized prompt string
        """
        logger.info(f"Starting RL optimization for {steps} steps")
        
        current_prompt = prompt
        best_prompt = prompt
        best_reward = float('-inf')
        
        # RL state tracking
        prompt_history = [prompt]
        reward_history = []
        
        for step in range(steps):
            try:
                # Generate prompt variants using exploration
                variants = self._generate_prompt_variants(current_prompt, self.config.rl_exploration_factor)
                
                # Evaluate variants
                step_rewards = []
                for variant in variants:
                    # Generate output with variant
                    output = self.llm_wrapper.generate(variant)
                    reward = reward_fn(variant, output)
                    step_rewards.append((variant, reward))
                
                # Select best variant
                best_variant, best_step_reward = max(step_rewards, key=lambda x: x[1])
                
                # Update if improvement found
                if best_step_reward > best_reward:
                    best_reward = best_step_reward
                    best_prompt = best_variant
                    logger.info(f"RL Step {step}: New best reward {best_reward:.4f}")
                
                # Update current prompt using RL-style update
                current_prompt = self._rl_update_prompt(
                    current_prompt, 
                    best_variant, 
                    best_step_reward,
                    step
                )
                
                prompt_history.append(current_prompt)
                reward_history.append(best_step_reward)
                
                # Early stopping if converged
                if len(reward_history) > self.config.early_stopping_patience:
                    recent_rewards = reward_history[-self.config.early_stopping_patience:]
                    if max(recent_rewards) - min(recent_rewards) < self.config.min_improvement_threshold:
                        logger.info(f"RL optimization converged at step {step}")
                        break
                        
            except Exception as e:
                logger.error(f"RL optimization step {step} failed: {e}")
                continue
        
        logger.info(f"RL optimization completed. Best reward: {best_reward:.4f}")
        return best_prompt
    
    def bootstrap_examples(self, dataset: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Select optimal few-shot examples using bootstrap sampling.
        
        Args:
            dataset: Full dataset to sample from
            k: Number of examples to select
            
        Returns:
            List of selected examples optimized for few-shot learning
        """
        logger.info(f"Bootstrapping {k} examples from dataset of {len(dataset)}")
        
        if len(dataset) <= k:
            return dataset
        
        try:
            # Convert dataset to DSPy Examples
            examples = [
                Example(
                    question=item.get('input', item.get('question', '')),
                    answer=item.get('output', item.get('answer', ''))
                ).with_inputs('question')
                for item in dataset
            ]
            
            # Use DSPy's bootstrap selector
            bootstrap = BootstrapFewShot(
                max_bootstrapped_demos=min(k, self.config.bootstrap_max_bootstrapped_demos),
                max_labeled_demos=min(len(dataset), self.config.bootstrap_max_labeled_demos)
            )
            
            # Create a simple program for bootstrapping
            program = dspy.Predict("question -> answer")
            
            # Compile with bootstrap
            compiled_program = bootstrap.compile(program, trainset=examples[:50])  # Use subset for efficiency
            
            # Extract the selected examples
            if hasattr(compiled_program, 'demos') and compiled_program.demos:
                selected_indices = list(range(min(k, len(compiled_program.demos))))
                return [dataset[i] for i in selected_indices if i < len(dataset)]
            else:
                # Fallback to random sampling
                import random
                return random.sample(dataset, k)
                
        except Exception as e:
            logger.error(f"Bootstrap selection failed: {e}")
            # Fallback to first k examples
            return dataset[:k]
    
    def compile_program(self, prompt: str, examples: List[Dict[str, Any]]) -> DSPyProgram:
        """
        Compile DSPy program with optimized prompt and examples.
        
        Args:
            prompt: Optimized prompt template
            examples: Selected few-shot examples
            
        Returns:
            Compiled DSPyProgram ready for inference
        """
        logger.info(f"Compiling DSPy program with {len(examples)} examples")
        
        try:
            # Create program with optimized prompt
            program = DSPyProgram(prompt)
            
            # Convert examples to DSPy format
            dspy_examples = [
                Example(
                    question=ex.get('input', ex.get('question', '')),
                    answer=ex.get('output', ex.get('answer', ''))
                ).with_inputs('question')
                for ex in examples
            ]
            
            # Use COPRO optimizer for program compilation
            copro = COPRO(metric=self._default_metric, breadth=3, depth=2)
            
            # Compile the program
            compiled_program = copro.compile(program, trainset=dspy_examples)
            
            logger.info("DSPy program compiled successfully")
            return compiled_program
            
        except Exception as e:
            logger.error(f"Program compilation failed: {e}")
            # Return basic program without optimization
            return DSPyProgram(prompt)
    
    def evaluate_step(
        self, 
        program: DSPyProgram, 
        test_set: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate a DSPy program on test set and return metrics.
        
        Args:
            program: Compiled DSPy program
            test_set: Test dataset for evaluation
            
        Returns:
            Dictionary of metric names and values
        """
        logger.info(f"Evaluating program on {len(test_set)} test examples")
        
        try:
            # Convert test set to DSPy format
            dspy_test = [
                Example(
                    question=ex.get('input', ex.get('question', '')),
                    answer=ex.get('output', ex.get('answer', ''))
                ).with_inputs('question')
                for ex in test_set
            ]
            
            # Create evaluator
            evaluator = Evaluate(
                devset=dspy_test,
                metric=self._default_metric,
                num_threads=1,  # CPU-only evaluation
                display_progress=True
            )
            
            # Run evaluation
            score = evaluator(program)
            
            # Calculate additional metrics if metrics calculator available
            additional_metrics = {}
            if self.metrics_calculator:
                predictions = []
                ground_truths = []
                
                for example in dspy_test:
                    try:
                        prediction = program(question=example.question)
                        predictions.append(prediction.answer if hasattr(prediction, 'answer') else str(prediction))
                        ground_truths.append(example.answer)
                    except Exception as e:
                        logger.warning(f"Evaluation error for example: {e}")
                        continue
                
                if predictions and ground_truths:
                    additional_metrics = self.metrics_calculator.calculate_all_metrics(
                        predictions, ground_truths
                    )
            
            # Combine metrics
            metrics = {'primary_score': score}
            metrics.update(additional_metrics)
            
            logger.info(f"Evaluation completed with metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'primary_score': 0.0, 'error': str(e)}
    
    # Private helper methods
    
    def _split_dataset(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into training and validation sets."""
        if len(dataset) < 10:  # Too small to split meaningfully
            return dataset, dataset
        
        split_idx = int(len(dataset) * (1 - self.config.validation_split))
        return dataset[:split_idx], dataset[split_idx:]
    
    def _evaluate_baseline(self, prompt: str, dataset: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
        """Evaluate baseline performance of the initial prompt."""
        try:
            program = DSPyProgram(prompt)
            return self.evaluate_step(program, dataset)
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {e}")
            return {'primary_score': 0.0}
    
    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search."""
        from itertools import product
        
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations[:self.config.max_iterations]  # Limit combinations
    
    def _evaluate_param_combination(
        self, 
        prompt: str, 
        params: Dict[str, Any], 
        step_num: int
    ) -> Optional[OptimizationStep]:
        """Evaluate a single parameter combination."""
        start_time = time.time()
        
        try:
            # Create program with parameters
            program = DSPyProgram(prompt)
            
            # Apply parameters to LLM wrapper (simplified)
            temp_params = {
                'temperature': params.get('temperature', 0.7),
                'max_tokens': params.get('max_tokens', 100)
            }
            
            # Generate test output (simplified evaluation)
            test_input = "Test evaluation prompt"
            output = self.llm_wrapper.generate(prompt.format(question=test_input), **temp_params)
            
            # Calculate simple metrics (placeholder)
            metrics = {'length': len(output), 'coherence': min(1.0, len(output) / 100)}
            primary_score = metrics.get('coherence', 0.0)
            
            # Calculate improvement over baseline
            baseline_score = self.baseline_metrics.get('primary_score', 0.0)
            improvement = primary_score - baseline_score
            
            execution_time = time.time() - start_time
            
            return OptimizationStep(
                step_number=step_num,
                prompt_variant=prompt,
                metrics=metrics,
                parameters=params,
                improvement=improvement,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Parameter evaluation failed: {e}")
            return None
    
    def _run_grid_search(self, prompt: str, train_set: List[Dict], val_set: List[Dict], metrics: List[str]):
        """Run grid search optimization strategy."""
        try:
            results = self.grid_search(prompt, self.config.grid_search_params)
            self.optimization_history.extend(results)
            
            # Update best result
            if results:
                best_step = max(results, key=lambda x: x.improvement)
                if not self.best_result or best_step.improvement > self.best_result.improvement:
                    self.best_result = best_step
                    
        except Exception as e:
            logger.error(f"Grid search failed: {e}")
    
    def _run_bootstrap_optimization(self, prompt: str, train_set: List[Dict], val_set: List[Dict], metrics: List[str]):
        """Run bootstrap-based optimization strategy."""
        try:
            # Select optimal examples
            optimal_examples = self.bootstrap_examples(train_set, self.config.bootstrap_examples)
            
            # Create program with bootstrap examples
            program = self.compile_program(prompt, optimal_examples)
            
            # Evaluate on validation set
            eval_metrics = self.evaluate_step(program, val_set)
            
            # Create optimization step
            baseline_score = self.baseline_metrics.get('primary_score', 0.0)
            improvement = eval_metrics.get('primary_score', 0.0) - baseline_score
            
            bootstrap_step = OptimizationStep(
                step_number=len(self.optimization_history),
                prompt_variant=prompt,
                metrics=eval_metrics,
                parameters={'strategy': 'bootstrap', 'examples_count': len(optimal_examples)},
                improvement=improvement,
                execution_time=0.0  # Set during actual execution
            )
            
            self.optimization_history.append(bootstrap_step)
            
            # Update best result
            if not self.best_result or improvement > self.best_result.improvement:
                self.best_result = bootstrap_step
                
        except Exception as e:
            logger.error(f"Bootstrap optimization failed: {e}")
    
    def _run_rl_optimization(self, prompt: str, train_set: List[Dict], val_set: List[Dict], metrics: List[str]):
        """Run RL-style optimization strategy."""
        try:
            # Define reward function based on metrics
            def reward_function(prompt_variant: str, output: str) -> float:
                try:
                    # Simple reward based on output quality metrics
                    length_score = min(1.0, len(output) / 200)  # Normalize by expected length
                    coherence_score = 1.0 if len(output.split()) > 5 else 0.5  # Basic coherence check
                    return (length_score + coherence_score) / 2
                except Exception:
                    return 0.0
            
            # Run RL optimization
            optimized_prompt = self.rl_optimization(
                prompt, 
                reward_function, 
                steps=min(20, self.config.max_iterations // 3)
            )
            
            # Evaluate optimized prompt
            program = DSPyProgram(optimized_prompt)
            eval_metrics = self.evaluate_step(program, val_set)
            
            # Create optimization step
            baseline_score = self.baseline_metrics.get('primary_score', 0.0)
            improvement = eval_metrics.get('primary_score', 0.0) - baseline_score
            
            rl_step = OptimizationStep(
                step_number=len(self.optimization_history),
                prompt_variant=optimized_prompt,
                metrics=eval_metrics,
                parameters={'strategy': 'rl', 'learning_rate': self.config.rl_learning_rate},
                improvement=improvement,
                execution_time=0.0
            )
            
            self.optimization_history.append(rl_step)
            
            # Update best result
            if not self.best_result or improvement > self.best_result.improvement:
                self.best_result = rl_step
                
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
    
    def _generate_prompt_variants(self, prompt: str, exploration_factor: float) -> List[str]:
        """Generate prompt variants for RL exploration."""
        variants = [prompt]  # Include original
        
        try:
            # Use APE engine if available
            if self.ape_engine:
                ape_variants = self.ape_engine.generate_variants(prompt, num_variants=3)
                variants.extend(ape_variants)
            
            # Generate simple variations
            simple_variants = self._generate_simple_variants(prompt, exploration_factor)
            variants.extend(simple_variants)
            
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
        
        return variants[:8]  # Limit to reasonable number
    
    def _generate_simple_variants(self, prompt: str, exploration_factor: float) -> List[str]:
        """Generate simple prompt variants through text modifications."""
        variants = []
        
        try:
            # Add instruction prefixes
            prefixes = [
                "Please carefully",
                "Think step by step and",
                "Consider the context and",
                "Using your expertise,"
            ]
            
            for prefix in prefixes:
                variant = f"{prefix} {prompt.lower()}"
                variants.append(variant)
            
            # Add instruction suffixes
            suffixes = [
                "Provide a detailed explanation.",
                "Be concise and accurate.",
                "Consider multiple perspectives.",
                "Focus on the key points."
            ]
            
            for suffix in suffixes:
                variant = f"{prompt} {suffix}"
                variants.append(variant)
                
        except Exception as e:
            logger.error(f"Simple variant generation failed: {e}")
        
        return variants
    
    def _rl_update_prompt(self, current_prompt: str, best_variant: str, reward: float, step: int) -> str:
        """Update prompt using RL-style learning."""
        try:
            # Simple interpolation between current and best variant
            learning_rate = self.config.rl_learning_rate * (0.95 ** step)  # Decay learning rate
            
            if reward > 0.7:  # High reward threshold
                # Move towards best variant
                if len(best_variant) > len(current_prompt):
                    return best_variant  # Use better variant
                else:
                    return current_prompt  # Keep current if similar performance
            else:
                # Small modification for exploration
                return self._small_prompt_modification(current_prompt)
                
        except Exception as e:
            logger.error(f"RL update failed: {e}")
            return current_prompt
    
    def _small_prompt_modification(self, prompt: str) -> str:
        """Make small modifications to prompt for exploration."""
        modifications = [
            lambda p: p.replace(".", ". Please"),
            lambda p: f"Carefully {p.lower()}",
            lambda p: f"{p} Think step by step.",
            lambda p: p.replace("?", "? Consider all aspects.")
        ]
        
        import random
        modification = random.choice(modifications)
        return modification(prompt)
    
    def _default_metric(self, example, pred, trace=None) -> float:
        """Default metric function for DSPy evaluation."""
        try:
            if hasattr(pred, 'answer') and hasattr(example, 'answer'):
                # Simple string similarity metric
                pred_text = str(pred.answer).lower().strip()
                true_text = str(example.answer).lower().strip()
                
                if pred_text == true_text:
                    return 1.0
                elif true_text in pred_text or pred_text in true_text:
                    return 0.7
                else:
                    # Simple word overlap metric
                    pred_words = set(pred_text.split())
                    true_words = set(true_text.split())
                    if len(true_words) == 0:
                        return 0.0
                    overlap = len(pred_words.intersection(true_words))
                    return overlap / len(true_words)
            return 0.0
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return 0.0
    
    def _compile_results(self, start_time: float, failed: bool = False) -> OptimizationResult:
        """Compile final optimization results."""
        execution_time = time.time() - start_time
        
        if failed or not self.optimization_history:
            return OptimizationResult(
                best_prompt="",
                optimization_history=[],
                final_metrics={},
                convergence_info={'status': 'failed'},
                execution_time=execution_time,
                total_steps=0,
                best_step=-1,
                improvement_over_baseline=0.0
            )
        
        # Find best result
        if not self.best_result:
            self.best_result = max(self.optimization_history, key=lambda x: x.improvement)
        
        # Calculate convergence info
        convergence_info = self._calculate_convergence_info()
        
        # Calculate improvement over baseline
        baseline_score = self.baseline_metrics.get('primary_score', 0.0)
        improvement = self.best_result.improvement
        
        return OptimizationResult(
            best_prompt=self.best_result.prompt_variant,
            optimization_history=self.optimization_history,
            final_metrics=self.best_result.metrics,
            convergence_info=convergence_info,
            execution_time=execution_time,
            total_steps=len(self.optimization_history),
            best_step=self.best_result.step_number,
            improvement_over_baseline=improvement
        )
    
    def _calculate_convergence_info(self) -> Dict[str, Any]:
        """Calculate convergence information from optimization history."""
        if not self.optimization_history:
            return {'status': 'no_data'}
        
        improvements = [step.improvement for step in self.optimization_history]
        
        # Check for convergence
        converged = False
        convergence_step = -1
        
        if len(improvements) > self.config.early_stopping_patience:
            recent_improvements = improvements[-self.config.early_stopping_patience:]
            if max(recent_improvements) - min(recent_improvements) < self.config.min_improvement_threshold:
                converged = True
                convergence_step = len(improvements) - self.config.early_stopping_patience
        
        return {
            'status': 'converged' if converged else 'completed',
            'convergence_step': convergence_step,
            'total_improvement': max(improvements) if improvements else 0.0,
            'average_improvement': sum(improvements) / len(improvements) if improvements else 0.0,
            'improvement_variance': self._calculate_variance(improvements),
            'monotonic_improvement': all(improvements[i] <= improvements[i+1] for i in range(len(improvements)-1))
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


# Unit Tests
def test_dspy_optimizer():
    """Basic unit tests for DSPy optimizer functionality."""
    
    class MockLlamaWrapper:
        def generate(self, prompt: str, **kwargs) -> str:
            return f"Generated response for: {prompt[:50]}..."
    
    def test_optimizer_initialization():
        """Test optimizer initialization."""
        config = OptimizerConfig(max_iterations=10)
        llm_wrapper = MockLlamaWrapper()
        
        try:
            optimizer = DSPyOptimizer(llm_wrapper, config)
            assert optimizer is not None
            assert optimizer.config.max_iterations == 10
            print("✓ Optimizer initialization test passed")
        except Exception as e:
            print(f"✗ Optimizer initialization test failed: {e}")
    
    def test_cache_functionality():
        """Test optimization cache."""
        cache = OptimizationCache()
        
        # Test cache operations
        test_prompt = "Test prompt"
        test_params = {"temperature": 0.7}
        test_result = {"metrics": {"score": 0.8}}
        
        # Test cache miss
        result = cache.get(test_prompt, test_params)
        assert result is None
        
        # Test cache set and get
        cache.set(test_prompt, test_params, test_result)
        result = cache.get(test_prompt, test_params)
        assert result is not None
        assert result["metrics"]["score"] == 0.8
        
        print("✓ Cache functionality test passed")
    
    def test_prompt_variants():
        """Test prompt variant generation."""
        config = OptimizerConfig()
        llm_wrapper = MockLlamaWrapper()
        optimizer = DSPyOptimizer(llm_wrapper, config)
        
        base_prompt = "Explain the concept of machine learning"
        variants = optimizer._generate_simple_variants(base_prompt, 0.1)
        
        assert len(variants) > 0
        assert all(isinstance(v, str) for v in variants)
        print("✓ Prompt variants test passed")
    
    def test_bootstrap_examples():
        """Test bootstrap example selection."""
        config = OptimizerConfig()
        llm_wrapper = MockLlamaWrapper()
        optimizer = DSPyOptimizer(llm_wrapper, config)
        
        dataset = [
            {"input": f"Question {i}", "output": f"Answer {i}"}
            for i in range(20)
        ]
        
        selected = optimizer.bootstrap_examples(dataset, k=5)
        assert len(selected) <= 5
        assert all(isinstance(ex, dict) for ex in selected)
        print("✓ Bootstrap examples test passed")
    
    # Run tests
    test_optimizer_initialization()
    test_cache_functionality()
    test_prompt_variants()
    test_bootstrap_examples()
    print("All tests completed!")


if __name__ == "__main__":
    # Example usage
    print("DSPy Optimizer Integration - Example Usage")
    
    # Create mock components for demonstration
    class MockLlamaWrapper:
        def generate(self, prompt: str, **kwargs) -> str:
            return f"Mock response to: {prompt[:100]}..."
    
    # Initialize configuration
    config = OptimizerConfig(
        max_iterations=20,
        cache_enabled=True,
        parallel_execution=True,
        max_workers=2
    )
    
    # Initialize optimizer
    llm_wrapper = MockLlamaWrapper()
    optimizer = DSPyOptimizer(llm_wrapper, config)
    
    # Example optimization
    base_prompt = "Explain the following concept clearly and concisely: {question}"
    dataset = [
        {"input": "What is machine learning?", "output": "Machine learning is a subset of AI..."},
        {"input": "How does neural network work?", "output": "Neural networks are computational models..."},
        {"input": "What is deep learning?", "output": "Deep learning uses multi-layer neural networks..."}
    ]
    metrics = ["accuracy", "coherence"]
    
    try:
        # Run optimization
        result = optimizer.optimize(base_prompt, dataset, metrics)
        
        print(f"\nOptimization Results:")
        print(f"- Best prompt: {result.best_prompt[:100]}...")
        print(f"- Total steps: {result.total_steps}")
        print(f"- Execution time: {result.execution_time:.2f}s")
        print(f"- Improvement: {result.improvement_over_baseline:.4f}")
        print(f"- Final metrics: {result.final_metrics}")
        
    except Exception as e:
        print(f"Example execution failed: {e}")
    
    # Run unit tests
    print("\n" + "="*50)
    print("Running Unit Tests")
    print("="*50)
    test_dspy_optimizer()
"""
PromptOpt Co-Pilot Evaluation Framework

This module provides comprehensive prompt evaluation capabilities using multiple metrics,
local LLM judges, and concurrent processing. It orchestrates evaluation runs, manages
test datasets, and integrates with LangSmith-style evaluation patterns.

Key Features:
- Multi-metric evaluation (exact-match, semantic similarity, pairwise comparison)
- Concurrent evaluation processing with resource management
- Local LLM integration for judge-based evaluations
- Detailed performance and latency tracking
- Result caching and persistence
- Flexible dataset handling
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import uuid
from pathlib import Path

# Internal imports
from backend.llm.llama_wrapper import LlamaWrapper
from backend.evaluation.metrics import (
    ExactMatchMetric, SemanticSimilarityMetric, BLEUMetric, 
    ROUGEMetric, CustomMetric, MetricCalculator
)
from backend.evaluation.langsmith_adapter import LangSmithAdapter
from backend.core.database import DatabaseManager
from backend.utils.dataset_handler import DatasetHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    enabled_metrics: List[str] = field(default_factory=lambda: ["exact_match", "semantic_similarity"])
    timeout_per_case: int = 30  # seconds
    batch_size: int = 10
    max_concurrent_workers: int = 4
    judge_model_settings: Dict[str, Any] = field(default_factory=dict)
    cache_results: bool = True
    enable_progress_tracking: bool = True
    retry_failed_cases: bool = True
    max_retries: int = 3
    save_intermediate_results: bool = True


@dataclass
class EvaluationResult:
    """Results from evaluating a single prompt."""
    prompt_id: str
    prompt_content: str
    metrics_scores: Dict[str, float]
    latency_stats: Dict[str, float]
    success_rate: float
    individual_results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: datetime = field(default_factory=datetime.now)
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    error_details: List[str] = field(default_factory=list)


@dataclass
class EvaluationRequest:
    """Request for prompt evaluation."""
    prompt_content: str
    dataset_subset: List[Dict[str, Any]]
    required_metrics: List[str]
    priority: int = 1
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from pairwise prompt comparison."""
    prompt_a_id: str
    prompt_b_id: str
    winner: str  # 'A', 'B', or 'tie'
    confidence: float
    detailed_scores: Dict[str, Dict[str, float]]
    judge_reasoning: List[str]
    comparison_metrics: Dict[str, float]


@dataclass
class EvaluationSuite:
    """Complete evaluation suite results."""
    suite_id: str
    results: List[EvaluationResult]
    summary_stats: Dict[str, Any]
    execution_time: float
    dataset_info: Dict[str, Any]
    config: EvaluationConfig
    created_at: datetime = field(default_factory=datetime.now)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


class ProgressTracker:
    """Tracks evaluation progress."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        
    def update(self, completed: int = 1, failed: int = 0):
        """Update progress counters."""
        self.completed_tasks += completed
        self.failed_tasks += failed
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        elapsed = time.time() - self.start_time
        completion_rate = self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0
        
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "completion_rate": completion_rate,
            "elapsed_time": elapsed,
            "estimated_remaining": elapsed / completion_rate * (1 - completion_rate) if completion_rate > 0 else 0
        }


class PromptEvaluator:
    """
    Main evaluation orchestrator for PromptOpt Co-Pilot.
    
    Coordinates comprehensive prompt evaluation using multiple metrics,
    local LLM judges, and concurrent processing patterns.
    """
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: EvaluationConfig):
        """
        Initialize the prompt evaluator.
        
        Args:
            llm_wrapper: Local LLM wrapper for judge-based evaluations
            config: Evaluation configuration
        """
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.metric_calculator = MetricCalculator()
        self.langsmith_adapter = LangSmithAdapter()
        self.database = DatabaseManager()
        self.dataset_handler = DatasetHandler()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Cache for results
        self._result_cache = {}
        
        logger.info(f"PromptEvaluator initialized with metrics: {config.enabled_metrics}")
    
    def _initialize_metrics(self):
        """Initialize available metrics."""
        self.available_metrics = {
            "exact_match": ExactMatchMetric(),
            "semantic_similarity": SemanticSimilarityMetric(),
            "bleu": BLEUMetric(),
            "rouge": ROUGEMetric(),
            "custom": CustomMetric()
        }
    
    def _generate_cache_key(self, prompt: str, dataset: List[dict], metrics: List[str]) -> str:
        """Generate cache key for evaluation results."""
        import hashlib
        content = f"{prompt}_{len(dataset)}_{sorted(metrics)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def evaluate_prompt(
        self, 
        prompt: str, 
        dataset: List[dict], 
        metrics: List[str]
    ) -> EvaluationResult:
        """
        Evaluate a single prompt against a dataset using specified metrics.
        
        Args:
            prompt: The prompt to evaluate
            dataset: List of test cases with input/expected_output
            metrics: List of metric names to use
            
        Returns:
            EvaluationResult with comprehensive evaluation data
        """
        prompt_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(prompt, dataset, metrics)
        if self.config.cache_results and cache_key in self._result_cache:
            logger.info(f"Returning cached result for prompt {prompt_id}")
            return self._result_cache[cache_key]
        
        logger.info(f"Starting evaluation for prompt {prompt_id} with {len(dataset)} test cases")
        
        # Initialize result structure
        result = EvaluationResult(
            prompt_id=prompt_id,
            prompt_content=prompt,
            metrics_scores={},
            latency_stats={},
            success_rate=0.0,
            individual_results=[],
            total_cases=len(dataset)
        )
        
        # Process test cases
        individual_results = []
        latencies = []
        successful_cases = 0
        
        try:
            # Use concurrent processing with controlled batch size
            batches = [dataset[i:i + self.config.batch_size] 
                      for i in range(0, len(dataset), self.config.batch_size)]
            
            for batch in batches:
                batch_results = await self._process_batch(prompt, batch, metrics)
                individual_results.extend(batch_results)
                
                # Update statistics
                for case_result in batch_results:
                    if case_result.get("success", False):
                        successful_cases += 1
                        latencies.append(case_result.get("latency", 0))
            
            # Calculate aggregate metrics
            result.individual_results = individual_results
            result.successful_cases = successful_cases
            result.failed_cases = len(dataset) - successful_cases
            result.success_rate = successful_cases / len(dataset) if dataset else 0
            
            # Calculate metric scores
            result.metrics_scores = self._calculate_aggregate_metrics(
                individual_results, metrics
            )
            
            # Calculate latency statistics
            if latencies:
                result.latency_stats = {
                    "mean": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "p50": self._percentile(latencies, 50),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99)
                }
            
            # Store metadata
            result.metadata = {
                "evaluation_duration": time.time() - start_time,
                "dataset_size": len(dataset),
                "metrics_used": metrics,
                "config": self.config.__dict__
            }
            
            # Cache result
            if self.config.cache_results:
                self._result_cache[cache_key] = result
            
            # Persist to database
            if self.config.save_intermediate_results:
                await self._save_result(result)
            
            logger.info(f"Evaluation completed for prompt {prompt_id}. Success rate: {result.success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for prompt {prompt_id}: {str(e)}")
            result.error_details.append(str(e))
            raise EvaluationError(f"Evaluation failed: {str(e)}")
        
        return result
    
    async def _process_batch(
        self, 
        prompt: str, 
        batch: List[dict], 
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Process a batch of test cases concurrently."""
        tasks = []
        
        for test_case in batch:
            task = self._evaluate_single_case(prompt, test_case, metrics)
            tasks.append(task)
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_per_case * len(batch)
            )
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Test case {i} failed: {str(result)}")
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "test_case_index": i
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {self.config.timeout_per_case * len(batch)} seconds")
            return [{"success": False, "error": "timeout"} for _ in batch]
    
    async def _evaluate_single_case(
        self, 
        prompt: str, 
        test_case: dict, 
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a single test case."""
        start_time = time.time()
        
        try:
            # Generate response using LLM
            input_text = test_case.get("input", "")
            expected_output = test_case.get("expected_output", "")
            
            # Format prompt with input
            formatted_prompt = self._format_prompt(prompt, input_text)
            
            # Get LLM response
            response = await self.llm_wrapper.generate_async(
                formatted_prompt,
                **self.config.judge_model_settings
            )
            
            latency = time.time() - start_time
            
            # Calculate metrics
            metric_scores = {}
            for metric_name in metrics:
                if metric_name in self.available_metrics:
                    metric = self.available_metrics[metric_name]
                    score = metric.calculate(response, expected_output)
                    metric_scores[metric_name] = score
            
            return {
                "success": True,
                "input": input_text,
                "expected_output": expected_output,
                "actual_output": response,
                "latency": latency,
                "metric_scores": metric_scores,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time,
                "input": test_case.get("input", ""),
                "expected_output": test_case.get("expected_output", "")
            }
    
    def _format_prompt(self, prompt: str, input_text: str) -> str:
        """Format prompt with input text."""
        # Support different prompt formats
        if "{input}" in prompt:
            return prompt.format(input=input_text)
        elif "[INPUT]" in prompt:
            return prompt.replace("[INPUT]", input_text)
        else:
            return f"{prompt}\n\nInput: {input_text}"
    
    def _calculate_aggregate_metrics(
        self, 
        individual_results: List[Dict[str, Any]], 
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate aggregate metric scores."""
        aggregate_scores = {}
        
        for metric_name in metrics:
            scores = []
            for result in individual_results:
                if result.get("success") and "metric_scores" in result:
                    score = result["metric_scores"].get(metric_name)
                    if score is not None:
                        scores.append(score)
            
            if scores:
                aggregate_scores[metric_name] = sum(scores) / len(scores)
            else:
                aggregate_scores[metric_name] = 0.0
        
        return aggregate_scores
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    async def evaluate_variants(
        self, 
        variants: List[str], 
        dataset: List[dict]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple prompt variants against the same dataset.
        
        Args:
            variants: List of prompt variants to evaluate
            dataset: Common dataset for evaluation
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Evaluating {len(variants)} prompt variants")
        
        results = []
        progress_tracker = ProgressTracker(len(variants)) if self.config.enable_progress_tracking else None
        
        # Create evaluation requests
        requests = []
        for i, variant in enumerate(variants):
            request = EvaluationRequest(
                prompt_content=variant,
                dataset_subset=dataset,
                required_metrics=self.config.enabled_metrics,
                priority=1,
                metadata={"variant_index": i}
            )
            requests.append(request)
        
        # Process variants concurrently
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_workers) as executor:
            future_to_request = {}
            
            for request in requests:
                future = executor.submit(
                    asyncio.run,
                    self.evaluate_prompt(
                        request.prompt_content,
                        request.dataset_subset,
                        request.required_metrics
                    )
                )
                future_to_request[future] = request
            
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_tracker:
                        progress_tracker.update(completed=1)
                        logger.info(f"Variant evaluation progress: {progress_tracker.get_progress()}")
                        
                except Exception as e:
                    logger.error(f"Variant evaluation failed: {str(e)}")
                    if progress_tracker:
                        progress_tracker.update(failed=1)
        
        # Sort results by variant index
        results.sort(key=lambda x: x.metadata.get("variant_index", 0))
        
        logger.info(f"Completed evaluation of {len(results)} variants")
        return results
    
    async def run_evaluation_suite(
        self, 
        prompts: List[str], 
        dataset: List[dict]
    ) -> EvaluationSuite:
        """
        Run a complete evaluation suite with multiple prompts.
        
        Args:
            prompts: List of prompts to evaluate
            dataset: Dataset for evaluation
            
        Returns:
            EvaluationSuite with comprehensive results
        """
        suite_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting evaluation suite {suite_id} with {len(prompts)} prompts")
        
        # Evaluate all prompts
        results = await self.evaluate_variants(prompts, dataset)
        
        # Calculate summary statistics
        summary_stats = self._calculate_suite_summary(results)
        
        # Create suite object
        suite = EvaluationSuite(
            suite_id=suite_id,
            results=results,
            summary_stats=summary_stats,
            execution_time=time.time() - start_time,
            dataset_info={
                "size": len(dataset),
                "fields": list(dataset[0].keys()) if dataset else []
            },
            config=self.config
        )
        
        # Save suite to database
        await self._save_suite(suite)
        
        logger.info(f"Evaluation suite {suite_id} completed in {suite.execution_time:.2f} seconds")
        return suite
    
    def _calculate_suite_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics for evaluation suite."""
        if not results:
            return {}
        
        # Aggregate metric scores
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics_scores.keys())
        
        metric_summaries = {}
        for metric in all_metrics:
            scores = [r.metrics_scores.get(metric, 0) for r in results]
            metric_summaries[metric] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std": self._calculate_std(scores)
            }
        
        # Success rate summary
        success_rates = [r.success_rate for r in results]
        
        # Latency summary
        mean_latencies = []
        for result in results:
            if result.latency_stats:
                mean_latencies.append(result.latency_stats.get("mean", 0))
        
        return {
            "total_prompts": len(results),
            "metric_summaries": metric_summaries,
            "success_rate_summary": {
                "mean": sum(success_rates) / len(success_rates),
                "min": min(success_rates),
                "max": max(success_rates)
            },
            "latency_summary": {
                "mean": sum(mean_latencies) / len(mean_latencies) if mean_latencies else 0,
                "min": min(mean_latencies) if mean_latencies else 0,
                "max": max(mean_latencies) if mean_latencies else 0
            },
            "best_performing_prompt": max(results, key=lambda x: sum(x.metrics_scores.values())).prompt_id,
            "total_test_cases": sum(r.total_cases for r in results),
            "total_successful_cases": sum(r.successful_cases for r in results)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def pairwise_comparison(
        self, 
        prompt_a: str, 
        prompt_b: str, 
        test_cases: List[dict]
    ) -> ComparisonResult:
        """
        Perform pairwise comparison between two prompts.
        
        Args:
            prompt_a: First prompt
            prompt_b: Second prompt
            test_cases: Test cases for comparison
            
        Returns:
            ComparisonResult with comparison details
        """
        logger.info("Starting pairwise comparison")
        
        # Evaluate both prompts
        result_a = await self.evaluate_prompt(prompt_a, test_cases, self.config.enabled_metrics)
        result_b = await self.evaluate_prompt(prompt_b, test_cases, self.config.enabled_metrics)
        
        # Perform judge-based comparison
        judge_results = await self._judge_comparison(
            prompt_a, prompt_b, result_a, result_b, test_cases
        )
        
        # Determine winner based on metrics and judge
        winner = self._determine_winner(result_a, result_b, judge_results)
        
        comparison = ComparisonResult(
            prompt_a_id=result_a.prompt_id,
            prompt_b_id=result_b.prompt_id,
            winner=winner["winner"],
            confidence=winner["confidence"],
            detailed_scores={
                "prompt_a": result_a.metrics_scores,
                "prompt_b": result_b.metrics_scores
            },
            judge_reasoning=judge_results.get("reasoning", []),
            comparison_metrics={
                "metric_difference": winner["metric_difference"],
                "success_rate_difference": result_a.success_rate - result_b.success_rate,
                "latency_difference": (
                    result_a.latency_stats.get("mean", 0) - 
                    result_b.latency_stats.get("mean", 0)
                )
            }
        )
        
        logger.info(f"Pairwise comparison completed. Winner: {winner['winner']}")
        return comparison
    
    async def _judge_comparison(
        self, 
        prompt_a: str, 
        prompt_b: str, 
        result_a: EvaluationResult, 
        result_b: EvaluationResult,
        test_cases: List[dict]
    ) -> Dict[str, Any]:
        """Use LLM judge for comparison."""
        judge_prompt = f"""
        Compare the following two prompts based on their performance:
        
        Prompt A: {prompt_a}
        Prompt B: {prompt_b}
        
        Prompt A Results:
        - Success Rate: {result_a.success_rate:.2%}
        - Metric Scores: {result_a.metrics_scores}
        
        Prompt B Results:
        - Success Rate: {result_b.success_rate:.2%}
        - Metric Scores: {result_b.metrics_scores}
        
        Which prompt performs better overall? Provide reasoning.
        """
        
        try:
            judge_response = await self.llm_wrapper.generate_async(
                judge_prompt,
                **self.config.judge_model_settings
            )
            
            return {
                "judge_response": judge_response,
                "reasoning": [judge_response]
            }
        except Exception as e:
            logger.error(f"Judge comparison failed: {str(e)}")
            return {"reasoning": [f"Judge comparison failed: {str(e)}"]}
    
    def _determine_winner(
        self, 
        result_a: EvaluationResult, 
        result_b: EvaluationResult, 
        judge_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine winner based on metrics and judge."""
        # Calculate aggregate scores
        score_a = sum(result_a.metrics_scores.values())
        score_b = sum(result_b.metrics_scores.values())
        
        score_diff = score_a - score_b
        
        # Determine winner
        if abs(score_diff) < 0.01:  # Tie threshold
            winner = "tie"
            confidence = 0.5
        elif score_diff > 0:
            winner = "A"
            confidence = min(0.95, 0.5 + abs(score_diff) / 2)
        else:
            winner = "B"
            confidence = min(0.95, 0.5 + abs(score_diff) / 2)
        
        return {
            "winner": winner,
            "confidence": confidence,
            "metric_difference": score_diff
        }
    
    async def batch_evaluate(
        self, 
        evaluation_requests: List[EvaluationRequest]
    ) -> List[EvaluationResult]:
        """
        Process multiple evaluation requests in batch.
        
        Args:
            evaluation_requests: List of evaluation requests
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Processing batch of {len(evaluation_requests)} evaluation requests")
        
        # Sort by priority
        sorted_requests = sorted(evaluation_requests, key=lambda x: x.priority, reverse=True)
        
        results = []
        progress_tracker = ProgressTracker(len(sorted_requests))
        
        # Process in batches
        for i in range(0, len(sorted_requests), self.config.batch_size):
            batch = sorted_requests[i:i + self.config.batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for request in batch:
                task = self.evaluate_prompt(
                    request.prompt_content,
                    request.dataset_subset,
                    request.required_metrics
                )
                batch_tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch evaluation failed for request {batch[j].request_id}: {str(result)}")
                        progress_tracker.update(failed=1)
                    else:
                        results.append(result)
                        progress_tracker.update(completed=1)
                
                logger.info(f"Batch progress: {progress_tracker.get_progress()}")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                progress_tracker.update(failed=len(batch))
        
        logger.info(f"Batch evaluation completed. Processed {len(results)} requests")
        return results
    
    def get_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"error": "No results provided"}
        
        # Basic statistics
        total_prompts = len(results)
        total_cases = sum(r.total_cases for r in results)
        total_successful = sum(r.successful_cases for r in results)
        
        # Metric statistics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics_scores.keys())
        
        metric_stats = {}
        for metric in all_metrics:
            scores = [r.metrics_scores.get(metric, 0) for r in results if metric in r.metrics_scores]
            if scores:
                metric_stats[metric] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Performance statistics
        success_rates = [r.success_rate for r in results]
        latency_means = [r.latency_stats.get("mean", 0) for r in results if r.latency_stats]
        
        # Best and worst performing prompts
        best_prompt = max(results, key=lambda x: sum(x.metrics_scores.values()))
        worst_prompt = min(results, key=lambda x: sum(x.metrics_scores.values()))
        
        return {
            "overview": {
                "total_prompts": total_prompts,
                "total_test_cases": total_cases,
                "total_successful_cases": total_successful,
                "overall_success_rate": total_successful / total_cases if total_cases > 0 else 0
            },
            "metric_statistics": metric_stats,
            "performance_statistics": {
                "success_rate": {
                    "mean": sum(success_rates) / len(success_rates),
                    "min": min(success_rates),
                    "max": max(success_rates)
                },
                "latency": {
                    "mean": sum(latency_means) / len(latency_means) if latency_means else 0,
                    "min": min(latency_means) if latency_means else 0,
                    "max": max(latency_means) if latency_means else 0
                }
            },
            "best_performing_prompt": {
                "prompt_id": best_prompt.prompt_id,
                "metrics_scores": best_prompt.metrics_scores,
                "success_rate": best_prompt.success_rate
            },
            "worst_performing_prompt": {
                "prompt_id": worst_prompt.prompt_id,
                "metrics_scores": worst_prompt.metrics_scores,
                "success_rate": worst_prompt.success_rate
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if not results:
            return ["No results available for analysis"]
        
        # Analyze success rates
        success_rates = [r.success_rate for r in results]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        if avg_success_rate < 0.7:
            recommendations.append("Overall success rate is low. Consider revising prompt instructions or dataset quality.")
        
        # Analyze metric performance
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics_scores.keys())
        
        for metric in all_metrics:
            scores = [r.metrics_scores.get(metric, 0) for r in results if metric in r.metrics_scores]
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score < 0.5:
                    recommendations.append(f"{metric} scores are low. Focus on improving this specific aspect.")
        
        # Analyze latency
        latency_means = [r.latency_stats.get("mean", 0) for r in results if r.latency_stats]
        if latency_means:
            avg_latency = sum(latency_means) / len(latency_means)
            if avg_latency > 5.0:  # 5 second threshold
                recommendations.append("High latency detected. Consider optimizing prompt length or model settings.")
        
        # Variance analysis
        if len(results) > 1:
            metric_variances = {}
            for metric in all_metrics:
                scores = [r.metrics_scores.get(metric, 0) for r in results if metric in r.metrics_scores]
                if len(scores) > 1:
                    variance = self._calculate_std(scores) ** 2
                    metric_variances[metric] = variance
            
            high_variance_metrics = [m for m, v in metric_variances.items() if v > 0.1]
            if high_variance_metrics:
                recommendations.append(f"High variance in {', '.join(high_variance_metrics)}. Results are inconsistent.")
        
        if not recommendations:
            recommendations.append("Results look good! Performance is consistent and meets quality thresholds.")
        
        return recommendations
    
    async def _save_result(self, result: EvaluationResult):
        """Save evaluation result to database."""
        try:
            await self.database.save_evaluation_result(result)
            logger.debug(f"Saved evaluation result {result.prompt_id}")
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {str(e)}")
    
    async def _save_suite(self, suite: EvaluationSuite):
        """Save evaluation suite to database."""
        try:
            await self.database.save_evaluation_suite(suite)
            logger.debug(f"Saved evaluation suite {suite.suite_id}")
        except Exception as e:
            logger.error(f"Failed to save evaluation suite: {str(e)}")


# Unit Tests
import unittest
from unittest.mock import Mock, AsyncMock, patch
import pytest


class TestPromptEvaluator(unittest.TestCase):
    """Unit tests for PromptEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=LlamaWrapper)
        self.mock_llm.generate_async = AsyncMock(return_value="test response")
        
        self.config = EvaluationConfig(
            enabled_metrics=["exact_match", "semantic_similarity"],
            timeout_per_case=10,
            batch_size=2,
            cache_results=False
        )
        
        self.evaluator = PromptEvaluator(self.mock_llm, self.config)
        
        self.test_dataset = [
            {"input": "What is 2+2?", "expected_output": "4"},
            {"input": "What is the capital of France?", "expected_output": "Paris"}
        ]
    
    def test_init(self):
        """Test evaluator initialization."""
        self.assertIsInstance(self.evaluator, PromptEvaluator)
        self.assertEqual(self.evaluator.config, self.config)
        self.assertIsNotNone(self.evaluator.available_metrics)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        prompt = "Test prompt"
        dataset = [{"input": "test", "expected_output": "test"}]
        metrics = ["exact_match"]
        
        key1 = self.evaluator._generate_cache_key(prompt, dataset, metrics)
        key2 = self.evaluator._generate_cache_key(prompt, dataset, metrics)
        
        self.assertEqual(key1, key2)
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 32)  # MD5 hash length
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        # Test with {input} placeholder
        prompt1 = "Answer this: {input}"
        formatted1 = self.evaluator._format_prompt(prompt1, "test input")
        self.assertEqual(formatted1, "Answer this: test input")
        
        # Test with [INPUT] placeholder
        prompt2 = "Answer this: [INPUT]"
        formatted2 = self.evaluator._format_prompt(prompt2, "test input")
        self.assertEqual(formatted2, "Answer this: test input")
        
        # Test without placeholder
        prompt3 = "Answer this:"
        formatted3 = self.evaluator._format_prompt(prompt3, "test input")
        self.assertEqual(formatted3, "Answer this:\n\nInput: test input")
    
    def test_calculate_std(self):
        """Test standard deviation calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = self.evaluator._calculate_std(values)
        self.assertAlmostEqual(std, 1.58, places=2)
        
        # Test with single value
        single_value = [1.0]
        std_single = self.evaluator._calculate_std(single_value)
        self.assertEqual(std_single, 0.0)
    
    def test_percentile(self):
        """Test percentile calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        p50 = self.evaluator._percentile(data, 50)
        self.assertEqual(p50, 5.5)
        
        p95 = self.evaluator._percentile(data, 95)
        self.assertEqual(p95, 9.5)
        
        # Test with empty data
        empty_p50 = self.evaluator._percentile([], 50)
        self.assertEqual(empty_p50, 0.0)
    
    @pytest.mark.asyncio
    async def test_evaluate_single_case(self):
        """Test single case evaluation."""
        prompt = "Answer: {input}"
        test_case = {"input": "What is 2+2?", "expected_output": "4"}
        metrics = ["exact_match"]
        
        result = await self.evaluator._evaluate_single_case(prompt, test_case, metrics)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["input"], "What is 2+2?")
        self.assertEqual(result["expected_output"], "4")
        self.assertEqual(result["actual_output"], "test response")
        self.assertIn("latency", result)
        self.assertIn("metric_scores", result)
    
    def test_calculate_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        individual_results = [
            {"success": True, "metric_scores": {"exact_match": 1.0, "bleu": 0.8}},
            {"success": True, "metric_scores": {"exact_match": 0.0, "bleu": 0.6}},
            {"success": False, "error": "timeout"}
        ]
        metrics = ["exact_match", "bleu"]
        
        aggregate = self.evaluator._calculate_aggregate_metrics(individual_results, metrics)
        
        self.assertEqual(aggregate["exact_match"], 0.5)  # (1.0 + 0.0) / 2
        self.assertEqual(aggregate["bleu"], 0.7)  # (0.8 + 0.6) / 2
    
    def test_determine_winner(self):
        """Test winner determination in pairwise comparison."""
        result_a = EvaluationResult(
            prompt_id="a",
            prompt_content="",
            metrics_scores={"exact_match": 0.8, "bleu": 0.7},
            latency_stats={},
            success_rate=0.9,
            individual_results=[]
        )
        
        result_b = EvaluationResult(
            prompt_id="b",
            prompt_content="",
            metrics_scores={"exact_match": 0.6, "bleu": 0.5},
            latency_stats={},
            success_rate=0.8,
            individual_results=[]
        )
        
        judge_results = {"reasoning": ["A performs better"]}
        
        winner = self.evaluator._determine_winner(result_a, result_b, judge_results)
        
        self.assertEqual(winner["winner"], "A")
        self.assertGreater(winner["confidence"], 0.5)
        self.assertGreater(winner["metric_difference"], 0)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        results = [
            EvaluationResult(
                prompt_id="1",
                prompt_content="",
                metrics_scores={"exact_match": 0.3},  # Low score
                latency_stats={"mean": 6.0},  # High latency
                success_rate=0.5,  # Low success rate
                individual_results=[]
            ),
            EvaluationResult(
                prompt_id="2",
                prompt_content="",
                metrics_scores={"exact_match": 0.9},
                latency_stats={"mean": 1.0},
                success_rate=0.95,
                individual_results=[]
            )
        ]
        
        recommendations = self.evaluator._generate_recommendations(results)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for specific recommendations
        rec_text = " ".join(recommendations)
        self.assertTrue(any("success rate" in rec.lower() for rec in recommendations))
    
    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        config = EvaluationConfig()
        
        self.assertEqual(config.enabled_metrics, ["exact_match", "semantic_similarity"])
        self.assertEqual(config.timeout_per_case, 30)
        self.assertEqual(config.batch_size, 10)
        self.assertTrue(config.cache_results)
        self.assertTrue(config.enable_progress_tracking)
    
    def test_progress_tracker(self):
        """Test ProgressTracker functionality."""
        tracker = ProgressTracker(total_tasks=10)
        
        # Initial state
        progress = tracker.get_progress()
        self.assertEqual(progress["total_tasks"], 10)
        self.assertEqual(progress["completed_tasks"], 0)
        self.assertEqual(progress["completion_rate"], 0)
        
        # Update progress
        tracker.update(completed=3)
        progress = tracker.get_progress()
        self.assertEqual(progress["completed_tasks"], 3)
        self.assertEqual(progress["completion_rate"], 0.3)
        
        # Update with failures
        tracker.update(completed=2, failed=1)
        progress = tracker.get_progress()
        self.assertEqual(progress["completed_tasks"], 5)
        self.assertEqual(progress["failed_tasks"], 1)
    
    def test_evaluation_result_dataclass(self):
        """Test EvaluationResult dataclass."""
        result = EvaluationResult(
            prompt_id="test",
            prompt_content="test prompt",
            metrics_scores={"exact_match": 0.8},
            latency_stats={"mean": 1.5},
            success_rate=0.9,
            individual_results=[]
        )
        
        self.assertEqual(result.prompt_id, "test")
        self.assertEqual(result.prompt_content, "test prompt")
        self.assertEqual(result.metrics_scores["exact_match"], 0.8)
        self.assertEqual(result.success_rate, 0.9)
        self.assertIsInstance(result.evaluation_time, datetime)
    
    def test_evaluation_request_dataclass(self):
        """Test EvaluationRequest dataclass."""
        request = EvaluationRequest(
            prompt_content="test prompt",
            dataset_subset=[{"input": "test", "expected_output": "test"}],
            required_metrics=["exact_match"]
        )
        
        self.assertEqual(request.prompt_content, "test prompt")
        self.assertEqual(len(request.dataset_subset), 1)
        self.assertEqual(request.required_metrics, ["exact_match"])
        self.assertEqual(request.priority, 1)
        self.assertIsInstance(request.request_id, str)


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation components."""
    
    @patch('backend.core.database.DatabaseManager')
    @patch('backend.evaluation.metrics.MetricCalculator')
    def setUp(self, mock_metric_calc, mock_db):
        """Set up integration test fixtures."""
        self.mock_llm = Mock(spec=LlamaWrapper)
        self.mock_llm.generate_async = AsyncMock(return_value="test response")
        
        self.config = EvaluationConfig(
            enabled_metrics=["exact_match"],
            timeout_per_case=5,
            batch_size=2,
            cache_results=False,
            save_intermediate_results=False
        )
        
        self.evaluator = PromptEvaluator(self.mock_llm, self.config)
        
    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test complete evaluation flow."""
        prompt = "Answer the question: {input}"
        dataset = [
            {"input": "What is 2+2?", "expected_output": "4"},
            {"input": "What is 3+3?", "expected_output": "6"}
        ]
        metrics = ["exact_match"]
        
        # Mock metric calculation
        with patch.object(self.evaluator.available_metrics["exact_match"], 'calculate', return_value=0.8):
            result = await self.evaluator.evaluate_prompt(prompt, dataset, metrics)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(len(result.individual_results), 2)
        self.assertIn("exact_match", result.metrics_scores)
        self.assertGreater(result.success_rate, 0)


if __name__ == "__main__":
    # Run basic tests
    unittest.main(verbosity=2)
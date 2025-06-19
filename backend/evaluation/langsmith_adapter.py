"""
LangSmith Evaluation Adapter for PromptOpt Co-Pilot

This module provides LangSmith-compatible evaluation patterns and interfaces for offline
prompt evaluation. It mimics LangSmith's cloud-based evaluation platform while operating
entirely offline with local LLMs and SQLite storage.

Author: PromptOpt Co-Pilot
Version: 1.0.0
"""

import json
import uuid
import time
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

# Internal imports
from backend.llm.llama_wrapper import LlamaWrapper
from backend.evaluation.metrics import (
    calculate_exact_match,
    calculate_semantic_similarity,
    calculate_bleu_score,
    calculate_rouge_score
)
from backend.core.database import DatabaseManager
from backend.core.models import EvaluationResult, Dataset, EvaluationRun

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LangSmithConfig:
    """Configuration for LangSmith adapter operations."""
    default_evaluators: List[str] = field(default_factory=lambda: ['exact_match', 'semantic_similarity'])
    batch_size: int = 10
    timeout_seconds: int = 300
    cache_enabled: bool = True
    max_workers: int = 4
    similarity_threshold: float = 0.8
    enable_progress_tracking: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv', 'html'])


@dataclass
class EvaluationSummary:
    """Summary of evaluation results in LangSmith format."""
    run_id: str
    name: str
    dataset_name: str
    total_examples: int
    completed_examples: int
    failed_examples: int
    average_score: float
    evaluator_scores: Dict[str, float]
    execution_time: float
    created_at: datetime
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangSmithEvaluator(ABC):
    """Abstract base class for LangSmith-style evaluators."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate a single prediction against reference."""
        pass
    
    def batch_evaluate(self, predictions: List[str], references: List[str], 
                      input_data: List[Dict[str, Any]] = None) -> List[Dict[str, float]]:
        """Evaluate multiple predictions in batch."""
        if input_data is None:
            input_data = [{}] * len(predictions)
        
        results = []
        for pred, ref, inp in zip(predictions, references, input_data):
            try:
                result = self.evaluate(pred, ref, inp)
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for {self.name}: {str(e)}")
                results.append({f"{self.name}_score": 0.0, "error": str(e)})
        
        return results


class ExactMatchEvaluator(LangSmithEvaluator):
    """Exact match evaluator for LangSmith compatibility."""
    
    def __init__(self, case_sensitive: bool = False, normalize_whitespace: bool = True):
        super().__init__("exact_match")
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
    
    def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate exact match between prediction and reference."""
        pred = prediction
        ref = reference
        
        if not self.case_sensitive:
            pred = pred.lower()
            ref = ref.lower()
        
        if self.normalize_whitespace:
            pred = ' '.join(pred.split())
            ref = ' '.join(ref.split())
        
        score = 1.0 if pred == ref else 0.0
        return {
            "exact_match_score": score,
            "exact_match_binary": score == 1.0
        }


class SemanticSimilarityEvaluator(LangSmithEvaluator):
    """Semantic similarity evaluator using embeddings."""
    
    def __init__(self, threshold: float = 0.8, llm_wrapper: LlamaWrapper = None):
        super().__init__("semantic_similarity")
        self.threshold = threshold
        self.llm_wrapper = llm_wrapper
    
    def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate semantic similarity between prediction and reference."""
        try:
            similarity_score = calculate_semantic_similarity(prediction, reference)
            
            return {
                "semantic_similarity_score": similarity_score,
                "semantic_similarity_binary": similarity_score >= self.threshold,
                "threshold": self.threshold
            }
        except Exception as e:
            logger.error(f"Semantic similarity evaluation failed: {str(e)}")
            return {
                "semantic_similarity_score": 0.0,
                "semantic_similarity_binary": False,
                "error": str(e)
            }


class CustomLLMEvaluator(LangSmithEvaluator):
    """Custom LLM-based evaluator for complex evaluation criteria."""
    
    def __init__(self, llm_wrapper: LlamaWrapper, evaluation_prompt: str, 
                 score_key: str = "score", binary_threshold: float = 0.5):
        super().__init__("custom_llm")
        self.llm_wrapper = llm_wrapper
        self.evaluation_prompt = evaluation_prompt
        self.score_key = score_key
        self.binary_threshold = binary_threshold
    
    def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate using custom LLM prompt."""
        try:
            # Format evaluation prompt
            prompt = self.evaluation_prompt.format(
                prediction=prediction,
                reference=reference,
                input=input_data.get('input', '') if input_data else ''
            )
            
            # Get LLM evaluation
            response = self.llm_wrapper.generate_response(prompt)
            
            # Extract score (assuming JSON format response)
            try:
                result = json.loads(response)
                score = float(result.get(self.score_key, 0.0))
            except (json.JSONDecodeError, ValueError):
                # Fallback: try to extract numeric score from text
                import re
                numbers = re.findall(r'\d+\.?\d*', response)
                score = float(numbers[0]) if numbers else 0.0
            
            # Normalize score to 0-1 range if needed
            if score > 1.0:
                score = score / 10.0 if score <= 10.0 else score / 100.0
            
            return {
                f"{self.score_key}": score,
                f"{self.score_key}_binary": score >= self.binary_threshold,
                "llm_response": response[:200]  # Truncated for storage
            }
            
        except Exception as e:
            logger.error(f"Custom LLM evaluation failed: {str(e)}")
            return {
                f"{self.score_key}": 0.0,
                f"{self.score_key}_binary": False,
                "error": str(e)
            }


class LatencyEvaluator(LangSmithEvaluator):
    """Latency evaluator for performance measurement."""
    
    def __init__(self, target_latency_ms: float = 1000.0):
        super().__init__("latency")
        self.target_latency_ms = target_latency_ms
    
    def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate latency performance."""
        latency_ms = input_data.get('latency_ms', 0.0) if input_data else 0.0
        
        # Calculate performance score (inverse of latency ratio)
        if latency_ms > 0:
            latency_score = min(1.0, self.target_latency_ms / latency_ms)
        else:
            latency_score = 0.0
        
        return {
            "latency_ms": latency_ms,
            "latency_score": latency_score,
            "meets_target": latency_ms <= self.target_latency_ms,
            "target_latency_ms": self.target_latency_ms
        }


class LangSmithDataset:
    """LangSmith-compatible dataset class for offline operation."""
    
    def __init__(self, name: str, data: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.size = len(data)
    
    @classmethod
    def create_from_csv(cls, file_path: str, name: str = None) -> 'LangSmithDataset':
        """Create dataset from CSV file."""
        try:
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
            dataset_name = name or Path(file_path).stem
            
            logger.info(f"Created dataset '{dataset_name}' from CSV with {len(data)} examples")
            return cls(dataset_name, data)
            
        except Exception as e:
            logger.error(f"Failed to create dataset from CSV: {str(e)}")
            raise ValueError(f"Invalid CSV file: {str(e)}")
    
    @classmethod
    def create_from_json(cls, data: List[Dict[str, Any]], name: str) -> 'LangSmithDataset':
        """Create dataset from JSON data."""
        if not cls.validate_dataset_schema(data):
            raise ValueError("Invalid dataset schema")
        
        logger.info(f"Created dataset '{name}' from JSON with {len(data)} examples")
        return cls(name, data)
    
    @staticmethod
    def validate_dataset_schema(data: List[Dict[str, Any]]) -> bool:
        """Validate dataset schema for LangSmith compatibility."""
        if not data:
            return False
        
        required_fields = {'input', 'output'}  # Minimum required fields
        
        for example in data:
            if not isinstance(example, dict):
                return False
            
            # Check if at least input field exists
            if 'input' not in example:
                logger.warning("Dataset example missing 'input' field")
                return False
        
        return True
    
    def add_example(self, example: Dict[str, Any]) -> None:
        """Add a single example to the dataset."""
        if not isinstance(example, dict) or 'input' not in example:
            raise ValueError("Invalid example format")
        
        self.data.append(example)
        self.size = len(self.data)
    
    def get_sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of examples from the dataset."""
        return self.data[:min(n, len(self.data))]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format."""
        return {
            'id': self.id,
            'name': self.name,
            'size': self.size,
            'data': self.data,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class LangSmithAdapter:
    """Main adapter class providing LangSmith-compatible evaluation interface."""
    
    def __init__(self, llm_wrapper: LlamaWrapper, config: LangSmithConfig = None):
        self.llm_wrapper = llm_wrapper
        self.config = config or LangSmithConfig()
        self.db_manager = DatabaseManager()
        self.datasets = {}
        self.evaluation_runs = {}
        
        # Initialize evaluators
        self.evaluators = {
            'exact_match': ExactMatchEvaluator(),
            'semantic_similarity': SemanticSimilarityEvaluator(
                threshold=self.config.similarity_threshold,
                llm_wrapper=self.llm_wrapper
            ),
            'latency': LatencyEvaluator()
        }
        
        logger.info("LangSmith adapter initialized")
    
    def create_dataset(self, name: str, data: List[Dict[str, Any]]) -> LangSmithDataset:
        """Create a new dataset for evaluation."""
        try:
            dataset = LangSmithDataset.create_from_json(data, name)
            self.datasets[dataset.id] = dataset
            
            # Store in database
            self.db_manager.store_dataset(dataset.to_dict())
            
            logger.info(f"Created dataset '{name}' with {len(data)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            raise
    
    def create_evaluation_run(self, name: str, dataset: LangSmithDataset) -> EvaluationRun:
        """Create a new evaluation run."""
        run = EvaluationRun(
            id=str(uuid.uuid4()),
            name=name,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            status='initialized',
            created_at=datetime.now(),
            total_examples=dataset.size
        )
        
        self.evaluation_runs[run.id] = run
        logger.info(f"Created evaluation run '{name}' with ID: {run.id}")
        
        return run
    
    def evaluate_prompt(self, prompt: str, dataset: LangSmithDataset, 
                       evaluators: List[str] = None) -> EvaluationResult:
        """Evaluate a single prompt against a dataset."""
        evaluators = evaluators or self.config.default_evaluators
        
        # Create evaluation run
        run = self.create_evaluation_run(f"Single prompt evaluation", dataset)
        
        try:
            run.status = 'running'
            start_time = time.time()
            
            results = []
            failed_count = 0
            
            for i, example in enumerate(dataset.data):
                try:
                    # Generate prediction
                    input_text = example.get('input', '')
                    formatted_prompt = prompt.format(input=input_text) if '{input}' in prompt else f"{prompt}\n{input_text}"
                    
                    prediction_start = time.time()
                    prediction = self.llm_wrapper.generate_response(formatted_prompt)
                    prediction_time = (time.time() - prediction_start) * 1000  # Convert to ms
                    
                    # Get reference output
                    reference = example.get('output', example.get('expected', ''))
                    
                    # Run evaluations
                    eval_results = {}
                    for eval_name in evaluators:
                        if eval_name in self.evaluators:
                            evaluator = self.evaluators[eval_name]
                            input_data = {**example, 'latency_ms': prediction_time}
                            eval_result = evaluator.evaluate(prediction, reference, input_data)
                            eval_results.update(eval_result)
                    
                    results.append({
                        'example_id': i,
                        'input': input_text,
                        'prediction': prediction,
                        'reference': reference,
                        'evaluations': eval_results,
                        'latency_ms': prediction_time
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate example {i}: {str(e)}")
                    failed_count += 1
                    results.append({
                        'example_id': i,
                        'error': str(e),
                        'evaluations': {}
                    })
            
            # Calculate summary metrics
            execution_time = time.time() - start_time
            completed_examples = len(results) - failed_count
            
            # Aggregate scores
            evaluator_scores = {}
            for eval_name in evaluators:
                scores = []
                for result in results:
                    if 'evaluations' in result:
                        score_key = f"{eval_name}_score"
                        if score_key in result['evaluations']:
                            scores.append(result['evaluations'][score_key])
                
                if scores:
                    evaluator_scores[eval_name] = sum(scores) / len(scores)
            
            average_score = sum(evaluator_scores.values()) / len(evaluator_scores) if evaluator_scores else 0.0
            
            # Update run status
            run.status = 'completed'
            run.completed_examples = completed_examples
            run.failed_examples = failed_count
            run.execution_time = execution_time
            
            # Create evaluation result
            evaluation_result = EvaluationResult(
                run_id=run.id,
                prompt=prompt,
                dataset_name=dataset.name,
                results=results,
                summary=EvaluationSummary(
                    run_id=run.id,
                    name=run.name,
                    dataset_name=dataset.name,
                    total_examples=dataset.size,
                    completed_examples=completed_examples,
                    failed_examples=failed_count,
                    average_score=average_score,
                    evaluator_scores=evaluator_scores,
                    execution_time=execution_time,
                    created_at=run.created_at,
                    status=run.status
                )
            )
            
            # Store results
            self.db_manager.store_evaluation_result(evaluation_result.to_dict())
            
            logger.info(f"Completed evaluation run {run.id} with average score: {average_score:.3f}")
            return evaluation_result
            
        except Exception as e:
            run.status = 'failed'
            logger.error(f"Evaluation run {run.id} failed: {str(e)}")
            raise
    
    def batch_evaluate(self, prompts: List[str], dataset: LangSmithDataset, 
                      evaluators: List[str] = None) -> List[EvaluationResult]:
        """Evaluate multiple prompts in batch."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit evaluation tasks
            future_to_prompt = {
                executor.submit(self.evaluate_prompt, prompt, dataset, evaluators): prompt
                for prompt in prompts
            }
            
            # Collect results
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                    logger.info(f"Completed evaluation for prompt: {prompt[:50]}...")
                except Exception as e:
                    logger.error(f"Failed to evaluate prompt '{prompt[:50]}...': {str(e)}")
        
        logger.info(f"Completed batch evaluation of {len(results)} prompts")
        return results
    
    def get_evaluation_results(self, run_id: str) -> EvaluationSummary:
        """Get evaluation results for a specific run."""
        try:
            # Retrieve from database
            result_data = self.db_manager.get_evaluation_result(run_id)
            if not result_data:
                raise ValueError(f"No evaluation results found for run ID: {run_id}")
            
            # Convert to EvaluationSummary
            summary_data = result_data.get('summary', {})
            return EvaluationSummary(**summary_data)
            
        except Exception as e:
            logger.error(f"Failed to get evaluation results: {str(e)}")
            raise
    
    def export_results(self, run_id: str, format: str = 'json') -> Dict[str, Any]:
        """Export evaluation results in specified format."""
        if format not in self.config.export_formats:
            raise ValueError(f"Unsupported export format: {format}")
        
        try:
            result_data = self.db_manager.get_evaluation_result(run_id)
            if not result_data:
                raise ValueError(f"No evaluation results found for run ID: {run_id}")
            
            if format == 'json':
                return self.export_to_langsmith_format(result_data)
            elif format == 'csv':
                return self._export_to_csv(result_data)
            elif format == 'html':
                return self._export_to_html(result_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            raise
    
    def export_to_langsmith_format(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export results in LangSmith-compatible format."""
        return {
            'run_id': result_data.get('run_id'),
            'run_name': result_data.get('summary', {}).get('name'),
            'dataset_name': result_data.get('dataset_name'),
            'created_at': result_data.get('summary', {}).get('created_at'),
            'status': result_data.get('summary', {}).get('status'),
            'metrics': {
                'total_examples': result_data.get('summary', {}).get('total_examples'),
                'completed_examples': result_data.get('summary', {}).get('completed_examples'),
                'failed_examples': result_data.get('summary', {}).get('failed_examples'),
                'average_score': result_data.get('summary', {}).get('average_score'),
                'execution_time': result_data.get('summary', {}).get('execution_time')
            },
            'evaluator_scores': result_data.get('summary', {}).get('evaluator_scores', {}),
            'results': result_data.get('results', []),
            'metadata': {
                'adapter_version': '1.0.0',
                'offline_mode': True,
                'export_timestamp': datetime.now().isoformat()
            }
        }
    
    def _export_to_csv(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export results to CSV format."""
        results = result_data.get('results', [])
        
        # Flatten results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'example_id': result.get('example_id'),
                'input': result.get('input', ''),
                'prediction': result.get('prediction', ''),
                'reference': result.get('reference', ''),
                'latency_ms': result.get('latency_ms', 0)
            }
            
            # Add evaluation scores
            evaluations = result.get('evaluations', {})
            for key, value in evaluations.items():
                flat_result[key] = value
            
            flattened_results.append(flat_result)
        
        # Convert to DataFrame and then to CSV string
        df = pd.DataFrame(flattened_results)
        csv_content = df.to_csv(index=False)
        
        return {
            'format': 'csv',
            'content': csv_content,
            'filename': f"evaluation_results_{result_data.get('run_id', 'unknown')}.csv"
        }
    
    def _export_to_html(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export results to HTML format."""
        summary = result_data.get('summary', {})
        results = result_data.get('results', [])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Results - {summary.get('name', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e9ecef; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .score {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Evaluation Results</h1>
                <p><strong>Run ID:</strong> {result_data.get('run_id', 'N/A')}</p>
                <p><strong>Dataset:</strong> {summary.get('dataset_name', 'N/A')}</p>
                <p><strong>Status:</strong> {summary.get('status', 'N/A')}</p>
                <p><strong>Created:</strong> {summary.get('created_at', 'N/A')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <strong>Total Examples:</strong> {summary.get('total_examples', 0)}
                </div>
                <div class="metric">
                    <strong>Completed:</strong> {summary.get('completed_examples', 0)}
                </div>
                <div class="metric">
                    <strong>Failed:</strong> {summary.get('failed_examples', 0)}
                </div>
                <div class="metric">
                    <strong>Average Score:</strong> <span class="score">{summary.get('average_score', 0):.3f}</span>
                </div>
                <div class="metric">
                    <strong>Execution Time:</strong> {summary.get('execution_time', 0):.2f}s
                </div>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Example ID</th>
                    <th>Input</th>
                    <th>Prediction</th>
                    <th>Reference</th>
                    <th>Scores</th>
                </tr>
        """
        
        for result in results[:100]:  # Limit to first 100 for HTML
            evaluations = result.get('evaluations', {})
            scores_html = '<br>'.join([f"{k}: {v}" for k, v in evaluations.items() if isinstance(v, (int, float))])
            
            html_content += f"""
                <tr>
                    <td>{result.get('example_id', 'N/A')}</td>
                    <td>{result.get('input', '')[:100]}</td>
                    <td>{result.get('prediction', '')[:100]}</td>
                    <td>{result.get('reference', '')[:100]}</td>
                    <td>{scores_html}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        return {
            'format': 'html',
            'content': html_content,
            'filename': f"evaluation_results_{result_data.get('run_id', 'unknown')}.html"
        }
    
    def add_custom_evaluator(self, name: str, evaluator: LangSmithEvaluator) -> None:
        """Add a custom evaluator to the adapter."""
        self.evaluators[name] = evaluator
        logger.info(f"Added custom evaluator: {name}")
    
    def list_evaluators(self) -> List[str]:
        """List all available evaluators."""
        return list(self.evaluators.keys())
    
    def get_dataset(self, dataset_id: str) -> Optional[LangSmithDataset]:
        """Get dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        return [dataset.to_dict() for dataset in self.datasets.values()]


def format_langsmith_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Format evaluation results in LangSmith-compatible structure."""
    formatted_results = []
    
    for result in results:
        formatted_result = {
            'run_id': result.run_id,
            'prompt': result.prompt,
            'dataset_name': result.dataset_name,
            'summary': result.summary.__dict__ if hasattr(result.summary, '__dict__') else result.summary,
            'results': result.results,
            'created_at': datetime.now().isoformat()
        }
        formatted_results.append(formatted_result)
    
    return {
        'evaluation_results': formatted_results,
        'total_runs': len(results),
        'format_version': '1.0.0',
        'exported_at': datetime.now().isoformat()
    }


def generate_evaluation_report(results: List[EvaluationResult]) -> str:
    """Generate a comprehensive evaluation report."""
    if not results:
        return "No evaluation results to report."
    
    report = []
    report.append("# LangSmith Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Evaluation Runs: {len(results)}")
    report.append("")
    
    # Summary statistics
    total_examples = sum(r.summary.total_examples for r in results)
    total_completed = sum(r.summary.completed_examples for r in results)
    total_failed = sum(r.summary.failed_examples for r in results)
    avg_score = sum(r.summary.average_score for r in results) / len(results)
    
    report.append("## Summary Statistics")
    report.append(f"- Total Examples Evaluated: {total_examples}")
    report.append(f"- Successfully Completed: {total_completed}")
    report.append(f"- Failed Evaluations: {total_failed}")
    report.append(f"- Overall Average Score: {avg_score:.3f}")
    report.append("")
    
    # Individual run details
    report.append("## Individual Run Results")
    for i, result in enumerate(results, 1):
        report.append(f"### Run {i}: {result.summary.name}")
        report.append(f"- Dataset: {result.dataset_name}")
        report.append(f"- Status: {result.summary.status}")
        report.append(f"- Examples: {result.summary.completed_examples}/{result.summary.total_examples}")
        report.append(f"- Average Score: {result.summary.average_score:.3f}")
        report.append(f"- Execution Time: {result.summary.execution_time:.2f}s")
        
        # Evaluator scores
        if result.summary.evaluator_scores:
            report.append("- Evaluator Scores:")
            for evaluator, score in result.summary.evaluator_scores.items():
                report.append(f"  - {evaluator}: {score:.3f}")
        
        report.append("")
    
    # Performance analysis
    report.append("## Performance Analysis")
    execution_times = [r.summary.execution_time for r in results]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        report.append(f"- Average Execution Time: {avg_time:.2f}s")
        report.append(f"- Fastest Run: {min_time:.2f}s")
        report.append(f"- Slowest Run: {max_time:.2f}s")
    
    report.append("")
    report.append("---")
    report.append("Report generated by PromptOpt Co-Pilot LangSmith Adapter")
    
    return "\n".join(report)


# Unit Tests for LangSmith Adapter
def test_exact_match_evaluator():
    """Test exact match evaluator functionality."""
    evaluator = ExactMatchEvaluator()
    
    # Test perfect match
    result = evaluator.evaluate("hello world", "hello world")
    assert result["exact_match_score"] == 1.0
    assert result["exact_match_binary"] is True
    
    # Test case insensitive match
    evaluator_case_insensitive = ExactMatchEvaluator(case_sensitive=False)
    result = evaluator_case_insensitive.evaluate("Hello World", "hello world")
    assert result["exact_match_score"] == 1.0
    
    # Test no match
    result = evaluator.evaluate("hello", "world")
    assert result["exact_match_score"] == 0.0
    assert result["exact_match_binary"] is False
    
    print("✓ ExactMatchEvaluator tests passed")


def test_semantic_similarity_evaluator():
    """Test semantic similarity evaluator functionality."""
    evaluator = SemanticSimilarityEvaluator(threshold=0.8)
    
    # Test similar sentences
    result = evaluator.evaluate("The cat is sleeping", "A cat is resting")
    assert "semantic_similarity_score" in result
    assert isinstance(result["semantic_similarity_score"], float)
    assert 0.0 <= result["semantic_similarity_score"] <= 1.0
    
    print("✓ SemanticSimilarityEvaluator tests passed")


def test_langsmith_dataset():
    """Test LangSmith dataset functionality."""
    # Test dataset creation
    data = [
        {"input": "What is AI?", "output": "Artificial Intelligence"},
        {"input": "Define ML", "output": "Machine Learning"}
    ]
    
    dataset = LangSmithDataset.create_from_json(data, "test_dataset")
    assert dataset.name == "test_dataset"
    assert dataset.size == 2
    assert len(dataset.data) == 2
    
    # Test schema validation
    valid_data = [{"input": "test", "output": "result"}]
    assert LangSmithDataset.validate_dataset_schema(valid_data) is True
    
    invalid_data = [{"no_input": "test"}]
    assert LangSmithDataset.validate_dataset_schema(invalid_data) is False
    
    # Test adding examples
    dataset.add_example({"input": "New question", "output": "New answer"})
    assert dataset.size == 3
    
    print("✓ LangSmithDataset tests passed")


def test_langsmith_adapter_initialization():
    """Test LangSmith adapter initialization."""
    from unittest.mock import Mock
    
    # Mock LlamaWrapper
    mock_llm = Mock()
    mock_llm.generate_response.return_value = "Mocked response"
    
    # Create adapter
    config = LangSmithConfig(batch_size=5, timeout_seconds=60)
    adapter = LangSmithAdapter(mock_llm, config)
    
    assert adapter.llm_wrapper is mock_llm
    assert adapter.config.batch_size == 5
    assert len(adapter.evaluators) >= 3  # At least basic evaluators
    
    # Test evaluator listing
    evaluators = adapter.list_evaluators()
    assert 'exact_match' in evaluators
    assert 'semantic_similarity' in evaluators
    assert 'latency' in evaluators
    
    print("✓ LangSmithAdapter initialization tests passed")


def test_custom_evaluator():
    """Test custom evaluator functionality."""
    class TestEvaluator(LangSmithEvaluator):
        def evaluate(self, prediction: str, reference: str, input_data: Dict[str, Any] = None) -> Dict[str, float]:
            return {"test_score": 0.5, "test_binary": True}
    
    from unittest.mock import Mock
    mock_llm = Mock()
    adapter = LangSmithAdapter(mock_llm)
    
    # Add custom evaluator
    custom_eval = TestEvaluator("test_evaluator")
    adapter.add_custom_evaluator("test", custom_eval)
    
    assert "test" in adapter.evaluators
    assert adapter.evaluators["test"] is custom_eval
    
    print("✓ Custom evaluator tests passed")


def run_all_tests():
    """Run all unit tests for the LangSmith adapter."""
    print("Running LangSmith Adapter Tests...")
    print("=" * 50)
    
    try:
        test_exact_match_evaluator()
        test_semantic_similarity_evaluator()
        test_langsmith_dataset()
        test_langsmith_adapter_initialization()
        test_custom_evaluator()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


# Example usage and integration guide
def example_usage():
    """Example usage of the LangSmith adapter."""
    from unittest.mock import Mock
    
    # Initialize mock LLM wrapper
    mock_llm = Mock()
    mock_llm.generate_response.return_value = "This is a test response"
    
    # Create configuration
    config = LangSmithConfig(
        default_evaluators=['exact_match', 'semantic_similarity'],
        batch_size=5,
        timeout_seconds=120
    )
    
    # Initialize adapter
    adapter = LangSmithAdapter(mock_llm, config)
    
    # Create dataset
    dataset_data = [
        {
            "input": "What is the capital of France?",
            "output": "Paris",
            "metadata": {"category": "geography"}
        },
        {
            "input": "What is 2+2?",
            "output": "4",
            "metadata": {"category": "math"}
        }
    ]
    
    dataset = adapter.create_dataset("example_dataset", dataset_data)
    print(f"Created dataset: {dataset.name} with {dataset.size} examples")
    
    # Single prompt evaluation
    prompt = "Answer the following question: {input}"
    result = adapter.evaluate_prompt(prompt, dataset)
    
    print(f"Evaluation completed:")
    print(f"- Run ID: {result.summary.run_id}")
    print(f"- Average Score: {result.summary.average_score:.3f}")
    print(f"- Completed Examples: {result.summary.completed_examples}")
    
    # Export results
    exported = adapter.export_results(result.summary.run_id, format='json')
    print(f"Results exported in LangSmith format")
    
    # Batch evaluation
    prompts = [
        "Answer this question: {input}",
        "Please respond to: {input}",
        "Here's the answer to '{input}':"
    ]
    
    batch_results = adapter.batch_evaluate(prompts, dataset)
    print(f"Batch evaluation completed for {len(batch_results)} prompts")
    
    # Generate report
    report = generate_evaluation_report(batch_results)
    print("\nGenerated Evaluation Report:")
    print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    # Run tests
    run_all_tests()
    
    # Show example usage
    print("\n" + "=" * 50)
    print("Example Usage:")
    print("=" * 50)
    example_usage()
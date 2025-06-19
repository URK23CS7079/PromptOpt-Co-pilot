"""
Evaluation API Routes for PromptOpt Co-Pilot

This module provides REST API endpoints for managing evaluation runs, datasets,
metrics, and results analysis in the PromptOpt Co-Pilot system.

Author: PromptOpt Co-Pilot Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from uuid import uuid4, UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

# Internal imports
from backend.evaluation.evaluator import PromptEvaluator
from backend.evaluation.metrics import MetricCalculator
from backend.utils.dataset_handler import DatasetHandler
from backend.core.database import get_db_session, get_async_db_session
from backend.core.models import EvaluationRun, Dataset, EvaluationResult
from backend.core.auth import get_current_user, User
from backend.core.rate_limiter import rate_limit
from backend.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Pydantic Models
class MetricConfig(BaseModel):
    """Configuration for evaluation metrics."""
    name: str = Field(..., description="Metric name")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Metric weight")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Metric parameters")
    
    @validator('name')
    def validate_metric_name(cls, v):
        valid_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'bleu', 'rouge', 'perplexity']
        if v not in valid_metrics:
            raise ValueError(f"Invalid metric name. Must be one of: {valid_metrics}")
        return v


class EvaluationRunRequest(BaseModel):
    """Request model for creating evaluation runs."""
    name: str = Field(..., min_length=1, max_length=255, description="Evaluation run name")
    description: Optional[str] = Field(None, max_length=1000, description="Run description")
    prompts: List[str] = Field(..., min_items=1, description="List of prompts to evaluate")
    dataset_id: UUID = Field(..., description="Dataset ID to use for evaluation")
    metrics: List[MetricConfig] = Field(..., min_items=1, description="Metrics configuration")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    batch_size: int = Field(32, ge=1, le=1000, description="Batch size for evaluation")
    timeout: int = Field(3600, ge=60, le=86400, description="Timeout in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Marketing Copy Evaluation",
                "description": "Evaluate marketing copy prompts for effectiveness",
                "prompts": [
                    "Write a compelling product description for {product}",
                    "Create an engaging marketing copy for {product}"
                ],
                "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "metrics": [
                    {"name": "accuracy", "weight": 0.5},
                    {"name": "bleu", "weight": 0.3, "parameters": {"n_gram": 4}},
                    {"name": "rouge", "weight": 0.2}
                ],
                "model_config": {"temperature": 0.7, "max_tokens": 150},
                "batch_size": 16,
                "timeout": 1800
            }
        }


class EvaluationRunResponse(BaseModel):
    """Response model for evaluation run information."""
    id: UUID
    name: str
    description: Optional[str]
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    prompts: List[str]
    dataset_id: UUID
    metrics: List[MetricConfig]
    model_config: Dict[str, Any]
    batch_size: int
    timeout: int
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class EvaluationResultResponse(BaseModel):
    """Response model for evaluation results."""
    run_id: UUID
    overall_score: float = Field(ge=0.0, le=1.0)
    metric_scores: Dict[str, float]
    prompt_results: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    execution_time: float
    sample_count: int
    error_count: int
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "overall_score": 0.85,
                "metric_scores": {
                    "accuracy": 0.89,
                    "bleu": 0.82,
                    "rouge": 0.84
                },
                "prompt_results": [
                    {
                        "prompt": "Write a compelling product description for {product}",
                        "score": 0.87,
                        "metrics": {"accuracy": 0.91, "bleu": 0.84, "rouge": 0.86}
                    }
                ],
                "summary_statistics": {
                    "mean_score": 0.85,
                    "std_score": 0.12,
                    "min_score": 0.72,
                    "max_score": 0.94
                },
                "execution_time": 245.5,
                "sample_count": 1000,
                "error_count": 3,
                "warnings": ["Some samples failed validation"]
            }
        }


class DatasetUploadRequest(BaseModel):
    """Request model for dataset upload."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    format: str = Field(..., regex="^(csv|json|jsonl)$")
    schema_config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Product Descriptions Dataset",
                "description": "Dataset containing product information and descriptions",
                "format": "csv",
                "schema_config": {
                    "input_column": "product_info",
                    "output_column": "description",
                    "delimiter": ","
                }
            }
        }


class DatasetResponse(BaseModel):
    """Response model for dataset information."""
    id: UUID
    name: str
    description: Optional[str]
    format: str
    size: int
    created_at: datetime
    updated_at: datetime
    schema_info: Dict[str, Any]
    statistics: Dict[str, Any]
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ComparisonRequest(BaseModel):
    """Request model for result comparison."""
    run_ids: List[UUID] = Field(..., min_items=2, max_items=10)
    comparison_metrics: List[str] = Field(default_factory=list)
    include_statistical_tests: bool = Field(True)
    
    class Config:
        schema_extra = {
            "example": {
                "run_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400-e29b-41d4-a716-446655440001"
                ],
                "comparison_metrics": ["accuracy", "bleu", "rouge"],
                "include_statistical_tests": True
            }
        }


class ComparisonResponse(BaseModel):
    """Response model for comparison results."""
    comparison_id: UUID
    run_ids: List[UUID]
    comparison_summary: Dict[str, Any]
    metric_comparisons: Dict[str, Dict[str, Any]]
    statistical_tests: Optional[Dict[str, Any]]
    recommendations: List[str]
    visualization_data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "comparison_id": "550e8400-e29b-41d4-a716-446655440002",
                "run_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400-e29b-41d4-a716-446655440001"
                ],
                "comparison_summary": {
                    "best_run": "550e8400-e29b-41d4-a716-446655440000",
                    "performance_difference": 0.05,
                    "statistical_significance": True
                },
                "metric_comparisons": {
                    "accuracy": {
                        "run_1": 0.89,
                        "run_2": 0.84,
                        "difference": 0.05,
                        "better": "run_1"
                    }
                },
                "statistical_tests": {
                    "t_test": {"p_value": 0.03, "significant": True}
                },
                "recommendations": [
                    "Run 1 shows significantly better performance",
                    "Consider using prompt from Run 1 for production"
                ],
                "visualization_data": {
                    "chart_type": "bar",
                    "data": []
                }
            }
        }


class MetricInfo(BaseModel):
    """Information about available metrics."""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    requirements: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "name": "bleu",
                "description": "BLEU score for text generation quality",
                "category": "text_generation",
                "parameters": {
                    "n_gram": {"type": "int", "default": 4, "range": [1, 4]},
                    "smoothing": {"type": "bool", "default": True}
                },
                "requirements": ["reference_text", "generated_text"]
            }
        }


# Global variables for managing evaluation runs
active_evaluations: Dict[UUID, asyncio.Task] = {}
evaluation_progress: Dict[UUID, Dict[str, Any]] = {}


# Utility functions
async def get_evaluation_service() -> PromptEvaluator:
    """Get evaluation service instance."""
    return PromptEvaluator()


async def get_metric_calculator() -> MetricCalculator:
    """Get metric calculator instance."""
    return MetricCalculator()


async def get_dataset_handler() -> DatasetHandler:
    """Get dataset handler instance."""
    return DatasetHandler()


def validate_dataset_exists(dataset_id: UUID, db: Session) -> Dataset:
    """Validate that dataset exists and return it."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


def validate_evaluation_run_exists(run_id: UUID, db: Session) -> EvaluationRun:
    """Validate that evaluation run exists and return it."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Evaluation run {run_id} not found")
    return run


async def run_evaluation_background(
    run_id: UUID,
    request: EvaluationRunRequest,
    dataset: Dataset,
    user_id: UUID,
    db_session: AsyncSession
):
    """Background task for running evaluation."""
    try:
        logger.info(f"Starting evaluation run {run_id}")
        
        # Update run status
        evaluation_progress[run_id] = {
            "status": "running",
            "progress": 0.0,
            "started_at": datetime.utcnow(),
            "current_step": "initialization"
        }
        
        # Initialize evaluator
        evaluator = await get_evaluation_service()
        metric_calculator = await get_metric_calculator()
        dataset_handler = await get_dataset_handler()
        
        # Load dataset
        evaluation_progress[run_id]["current_step"] = "loading_dataset"
        evaluation_progress[run_id]["progress"] = 10.0
        
        dataset_data = await dataset_handler.load_dataset(dataset.file_path)
        
        # Prepare evaluation
        evaluation_progress[run_id]["current_step"] = "preparing_evaluation"
        evaluation_progress[run_id]["progress"] = 20.0
        
        # Configure metrics
        configured_metrics = []
        for metric_config in request.metrics:
            metric = await metric_calculator.configure_metric(
                metric_config.name,
                metric_config.parameters
            )
            configured_metrics.append((metric, metric_config.weight))
        
        # Run evaluation
        evaluation_progress[run_id]["current_step"] = "running_evaluation"
        evaluation_progress[run_id]["progress"] = 30.0
        
        results = await evaluator.evaluate_prompts(
            prompts=request.prompts,
            dataset=dataset_data,
            metrics=configured_metrics,
            model_config=request.model_config,
            batch_size=request.batch_size,
            progress_callback=lambda p: update_progress(run_id, 30.0 + (p * 0.6))
        )
        
        # Calculate final scores
        evaluation_progress[run_id]["current_step"] = "calculating_scores"
        evaluation_progress[run_id]["progress"] = 90.0
        
        final_results = await metric_calculator.calculate_final_scores(
            results, configured_metrics
        )
        
        # Save results to database
        evaluation_progress[run_id]["current_step"] = "saving_results"
        evaluation_progress[run_id]["progress"] = 95.0
        
        async with db_session() as db:
            # Update run status
            run = await db.get(EvaluationRun, run_id)
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            
            # Save results
            result = EvaluationResult(
                run_id=run_id,
                results=final_results,
                created_at=datetime.utcnow()
            )
            db.add(result)
            await db.commit()
        
        # Update progress
        evaluation_progress[run_id].update({
            "status": "completed",
            "progress": 100.0,
            "completed_at": datetime.utcnow(),
            "current_step": "completed"
        })
        
        logger.info(f"Evaluation run {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation run {run_id} failed: {str(e)}")
        
        # Update run status
        evaluation_progress[run_id] = {
            "status": "failed",
            "progress": 0.0,
            "error": str(e),
            "failed_at": datetime.utcnow()
        }
        
        # Update database
        async with db_session() as db:
            run = await db.get(EvaluationRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = str(e)
                await db.commit()
    
    finally:
        # Clean up active evaluation
        if run_id in active_evaluations:
            del active_evaluations[run_id]


def update_progress(run_id: UUID, progress: float):
    """Update evaluation progress."""
    if run_id in evaluation_progress:
        evaluation_progress[run_id]["progress"] = progress


# API Endpoints

@router.post("/runs", response_model=EvaluationRunResponse)
@rate_limit(max_requests=10, window_seconds=60)
async def create_evaluation_run(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    db_async: AsyncSession = Depends(get_async_db_session)
):
    """
    Create a new evaluation run.
    
    This endpoint creates a new evaluation run with the specified prompts,
    dataset, and metrics configuration. The evaluation runs asynchronously
    in the background.
    """
    try:
        # Validate dataset exists
        dataset = validate_dataset_exists(request.dataset_id, db)
        
        # Create evaluation run
        run_id = UUID(str(uuid4()))
        
        # Estimate completion time
        estimated_time = timedelta(
            seconds=len(request.prompts) * dataset.size * 0.1  # Rough estimate
        )
        estimated_completion = datetime.utcnow() + estimated_time
        
        # Create database record
        evaluation_run = EvaluationRun(
            id=run_id,
            name=request.name,
            description=request.description,
            status="pending",
            user_id=current_user.id,
            prompts=request.prompts,
            dataset_id=request.dataset_id,
            metrics_config=[metric.dict() for metric in request.metrics],
            model_config=request.model_config,
            batch_size=request.batch_size,
            timeout=request.timeout,
            created_at=datetime.utcnow(),
            estimated_completion=estimated_completion
        )
        
        db.add(evaluation_run)
        db.commit()
        
        # Start background evaluation
        task = background_tasks.add_task(
            run_evaluation_background,
            run_id,
            request,
            dataset,
            current_user.id,
            db_async
        )
        active_evaluations[run_id] = task
        
        # Initialize progress tracking
        evaluation_progress[run_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.utcnow()
        }
        
        # Prepare response
        response = EvaluationRunResponse(
            id=run_id,
            name=request.name,
            description=request.description,
            status="pending",
            progress=0.0,
            created_at=evaluation_run.created_at,
            started_at=None,
            completed_at=None,
            estimated_completion=estimated_completion,
            prompts=request.prompts,
            dataset_id=request.dataset_id,
            metrics=request.metrics,
            model_config=request.model_config,
            batch_size=request.batch_size,
            timeout=request.timeout
        )
        
        logger.info(f"Created evaluation run {run_id} for user {current_user.id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create evaluation run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation run: {str(e)}")


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get details of a specific evaluation run.
    
    Returns the current status, progress, and details of the evaluation run.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get current progress
        progress_info = evaluation_progress.get(run_id, {})
        
        # Prepare response
        response = EvaluationRunResponse(
            id=run.id,
            name=run.name,
            description=run.description,
            status=progress_info.get("status", run.status),
            progress=progress_info.get("progress", 0.0),
            created_at=run.created_at,
            started_at=progress_info.get("started_at", run.started_at),
            completed_at=progress_info.get("completed_at", run.completed_at),
            estimated_completion=run.estimated_completion,
            prompts=run.prompts,
            dataset_id=run.dataset_id,
            metrics=[MetricConfig(**config) for config in run.metrics_config],
            model_config=run.model_config,
            batch_size=run.batch_size,
            timeout=run.timeout,
            resource_usage=progress_info.get("resource_usage", {}),
            error_message=progress_info.get("error", run.error_message)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation run: {str(e)}")


@router.get("/runs/{run_id}/results", response_model=EvaluationResultResponse)
async def get_evaluation_results(
    run_id: UUID,
    format: str = Query("summary", regex="^(summary|detailed|export)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get results of a completed evaluation run.
    
    Returns detailed evaluation results including metrics, scores, and analysis.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if run is completed
        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Evaluation run is not completed. Status: {run.status}"
            )
        
        # Get results from database
        result = db.query(EvaluationResult).filter(
            EvaluationResult.run_id == run_id
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Format results based on requested format
        if format == "export":
            # Return raw results for export
            return StreamingResponse(
                iter([str(result.results)]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=evaluation_results_{run_id}.json"}
            )
        
        # Prepare response
        response = EvaluationResultResponse(
            run_id=run_id,
            overall_score=result.results.get("overall_score", 0.0),
            metric_scores=result.results.get("metric_scores", {}),
            prompt_results=result.results.get("prompt_results", []),
            summary_statistics=result.results.get("summary_statistics", {}),
            execution_time=result.results.get("execution_time", 0.0),
            sample_count=result.results.get("sample_count", 0),
            error_count=result.results.get("error_count", 0),
            warnings=result.results.get("warnings", [])
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation results for {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation results: {str(e)}")


@router.post("/runs/{run_id}/stop")
async def stop_evaluation_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Stop a running evaluation run.
    
    Gracefully stops the evaluation and returns partial results if available.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if run can be stopped
        if run.status not in ["pending", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop evaluation run with status: {run.status}"
            )
        
        # Stop the background task
        if run_id in active_evaluations:
            task = active_evaluations[run_id]
            task.cancel()
            del active_evaluations[run_id]
        
        # Update run status
        run.status = "stopped"
        run.completed_at = datetime.utcnow()
        db.commit()
        
        # Update progress
        if run_id in evaluation_progress:
            evaluation_progress[run_id]["status"] = "stopped"
            evaluation_progress[run_id]["stopped_at"] = datetime.utcnow()
        
        logger.info(f"Stopped evaluation run {run_id}")
        
        return {"message": f"Evaluation run {run_id} stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop evaluation run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop evaluation run: {str(e)}")


@router.get("/runs", response_model=List[EvaluationRunResponse])
async def list_evaluation_runs(
    status: Optional[str] = Query(None, regex="^(pending|running|completed|failed|stopped)$"),
    dataset_id: Optional[UUID] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", regex="^(created_at|name|status|progress)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List evaluation runs with filtering and pagination.
    
    Returns a paginated list of evaluation runs belonging to the current user.
    """
    try:
        # Build query
        query = db.query(EvaluationRun).filter(
            EvaluationRun.user_id == current_user.id
        )
        
        # Apply filters
        if status:
            query = query.filter(EvaluationRun.status == status)
        
        if dataset_id:
            query = query.filter(EvaluationRun.dataset_id == dataset_id)
        
        if start_date:
            query = query.filter(EvaluationRun.created_at >= start_date)
        
        if end_date:
            query = query.filter(EvaluationRun.created_at <= end_date)
        
        # Apply sorting
        if sort_by == "created_at":
            order_col = EvaluationRun.created_at
        elif sort_by == "name":
            order_col = EvaluationRun.name
        elif sort_by == "status":
            order_col = EvaluationRun.status
        else:
            order_col = EvaluationRun.created_at
        
        if sort_order == "desc":
            query = query.order_by(order_col.desc())
        else:
            query = query.order_by(order_col.asc())
        
        # Apply pagination
        offset = (page - 1) * page_size
        runs = query.offset(offset).limit(page_size).all()
        
        # Prepare response
        response = []
        for run in runs:
            progress_info = evaluation_progress.get(run.id, {})
            
            run_response = EvaluationRunResponse(
                id=run.id,
                name=run.name,
                description=run.description,
                status=progress_info.get("status", run.status),
                progress=progress_info.get("progress", 0.0),
                created_at=run.created_at,
                started_at=progress_info.get("started_at", run.started_at),
                completed_at=progress_info.get("completed_at", run.completed_at),
                estimated_completion=run.estimated_completion,
                prompts=run.prompts,
                dataset_id=run.dataset_id,
                metrics=[MetricConfig(**config) for config in run.metrics_config],
                model_config=run.model_config,
                batch_size=run.batch_size,
                timeout=run.timeout,
                resource_usage=progress_info.get("resource_usage", {}),
                error_message=progress_info.get("error", run.error_message)
            )
            response.append(run_response)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list evaluation runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list evaluation runs: {str(e)}")


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete an evaluation run and its associated results.
    
    Permanently removes the evaluation run and all associated data.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Stop if running
        if run.status in ["pending", "running"] and run_id in active_evaluations:
            task = active_evaluations[run_id]
            task.cancel()
            del active_evaluations[run_id]
        
        # Delete results
        db.query(EvaluationResult).filter(
            EvaluationResult.run_id == run_id
        ).delete()
        
        # Delete run
        db.delete(run)
        db.commit()
        
        # Clean up progress tracking
        if run_id in evaluation_progress:
            del evaluation_progress[run_id]
        
        logger.info(f"Deleted evaluation run {run_id}")
        
        return {"message": f"Evaluation run {run_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation run: {str(e)}")


@router.post("/datasets", response_model=DatasetResponse)
@rate_limit(max_requests=5, window_seconds=60)
async def upload_dataset(
    file: UploadFile = File(...),
    request: DatasetUploadRequest = Depends(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Upload a new evaluation dataset.
    
    Accepts CSV, JSON, or JSONL files and validates the dataset structure.
    """
    try:
        # Validate file format
        if not file.filename.endswith(('.csv', '.json', '.jsonl')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only CSV, JSON, and JSONL files are supported."
            )
        
        # Validate file size (max 100MB)
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 100MB limit"
            )
        
        # Initialize dataset handler
        dataset_handler = await get_dataset_handler()
        
        # Validate and process dataset
        dataset_info = await dataset_handler.validate_dataset(
            content,
            request.format,
            request.schema_config
        )
        
        # Save dataset file
        dataset_id = UUID(str(uuid4()))
        file_path = await dataset_handler.save_dataset(
            dataset_id,
            content,
            request.format
        )
        
        # Create database record
        dataset = Dataset(
            id=dataset_id,
            name=request.name,
            description=request.description,
            format=request.format,
            file_path=file_path,
            size=dataset_info["size"],
            schema_info=dataset_info["schema"],
            statistics=dataset_info["statistics"],
            user_id=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(dataset)
        db.commit()
        
        # Prepare response
        response = DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            format=dataset.format,
            size=dataset.size,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            schema_info=dataset.schema_info,
            statistics=dataset.statistics,
            sample_data=dataset_info.get("sample_data", [])
        )
        
        logger.info(f"Uploaded dataset {dataset_id} for user {current_user.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", regex="^(created_at|name|size)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    List available datasets.
    
    Returns a paginated list of datasets belonging to the current user.
    """
    try:
        # Build query
        query = db.query(Dataset).filter(Dataset.user_id == current_user.id)
        
        # Apply sorting
        if sort_by == "created_at":
            order_col = Dataset.created_at
        elif sort_by == "name":
            order_col = Dataset.name
        elif sort_by == "size":
            order_col = Dataset.size
        else:
            order_col = Dataset.created_at
        
        if sort_order == "desc":
            query = query.order_by(order_col.desc())
        else:
            query = query.order_by(order_col.asc())
        
        # Apply pagination
        offset = (page - 1) * page_size
        datasets = query.offset(offset).limit(page_size).all()
        
        # Prepare response
        response = []
        for dataset in datasets:
            dataset_response = DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                format=dataset.format,
                size=dataset.size,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
                schema_info=dataset.schema_info,
                statistics=dataset.statistics
            )
            response.append(dataset_response)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset_details(
    dataset_id: UUID,
    include_sample: bool = Query(True),
    sample_size: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get detailed information about a specific dataset.
    
    Returns dataset metadata, schema information, and optionally sample data.
    """
    try:
        # Get dataset from database
        dataset = validate_dataset_exists(dataset_id, db)
        
        # Check user permission
        if dataset.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get sample data if requested
        sample_data = []
        if include_sample:
            dataset_handler = await get_dataset_handler()
            sample_data = await dataset_handler.get_sample_data(
                dataset.file_path,
                dataset.format,
                sample_size
            )
        
        # Prepare response
        response = DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            format=dataset.format,
            size=dataset.size,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            schema_info=dataset.schema_info,
            statistics=dataset.statistics,
            sample_data=sample_data
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {str(e)}")


@router.post("/compare", response_model=ComparisonResponse)
@rate_limit(max_requests=5, window_seconds=60)
async def compare_evaluation_results(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Compare results from multiple evaluation runs.
    
    Generates detailed comparison analysis with statistical tests and recommendations.
    """
    try:
        # Validate all runs exist and belong to user
        runs = []
        results = []
        
        for run_id in request.run_ids:
            run = validate_evaluation_run_exists(run_id, db)
            
            # Check user permission
            if run.user_id != current_user.id:
                raise HTTPException(status_code=403, detail=f"Access denied to run {run_id}")
            
            # Check if run is completed
            if run.status != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Run {run_id} is not completed. Status: {run.status}"
                )
            
            # Get results
            result = db.query(EvaluationResult).filter(
                EvaluationResult.run_id == run_id
            ).first()
            
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Results not found for run {run_id}"
                )
            
            runs.append(run)
            results.append(result)
        
        # Initialize metric calculator for comparison
        metric_calculator = await get_metric_calculator()
        
        # Perform comparison analysis
        comparison_id = UUID(str(uuid4()))
        
        comparison_analysis = await metric_calculator.compare_results(
            results,
            request.comparison_metrics,
            request.include_statistical_tests
        )
        
        # Generate recommendations
        recommendations = await metric_calculator.generate_recommendations(
            comparison_analysis
        )
        
        # Prepare visualization data
        visualization_data = await metric_calculator.prepare_visualization_data(
            comparison_analysis
        )
        
        # Prepare response
        response = ComparisonResponse(
            comparison_id=comparison_id,
            run_ids=request.run_ids,
            comparison_summary=comparison_analysis.get("summary", {}),
            metric_comparisons=comparison_analysis.get("metric_comparisons", {}),
            statistical_tests=comparison_analysis.get("statistical_tests") if request.include_statistical_tests else None,
            recommendations=recommendations,
            visualization_data=visualization_data
        )
        
        logger.info(f"Generated comparison {comparison_id} for runs {request.run_ids}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare evaluation results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare results: {str(e)}")


@router.get("/metrics", response_model=List[MetricInfo])
async def list_available_metrics():
    """
    List all available evaluation metrics.
    
    Returns information about supported metrics, their parameters, and requirements.
    """
    try:
        # Initialize metric calculator
        metric_calculator = await get_metric_calculator()
        
        # Get available metrics
        metrics = await metric_calculator.get_available_metrics()
        
        # Prepare response
        response = []
        for metric_name, metric_info in metrics.items():
            metric_response = MetricInfo(
                name=metric_name,
                description=metric_info.get("description", ""),
                category=metric_info.get("category", "general"),
                parameters=metric_info.get("parameters", {}),
                requirements=metric_info.get("requirements", [])
            )
            response.append(metric_response)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list available metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list metrics: {str(e)}")


# Additional utility endpoints

@router.get("/runs/{run_id}/logs")
async def get_evaluation_logs(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get execution logs for an evaluation run.
    
    Returns detailed logs and debugging information for the evaluation run.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get logs (implementation depends on logging system)
        logs = []  # This would be implemented based on your logging infrastructure
        
        return {
            "run_id": run_id,
            "logs": logs,
            "log_level": "INFO",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation logs for {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation logs: {str(e)}")


@router.post("/runs/{run_id}/export")
async def export_evaluation_results(
    run_id: UUID,
    format: str = Query("json", regex="^(json|csv|xlsx)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Export evaluation results in various formats.
    
    Supports JSON, CSV, and Excel export formats.
    """
    try:
        # Get run from database
        run = validate_evaluation_run_exists(run_id, db)
        
        # Check user permission
        if run.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if run is completed
        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Evaluation run is not completed. Status: {run.status}"
            )
        
        # Get results
        result = db.query(EvaluationResult).filter(
            EvaluationResult.run_id == run_id
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Results not found")
        
        # Initialize dataset handler for export
        dataset_handler = await get_dataset_handler()
        
        # Export results
        exported_data = await dataset_handler.export_results(
            result.results,
            format
        )
        
        # Determine content type and filename
        if format == "json":
            content_type = "application/json"
            filename = f"evaluation_results_{run_id}.json"
        elif format == "csv":
            content_type = "text/csv"
            filename = f"evaluation_results_{run_id}.csv"
        elif format == "xlsx":
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"evaluation_results_{run_id}.xlsx"
        
        return StreamingResponse(
            iter([exported_data]),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export evaluation results for {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export results: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the evaluation service.
    
    Returns the current status of the evaluation system.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "active_evaluations": len(active_evaluations),
            "service": "evaluation_api",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.error(f"Validation error: {str(exc)}")
    return HTTPException(status_code=400, detail=str(exc))


@router.exception_handler(asyncio.TimeoutError)
async def timeout_error_handler(request, exc):
    """Handle timeout errors."""
    logger.error(f"Timeout error: {str(exc)}")
    return HTTPException(status_code=408, detail="Request timeout")


# Cleanup function (should be called on shutdown)
async def cleanup_evaluation_routes():
    """Clean up active evaluations on shutdown."""
    logger.info("Cleaning up active evaluations...")
    
    for run_id, task in active_evaluations.items():
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error cleaning up evaluation {run_id}: {str(e)}")
    
    active_evaluations.clear()
    evaluation_progress.clear()
    
    logger.info("Evaluation cleanup completed")


# Include router in main application
# app.include_router(router)

if __name__ == "__main__":
    # This would be used for testing the module
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="PromptOpt Co-Pilot Evaluation API")
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
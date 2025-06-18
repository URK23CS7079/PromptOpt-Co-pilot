"""
Optimization API Routes for PromptOpt Co-Pilot

This module provides REST API endpoints for managing prompt optimization workflows,
including starting optimization runs, monitoring progress, retrieving results,
and managing optimization history.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
import psutil

# Internal imports
from backend.optimization.dspy_optimizer import DSPyOptimizer
from backend.optimization.ape_engine import APEEngine
from backend.evaluation.evaluator import PromptEvaluator
from backend.core.database import DatabaseManager
from backend.llm.model_manager import ModelManager
from backend.core.exceptions import (
    OptimizationError,
    ResourceError,
    ValidationError,
    NotFoundError
)
from backend.core.config import settings
from backend.core.rate_limiter import RateLimiter
from backend.core.websocket_manager import WebSocketManager

# Configure logging
logger = logging.getLogger(__name__)

# Global state management
active_jobs: Dict[str, Dict] = {}
job_results: Dict[str, Dict] = {}
optimization_queue = asyncio.Queue()

router = APIRouter(prefix="/optimization", tags=["optimization"])

# Rate limiter setup
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# =====================================================
# Enums and Constants
# =====================================================

class OptimizationStatus(str, Enum):
    """Status enum for optimization jobs"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELLING = "cancelling"

class OptimizationMethod(str, Enum):
    """Available optimization methods"""
    DSPY = "dspy"
    APE = "ape"
    HYBRID = "hybrid"

class ExecutionMode(str, Enum):
    """Execution mode for optimization"""
    SYNC = "sync"
    ASYNC = "async"

# =====================================================
# Pydantic Models
# =====================================================

class OptimizationConfig(BaseModel):
    """Configuration parameters for optimization"""
    method: OptimizationMethod = Field(
        default=OptimizationMethod.DSPY,
        description="Optimization method to use"
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of optimization iterations"
    )
    population_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Population size for evolutionary methods"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for evolutionary optimization"
    )
    early_stopping_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Early stopping threshold for convergence"
    )
    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Patience for early stopping"
    )
    timeout_minutes: int = Field(
        default=60,
        ge=1,
        le=480,
        description="Maximum optimization time in minutes"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers"
    )

class DatasetConfig(BaseModel):
    """Dataset configuration for optimization"""
    dataset_id: Optional[str] = Field(
        None,
        description="ID of existing dataset"
    )
    dataset_content: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Inline dataset content"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Fraction of data for validation"
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle dataset before splitting"
    )
    
    @root_validator
    def validate_dataset_source(cls, values):
        dataset_id = values.get('dataset_id')
        dataset_content = values.get('dataset_content')
        
        if not dataset_id and not dataset_content:
            raise ValueError("Either dataset_id or dataset_content must be provided")
        
        if dataset_id and dataset_content:
            raise ValueError("Cannot specify both dataset_id and dataset_content")
        
        if dataset_content and len(dataset_content) < 5:
            raise ValueError("Inline dataset must contain at least 5 examples")
        
        return values

class OptimizationRequest(BaseModel):
    """Request model for starting optimization"""
    base_prompt: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Base prompt to optimize"
    )
    dataset_config: DatasetConfig = Field(
        ...,
        description="Dataset configuration"
    )
    optimization_config: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization parameters"
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.ASYNC,
        description="Execution mode (sync/async)"
    )
    model_name: str = Field(
        default="llama2-7b",
        description="LLM model to use for optimization"
    )
    evaluation_metrics: List[str] = Field(
        default=["accuracy", "relevance"],
        description="Metrics to optimize for"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @validator('base_prompt')
    def validate_base_prompt(cls, v):
        if not v.strip():
            raise ValueError("Base prompt cannot be empty")
        return v.strip()
    
    @validator('evaluation_metrics')
    def validate_metrics(cls, v):
        allowed_metrics = {
            "accuracy", "relevance", "coherence", "fluency",
            "diversity", "novelty", "semantic_similarity"
        }
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
        return v

class BatchOptimizationRequest(BaseModel):
    """Request model for batch optimization"""
    requests: List[OptimizationRequest] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of optimization requests"
    )
    shared_config: Optional[OptimizationConfig] = Field(
        None,
        description="Shared configuration for all requests"
    )
    batch_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Batch-level metadata"
    )

class OptimizationProgress(BaseModel):
    """Progress information for optimization"""
    current_iteration: int = Field(..., description="Current iteration number")
    total_iterations: int = Field(..., description="Total planned iterations")
    progress_percentage: float = Field(..., description="Progress as percentage")
    best_score: Optional[float] = Field(None, description="Best score achieved so far")
    current_score: Optional[float] = Field(None, description="Current iteration score")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage stats")

class OptimizationStatusResponse(BaseModel):
    """Response model for optimization status"""
    job_id: str = Field(..., description="Unique job identifier")
    status: OptimizationStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    progress: Optional[OptimizationProgress] = Field(None, description="Progress information")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Job metadata")

class PromptVariant(BaseModel):
    """Individual prompt variant result"""
    variant_id: str = Field(..., description="Unique variant identifier")
    prompt_text: str = Field(..., description="Optimized prompt text")
    score: float = Field(..., description="Overall performance score")
    metrics: Dict[str, float] = Field(..., description="Individual metric scores")
    generation_method: str = Field(..., description="Method used to generate variant")
    iteration: int = Field(..., description="Iteration when variant was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Variant metadata")

class OptimizationResult(BaseModel):
    """Complete optimization results"""
    job_id: str = Field(..., description="Job identifier")
    base_prompt: str = Field(..., description="Original base prompt")
    best_variant: PromptVariant = Field(..., description="Best performing variant")
    top_variants: List[PromptVariant] = Field(..., description="Top N performing variants")
    optimization_history: List[Dict[str, Any]] = Field(..., description="Iteration history")
    final_metrics: Dict[str, float] = Field(..., description="Final aggregated metrics")
    total_iterations: int = Field(..., description="Total iterations completed")
    total_variants_generated: int = Field(..., description="Total variants generated")
    optimization_time: float = Field(..., description="Total optimization time in seconds")
    convergence_achieved: bool = Field(..., description="Whether optimization converged")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

class OptimizationHistoryItem(BaseModel):
    """Historical optimization run summary"""
    job_id: str = Field(..., description="Job identifier")
    base_prompt: str = Field(..., description="Base prompt (truncated)")
    status: OptimizationStatus = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    best_score: Optional[float] = Field(None, description="Best achieved score")
    total_variants: int = Field(default=0, description="Total variants generated")
    optimization_method: str = Field(..., description="Optimization method used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Job metadata")

class OptimizationHistoryResponse(BaseModel):
    """Response model for optimization history"""
    items: List[OptimizationHistoryItem] = Field(..., description="History items")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether more pages exist")

# =====================================================
# Dependencies
# =====================================================

async def get_database() -> DatabaseManager:
    """Get database manager instance"""
    return DatabaseManager()

async def get_model_manager() -> ModelManager:
    """Get model manager instance"""
    return ModelManager()

async def get_websocket_manager() -> WebSocketManager:
    """Get websocket manager instance"""
    return WebSocketManager()

async def check_rate_limit(request_id: str = None):
    """Check rate limiting"""
    if not await rate_limiter.check_limit(request_id or "default"):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

async def check_system_resources():
    """Check if system has sufficient resources for optimization"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    if cpu_percent > 90:
        raise HTTPException(
            status_code=503,
            detail="System CPU usage too high. Please try again later."
        )
    
    if memory.percent > 85:
        raise HTTPException(
            status_code=503,
            detail="System memory usage too high. Please try again later."
        )
    
    if disk.percent > 95:
        raise HTTPException(
            status_code=503,
            detail="System disk usage too high. Please try again later."
        )

# =====================================================
# Helper Functions
# =====================================================

def generate_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())

async def validate_model_availability(model_name: str, model_manager: ModelManager):
    """Validate that the requested model is available"""
    available_models = await model_manager.list_available_models()
    if model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not available. "
                   f"Available models: {', '.join(available_models)}"
        )

async def get_resource_usage() -> Dict[str, Any]:
    """Get current system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "active_jobs": len([j for j in active_jobs.values() 
                          if j["status"] == OptimizationStatus.RUNNING])
    }

async def cleanup_job_resources(job_id: str):
    """Cleanup resources associated with a job"""
    try:
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
            
            # Cancel any running tasks
            if "task" in job_info:
                job_info["task"].cancel()
            
            # Update status
            job_info["status"] = OptimizationStatus.STOPPED
            job_info["completed_at"] = datetime.utcnow()
            
            # Move to results if needed
            if job_id not in job_results:
                job_results[job_id] = job_info
            
            # Remove from active jobs
            del active_jobs[job_id]
            
        logger.info(f"Cleaned up resources for job {job_id}")
    except Exception as e:
        logger.error(f"Error cleaning up job {job_id}: {e}")

# =====================================================
# Optimization Engine Functions
# =====================================================

async def run_optimization(
    job_id: str,
    request: OptimizationRequest,
    db: DatabaseManager,
    model_manager: ModelManager,
    websocket_manager: WebSocketManager
):
    """Run the actual optimization process"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = OptimizationStatus.RUNNING
        active_jobs[job_id]["started_at"] = datetime.utcnow()
        
        # Initialize components
        if request.optimization_config.method == OptimizationMethod.DSPY:
            optimizer = DSPyOptimizer(
                model_manager=model_manager,
                config=request.optimization_config.dict()
            )
        elif request.optimization_config.method == OptimizationMethod.APE:
            optimizer = APEEngine(
                model_manager=model_manager,
                config=request.optimization_config.dict()
            )
        else:  # HYBRID
            optimizer = DSPyOptimizer(  # Default to DSPy for hybrid
                model_manager=model_manager,
                config=request.optimization_config.dict()
            )
        
        evaluator = PromptEvaluator(
            model_manager=model_manager,
            metrics=request.evaluation_metrics
        )
        
        # Load dataset
        if request.dataset_config.dataset_id:
            dataset = await db.load_dataset(request.dataset_config.dataset_id)
        else:
            dataset = request.dataset_config.dataset_content
        
        # Run optimization with progress callbacks
        async def progress_callback(iteration: int, best_score: float, current_score: float):
            progress = OptimizationProgress(
                current_iteration=iteration,
                total_iterations=request.optimization_config.max_iterations,
                progress_percentage=(iteration / request.optimization_config.max_iterations) * 100,
                best_score=best_score,
                current_score=current_score,
                estimated_completion=datetime.utcnow() + timedelta(
                    minutes=((request.optimization_config.max_iterations - iteration) * 2)
                ),
                resource_usage=await get_resource_usage()
            )
            
            active_jobs[job_id]["progress"] = progress
            
            # Send WebSocket update
            await websocket_manager.send_to_job(job_id, {
                "type": "progress_update",
                "data": progress.dict()
            })
        
        # Execute optimization
        results = await optimizer.optimize(
            base_prompt=request.base_prompt,
            dataset=dataset,
            evaluator=evaluator,
            progress_callback=progress_callback
        )
        
        # Process results
        variants = []
        for i, (prompt, score, metrics) in enumerate(results.get("variants", [])):
            variant = PromptVariant(
                variant_id=f"{job_id}-variant-{i}",
                prompt_text=prompt,
                score=score,
                metrics=metrics,
                generation_method=request.optimization_config.method.value,
                iteration=metrics.get("iteration", 0),
                metadata={}
            )
            variants.append(variant)
        
        # Create final result
        optimization_result = OptimizationResult(
            job_id=job_id,
            base_prompt=request.base_prompt,
            best_variant=variants[0] if variants else None,
            top_variants=variants[:10],
            optimization_history=results.get("history", []),
            final_metrics=results.get("final_metrics", {}),
            total_iterations=results.get("total_iterations", 0),
            total_variants_generated=len(variants),
            optimization_time=results.get("optimization_time", 0),
            convergence_achieved=results.get("converged", False),
            metadata=request.metadata
        )
        
        # Store results
        job_results[job_id] = {
            "status": OptimizationStatus.COMPLETED,
            "result": optimization_result,
            "completed_at": datetime.utcnow()
        }
        
        # Update active job
        active_jobs[job_id]["status"] = OptimizationStatus.COMPLETED
        active_jobs[job_id]["completed_at"] = datetime.utcnow()
        
        # Save to database
        await db.save_optimization_result(job_id, optimization_result.dict())
        
        # Send completion notification
        await websocket_manager.send_to_job(job_id, {
            "type": "optimization_complete",
            "data": {"job_id": job_id, "status": "completed"}
        })
        
        logger.info(f"Optimization job {job_id} completed successfully")
        
    except asyncio.CancelledError:
        logger.info(f"Optimization job {job_id} was cancelled")
        active_jobs[job_id]["status"] = OptimizationStatus.STOPPED
        raise
    except Exception as e:
        logger.error(f"Optimization job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = OptimizationStatus.FAILED
        active_jobs[job_id]["error_message"] = str(e)
        active_jobs[job_id]["completed_at"] = datetime.utcnow()
        
        # Send error notification
        await websocket_manager.send_to_job(job_id, {
            "type": "optimization_error",
            "data": {"job_id": job_id, "error": str(e)}
        })
        
        raise

# =====================================================
# API Endpoints  
# =====================================================

@router.post("/start", response_model=Dict[str, str])
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_database),
    model_manager: ModelManager = Depends(get_model_manager),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    _: None = Depends(check_rate_limit),
    __: None = Depends(check_system_resources)
):
    """
    Start a new optimization run
    
    This endpoint initiates a prompt optimization workflow with the specified
    parameters. It validates inputs, checks system resources, and either runs
    the optimization synchronously or queues it for asynchronous execution.
    """
    try:
        # Validate model availability
        await validate_model_availability(request.model_name, model_manager)
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Initialize job tracking
        job_info = {
            "job_id": job_id,
            "status": OptimizationStatus.QUEUED,
            "created_at": datetime.utcnow(),
            "request": request,
            "progress": None,
            "error_message": None,
            "metadata": request.metadata
        }
        
        active_jobs[job_id] = job_info
        
        # Save job to database
        await db.save_optimization_job(job_id, job_info)
        
        if request.execution_mode == ExecutionMode.SYNC:
            # Run synchronously (with timeout)
            try:
                timeout_seconds = request.optimization_config.timeout_minutes * 60
                await asyncio.wait_for(
                    run_optimization(job_id, request, db, model_manager, websocket_manager),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                active_jobs[job_id]["status"] = OptimizationStatus.FAILED
                active_jobs[job_id]["error_message"] = "Optimization timed out"
                raise HTTPException(
                    status_code=408,
                    detail="Optimization timed out"
                )
        else:
            # Run asynchronously
            task = background_tasks.add_task(
                run_optimization,
                job_id, request, db, model_manager, websocket_manager
            )
            active_jobs[job_id]["task"] = task
        
        logger.info(f"Started optimization job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "started",
            "execution_mode": request.execution_mode.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start optimization: {str(e)}"
        )

@router.get("/{job_id}/status", response_model=OptimizationStatusResponse)
async def get_optimization_status(
    job_id: str = Path(..., description="Job ID to check status for"),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the current status and progress of an optimization job
    
    Returns detailed status information including progress, resource usage,
    and estimated completion time for running optimizations.
    """
    try:
        # Check active jobs first
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
            return OptimizationStatusResponse(
                job_id=job_id,
                status=job_info["status"],
                created_at=job_info["created_at"],
                started_at=job_info.get("started_at"),
                completed_at=job_info.get("completed_at"),
                progress=job_info.get("progress"),
                error_message=job_info.get("error_message"),
                metadata=job_info.get("metadata", {})
            )
        
        # Check completed jobs
        if job_id in job_results:
            job_info = job_results[job_id]
            return OptimizationStatusResponse(
                job_id=job_id,
                status=job_info["status"],
                created_at=job_info.get("created_at", datetime.utcnow()),
                completed_at=job_info.get("completed_at"),
                progress=None,
                error_message=job_info.get("error_message"),
                metadata=job_info.get("metadata", {})
            )
        
        # Check database for historical jobs
        job_info = await db.get_optimization_job(job_id)
        if job_info:
            return OptimizationStatusResponse(**job_info)
        
        raise HTTPException(
            status_code=404,
            detail=f"Optimization job {job_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve job status"
        )

@router.get("/{job_id}/results", response_model=OptimizationResult)
async def get_optimization_results(
    job_id: str = Path(..., description="Job ID to get results for"),
    page: int = Query(1, ge=1, description="Page number for variant pagination"),
    page_size: int = Query(10, ge=1, le=100, description="Number of variants per page"),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the complete results of a completed optimization job
    
    Returns the best variants, metrics, and optimization history.
    Supports pagination for large result sets.
    """
    try:
        # Check if job exists and is completed
        result = None
        
        if job_id in job_results:
            if job_results[job_id]["status"] != OptimizationStatus.COMPLETED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job {job_id} has not completed yet"
                )
            result = job_results[job_id]["result"]
        else:
            # Check database
            result_data = await db.get_optimization_result(job_id)
            if not result_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Results for job {job_id} not found"
                )
            result = OptimizationResult(**result_data)
        
        # Apply pagination to top_variants
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_result = result.copy()
        paginated_result.top_variants = result.top_variants[start_idx:end_idx]
        
        # Add pagination metadata
        paginated_result.metadata.update({
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_variants": len(result.top_variants),
                "has_next": end_idx < len(result.top_variants)
            }
        })
        
        return paginated_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve optimization results"
        )

@router.post("/{job_id}/stop")
async def stop_optimization(
    job_id: str = Path(..., description="Job ID to stop"),
    db: DatabaseManager = Depends(get_database),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Stop a running optimization job
    
    Gracefully terminates the optimization process and returns any partial
    results that have been generated so far.
    """
    try:
        if job_id not in active_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Active job {job_id} not found"
            )
        
        job_info = active_jobs[job_id]
        
        if job_info["status"] not in [OptimizationStatus.QUEUED, OptimizationStatus.RUNNING]:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be stopped (status: {job_info['status']})"
            )
        
        # Mark job as cancelling
        job_info["status"] = OptimizationStatus.CANCELLING
        
        # Cancel the task if it exists
        if "task" in job_info:
            job_info["task"].cancel()
        
        # Cleanup resources
        await cleanup_job_resources(job_id)
        
        # Update database
        await db.update_job_status(job_id, OptimizationStatus.STOPPED)
        
        # Send WebSocket notification
        await websocket_manager.send_to_job(job_id, {
            "type": "optimization_stopped",
            "data": {"job_id": job_id}
        })
        
        logger.info(f"Stopped optimization job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "stopped",
            "message": "Optimization job stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to stop optimization job"
        )

@router.get("/history", response_model=OptimizationHistoryResponse)
async def get_optimization_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[OptimizationStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    search: Optional[str] = Query(None, description="Search in base prompt"),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get optimization history with filtering and pagination
    
    Returns a paginated list of historical optimization runs with
    summary information and filtering capabilities.
    """
    try:
        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        if search:
            filters["search"] = search
        
        # Get total count
        total_count = await db.count_optimization_jobs(filters)
        
        # Get paginated results
        offset = (page - 1) * page_size
        jobs = await db.get_optimization_jobs(
            filters=filters,
            limit=page_size,
            offset=offset,
            order_by="created_at",
            order_dir="desc"
        )
        
        # Convert to history items
        history_items = []
        for job in jobs:
            # Truncate base prompt for display
            base_prompt = job.get("base_prompt", "")
            if len(base_prompt) > 100:
                base_prompt = base_prompt[:97] + "..."
            
            history_item = OptimizationHistoryItem(
                job_id=job["job_id"],
                base_prompt=base_prompt,
                status=OptimizationStatus(job["status"]),
                created_at=job["created_at"],
                completed_at=job.get("completed_at"),
                best_score=job.get("best_score"),
                total_variants=job.get("total_variants", 0),
                optimization_method=job.get("optimization_method", "unknown"),
                metadata=job.get("metadata", {})
            )
            history_items.append(history_item)
        
        # Calculate pagination info
        has_next = offset + page_size < total_count
        
        return OptimizationHistoryResponse(
            items=history_items,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve optimization history"
        )

@router.delete("/{job_id}")
async def delete_optimization(
    job_id: str = Path(..., description="Job ID to delete"),
    force: bool = Query(False, description="Force delete even if job is running"),
    db: DatabaseManager = Depends(get_database),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    Delete an optimization job and cleanup associated resources
    
    Removes all data associated with the optimization job including
    results, history, and cached data. Use with caution.
    """
    try:
        # Check if job exists
        job_exists = (
            job_id in active_jobs or 
            job_id in job_results or 
            await db.optimization_job_exists(job_id)
        )
        
        if not job_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization job {job_id} not found"
            )
        
        # Check if job is running
        if job_id in active_jobs:
            job_status = active_jobs[job_id]["status"]
            if job_status in [OptimizationStatus.RUNNING, OptimizationStatus.QUEUED]:
                if not force:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot delete running job {job_id}. Use force=true to force deletion."
                    )
                else:
                    # Stop the job first
                    await cleanup_job_resources(job_id)
        
        # Remove from memory
        if job_id in active_jobs:
            del active_jobs[job_id]
        if job_id in job_results:
            del job_results[job_id]
        
        # Remove from database
        await db.delete_optimization_job(job_id)
        await db.delete_optimization_result(job_id)
        
        # Send WebSocket notification
        await websocket_manager.send_to_job(job_id, {
            "type": "optimization_deleted",
            "data": {"job_id": job_id}
        })
        
        logger.info(f"Deleted optimization job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "deleted",
            "message": "Optimization job deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete optimization job"
        )

@router.post("/batch", response_model=Dict[str, Any])
async def batch_optimization(
    request: BatchOptimizationRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_database),
    model_manager: ModelManager = Depends(get_model_manager),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
    _: None = Depends(check_rate_limit),
    __: None = Depends(check_system_resources)
):
    """
    Start multiple optimization jobs in batch
    
    Processes multiple optimization requests with optional shared configuration.
    All jobs are executed asynchronously and can be monitored individually.
    """
    try:
        batch_id = generate_job_id()
        job_ids = []
        failed_requests = []
        
        # Apply shared config if provided
        processed_requests = []
        for i, opt_request in enumerate(request.requests):
            try:
                if request.shared_config:
                    # Merge shared config with individual request config
                    merged_config = opt_request.optimization_config.dict()
                    shared_dict = request.shared_config.dict()
                    
                    # Only override if not explicitly set in individual request
                    for key, value in shared_dict.items():
                        if key not in merged_config or merged_config[key] == OptimizationConfig().dict()[key]:
                            merged_config[key] = value
                    
                    opt_request.optimization_config = OptimizationConfig(**merged_config)
                
                # Force async execution for batch jobs
                opt_request.execution_mode = ExecutionMode.ASYNC
                
                # Add batch metadata
                if not opt_request.metadata:
                    opt_request.metadata = {}
                opt_request.metadata.update({
                    "batch_id": batch_id,
                    "batch_index": i,
                    **request.batch_metadata
                })
                
                processed_requests.append(opt_request)
                
            except Exception as e:
                failed_requests.append({
                    "index": i,
                    "error": str(e),
                    "request": opt_request.dict() if hasattr(opt_request, 'dict') else str(opt_request)
                })
        
        # Start individual optimization jobs
        for i, opt_request in enumerate(processed_requests):
            try:
                # Validate model availability
                await validate_model_availability(opt_request.model_name, model_manager)
                
                # Generate job ID
                job_id = generate_job_id()
                job_ids.append(job_id)
                
                # Initialize job tracking
                job_info = {
                    "job_id": job_id,
                    "status": OptimizationStatus.QUEUED,
                    "created_at": datetime.utcnow(),
                    "request": opt_request,
                    "progress": None,
                    "error_message": None,
                    "metadata": opt_request.metadata
                }
                
                active_jobs[job_id] = job_info
                
                # Save job to database
                await db.save_optimization_job(job_id, job_info)
                
                # Start optimization task
                task = background_tasks.add_task(
                    run_optimization,
                    job_id, opt_request, db, model_manager, websocket_manager
                )
                active_jobs[job_id]["task"] = task
                
                logger.info(f"Started batch optimization job {job_id} (batch {batch_id})")
                
            except Exception as e:
                failed_requests.append({
                    "index": i,
                    "error": str(e),
                    "request": opt_request.dict()
                })
        
        # Save batch information
        batch_info = {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_requests": len(request.requests),
            "successful_jobs": len(job_ids),
            "failed_requests": len(failed_requests),
            "created_at": datetime.utcnow(),
            "metadata": request.batch_metadata
        }
        
        await db.save_batch_info(batch_id, batch_info)
        
        return {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total_requests": len(request.requests),
            "successful_jobs": len(job_ids),
            "failed_requests": failed_requests,
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start batch optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch optimization: {str(e)}"
        )

@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str = Path(..., description="Batch ID to check status for"),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the status of a batch optimization
    
    Returns aggregate status information for all jobs in the batch,
    including completion counts and overall progress.
    """
    try:
        # Get batch info from database
        batch_info = await db.get_batch_info(batch_id)
        if not batch_info:
            raise HTTPException(
                status_code=404,
                detail=f"Batch {batch_id} not found"
            )
        
        job_ids = batch_info["job_ids"]
        job_statuses = {}
        
        # Get status for each job in batch
        for job_id in job_ids:
            if job_id in active_jobs:
                job_statuses[job_id] = {
                    "status": active_jobs[job_id]["status"],
                    "progress": active_jobs[job_id].get("progress")
                }
            elif job_id in job_results:
                job_statuses[job_id] = {
                    "status": job_results[job_id]["status"],
                    "progress": None
                }
            else:
                # Check database
                job_info = await db.get_optimization_job(job_id)
                if job_info:
                    job_statuses[job_id] = {
                        "status": job_info["status"],
                        "progress": None
                    }
        
        # Calculate aggregate statistics
        status_counts = {}
        total_progress = 0
        jobs_with_progress = 0
        
        for job_id, job_info in job_statuses.items():
            status = job_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if job_info["progress"]:
                total_progress += job_info["progress"].progress_percentage
                jobs_with_progress += 1
        
        # Calculate overall progress
        overall_progress = 0
        if jobs_with_progress > 0:
            avg_active_progress = total_progress / jobs_with_progress
            completed_jobs = status_counts.get(OptimizationStatus.COMPLETED, 0)
            total_jobs = len(job_ids)
            
            # Weight completed jobs as 100% and active jobs by their progress
            overall_progress = (
                (completed_jobs * 100 + 
                 status_counts.get(OptimizationStatus.RUNNING, 0) * avg_active_progress) 
                / total_jobs
            )
        elif status_counts.get(OptimizationStatus.COMPLETED, 0) > 0:
            overall_progress = (
                status_counts[OptimizationStatus.COMPLETED] / len(job_ids) * 100
            )
        
        # Determine overall batch status
        if status_counts.get(OptimizationStatus.COMPLETED, 0) == len(job_ids):
            batch_status = "completed"
        elif status_counts.get(OptimizationStatus.FAILED, 0) == len(job_ids):
            batch_status = "failed"
        elif any(status in [OptimizationStatus.RUNNING, OptimizationStatus.QUEUED] 
                for status in status_counts.keys()):
            batch_status = "running"
        else:
            batch_status = "partial"
        
        return {
            "batch_id": batch_id,
            "batch_status": batch_status,
            "overall_progress": round(overall_progress, 2),
            "job_statuses": job_statuses,
            "status_summary": status_counts,
            "total_jobs": len(job_ids),
            "created_at": batch_info["created_at"],
            "metadata": batch_info.get("metadata", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status for {batch_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve batch status"
        )

@router.get("/stats/system")
async def get_system_stats():
    """
    Get current system statistics and resource usage
    
    Returns information about system resources, active jobs,
    and optimization queue status for monitoring purposes.
    """
    try:
        # Get system resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Count job statuses
        status_counts = {}
        for job_info in active_jobs.values():
            status = job_info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate queue metrics
        queue_size = optimization_queue.qsize() if hasattr(optimization_queue, 'qsize') else 0
        
        # Get recent performance metrics
        recent_jobs = list(active_jobs.values())[-10:]  # Last 10 jobs
        avg_completion_time = 0
        if recent_jobs:
            completion_times = []
            for job in recent_jobs:
                if (job.get("completed_at") and job.get("started_at") and 
                    job["status"] == OptimizationStatus.COMPLETED):
                    duration = (job["completed_at"] - job["started_at"]).total_seconds()
                    completion_times.append(duration)
            
            if completion_times:
                avg_completion_time = sum(completion_times) / len(completion_times)
        
        return {
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "job_statistics": {
                "active_jobs": len(active_jobs),
                "completed_results": len(job_results),
                "queue_size": queue_size,
                "status_breakdown": status_counts
            },
            "performance_metrics": {
                "average_completion_time_seconds": round(avg_completion_time, 2),
                "recent_job_count": len(recent_jobs)
            },
            "capacity_status": {
                "can_accept_jobs": (
                    cpu_percent < 80 and 
                    memory.percent < 80 and 
                    len([j for j in active_jobs.values() 
                         if j["status"] == OptimizationStatus.RUNNING]) < 5
                ),
                "recommended_action": (
                    "normal" if cpu_percent < 70 and memory.percent < 70
                    else "caution" if cpu_percent < 85 and memory.percent < 85
                    else "high_load"
                )
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system statistics"
        )

# =====================================================
# WebSocket Integration
# =====================================================

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(
    websocket,
    job_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time optimization updates
    
    Provides real-time progress updates, status changes, and completion
    notifications for optimization jobs.
    """
    await websocket_manager.connect(websocket, job_id)
    
    try:
        # Send initial status if job exists
        if job_id in active_jobs:
            initial_status = {
                "type": "initial_status",
                "data": {
                    "job_id": job_id,
                    "status": active_jobs[job_id]["status"],
                    "progress": active_jobs[job_id].get("progress")
                }
            }
            await websocket.send_json(initial_status)
        
        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            
            # Handle client commands (like requesting status updates)
            try:
                message = eval(data)  # Basic JSON parsing
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            except:
                pass  # Ignore malformed messages
                
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await websocket_manager.disconnect(websocket, job_id)

# =====================================================
# Startup and Cleanup Handlers
# =====================================================

@router.on_event("startup")
async def startup_optimization_service():
    """Initialize optimization service on startup"""
    logger.info("Starting optimization service...")
    
    # Recover active jobs from database on startup
    try:
        db = DatabaseManager()
        incomplete_jobs = await db.get_incomplete_optimization_jobs()
        
        for job_info in incomplete_jobs:
            job_id = job_info["job_id"]
            
            # Mark as failed if they were running during shutdown
            if job_info["status"] in [OptimizationStatus.RUNNING, OptimizationStatus.QUEUED]:
                job_info["status"] = OptimizationStatus.FAILED
                job_info["error_message"] = "Server restart during optimization"
                job_info["completed_at"] = datetime.utcnow()
                
                await db.update_job_status(job_id, OptimizationStatus.FAILED)
                
            # Don't restore to active_jobs - they're considered failed
            logger.info(f"Marked job {job_id} as failed due to server restart")
        
        logger.info(f"Processed {len(incomplete_jobs)} incomplete jobs on startup")
        
    except Exception as e:
        logger.error(f"Error during startup recovery: {e}")

@router.on_event("shutdown")
async def shutdown_optimization_service():
    """Cleanup optimization service on shutdown"""
    logger.info("Shutting down optimization service...")
    
    # Cancel all active jobs
    for job_id, job_info in active_jobs.items():
        try:
            if "task" in job_info:
                job_info["task"].cancel()
            
            # Update status in database
            db = DatabaseManager()
            await db.update_job_status(job_id, OptimizationStatus.FAILED)
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id} during shutdown: {e}")
    
    logger.info("Optimization service shutdown complete")

# =====================================================
# Error Handlers
# =====================================================

@router.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "type": "validation_error"
        }
    )

@router.exception_handler(OptimizationError)
async def optimization_error_handler(request, exc):
    """Handle optimization-specific errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Optimization Error",
            "detail": str(exc),
            "type": "optimization_error"
        }
    )

@router.exception_handler(ResourceError)
async def resource_error_handler(request, exc):
    """Handle resource-related errors"""
    return JSONResponse(
        status_code=503,
        content={
            "error": "Resource Error",
            "detail": str(exc),
            "type": "resource_error",
            "retry_after": 300  # Suggest retry after 5 minutes
        }
    )

# Export router for inclusion in main FastAPI app
__all__ = ["router"]
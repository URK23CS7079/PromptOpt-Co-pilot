"""
PromptOpt Co-Pilot - Prompt Management API Routes

This module provides comprehensive REST API endpoints for managing prompts and their variants,
including CRUD operations, search functionality, variant generation, and quick testing capabilities.

Author: PromptOpt Co-Pilot
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from backend.core.database import get_db
from backend.core.models import Prompt, PromptVariant, OptimizationSession, User
from backend.core.auth import get_current_user
from backend.optimization.variant_generator import VariantGenerator, VariantStrategy
from backend.evaluation.evaluator import Evaluator, EvaluationConfig
from backend.llm.model_manager import ModelManager
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/prompts", tags=["prompts"])
security = HTTPBearer()

# Request/Response Models
class PromptCreateRequest(BaseModel):
    """Request model for creating a new prompt."""
    
    content: str = Field(..., min_length=1, max_length=10000, description="Prompt content")
    title: str = Field(..., min_length=1, max_length=200, description="Prompt title")
    description: Optional[str] = Field(None, max_length=1000, description="Prompt description")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    model_type: str = Field(..., description="Target model type (gpt-4, claude-3, etc.)")
    use_case: Optional[str] = Field(None, description="Primary use case category")
    
    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Prompt content cannot be empty')
        return v.strip()


class PromptUpdateRequest(BaseModel):
    """Request model for updating an existing prompt."""
    
    content: Optional[str] = Field(None, min_length=1, max_length=10000)
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_type: Optional[str] = None
    use_case: Optional[str] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return [tag.strip().lower() for tag in v if tag.strip()] if v else v


class VariantCreateRequest(BaseModel):
    """Request model for creating prompt variants."""
    
    strategy: VariantStrategy = Field(..., description="Variant generation strategy")
    count: int = Field(default=3, ge=1, le=10, description="Number of variants to generate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    
    class Config:
        use_enum_values = True


class PromptTestRequest(BaseModel):
    """Request model for quick prompt testing."""
    
    test_cases: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10)
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["relevance", "clarity"])
    model_config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PromptSearchRequest(BaseModel):
    """Request model for searching prompts."""
    
    query: Optional[str] = Field(None, description="Search query")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    model_type: Optional[str] = Field(None, description="Filter by model type")
    use_case: Optional[str] = Field(None, description="Filter by use case")
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_performance: Optional[float] = Field(None, ge=0.0, le=1.0)


class PromptResponse(BaseModel):
    """Response model for prompt data."""
    
    id: UUID
    title: str
    content: str
    description: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    model_type: str
    use_case: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: UUID
    variant_count: int
    best_performance_score: Optional[float]
    optimization_count: int
    
    class Config:
        from_attributes = True


class VariantResponse(BaseModel):
    """Response model for prompt variant data."""
    
    id: UUID
    prompt_id: UUID
    content: str
    strategy: str
    parameters: Dict[str, Any]
    performance_score: Optional[float]
    evaluation_metrics: Dict[str, float]
    created_at: datetime
    is_best: bool
    
    class Config:
        from_attributes = True


class PromptListResponse(BaseModel):
    """Response model for paginated prompt list."""
    
    prompts: List[PromptResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PromptTestResponse(BaseModel):
    """Response model for prompt testing results."""
    
    test_id: UUID
    overall_score: float
    metric_scores: Dict[str, float]
    test_results: List[Dict[str, Any]]
    execution_time: float
    timestamp: datetime


# Helper Functions
async def get_prompt_or_404(
    prompt_id: UUID,
    db: Session,
    user: User,
    include_variants: bool = False
) -> Prompt:
    """Get prompt by ID or raise 404 error."""
    query = db.query(Prompt).filter(
        and_(
            Prompt.id == prompt_id,
            Prompt.created_by == user.id,
            Prompt.deleted_at.is_(None)
        )
    )
    
    if include_variants:
        query = query.options(joinedload(Prompt.variants))
    
    prompt = query.first()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt not found"
        )
    return prompt


def build_search_query(db: Session, user: User, search_params: PromptSearchRequest):
    """Build SQLAlchemy query for prompt search."""
    query = db.query(Prompt).filter(
        and_(
            Prompt.created_by == user.id,
            Prompt.deleted_at.is_(None)
        )
    )
    
    # Text search
    if search_params.query:
        search_term = f"%{search_params.query}%"
        query = query.filter(
            or_(
                Prompt.title.ilike(search_term),
                Prompt.content.ilike(search_term),
                Prompt.description.ilike(search_term)
            )
        )
    
    # Tag filter
    if search_params.tags:
        for tag in search_params.tags:
            query = query.filter(Prompt.tags.contains([tag]))
    
    # Model type filter
    if search_params.model_type:
        query = query.filter(Prompt.model_type == search_params.model_type)
    
    # Use case filter
    if search_params.use_case:
        query = query.filter(Prompt.use_case == search_params.use_case)
    
    # Date range filter
    if search_params.date_from:
        query = query.filter(Prompt.created_at >= search_params.date_from)
    
    if search_params.date_to:
        query = query.filter(Prompt.created_at <= search_params.date_to)
    
    # Performance filter
    if search_params.min_performance is not None:
        query = query.filter(Prompt.best_performance_score >= search_params.min_performance)
    
    return query


# API Endpoints
@router.post("/", response_model=PromptResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    request: PromptCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new prompt.
    
    Creates a new prompt with the provided content, metadata, and tags.
    Validates the prompt structure and stores it in the database.
    """
    try:
        # Create prompt instance
        prompt = Prompt(
            id=uuid4(),
            title=request.title,
            content=request.content,
            description=request.description,
            tags=request.tags,
            metadata=request.metadata,
            model_type=request.model_type,
            use_case=request.use_case,
            created_by=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
        
        logger.info(f"Created prompt {prompt.id} by user {current_user.id}")
        
        # Build response with additional computed fields
        response_data = PromptResponse.from_orm(prompt)
        response_data.variant_count = 0
        response_data.optimization_count = 0
        response_data.best_performance_score = None
        
        return response_data
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error creating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt creation failed due to data constraints"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a prompt by ID.
    
    Returns detailed information about a specific prompt including
    variant count and optimization history.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user)
        
        # Get additional computed fields
        variant_count = db.query(PromptVariant).filter(
            PromptVariant.prompt_id == prompt_id
        ).count()
        
        optimization_count = db.query(OptimizationSession).filter(
            OptimizationSession.prompt_id == prompt_id
        ).count()
        
        best_score = db.query(func.max(PromptVariant.performance_score)).filter(
            PromptVariant.prompt_id == prompt_id
        ).scalar()
        
        # Build response
        response_data = PromptResponse.from_orm(prompt)
        response_data.variant_count = variant_count
        response_data.optimization_count = optimization_count
        response_data.best_performance_score = best_score
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
    prompt_id: UUID,
    request: PromptUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing prompt.
    
    Updates prompt content, metadata, or tags while maintaining version history.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user)
        
        # Update fields that are provided
        update_data = request.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(prompt, field, value)
        
        prompt.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(prompt)
        
        logger.info(f"Updated prompt {prompt_id} by user {current_user.id}")
        
        # Build response with computed fields
        variant_count = db.query(PromptVariant).filter(
            PromptVariant.prompt_id == prompt_id
        ).count()
        
        optimization_count = db.query(OptimizationSession).filter(
            OptimizationSession.prompt_id == prompt_id
        ).count()
        
        best_score = db.query(func.max(PromptVariant.performance_score)).filter(
            PromptVariant.prompt_id == prompt_id
        ).scalar()
        
        response_data = PromptResponse.from_orm(prompt)
        response_data.variant_count = variant_count
        response_data.optimization_count = optimization_count
        response_data.best_performance_score = best_score
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
    prompt_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a prompt.
    
    Performs soft delete and checks for active optimization sessions.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user)
        
        # Check for active optimization sessions
        active_sessions = db.query(OptimizationSession).filter(
            and_(
                OptimizationSession.prompt_id == prompt_id,
                OptimizationSession.status.in_(["running", "pending"])
            )
        ).count()
        
        if active_sessions > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete prompt with active optimization sessions"
            )
        
        # Soft delete
        prompt.deleted_at = datetime.utcnow()
        
        # Soft delete all variants
        db.query(PromptVariant).filter(
            PromptVariant.prompt_id == prompt_id
        ).update({"deleted_at": datetime.utcnow()})
        
        db.commit()
        
        logger.info(f"Deleted prompt {prompt_id} by user {current_user.id}")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/", response_model=PromptListResponse)
async def list_prompts(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List prompts with pagination and filtering.
    
    Returns a paginated list of prompts with summary statistics.
    """
    try:
        # Base query
        query = db.query(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        )
        
        # Sorting
        sort_field = getattr(Prompt, sort_by, Prompt.created_at)
        if sort_order == "desc":
            query = query.order_by(desc(sort_field))
        else:
            query = query.order_by(sort_field)
        
        # Get total count
        total = query.count()
        
        # Pagination
        offset = (page - 1) * per_page
        prompts = query.offset(offset).limit(per_page).all()
        
        # Build response data
        prompt_responses = []
        for prompt in prompts:
            # Get additional data for each prompt
            variant_count = db.query(PromptVariant).filter(
                PromptVariant.prompt_id == prompt.id
            ).count()
            
            optimization_count = db.query(OptimizationSession).filter(
                OptimizationSession.prompt_id == prompt.id
            ).count()
            
            best_score = db.query(func.max(PromptVariant.performance_score)).filter(
                PromptVariant.prompt_id == prompt.id
            ).scalar()
            
            response_data = PromptResponse.from_orm(prompt)
            response_data.variant_count = variant_count
            response_data.optimization_count = optimization_count
            response_data.best_performance_score = best_score
            
            prompt_responses.append(response_data)
        
        # Calculate pagination metadata
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        return PromptListResponse(
            prompts=prompt_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/{prompt_id}/variants", response_model=List[VariantResponse])
async def create_variants(
    prompt_id: UUID,
    request: VariantCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create prompt variants using specified strategy.
    
    Generates variants using APE, StablePrompt, or custom methods.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user)
        
        # Initialize variant generator
        variant_generator = VariantGenerator()
        model_manager = ModelManager()
        
        # Generate variants
        generated_variants = await variant_generator.generate_variants(
            prompt_content=prompt.content,
            strategy=request.strategy,
            count=request.count,
            parameters=request.parameters,
            model_manager=model_manager
        )
        
        # Store variants in database
        variant_responses = []
        for variant_content in generated_variants:
            variant = PromptVariant(
                id=uuid4(),
                prompt_id=prompt_id,
                content=variant_content,
                strategy=request.strategy.value,
                parameters=request.parameters,
                created_at=datetime.utcnow(),
                performance_score=None,
                evaluation_metrics={}
            )
            
            db.add(variant)
            db.flush()  # Get the ID
            
            variant_response = VariantResponse.from_orm(variant)
            variant_response.is_best = False
            variant_responses.append(variant_response)
        
        db.commit()
        
        logger.info(f"Created {len(variant_responses)} variants for prompt {prompt_id}")
        
        return variant_responses
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating variants for prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate variants"
        )


@router.get("/{prompt_id}/variants", response_model=List[VariantResponse])
async def get_variants(
    prompt_id: UUID,
    sort_by: str = Query("performance_score", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all variants for a prompt.
    
    Returns variants with performance metrics, sorted by performance scores.
    """
    try:
        # Verify prompt ownership
        await get_prompt_or_404(prompt_id, db, current_user)
        
        # Query variants
        query = db.query(PromptVariant).filter(
            and_(
                PromptVariant.prompt_id == prompt_id,
                PromptVariant.deleted_at.is_(None)
            )
        )
        
        # Sorting
        sort_field = getattr(PromptVariant, sort_by, PromptVariant.performance_score)
        if sort_order == "desc":
            # Handle NULL values in performance_score
            if sort_by == "performance_score":
                query = query.order_by(sort_field.desc().nullslast())
            else:
                query = query.order_by(desc(sort_field))
        else:
            if sort_by == "performance_score":
                query = query.order_by(sort_field.asc().nullsfirst())
            else:
                query = query.order_by(sort_field)
        
        variants = query.all()
        
        # Find best performing variant
        best_score = max((v.performance_score for v in variants if v.performance_score), default=None)
        
        # Build response
        variant_responses = []
        for variant in variants:
            response_data = VariantResponse.from_orm(variant)
            response_data.is_best = (
                variant.performance_score is not None and 
                variant.performance_score == best_score
            )
            variant_responses.append(response_data)
        
        return variant_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving variants for prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/search", response_model=PromptListResponse)
async def search_prompts(
    search_params: PromptSearchRequest = Depends(),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Search prompts with advanced filtering.
    
    Supports full-text search, tag filtering, and performance-based filtering.
    """
    try:
        # Build search query
        query = build_search_query(db, current_user, search_params)
        
        # Get total count
        total = query.count()
        
        # Pagination
        offset = (page - 1) * per_page
        prompts = query.order_by(desc(Prompt.created_at)).offset(offset).limit(per_page).all()
        
        # Build response data
        prompt_responses = []
        for prompt in prompts:
            variant_count = db.query(PromptVariant).filter(
                PromptVariant.prompt_id == prompt.id
            ).count()
            
            optimization_count = db.query(OptimizationSession).filter(
                OptimizationSession.prompt_id == prompt.id
            ).count()
            
            best_score = db.query(func.max(PromptVariant.performance_score)).filter(
                PromptVariant.prompt_id == prompt.id
            ).scalar()
            
            response_data = PromptResponse.from_orm(prompt)
            response_data.variant_count = variant_count
            response_data.optimization_count = optimization_count
            response_data.best_performance_score = best_score
            
            prompt_responses.append(response_data)
        
        # Calculate pagination metadata
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        return PromptListResponse(
            prompts=prompt_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Error searching prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/{prompt_id}/test", response_model=PromptTestResponse)
async def test_prompt(
    prompt_id: UUID,
    request: PromptTestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Quick test of a prompt against sample data.
    
    Provides immediate performance feedback without full optimization.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user)
        
        # Initialize evaluator
        evaluator = Evaluator()
        model_manager = ModelManager()
        
        # Configure evaluation
        eval_config = EvaluationConfig(
            metrics=request.evaluation_metrics,
            model_config=request.model_config
        )
        
        # Run evaluation
        start_time = datetime.utcnow()
        
        results = await evaluator.evaluate_prompt(
            prompt_content=prompt.content,
            test_cases=request.test_cases,
            config=eval_config,
            model_manager=model_manager
        )
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate overall score
        metric_scores = results.get("metric_scores", {})
        overall_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0.0
        
        test_response = PromptTestResponse(
            test_id=uuid4(),
            overall_score=overall_score,
            metric_scores=metric_scores,
            test_results=results.get("test_results", []),
            execution_time=execution_time,
            timestamp=end_time
        )
        
        logger.info(f"Tested prompt {prompt_id} with score {overall_score}")
        
        return test_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test prompt"
        )


# Batch Operations
@router.post("/batch/delete", status_code=status.HTTP_204_NO_CONTENT)
async def batch_delete_prompts(
    prompt_ids: List[UUID],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete multiple prompts in batch.
    
    Performs soft delete on all specified prompts.
    """
    try:
        if len(prompt_ids) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 prompts can be deleted at once"
            )
        
        # Verify ownership and check for active sessions
        prompts = db.query(Prompt).filter(
            and_(
                Prompt.id.in_(prompt_ids),
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        ).all()
        
        if len(prompts) != len(prompt_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Some prompts not found or not owned by user"
            )
        
        # Check for active optimization sessions
        active_sessions = db.query(OptimizationSession).filter(
            and_(
                OptimizationSession.prompt_id.in_(prompt_ids),
                OptimizationSession.status.in_(["running", "pending"])
            )
        ).count()
        
        if active_sessions > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete prompts with active optimization sessions"
            )
        
        # Soft delete prompts and variants
        db.query(Prompt).filter(Prompt.id.in_(prompt_ids)).update(
            {"deleted_at": datetime.utcnow()}, synchronize_session=False
        )
        
        db.query(PromptVariant).filter(
            PromptVariant.prompt_id.in_(prompt_ids)
        ).update(
            {"deleted_at": datetime.utcnow()}, synchronize_session=False
        )
        
        db.commit()
        
        logger.info(f"Batch deleted {len(prompt_ids)} prompts by user {current_user.id}")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in batch delete: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Statistics endpoint
@router.get("/stats/summary")
async def get_prompt_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get summary statistics for user's prompts.
    
    Returns counts, performance metrics, and usage statistics.
    """
    try:
        # Basic counts
        total_prompts = db.query(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        ).count()
        
        total_variants = db.query(PromptVariant).join(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None),
                PromptVariant.deleted_at.is_(None)
            )
        ).count()
        
        total_optimizations = db.query(OptimizationSession).join(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        ).count()
        
        # Performance statistics
        avg_performance = db.query(func.avg(PromptVariant.performance_score)).join(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None),
                PromptVariant.deleted_at.is_(None),
                PromptVariant.performance_score.is_not(None)
            )
        ).scalar() or 0.0
        
        best_performance = db.query(func.max(PromptVariant.performance_score)).join(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None),
                PromptVariant.deleted_at.is_(None)
            )
        ).scalar() or 0.0
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_prompts = db.query(Prompt).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None),
                Prompt.created_at >= thirty_days_ago
            )
        ).count()
        
        # Tag usage
        tag_query = db.query(Prompt.tags).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        ).all()
        
        tag_counts = {}
        for (tags,) in tag_query:
            if tags:
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Most used tags (top 10)
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Model type distribution
        model_distribution = db.query(
            Prompt.model_type,
            func.count(Prompt.id)
        ).filter(
            and_(
                Prompt.created_by == current_user.id,
                Prompt.deleted_at.is_(None)
            )
        ).group_by(Prompt.model_type).all()
        
        return {
            "total_prompts": total_prompts,
            "total_variants": total_variants,
            "total_optimizations": total_optimizations,
            "average_performance": round(avg_performance, 3),
            "best_performance": round(best_performance, 3),
            "recent_prompts": recent_prompts,
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
            "model_distribution": [{"model": model, "count": count} for model, count in model_distribution]
        }
        
    except Exception as e:
        logger.error(f"Error getting prompt stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Export endpoint
@router.get("/{prompt_id}/export")
async def export_prompt(
    prompt_id: UUID,
    format: str = Query("json", regex="^(json|yaml|txt)$", description="Export format"),
    include_variants: bool = Query(False, description="Include variants in export"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export prompt data in various formats.
    
    Supports JSON, YAML, and plain text export formats.
    """
    try:
        prompt = await get_prompt_or_404(prompt_id, db, current_user, include_variants=include_variants)
        
        # Prepare export data
        export_data = {
            "id": str(prompt.id),
            "title": prompt.title,
            "content": prompt.content,
            "description": prompt.description,
            "tags": prompt.tags,
            "metadata": prompt.metadata,
            "model_type": prompt.model_type,
            "use_case": prompt.use_case,
            "created_at": prompt.created_at.isoformat(),
            "updated_at": prompt.updated_at.isoformat()
        }
        
        if include_variants and prompt.variants:
            export_data["variants"] = [
                {
                    "id": str(variant.id),
                    "content": variant.content,
                    "strategy": variant.strategy,
                    "parameters": variant.parameters,
                    "performance_score": variant.performance_score,
                    "evaluation_metrics": variant.evaluation_metrics,
                    "created_at": variant.created_at.isoformat()
                }
                for variant in prompt.variants
                if variant.deleted_at is None
            ]
        
        # Format response based on requested format
        if format == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=prompt_{prompt_id}.json"}
            )
        elif format == "yaml":
            import yaml
            from fastapi.responses import Response
            yaml_content = yaml.dump(export_data, default_flow_style=False)
            return Response(
                content=yaml_content,
                media_type="application/x-yaml",
                headers={"Content-Disposition": f"attachment; filename=prompt_{prompt_id}.yaml"}
            )
        else:  # txt format
            from fastapi.responses import Response
            txt_content = f"Title: {prompt.title}\n\n"
            txt_content += f"Description: {prompt.description or 'N/A'}\n\n"
            txt_content += f"Tags: {', '.join(prompt.tags)}\n\n"
            txt_content += f"Model Type: {prompt.model_type}\n\n"
            txt_content += f"Content:\n{prompt.content}\n"
            
            if include_variants and prompt.variants:
                txt_content += "\n\nVariants:\n"
                for i, variant in enumerate(prompt.variants, 1):
                    if variant.deleted_at is None:
                        txt_content += f"\n--- Variant {i} ---\n"
                        txt_content += f"Strategy: {variant.strategy}\n"
                        txt_content += f"Performance: {variant.performance_score or 'N/A'}\n"
                        txt_content += f"Content:\n{variant.content}\n"
            
            return Response(
                content=txt_content,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=prompt_{prompt_id}.txt"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Export failed"
        )


# Import endpoint
@router.post("/import")
async def import_prompts(
    file_content: str,
    format: str = Query("json", regex="^(json|yaml)$", description="Import format"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Import prompts from JSON or YAML format.
    
    Supports batch import of prompts with validation.
    """
    try:
        # Parse content based on format
        if format == "json":
            import json
            data = json.loads(file_content)
        else:  # yaml
            import yaml
            data = yaml.safe_load(file_content)
        
        # Handle both single prompt and array of prompts
        if not isinstance(data, list):
            data = [data]
        
        if len(data) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 prompts can be imported at once"
            )
        
        imported_prompts = []
        
        for prompt_data in data:
            # Validate required fields
            if not prompt_data.get("title") or not prompt_data.get("content"):
                continue
            
            # Create prompt
            prompt = Prompt(
                id=uuid4(),
                title=prompt_data["title"][:200],  # Truncate if too long
                content=prompt_data["content"][:10000],  # Truncate if too long
                description=prompt_data.get("description", "")[:1000],
                tags=prompt_data.get("tags", [])[:10],  # Limit tags
                metadata=prompt_data.get("metadata", {}),
                model_type=prompt_data.get("model_type", "gpt-4"),
                use_case=prompt_data.get("use_case"),
                created_by=current_user.id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(prompt)
            db.flush()  # Get the ID
            
            imported_prompts.append({
                "id": str(prompt.id),
                "title": prompt.title
            })
        
        db.commit()
        
        logger.info(f"Imported {len(imported_prompts)} prompts for user {current_user.id}")
        
        return {
            "imported_count": len(imported_prompts),
            "prompts": imported_prompts
        }
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON format"
        )
    except yaml.YAMLError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid YAML format"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error importing prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Import failed"
        )


# Additional utility imports needed
from datetime import timedelta
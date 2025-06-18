"""
Database layer for PromptOpt Co-Pilot offline prompt optimization system.

This module provides SQLite database operations, schema management, and data access
layer using SQLAlchemy ORM. It supports prompts, variants, evaluations, optimizations,
and datasets with proper indexing and transaction management.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

from sqlalchemy import (
    create_engine, 
    Column, 
    Integer, 
    String, 
    Text, 
    DateTime, 
    Float, 
    Boolean,
    ForeignKey,
    Index,
    JSON,
    event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.pool import StaticPool

# Import from config (assuming it exists)
try:
    from backend.core.config import Settings
except ImportError:
    # Fallback for standalone usage
    class Settings:
        database_url: str = "sqlite:///promptopt.db"
        database_pool_size: int = 10

# Configure logging
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class ValidationError(DatabaseError):
    """Exception raised for data validation errors."""
    pass


# Database Models
class Prompt(Base):
    """
    Base prompt model storing original prompts and metadata.
    
    Attributes:
        id: Primary key
        content: The prompt text content
        created_at: Timestamp when prompt was created
        updated_at: Timestamp when prompt was last modified
        tags: JSON field for storing prompt tags/categories
        user_id: Optional user identifier for multi-user scenarios
        description: Optional description of the prompt's purpose
        is_active: Flag to mark if prompt is currently active
    """
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    tags = Column(JSON, default=list)
    user_id = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    variants = relationship("PromptVariant", back_populates="prompt", cascade="all, delete-orphan")
    evaluation_runs = relationship("EvaluationRun", back_populates="prompt", cascade="all, delete-orphan")
    optimization_sessions = relationship("OptimizationSession", back_populates="base_prompt")
    
    # Indexes
    __table_args__ = (
        Index('idx_prompts_user_id', 'user_id'),
        Index('idx_prompts_created_at', 'created_at'),
        Index('idx_prompts_is_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Prompt(id={self.id}, content='{self.content[:50]}...', user_id='{self.user_id}')>"


class PromptVariant(Base):
    """
    Prompt variant model storing different versions/optimizations of base prompts.
    
    Attributes:
        id: Primary key
        prompt_id: Foreign key to base prompt
        content: The variant prompt text
        generation_method: Method used to generate this variant (manual, dspy, etc.)
        parameters: JSON field storing generation parameters
        created_at: Timestamp when variant was created
        performance_score: Optional cached performance score
        is_best: Flag indicating if this is the best performing variant
    """
    __tablename__ = 'prompt_variants'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    content = Column(Text, nullable=False)
    generation_method = Column(String(50), nullable=False, default='manual')
    parameters = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    performance_score = Column(Float, nullable=True)
    is_best = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    prompt = relationship("Prompt", back_populates="variants")
    evaluation_results = relationship("EvaluationResult", back_populates="variant")
    
    # Indexes
    __table_args__ = (
        Index('idx_variants_prompt_id', 'prompt_id'),
        Index('idx_variants_generation_method', 'generation_method'),
        Index('idx_variants_performance_score', 'performance_score'),
        Index('idx_variants_is_best', 'is_best'),
    )
    
    def __repr__(self):
        return f"<PromptVariant(id={self.id}, prompt_id={self.prompt_id}, method='{self.generation_method}')>"


class Dataset(Base):
    """
    Dataset model for storing evaluation datasets.
    
    Attributes:
        id: Primary key
        name: Human-readable dataset name
        file_path: Path to the dataset file
        samples_count: Number of samples in the dataset
        created_at: Timestamp when dataset was created
        description: Optional description of the dataset
        format_type: Type of dataset format (json, csv, etc.)
        metadata: Additional dataset metadata
    """
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    samples_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(Text, nullable=True)
    format_type = Column(String(20), nullable=False, default='json')
    metadata = Column(JSON, default=dict)
    
    # Relationships
    evaluation_runs = relationship("EvaluationRun", back_populates="dataset")
    
    # Indexes
    __table_args__ = (
        Index('idx_datasets_name', 'name'),
        Index('idx_datasets_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', samples={self.samples_count})>"


class EvaluationRun(Base):
    """
    Evaluation run model storing information about evaluation sessions.
    
    Attributes:
        id: Primary key
        prompt_id: Foreign key to base prompt being evaluated
        dataset_id: Foreign key to dataset used for evaluation
        metrics: JSON field storing evaluation metrics configuration
        timestamp: When the evaluation was run
        status: Current status of the evaluation (running, completed, failed)
        total_samples: Total number of samples evaluated
        completed_samples: Number of samples completed
        error_message: Error message if evaluation failed
    """
    __tablename__ = 'evaluation_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    metrics = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default='pending', nullable=False)
    total_samples = Column(Integer, default=0, nullable=False)
    completed_samples = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    prompt = relationship("Prompt", back_populates="evaluation_runs")
    dataset = relationship("Dataset", back_populates="evaluation_runs")
    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_eval_runs_prompt_id', 'prompt_id'),
        Index('idx_eval_runs_dataset_id', 'dataset_id'),
        Index('idx_eval_runs_timestamp', 'timestamp'),
        Index('idx_eval_runs_status', 'status'),
    )
    
    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, prompt_id={self.prompt_id}, status='{self.status}')>"


class EvaluationResult(Base):
    """
    Individual evaluation result for a specific prompt variant on a sample.
    
    Attributes:
        id: Primary key
        run_id: Foreign key to evaluation run
        variant_id: Foreign key to prompt variant
        score: Overall evaluation score
        latency: Response latency in milliseconds
        exact_match: Boolean indicating exact match with expected output
        sample_index: Index of the sample in the dataset
        input_data: JSON field storing the input data
        output_data: JSON field storing the model output
        expected_output: JSON field storing expected output
        metrics_breakdown: Detailed metrics breakdown
    """
    __tablename__ = 'evaluation_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('evaluation_runs.id'), nullable=False)
    variant_id = Column(Integer, ForeignKey('prompt_variants.id'), nullable=False)
    score = Column(Float, nullable=False)
    latency = Column(Float, nullable=True)  # in milliseconds
    exact_match = Column(Boolean, nullable=True)
    sample_index = Column(Integer, nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    expected_output = Column(JSON, nullable=True)
    metrics_breakdown = Column(JSON, default=dict)
    
    # Relationships
    run = relationship("EvaluationRun", back_populates="results")
    variant = relationship("PromptVariant", back_populates="evaluation_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_eval_results_run_id', 'run_id'),
        Index('idx_eval_results_variant_id', 'variant_id'),
        Index('idx_eval_results_score', 'score'),
        Index('idx_eval_results_sample_index', 'sample_index'),
    )
    
    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, run_id={self.run_id}, score={self.score})>"


class OptimizationSession(Base):
    """
    Optimization session model tracking prompt optimization processes.
    
    Attributes:
        id: Primary key
        base_prompt_id: Foreign key to the original prompt being optimized
        best_variant_id: Foreign key to the best performing variant found
        settings: JSON field storing optimization settings/parameters
        completed_at: Timestamp when optimization completed
        status: Current optimization status
        total_iterations: Total number of optimization iterations
        current_iteration: Current iteration number
        best_score: Best score achieved during optimization
        optimization_method: Method used for optimization (dspy, manual, etc.)
    """
    __tablename__ = 'optimization_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    base_prompt_id = Column(Integer, ForeignKey('prompts.id'), nullable=False)
    best_variant_id = Column(Integer, ForeignKey('prompt_variants.id'), nullable=True)
    settings = Column(JSON, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='pending', nullable=False)
    total_iterations = Column(Integer, default=0, nullable=False)
    current_iteration = Column(Integer, default=0, nullable=False)
    best_score = Column(Float, nullable=True)
    optimization_method = Column(String(50), nullable=False, default='dspy')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    base_prompt = relationship("Prompt", back_populates="optimization_sessions")
    best_variant = relationship("PromptVariant")
    
    # Indexes
    __table_args__ = (
        Index('idx_opt_sessions_base_prompt_id', 'base_prompt_id'),
        Index('idx_opt_sessions_status', 'status'),
        Index('idx_opt_sessions_created_at', 'created_at'),
        Index('idx_opt_sessions_best_score', 'best_score'),
    )
    
    def __repr__(self):
        return f"<OptimizationSession(id={self.id}, base_prompt_id={self.base_prompt_id}, status='{self.status}')>"


class DatabaseManager:
    """
    Database manager class handling SQLite operations and session management.
    
    Provides connection management, session handling, and transaction support
    for the PromptOpt Co-Pilot application.
    """
    
    def __init__(self, db_path: str, echo: bool = False):
        """
        Initialize database manager with SQLite connection.
        
        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
            
        Raises:
            DatabaseError: If database initialization fails
        """
        self.db_path = db_path
        self.echo = echo
        self._engine = None
        self._session_factory = None
        
        try:
            self._setup_database()
            logger.info(f"Database manager initialized with path: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _setup_database(self):
        """Setup database engine and session factory."""
        # Create database directory if it doesn't exist
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine with connection pooling
        self._engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=self.echo,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Enable foreign key constraints for SQLite
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
    
    def create_tables(self):
        """
        Create all database tables if they don't exist.
        
        Raises:
            DatabaseError: If table creation fails
        """
        try:
            Base.metadata.create_all(self._engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}")
    
    def drop_tables(self):
        """
        Drop all database tables. Use with caution!
        
        Raises:
            DatabaseError: If table dropping fails
        """
        try:
            Base.metadata.drop_all(self._engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table dropping failed: {e}")
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic transaction management.
        
        Yields:
            Session: SQLAlchemy session object
            
        Raises:
            DatabaseError: If session creation or transaction fails
        """
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise DatabaseError(f"Transaction failed: {e}")
        finally:
            session.close()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute raw SQL query and return results.
        
        Args:
            sql: SQL query string
            params: Optional parameters for the query
            
        Returns:
            List of dictionaries containing query results
            
        Raises:
            DatabaseError: If query execution fails
        """
        try:
            with self._engine.connect() as connection:
                result = connection.execute(sql, params or {})
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise DatabaseError(f"SQL execution failed: {e}")
    
    def get_table_info(self, table_name: str) -> List[Dict]:
        """
        Get information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        sql = f"PRAGMA table_info({table_name})"
        return self.execute_raw_sql(sql)
    
    def vacuum_database(self):
        """
        Vacuum the database to reclaim space and optimize performance.
        
        Raises:
            DatabaseError: If vacuum operation fails
        """
        try:
            with self._engine.connect() as connection:
                connection.execute("VACUUM")
            logger.info("Database vacuum completed successfully")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            raise DatabaseError(f"Vacuum operation failed: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including table sizes and row counts.
        
        Returns:
            Dictionary containing database statistics
        """
        stats = {}
        
        try:
            with self.get_session() as session:
                # Get row counts for each table
                stats['prompts_count'] = session.query(Prompt).count()
                stats['variants_count'] = session.query(PromptVariant).count()
                stats['datasets_count'] = session.query(Dataset).count()
                stats['evaluation_runs_count'] = session.query(EvaluationRun).count()
                stats['evaluation_results_count'] = session.query(EvaluationResult).count()
                stats['optimization_sessions_count'] = session.query(OptimizationSession).count()
                
                # Get database file size
                db_file = Path(self.db_path)
                if db_file.exists():
                    stats['database_size_bytes'] = db_file.stat().st_size
                    stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)
                
                stats['last_updated'] = datetime.utcnow().isoformat()
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def close(self):
        """
        Close database connections and cleanup resources.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Database initialization and utility functions
def init_database(settings: Settings) -> DatabaseManager:
    """
    Initialize database with proper schema and return database manager.
    
    Args:
        settings: Application settings containing database configuration
        
    Returns:
        DatabaseManager: Configured database manager instance
        
    Raises:
        DatabaseError: If database initialization fails
    """
    try:
        # Extract database path from URL (assuming sqlite:///path format)
        db_path = settings.database_url.replace('sqlite:///', '')
        
        # Initialize database manager
        db_manager = DatabaseManager(db_path, echo=False)
        
        # Create tables
        db_manager.create_tables()
        
        logger.info("Database initialized successfully")
        return db_manager
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Failed to initialize database: {e}")


# Data access layer utility functions
class PromptRepository:
    """Repository class for prompt-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_prompt(self, content: str, user_id: Optional[str] = None, 
                     tags: Optional[List[str]] = None, description: Optional[str] = None) -> Prompt:
        """Create a new prompt."""
        with self.db.get_session() as session:
            prompt = Prompt(
                content=content,
                user_id=user_id,
                tags=tags or [],
                description=description
            )
            session.add(prompt)
            session.flush()
            session.refresh(prompt)
            return prompt
    
    def get_prompt(self, prompt_id: int) -> Optional[Prompt]:
        """Get prompt by ID."""
        with self.db.get_session() as session:
            return session.query(Prompt).filter(Prompt.id == prompt_id).first()
    
    def list_prompts(self, user_id: Optional[str] = None, active_only: bool = True) -> List[Prompt]:
        """List prompts with optional filtering."""
        with self.db.get_session() as session:
            query = session.query(Prompt)
            if user_id:
                query = query.filter(Prompt.user_id == user_id)
            if active_only:
                query = query.filter(Prompt.is_active == True)
            return query.order_by(Prompt.created_at.desc()).all()
    
    def update_prompt(self, prompt_id: int, **kwargs) -> Optional[Prompt]:
        """Update prompt with given parameters."""
        with self.db.get_session() as session:
            prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()
            if prompt:
                for key, value in kwargs.items():
                    if hasattr(prompt, key):
                        setattr(prompt, key, value)
                prompt.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(prompt)
            return prompt
    
    def delete_prompt(self, prompt_id: int) -> bool:
        """Soft delete prompt by marking as inactive."""
        with self.db.get_session() as session:
            prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()
            if prompt:
                prompt.is_active = False
                prompt.updated_at = datetime.utcnow()
                return True
            return False


# Example usage and testing
if __name__ == "__main__":
    # Basic testing
    import tempfile
    import os
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        test_db_path = tmp_file.name
    
    try:
        # Initialize database
        db_manager = DatabaseManager(test_db_path, echo=True)
        db_manager.create_tables()
        
        # Test basic operations
        with db_manager.get_session() as session:
            # Create test prompt
            prompt = Prompt(
                content="What is the capital of France?",
                user_id="test_user",
                tags=["geography", "capitals"],
                description="Test geography prompt"
            )
            session.add(prompt)
            session.flush()
            
            # Create test variant
            variant = PromptVariant(
                prompt_id=prompt.id,
                content="Please tell me the capital city of France.",
                generation_method="manual",
                parameters={"temperature": 0.7}
            )
            session.add(variant)
            session.flush()
            
            # Create test dataset
            dataset = Dataset(
                name="Geography Test Dataset",
                file_path="/tmp/geography_dataset.json",
                samples_count=100,
                format_type="json",
                description="Test dataset for geography questions"
            )
            session.add(dataset)
            session.flush()
            
            print(f"Created prompt: {prompt}")
            print(f"Created variant: {variant}")
            print(f"Created dataset: {dataset}")
        
        # Test repository
        repo = PromptRepository(db_manager)
        prompts = repo.list_prompts()
        print(f"Found {len(prompts)} prompts")
        
        # Get database stats
        stats = db_manager.get_database_stats()
        print(f"Database stats: {stats}")
        
        print("Database testing completed successfully!")
        
    finally:
        # Cleanup
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
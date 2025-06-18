"""
PromptOpt Co-Pilot Configuration Module

This module provides centralized configuration management for the PromptOpt Co-Pilot
application, handling database settings, model configurations, API settings, and
optimization parameters with comprehensive validation.

Author: Santhosh Kumar
Version: 1.0.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache

try:
    from pydantic import BaseSettings, Field, validator, root_validator
    from pydantic.env_settings import SettingsSourceCallable
except ImportError:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    
    This class manages all configuration for PromptOpt Co-Pilot including:
    - Database connection settings
    - Local LLM model configurations
    - API server settings
    - DSPy optimization parameters
    - Logging and monitoring settings
    """
    
    # =====================================
    # Environment & Application Settings
    # =====================================
    
    environment: str = Field(
        default="development",
        description="Application environment (development, production, testing)"
    )
    
    app_name: str = Field(
        default="PromptOpt Co-Pilot",
        description="Application name for logging and UI"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode with verbose logging"
    )
    
    # =====================================
    # Directory Settings
    # =====================================
    
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Base application directory"
    )
    
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".promptopt",
        description="User data directory for storing databases and models"
    )
    
    models_dir: Path = Field(
        default_factory=lambda: Path.home() / ".promptopt" / "models",
        description="Directory for storing GGUF model files"
    )
    
    logs_dir: Path = Field(
        default_factory=lambda: Path.home() / ".promptopt" / "logs",
        description="Directory for application logs"
    )
    
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".promptopt" / "cache",
        description="Directory for caching optimization results"
    )
    
    # =====================================
    # Database Settings
    # =====================================
    
    database_url: str = Field(
        default="sqlite:///promptopt.db",
        description="SQLite database URL (relative to data_dir)"
    )
    
    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )
    
    database_pool_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Database connection timeout in seconds"
    )
    
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    
    # =====================================
    # Model Settings
    # =====================================
    
    default_model_path: Optional[str] = Field(
        default=None,
        description="Path to default GGUF model file"
    )
    
    model_context_window: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description="Default context window size for models"
    )
    
    model_memory_limit_gb: float = Field(
        default=8.0,
        ge=1.0,
        le=64.0,
        description="Memory limit for model loading in GB"
    )
    
    model_threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of threads for model inference"
    )
    
    model_batch_size: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Batch size for model processing"
    )
    
    supported_model_formats: List[str] = Field(
        default=["gguf", "ggml"],
        description="Supported model file formats"
    )
    
    # =====================================
    # API Server Settings
    # =====================================
    
    api_host: str = Field(
        default="localhost",
        description="API server host address"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    
    api_workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of API server workers"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for frontend"
    )
    
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    api_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="API request timeout in seconds"
    )
    
    # =====================================
    # DSPy Optimization Settings
    # =====================================
    
    dspy_max_iterations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum optimization iterations for DSPy"
    )
    
    dspy_num_candidates: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of candidate prompts to generate"
    )
    
    dspy_bootstrap_examples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of bootstrap examples for training"
    )
    
    evaluation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum evaluation score threshold"
    )
    
    optimization_timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Optimization timeout in seconds"
    )
    
    # =====================================
    # Logging Settings
    # =====================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    log_file_max_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size in MB"
    )
    
    log_file_backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of log file backups to keep"
    )
    
    enable_file_logging: bool = Field(
        default=True,
        description="Enable logging to files"
    )
    
    enable_console_logging: bool = Field(
        default=True,
        description="Enable console logging"
    )
    
    # =====================================
    # PromptLayer Integration Settings
    # =====================================
    
    promptlayer_enabled: bool = Field(
        default=False,
        description="Enable PromptLayer integration for tracking"
    )
    
    promptlayer_api_key: Optional[str] = Field(
        default=None,
        description="PromptLayer API key for tracking"
    )
    
    promptlayer_tags: List[str] = Field(
        default=["promptopt", "local"],
        description="Default tags for PromptLayer tracking"
    )
    
    # =====================================
    # Performance Settings
    # =====================================
    
    max_concurrent_optimizations: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Maximum concurrent optimization processes"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable result caching"
    )
    
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Cache time-to-live in hours"
    )
    
    # =====================================
    # Validation Methods
    # =====================================
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = ['development', 'production', 'testing']
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @field_validator('default_model_path')
    @classmethod
    def validate_model_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate model path if provided."""
        if v is not None and not Path(v).exists():
            logging.warning(f"Model path does not exist: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_directories(self) -> 'Settings':
        """Ensure all directory paths are absolute."""
        # Convert relative paths to absolute
        if not self.data_dir.is_absolute():
            self.data_dir = Path.cwd() / self.data_dir
        
        if not self.models_dir.is_absolute():
            self.models_dir = self.data_dir / "models"
        
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.data_dir / "logs"
        
        if not self.cache_dir.is_absolute():
            self.cache_dir = self.data_dir / "cache"
        
        return self
    
    @property
    def database_path(self) -> Path:
        """Get the full database file path."""
        if self.database_url.startswith("sqlite:///"):
            db_file = self.database_url.replace("sqlite:///", "")
            if not Path(db_file).is_absolute():
                return self.data_dir / db_file
            return Path(db_file)
        return self.data_dir / "promptopt.db"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "PROMPTOPT_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Returns:
        Settings: Application settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Create required directories
        create_directories(_settings)
        
        # Setup logging
        setup_logging(_settings)
        
        logging.info(f"Settings loaded for environment: {_settings.environment}")
        logging.debug(f"Data directory: {_settings.data_dir}")
        logging.debug(f"Models directory: {_settings.models_dir}")
    
    return _settings


def validate_model_path(model_path: Union[str, Path]) -> bool:
    """
    Validate if a model file exists and is supported.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if model file is valid
    """
    path = Path(model_path)
    
    if not path.exists():
        logging.error(f"Model file does not exist: {path}")
        return False
    
    if not path.is_file():
        logging.error(f"Model path is not a file: {path}")
        return False
    
    settings = get_settings()
    supported_extensions = [f".{fmt}" for fmt in settings.supported_model_formats]
    
    if path.suffix.lower() not in supported_extensions:
        logging.error(f"Unsupported model format: {path.suffix}")
        return False
    
    # Check file size (basic validation)
    file_size_gb = path.stat().st_size / (1024**3)
    if file_size_gb > settings.model_memory_limit_gb:
        logging.warning(f"Model file size ({file_size_gb:.2f}GB) exceeds memory limit")
    
    logging.info(f"Model file validated: {path}")
    return True


def create_directories(settings: Settings) -> None:
    """
    Create required application directories.
    
    Args:
        settings: Application settings instance
    """
    directories = [
        settings.data_dir,
        settings.models_dir,
        settings.logs_dir,
        settings.cache_dir
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created/verified directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            raise


def setup_logging(settings: Settings) -> None:
    """
    Setup application logging configuration.
    
    Args:
        settings: Application settings instance
    """
    import logging.handlers
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, settings.log_level)
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(settings.log_format)
    
    # Console handler
    if settings.enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if settings.enable_file_logging:
        log_file = settings.logs_dir / "promptopt.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.log_file_max_size_mb * 1024 * 1024,
            backupCount=settings.log_file_backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(
        logging.INFO if settings.database_echo else logging.WARNING
    )


def get_database_url(settings: Optional[Settings] = None) -> str:
    """
    Get the complete database URL with absolute path.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        str: Complete database URL
    """
    if settings is None:
        settings = get_settings()
    
    db_path = settings.database_path
    return f"sqlite:///{db_path}"


def validate_settings() -> Dict[str, Any]:
    """
    Validate all settings and return validation report.
    
    Returns:
        Dict[str, Any]: Validation report with errors and warnings
    """
    settings = get_settings()
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Check model path if provided
    if settings.default_model_path:
        if not validate_model_path(settings.default_model_path):
            report["errors"].append(f"Invalid default model path: {settings.default_model_path}")
            report["valid"] = False
    
    # Check directory permissions
    for dir_name, directory in [
        ("data", settings.data_dir),
        ("models", settings.models_dir),
        ("logs", settings.logs_dir),
        ("cache", settings.cache_dir)
    ]:
        if not directory.exists():
            report["warnings"].append(f"{dir_name} directory does not exist: {directory}")
        elif not os.access(directory, os.W_OK):
            report["errors"].append(f"No write permission for {dir_name} directory: {directory}")
            report["valid"] = False
    
    # Check database path
    db_path = settings.database_path
    if db_path.exists() and not os.access(db_path, os.W_OK):
        report["errors"].append(f"No write permission for database: {db_path}")
        report["valid"] = False
    
    # Add info
    report["info"]["environment"] = settings.environment
    report["info"]["data_dir"] = str(settings.data_dir)
    report["info"]["database_path"] = str(settings.database_path)
    report["info"]["models_dir"] = str(settings.models_dir)
    
    return report


# Environment-specific configuration helpers
def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_settings().environment == "testing"


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().is_production


def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().is_development


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print(f"PromptOpt Co-Pilot Configuration")
    print(f"Environment: {settings.environment}")
    print(f"Data Directory: {settings.data_dir}")
    print(f"Models Directory: {settings.models_dir}")
    print(f"Database URL: {get_database_url(settings)}")
    
    # Validate settings
    validation_report = validate_settings()
    print(f"\nValidation Report:")
    print(f"Valid: {validation_report['valid']}")
    if validation_report['errors']:
        print(f"Errors: {validation_report['errors']}")
    if validation_report['warnings']:
        print(f"Warnings: {validation_report['warnings']}")
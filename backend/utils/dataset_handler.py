"""
Dataset Handler for PromptOpt Co-Pilot

This module provides comprehensive dataset loading, processing, and validation
capabilities for prompt optimization workflows.
"""

import json
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
import logging
import hashlib
import statistics
from enum import Enum
import re
import numpy as np
from datetime import datetime
import pickle
import os

# Mock imports for demonstration - replace with actual imports
try:
    from backend.core.config import get_dataset_settings
    from backend.core.database import DatabaseManager
except ImportError:
    # Mock classes for demonstration
    class DatabaseManager:
        def store_dataset_metadata(self, metadata: dict) -> None:
            pass
        
        def get_dataset_metadata(self, dataset_id: str) -> Optional[dict]:
            return None
    
    def get_dataset_settings() -> dict:
        return {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'cache_enabled': True,
            'cache_dir': './cache',
            'default_encoding': 'utf-8',
            'max_rows': 10000,
            'validation_strict': True
        }


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """Supported dataset formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"


class SamplingStrategy(Enum):
    """Dataset sampling strategies."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    BALANCED = "balanced"


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""
    DROP = "drop"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_ZERO = "fill_zero"
    FILL_CUSTOM = "fill_custom"


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    supported_formats: List[str] = field(default_factory=lambda: [f.value for f in DatasetFormat])
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    processing_options: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_rows: int = 10000
    default_encoding: str = "utf-8"
    validation_strict: bool = True


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion to the validation result."""
        self.suggestions.append(suggestion)


@dataclass
class DatasetInfo:
    """Metadata and statistics about a dataset."""
    name: str
    size: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Dict[str, Any]
    schema: Dict[str, Any]
    created_at: datetime
    file_path: str
    format: str
    checksum: str


@dataclass
class Dataset:
    """Dataset container with metadata."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    
    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self.data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset records."""
        return iter(self.data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        return pd.DataFrame(self.data)
    
    def get_column_names(self) -> List[str]:
        """Get all column names in the dataset."""
        if not self.data:
            return []
        return list(self.data[0].keys())
    
    def sample(self, n: int, random_state: Optional[int] = None) -> 'Dataset':
        """Sample n records from the dataset."""
        if random_state:
            np.random.seed(random_state)
        
        if n >= len(self.data):
            return self
        
        indices = np.random.choice(len(self.data), n, replace=False)
        sampled_data = [self.data[i] for i in indices]
        
        return Dataset(
            data=sampled_data,
            metadata=self.metadata.copy(),
            schema=self.schema.copy(),
            source_path=self.source_path
        )


class DatasetHandler:
    """Main class for handling dataset operations."""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize the dataset handler."""
        self.config = config or DatasetConfig()
        self.db_manager = DatabaseManager()
        self.settings = get_dataset_settings()
        self._cache = {}
        
        # Create cache directory if enabled
        if self.config.cache_enabled:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def load_dataset(self, file_path: str, format: Optional[str] = None, **kwargs) -> Dataset:
        """
        Load dataset from file with automatic format detection.
        
        Args:
            file_path: Path to the dataset file
            format: Dataset format (auto-detected if None)
            **kwargs: Additional arguments for format-specific loaders
            
        Returns:
            Dataset object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
            IOError: If file cannot be read
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if file_path.stat().st_size > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed size: {self.config.max_file_size}")
        
        # Auto-detect format if not provided
        if format is None:
            format = self._detect_format(file_path)
        
        if format not in self.config.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        # Check cache first
        cache_key = self._get_cache_key(file_path)
        if self.config.cache_enabled and cache_key in self._cache:
            logger.info(f"Loading dataset from cache: {file_path}")
            return self._cache[cache_key]
        
        logger.info(f"Loading dataset from file: {file_path} (format: {format})")
        
        # Load data based on format
        try:
            if format == DatasetFormat.CSV.value:
                data = self.load_csv(str(file_path), **kwargs)
            elif format == DatasetFormat.JSON.value:
                data = self.load_json(str(file_path))
            elif format == DatasetFormat.JSONL.value:
                data = self.load_jsonl(str(file_path))
            elif format == DatasetFormat.PARQUET.value:
                data = self.load_parquet(str(file_path), **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Create dataset object
            dataset = Dataset(
                data=data,
                metadata={
                    'source_path': str(file_path),
                    'format': format,
                    'loaded_at': datetime.now().isoformat(),
                    'size': len(data)
                },
                schema=self._infer_schema(data),
                source_path=str(file_path)
            )
            
            # Cache the dataset
            if self.config.cache_enabled:
                self._cache[cache_key] = dataset
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise IOError(f"Failed to load dataset: {e}")
    
    def validate_dataset(self, dataset: Dataset) -> ValidationResult:
        """
        Validate dataset structure and content.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(is_valid=True)
        
        # Check if dataset is empty
        if not dataset.data:
            result.add_error("Dataset is empty")
            return result
        
        # Check for consistent column structure
        if not self._validate_column_consistency(dataset.data):
            result.add_error("Inconsistent column structure across records")
        
        # Check for required fields based on schema
        if dataset.schema:
            self._validate_schema_compliance(dataset.data, dataset.schema, result)
        
        # Check for data quality issues
        self._validate_data_quality(dataset.data, result)
        
        # Check dataset size limits
        if len(dataset.data) > self.config.max_rows:
            result.add_warning(f"Dataset size ({len(dataset.data)}) exceeds recommended limit ({self.config.max_rows})")
        
        # Add suggestions for optimization
        self._add_optimization_suggestions(dataset, result)
        
        return result
    
    def process_dataset(self, dataset: Dataset, processing_config: Dict[str, Any]) -> Dataset:
        """
        Apply transformations to the dataset.
        
        Args:
            dataset: Dataset to process
            processing_config: Configuration for processing steps
            
        Returns:
            Processed dataset
        """
        data = dataset.data.copy()
        
        # Apply normalization
        if processing_config.get('normalize_text', False):
            data = self.normalize_text_fields(data)
        
        # Handle missing values
        missing_strategy = processing_config.get('missing_value_strategy')
        if missing_strategy:
            data = self.handle_missing_values(data, missing_strategy)
        
        # Apply filters
        filters = processing_config.get('filters')
        if filters:
            data = self.apply_filters(data, filters)
        
        # Apply custom transformations
        transformations = processing_config.get('transformations', [])
        for transform in transformations:
            data = self._apply_transformation(data, transform)
        
        return Dataset(
            data=data,
            metadata={**dataset.metadata, 'processed_at': datetime.now().isoformat()},
            schema=dataset.schema,
            source_path=dataset.source_path
        )
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8, 
                     seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into training and testing sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio of data for training (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0.0 and 1.0")
        
        if seed:
            np.random.seed(seed)
        
        data = dataset.data.copy()
        np.random.shuffle(data)
        
        split_index = int(len(data) * train_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        train_dataset = Dataset(
            data=train_data,
            metadata={**dataset.metadata, 'split': 'train', 'split_ratio': train_ratio},
            schema=dataset.schema,
            source_path=dataset.source_path
        )
        
        test_dataset = Dataset(
            data=test_data,
            metadata={**dataset.metadata, 'split': 'test', 'split_ratio': 1.0 - train_ratio},
            schema=dataset.schema,
            source_path=dataset.source_path
        )
        
        return train_dataset, test_dataset
    
    def sample_dataset(self, dataset: Dataset, n_samples: int, 
                      strategy: str = SamplingStrategy.RANDOM.value,
                      **kwargs) -> Dataset:
        """
        Sample records from the dataset.
        
        Args:
            dataset: Dataset to sample from
            n_samples: Number of samples to draw
            strategy: Sampling strategy
            **kwargs: Additional arguments for sampling strategy
            
        Returns:
            Sampled dataset
        """
        if n_samples >= len(dataset.data):
            return dataset
        
        if strategy == SamplingStrategy.RANDOM.value:
            return self._random_sample(dataset, n_samples, **kwargs)
        elif strategy == SamplingStrategy.STRATIFIED.value:
            return self._stratified_sample(dataset, n_samples, **kwargs)
        elif strategy == SamplingStrategy.SYSTEMATIC.value:
            return self._systematic_sample(dataset, n_samples)
        elif strategy == SamplingStrategy.BALANCED.value:
            return self._balanced_sample(dataset, n_samples, **kwargs)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
    
    def get_dataset_info(self, dataset: Dataset) -> DatasetInfo:
        """
        Get comprehensive information about the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            DatasetInfo with metadata and statistics
        """
        df = dataset.to_dataframe()
        
        # Calculate statistics
        stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats[col] = {
                    'mean': float(df[col].mean()) if not df[col].empty else 0,
                    'median': float(df[col].median()) if not df[col].empty else 0,
                    'std': float(df[col].std()) if not df[col].empty else 0,
                    'min': float(df[col].min()) if not df[col].empty else 0,
                    'max': float(df[col].max()) if not df[col].empty else 0
                }
            elif df[col].dtype == 'object':
                stats[col] = {
                    'unique_count': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'avg_length': float(df[col].astype(str).str.len().mean()) if not df[col].empty else 0
                }
        
        # Calculate checksum
        checksum = self._calculate_checksum(dataset.data)
        
        return DatasetInfo(
            name=Path(dataset.source_path).stem if dataset.source_path else "unknown",
            size=len(dataset.data),
            columns=dataset.get_column_names(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values={col: int(df[col].isnull().sum()) for col in df.columns},
            statistics=stats,
            schema=dataset.schema,
            created_at=datetime.now(),
            file_path=dataset.source_path,
            format=dataset.metadata.get('format', 'unknown'),
            checksum=checksum
        )
    
    # Format-specific loaders
    
    def load_csv(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        try:
            encoding = kwargs.get('encoding', self.config.default_encoding)
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            return df.to_dict('records')
        except Exception as e:
            raise IOError(f"Failed to load CSV file: {e}")
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding=self.config.default_encoding) as f:
                data = json.load(f)
            
            # Ensure data is a list of dictionaries
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON data must be a list or dictionary")
            
            return data
        except Exception as e:
            raise IOError(f"Failed to load JSON file: {e}")
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        try:
            data = []
            with open(file_path, 'r', encoding=self.config.default_encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            return data
        except Exception as e:
            raise IOError(f"Failed to load JSONL file: {e}")
    
    def load_parquet(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Load data from Parquet file."""
        try:
            df = pd.read_parquet(file_path, **kwargs)
            return df.to_dict('records')
        except Exception as e:
            raise IOError(f"Failed to load Parquet file: {e}")
    
    # Data processing functions
    
    def normalize_text_fields(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize text fields in the dataset."""
        normalized_data = []
        
        for record in data:
            normalized_record = {}
            for key, value in record.items():
                if isinstance(value, str):
                    # Basic text normalization
                    normalized_value = value.strip().lower()
                    normalized_value = re.sub(r'\s+', ' ', normalized_value)
                    normalized_record[key] = normalized_value
                else:
                    normalized_record[key] = value
            normalized_data.append(normalized_record)
        
        return normalized_data
    
    def handle_missing_values(self, data: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        """Handle missing values in the dataset."""
        if not data:
            return data
        
        df = pd.DataFrame(data)
        
        if strategy == MissingValueStrategy.DROP.value:
            df = df.dropna()
        elif strategy == MissingValueStrategy.FILL_MEAN.value:
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == MissingValueStrategy.FILL_MEDIAN.value:
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == MissingValueStrategy.FILL_MODE.value:
            df = df.fillna(df.mode().iloc[0])
        elif strategy == MissingValueStrategy.FILL_ZERO.value:
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported missing value strategy: {strategy}")
        
        return df.to_dict('records')
    
    def apply_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to the dataset."""
        filtered_data = []
        
        for record in data:
            include_record = True
            
            for field, conditions in filters.items():
                if field not in record:
                    continue
                
                value = record[field]
                
                # Handle different filter conditions
                if isinstance(conditions, dict):
                    for op, expected in conditions.items():
                        if op == 'eq' and value != expected:
                            include_record = False
                            break
                        elif op == 'ne' and value == expected:
                            include_record = False
                            break
                        elif op == 'gt' and value <= expected:
                            include_record = False
                            break
                        elif op == 'lt' and value >= expected:
                            include_record = False
                            break
                        elif op == 'in' and value not in expected:
                            include_record = False
                            break
                        elif op == 'not_in' and value in expected:
                            include_record = False
                            break
                else:
                    # Simple equality filter
                    if value != conditions:
                        include_record = False
                
                if not include_record:
                    break
            
            if include_record:
                filtered_data.append(record)
        
        return filtered_data
    
    # Private helper methods
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect dataset format from file extension."""
        extension = file_path.suffix.lower()
        
        format_map = {
            '.csv': DatasetFormat.CSV.value,
            '.json': DatasetFormat.JSON.value,
            '.jsonl': DatasetFormat.JSONL.value,
            '.parquet': DatasetFormat.PARQUET.value
        }
        
        return format_map.get(extension, DatasetFormat.JSON.value)
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for the dataset."""
        stat = file_path.stat()
        return f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    
    def _infer_schema(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer schema from dataset."""
        if not data:
            return {}
        
        schema = {}
        df = pd.DataFrame(data)
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema[col] = {
                'type': dtype,
                'nullable': df[col].isnull().any(),
                'unique_count': df[col].nunique()
            }
        
        return schema
    
    def _validate_column_consistency(self, data: List[Dict[str, Any]]) -> bool:
        """Check if all records have consistent column structure."""
        if not data:
            return True
        
        first_keys = set(data[0].keys())
        return all(set(record.keys()) == first_keys for record in data)
    
    def _validate_schema_compliance(self, data: List[Dict[str, Any]], 
                                  schema: Dict[str, Any], result: ValidationResult) -> None:
        """Validate data against schema."""
        if not data or not schema:
            return
        
        df = pd.DataFrame(data)
        
        for col, col_schema in schema.items():
            if col not in df.columns:
                result.add_error(f"Required column '{col}' is missing")
                continue
            
            # Check data type compliance
            expected_type = col_schema.get('type')
            if expected_type and str(df[col].dtype) != expected_type:
                result.add_warning(f"Column '{col}' has type {df[col].dtype}, expected {expected_type}")
            
            # Check nullability
            if not col_schema.get('nullable', True) and df[col].isnull().any():
                result.add_error(f"Column '{col}' contains null values but is marked as non-nullable")
    
    def _validate_data_quality(self, data: List[Dict[str, Any]], result: ValidationResult) -> None:
        """Validate data quality."""
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Check for completely empty columns
        for col in df.columns:
            if df[col].isnull().all():
                result.add_warning(f"Column '{col}' is completely empty")
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            result.add_warning(f"Found {duplicate_count} duplicate records")
        
        # Check for high missing value rates
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            if missing_rate > 0.5:
                result.add_warning(f"Column '{col}' has high missing value rate: {missing_rate:.2%}")
    
    def _add_optimization_suggestions(self, dataset: Dataset, result: ValidationResult) -> None:
        """Add optimization suggestions."""
        df = dataset.to_dataframe()
        
        # Suggest data type optimizations
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:
                    result.add_suggestion(f"Consider converting '{col}' to categorical type")
        
        # Suggest sampling for large datasets
        if len(dataset.data) > 10000:
            result.add_suggestion("Consider sampling the dataset for faster processing")
    
    def _calculate_checksum(self, data: List[Dict[str, Any]]) -> str:
        """Calculate checksum for dataset."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _apply_transformation(self, data: List[Dict[str, Any]], transform: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a single transformation to the data."""
        # This is a placeholder for custom transformations
        # Implementation would depend on specific transformation requirements
        return data
    
    def _random_sample(self, dataset: Dataset, n_samples: int, **kwargs) -> Dataset:
        """Perform random sampling."""
        seed = kwargs.get('seed')
        return dataset.sample(n_samples, seed)
    
    def _stratified_sample(self, dataset: Dataset, n_samples: int, **kwargs) -> Dataset:
        """Perform stratified sampling."""
        stratify_column = kwargs.get('stratify_column')
        if not stratify_column:
            raise ValueError("stratify_column is required for stratified sampling")
        
        df = dataset.to_dataframe()
        if stratify_column not in df.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in dataset")
        
        # Perform stratified sampling
        sampled_df = df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(n_samples * len(x) / len(df)))))
        )
        
        return Dataset(
            data=sampled_df.to_dict('records'),
            metadata=dataset.metadata.copy(),
            schema=dataset.schema.copy(),
            source_path=dataset.source_path
        )
    
    def _systematic_sample(self, dataset: Dataset, n_samples: int) -> Dataset:
        """Perform systematic sampling."""
        interval = len(dataset.data) // n_samples
        if interval == 0:
            interval = 1
        
        sampled_data = [dataset.data[i] for i in range(0, len(dataset.data), interval)][:n_samples]
        
        return Dataset(
            data=sampled_data,
            metadata=dataset.metadata.copy(),
            schema=dataset.schema.copy(),
            source_path=dataset.source_path
        )
    
    def _balanced_sample(self, dataset: Dataset, n_samples: int, **kwargs) -> Dataset:
        """Perform balanced sampling."""
        balance_column = kwargs.get('balance_column')
        if not balance_column:
            raise ValueError("balance_column is required for balanced sampling")
        
        df = dataset.to_dataframe()
        if balance_column not in df.columns:
            raise ValueError(f"Balance column '{balance_column}' not found in dataset")
        
        # Calculate samples per class
        unique_values = df[balance_column].unique()
        samples_per_class = n_samples // len(unique_values)
        
        sampled_data = []
        for value in unique_values:
            class_data = df[df[balance_column] == value]
            sampled_class = class_data.sample(min(len(class_data), samples_per_class))
            sampled_data.extend(sampled_class.to_dict('records'))
        
        return Dataset(
            data=sampled_data,
            metadata=dataset.metadata.copy(),
            schema=dataset.schema.copy(),
            source_path=dataset.source_path
        )


# Utility functions for testing and examples

def create_sample_dataset(format_type: str = "json", size: int = 100) -> str:
    """Create a sample dataset for testing."""
    import tempfile
    import random
    
    # Generate sample data
    data = []
    for i in range(size):
        record = {
            "id": i,
            "input": f"Sample input text {i}",
            "output": f"Sample output {i}",
            "category": random.choice(["A", "B", "C"]),
            "score": random.uniform(0, 1),
            "metadata": {"source": "generated", "index": i}
        }
        data.append(record)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as f:
        if format_type == "json":
            json.dump(data, f, indent=2)
        elif format_type == "jsonl":
            for record in data:
                f.write(json.dumps(record) + '\n')
        elif format_type == "csv":
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
    
    return f.name


if __name__ == "__main__":
    # Example usage
    handler = DatasetHandler()
    
    # Create sample dataset
    sample_file = create_sample_dataset("json", 50)
    
    try:
        # Load dataset
        dataset = handler.load_dataset(sample_file)
        print(f"Loaded dataset with {len(dataset)} records")
        
        # Validate dataset
        validation_result = handler.validate_dataset(dataset)
        print(f"Validation result: {'Valid' if validation_result.is_valid else 'Invalid'}")
        if validation_result.errors:
            print("Errors:", validation_result.errors)
        if validation_result.warnings:
            print("Warnings:", validation_result.warnings)
        if validation_result.suggestions:
            print("Suggestions:", validation_result.suggestions)
        
        # Get dataset info
        dataset_info = handler.get_dataset_info(dataset)
        print(f"Dataset info: {dataset_info.name}, {dataset_info.size} records")
        print(f"Columns: {dataset_info.columns}")
        print(f"Missing values: {dataset_info.missing_values}")
        
        # Process dataset
        processing_config = {
            'normalize_text': True,
            'missing_value_strategy': 'fill_zero',
            'filters': {
                'category': {'in': ['A', 'B']}
            }
        }
        processed_dataset = handler.process_dataset(dataset, processing_config)
        print(f"Processed dataset has {len(processed_dataset)} records")
        
        # Split dataset
        train_dataset, test_dataset = handler.split_dataset(processed_dataset, train_ratio=0.8, seed=42)
        print(f"Train set: {len(train_dataset)} records, Test set: {len(test_dataset)} records")
        
        # Sample dataset
        sampled_dataset = handler.sample_dataset(
            dataset, 
            n_samples=20, 
            strategy=SamplingStrategy.STRATIFIED.value,
            stratify_column='category'
        )
        print(f"Sampled dataset has {len(sampled_dataset)} records")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        import os
        if os.path.exists(sample_file):
            os.unlink(sample_file)
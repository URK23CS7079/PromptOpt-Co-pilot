"""
Model Management System for PromptOpt Co-Pilot

This module provides centralized model discovery, validation, and lifecycle management
for local GGUF models used in prompt optimization tasks.

Key Features:
- Automatic GGUF model discovery
- Memory requirement analysis and RAM fitting hints
- Model metadata extraction and caching
- Efficient model loading/unloading with resource management
- System resource monitoring and recommendations
"""

import os
import re
import struct
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager

import psutil

# Internal imports
from backend.llm.llama_wrapper import LlamaWrapper
from backend.core.config import get_model_settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a discovered GGUF model."""
    
    name: str
    path: str
    size_bytes: int
    quantization: Optional[str] = None
    parameter_count: Optional[str] = None
    fits_in_ram: bool = False
    architecture: Optional[str] = None
    vocab_size: Optional[int] = None
    context_length: Optional[int] = None
    last_modified: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Get model size in gigabytes."""
        return self.size_bytes / (1024 * 1024 * 1024)


@dataclass
class MemoryRequirement:
    """Memory requirements and system compatibility analysis."""
    
    required_ram_gb: float
    available_ram_gb: float
    fits_in_ram: bool
    recommendation: str
    overhead_gb: float = 0.0
    swap_available_gb: float = 0.0
    confidence: float = 1.0  # 0.0 to 1.0, how confident we are in the estimate


class ModelDiscoveryError(Exception):
    """Raised when model discovery fails."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelManager:
    """
    Centralized manager for GGUF model discovery, validation, and lifecycle management.
    
    This class handles:
    - Automatic discovery of GGUF files in configured directories
    - Model metadata extraction and caching
    - Memory requirement analysis
    - Model loading/unloading with resource management
    - System compatibility recommendations
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the ModelManager.
        
        Args:
            models_dir: Directory containing GGUF models. If None, uses config default.
        """
        self.models_dir = Path(models_dir or get_model_settings().get("models_dir", "./models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for model info and metadata
        self._model_cache: Dict[str, ModelInfo] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Loaded models tracking
        self._loaded_models: Dict[str, LlamaWrapper] = {}
        self._loading_futures: Dict[str, Future] = {}
        self._load_lock = threading.RLock()
        
        # Thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelManager")
        
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
    
    def discover_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        Discover all GGUF models in the configured directory.
        
        Args:
            force_refresh: If True, bypass cache and re-scan filesystem
            
        Returns:
            List of ModelInfo objects for discovered models
            
        Raises:
            ModelDiscoveryError: If discovery fails
        """
        try:
            with self._cache_lock:
                if not force_refresh and self._model_cache:
                    return list(self._model_cache.values())
                
                logger.info(f"Discovering models in {self.models_dir}")
                discovered_models = []
                
                # Find all .gguf files recursively
                gguf_files = list(self.models_dir.rglob("*.gguf"))
                
                if not gguf_files:
                    logger.warning(f"No GGUF files found in {self.models_dir}")
                    return []
                
                for gguf_path in gguf_files:
                    try:
                        model_info = self.get_model_info(str(gguf_path))
                        discovered_models.append(model_info)
                        self._model_cache[model_info.name] = model_info
                        logger.debug(f"Discovered model: {model_info.name}")
                    except Exception as e:
                        logger.error(f"Failed to process {gguf_path}: {e}")
                        continue
                
                logger.info(f"Discovered {len(discovered_models)} models")
                return discovered_models
                
        except Exception as e:
            raise ModelDiscoveryError(f"Model discovery failed: {e}") from e
    
    def get_model_info(self, model_path: str) -> ModelInfo:
        """
        Extract comprehensive information about a GGUF model.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            ModelInfo object with model details
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If file is not a valid GGUF model
        """
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not path.suffix.lower() == '.gguf':
            raise ValueError(f"File is not a GGUF model: {model_path}")
        
        # Extract model name from filename
        model_name = path.stem
        
        # Get file stats
        stat = path.stat()
        size_bytes = stat.st_size
        last_modified = stat.st_mtime
        
        # Parse GGUF metadata
        try:
            metadata = self.parse_gguf_metadata(model_path)
        except Exception as e:
            logger.warning(f"Failed to parse GGUF metadata for {model_path}: {e}")
            metadata = {}
        
        # Extract key information from metadata
        quantization = self._extract_quantization(metadata, model_name)
        parameter_count = self._extract_parameter_count(metadata, model_name)
        architecture = metadata.get('general.architecture', 'unknown')
        vocab_size = metadata.get('tokenizer.ggml.tokens', {}).get('len', None)
        context_length = metadata.get('llama.context_length', None)
        
        # Check memory requirements
        memory_req = self.check_memory_requirements(model_path)
        
        return ModelInfo(
            name=model_name,
            path=str(path),
            size_bytes=size_bytes,
            quantization=quantization,
            parameter_count=parameter_count,
            fits_in_ram=memory_req.fits_in_ram,
            architecture=architecture,
            vocab_size=vocab_size,
            context_length=context_length,
            last_modified=last_modified,
            metadata=metadata
        )
    
    def check_memory_requirements(self, model_path: str) -> MemoryRequirement:
        """
        Analyze memory requirements for a model and system compatibility.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            MemoryRequirement object with analysis results
        """
        path = Path(model_path)
        model_size_bytes = path.stat().st_size
        
        # Get system memory info
        memory_info = self.get_system_memory()
        available_ram_gb = memory_info['available_gb']
        total_ram_gb = memory_info['total_gb']
        swap_gb = memory_info.get('swap_available_gb', 0.0)
        
        # Estimate memory requirements
        required_ram_gb = self.estimate_inference_memory(model_size_bytes)
        
        # Calculate overhead (context, KV cache, etc.)
        overhead_gb = max(0.5, required_ram_gb * 0.2)  # At least 0.5GB, or 20% of model size
        total_required = required_ram_gb + overhead_gb
        
        # Determine if it fits in RAM
        fits_in_ram = total_required <= available_ram_gb
        
        # Generate recommendation
        if fits_in_ram:
            if available_ram_gb - total_required > 2.0:
                recommendation = "Excellent fit - plenty of RAM available"
            elif available_ram_gb - total_required > 1.0:
                recommendation = "Good fit - adequate RAM available"
            else:
                recommendation = "Tight fit - may impact system performance"
        else:
            deficit = total_required - available_ram_gb
            if deficit <= swap_gb:
                recommendation = f"Will use swap ({deficit:.1f}GB) - expect slower performance"
            else:
                recommendation = f"Insufficient memory - need {deficit:.1f}GB more RAM"
        
        # Confidence based on model size and quantization detection
        confidence = 0.8  # Base confidence
        if 'q4' in path.name.lower():
            confidence = 0.9
        elif 'q8' in path.name.lower():
            confidence = 0.85
        elif model_size_bytes < 1024 * 1024 * 1024:  # < 1GB
            confidence = 0.95
        
        return MemoryRequirement(
            required_ram_gb=required_ram_gb,
            available_ram_gb=available_ram_gb,
            fits_in_ram=fits_in_ram,
            recommendation=recommendation,
            overhead_gb=overhead_gb,
            swap_available_gb=swap_gb,
            confidence=confidence
        )
    
    def load_model(self, model_name: str, **kwargs) -> LlamaWrapper:
        """
        Load a model into memory and return a LlamaWrapper instance.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments passed to LlamaWrapper
            
        Returns:
            LlamaWrapper instance for the loaded model
            
        Raises:
            ModelLoadError: If model loading fails
            ValueError: If model not found
        """
        with self._load_lock:
            # Check if already loaded
            if model_name in self._loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return self._loaded_models[model_name]
            
            # Check if currently loading
            if model_name in self._loading_futures:
                logger.info(f"Model {model_name} is currently loading, waiting...")
                future = self._loading_futures[model_name]
                try:
                    return future.result(timeout=300)  # 5 minute timeout
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    self._loading_futures.pop(model_name, None)
                    raise ModelLoadError(f"Model loading failed: {e}") from e
            
            # Find model info
            models = self.discover_models()
            model_info = next((m for m in models if m.name == model_name), None)
            
            if not model_info:
                raise ValueError(f"Model not found: {model_name}")
            
            # Check memory before loading
            memory_req = self.check_memory_requirements(model_info.path)
            if not memory_req.fits_in_ram:
                logger.warning(f"Loading {model_name} may cause memory issues: {memory_req.recommendation}")
            
            # Start async loading
            future = self._executor.submit(self._load_model_async, model_info, **kwargs)
            self._loading_futures[model_name] = future
            
            try:
                model_wrapper = future.result(timeout=300)  # 5 minute timeout
                self._loaded_models[model_name] = model_wrapper
                self._loading_futures.pop(model_name, None)
                
                logger.info(f"Successfully loaded model: {model_name}")
                return model_wrapper
                
            except Exception as e:
                self._loading_futures.pop(model_name, None)
                logger.error(f"Failed to load model {model_name}: {e}")
                raise ModelLoadError(f"Failed to load {model_name}: {e}") from e
    
    def _load_model_async(self, model_info: ModelInfo, **kwargs) -> LlamaWrapper:
        """
        Async helper for loading models.
        
        Args:
            model_info: Information about the model to load
            **kwargs: Additional arguments for LlamaWrapper
            
        Returns:
            LlamaWrapper instance
        """
        logger.info(f"Loading model: {model_info.name} from {model_info.path}")
        
        # Default parameters optimized for inference
        load_params = {
            'model_path': model_info.path,
            'n_ctx': kwargs.get('n_ctx', 2048),
            'n_batch': kwargs.get('n_batch', 512),
            'n_threads': kwargs.get('n_threads', min(8, os.cpu_count() or 4)),
            'verbose': kwargs.get('verbose', False)
        }
        
        # Add any additional parameters
        load_params.update({k: v for k, v in kwargs.items() if k not in load_params})
        
        return LlamaWrapper(**load_params)
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if it wasn't loaded
        """
        with self._load_lock:
            if model_name not in self._loaded_models:
                logger.warning(f"Model {model_name} is not loaded")
                return False
            
            try:
                model_wrapper = self._loaded_models.pop(model_name)
                # Cleanup model wrapper if it has cleanup methods
                if hasattr(model_wrapper, 'cleanup'):
                    model_wrapper.cleanup()
                elif hasattr(model_wrapper, 'close'):
                    model_wrapper.close()
                
                logger.info(f"Unloaded model: {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                return False
    
    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of all discoverable models.
        
        Returns:
            List of ModelInfo objects for available models
        """
        return self.discover_models()
    
    def get_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model names.
        
        Returns:
            List of loaded model names
        """
        with self._load_lock:
            return list(self._loaded_models.keys())
    
    def recommend_model(self) -> Optional[str]:
        """
        Recommend the best model for the current system based on available RAM and models.
        
        Returns:
            Name of recommended model, or None if no suitable model found
        """
        models = self.discover_models()
        if not models:
            return None
        
        # Filter models that fit in RAM
        suitable_models = [m for m in models if m.fits_in_ram]
        
        if not suitable_models:
            logger.warning("No models fit in available RAM")
            # Fall back to smallest model
            suitable_models = sorted(models, key=lambda m: m.size_bytes)[:1]
        
        # Prefer models with good quantization (Q4, Q5) and reasonable size
        def model_score(model: ModelInfo) -> Tuple[int, int, int]:
            """Score models for recommendation (higher is better)."""
            # Quantization preference: Q4/Q5 > Q8 > Q2/Q3 > others
            quant_score = 0
            if model.quantization:
                if 'q4' in model.quantization.lower() or 'q5' in model.quantization.lower():
                    quant_score = 3
                elif 'q8' in model.quantization.lower():
                    quant_score = 2
                elif 'q2' in model.quantization.lower() or 'q3' in model.quantization.lower():
                    quant_score = 1
            
            # Size preference: 3-7B models preferred
            size_gb = model.size_gb
            if 2 <= size_gb <= 8:
                size_score = 3
            elif 1 <= size_gb <= 10:
                size_score = 2
            else:
                size_score = 1
            
            # RAM fit preference
            ram_score = 3 if model.fits_in_ram else 1
            
            return (ram_score, quant_score, size_score)
        
        # Select best model
        best_model = max(suitable_models, key=model_score)
        
        logger.info(f"Recommended model: {best_model.name} ({best_model.size_gb:.1f}GB)")
        return best_model.name
    
    def parse_gguf_metadata(self, path: str) -> Dict[str, Any]:
        """
        Parse metadata from a GGUF file header.
        
        Args:
            path: Path to the GGUF file
            
        Returns:
            Dictionary containing extracted metadata
            
        Raises:
            ValueError: If file is not a valid GGUF file
        """
        # Check cache first
        with self._cache_lock:
            if path in self._metadata_cache:
                return self._metadata_cache[path].copy()
        
        try:
            metadata = {}
            
            with open(path, 'rb') as f:
                # Read GGUF header
                magic = f.read(4)
                if magic != b'GGUF':
                    raise ValueError("Not a valid GGUF file")
                
                # Read version
                version = struct.unpack('<I', f.read(4))[0]
                metadata['gguf_version'] = version
                
                # Read tensor count and metadata count
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                metadata['tensor_count'] = tensor_count
                metadata['metadata_count'] = metadata_count
                
                # Read metadata key-value pairs
                for _ in range(metadata_count):
                    try:
                        key = self._read_gguf_string(f)
                        value_type = struct.unpack('<I', f.read(4))[0]
                        value = self._read_gguf_value(f, value_type)
                        metadata[key] = value
                    except Exception as e:
                        logger.debug(f"Failed to read metadata pair: {e}")
                        continue
            
            # Cache the result
            with self._cache_lock:
                self._metadata_cache[path] = metadata.copy()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to parse GGUF metadata from {path}: {e}")
            return {}
    
    def _read_gguf_string(self, f) -> str:
        """Read a string from GGUF file."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8', errors='ignore')
    
    def _read_gguf_value(self, f, value_type: int) -> Any:
        """Read a value from GGUF file based on type."""
        # Simplified type reading - extend as needed
        if value_type == 8:  # String type
            return self._read_gguf_string(f)
        elif value_type == 4:  # Uint32
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == 5:  # Int32
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == 6:  # Float32
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == 0:  # Bool
            return struct.unpack('<?', f.read(1))[0]
        else:
            # Skip unknown types
            logger.debug(f"Skipping unknown GGUF value type: {value_type}")
            return None
    
    def _extract_quantization(self, metadata: Dict[str, Any], filename: str) -> Optional[str]:
        """Extract quantization information from metadata or filename."""
        # Try metadata first
        if 'general.file_type' in metadata:
            file_type = metadata['general.file_type']
            # Map GGUF file types to quantization names
            type_map = {
                0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1',
                6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K',
                11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L',
                14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M',
                18: 'Q6_K', 19: 'Q8_K'
            }
            if file_type in type_map:
                return type_map[file_type]
        
        # Fall back to filename pattern matching
        filename_lower = filename.lower()
        quant_patterns = [
            r'q([2-8])_k[_-]?([sml])?',
            r'q([2-8])_([01])',
            r'q([2-8])',
            r'f(16|32)'
        ]
        
        for pattern in quant_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return match.group(0).upper()
        
        return None
    
    def _extract_parameter_count(self, metadata: Dict[str, Any], filename: str) -> Optional[str]:
        """Extract parameter count from metadata or filename."""
        # Try metadata
        if 'llama.block_count' in metadata and 'llama.embedding_length' in metadata:
            # Rough estimation for transformer models
            layers = metadata['llama.block_count']
            hidden_size = metadata['llama.embedding_length']
            # Very rough parameter estimation
            params_approx = (layers * hidden_size * hidden_size * 12) // 1000000
            if params_approx >= 1000:
                return f"{params_approx // 1000}B"
            else:
                return f"{params_approx}M"
        
        # Fall back to filename patterns
        filename_lower = filename.lower()
        param_patterns = [
            r'(\d+)b',  # 7b, 13b, etc.
            r'(\d+)\.?\d*b',  # 7.5b, etc.
            r'(\d+)m'   # 125m, etc.
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return match.group(0).upper()
        
        return None
    
    @staticmethod
    def estimate_inference_memory(model_size_bytes: int) -> float:
        """
        Estimate memory requirements for model inference.
        
        Args:
            model_size_bytes: Size of the model file in bytes
            
        Returns:
            Estimated RAM requirement in GB
        """
        # Base model size
        model_gb = model_size_bytes / (1024 ** 3)
        
        # Add overhead for:
        # - Model loading and decompression: 1.2x
        # - KV cache and context: +0.5GB to +2GB based on model size
        # - System overhead: +0.3GB
        
        overhead_multiplier = 1.2
        context_overhead = min(2.0, max(0.5, model_gb * 0.3))
        system_overhead = 0.3
        
        total_gb = (model_gb * overhead_multiplier) + context_overhead + system_overhead
        
        return round(total_gb, 2)
    
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """
        Get current system memory information.
        
        Returns:
            Dictionary with memory information in GB
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': round(memory.total / (1024 ** 3), 2),
                'available_gb': round(memory.available / (1024 ** 3), 2),
                'used_gb': round(memory.used / (1024 ** 3), 2),
                'percent_used': memory.percent,
                'swap_total_gb': round(swap.total / (1024 ** 3), 2),
                'swap_available_gb': round((swap.total - swap.used) / (1024 ** 3), 2),
                'swap_percent_used': swap.percent
            }
        except Exception as e:
            logger.error(f"Failed to get system memory info: {e}")
            return {
                'total_gb': 8.0,  # Fallback values
                'available_gb': 4.0,
                'used_gb': 4.0,
                'percent_used': 50.0,
                'swap_total_gb': 0.0,
                'swap_available_gb': 0.0,
                'swap_percent_used': 0.0
            }
    
    @contextmanager
    def temporary_model(self, model_name: str, **kwargs):
        """
        Context manager for temporary model loading.
        
        Args:
            model_name: Name of model to load temporarily
            **kwargs: Arguments for model loading
            
        Yields:
            LlamaWrapper instance
        """
        model = None
        try:
            model = self.load_model(model_name, **kwargs)
            yield model
        finally:
            if model:
                self.unload_model(model_name)
    
    def cleanup(self):
        """Clean up resources and unload all models."""
        logger.info("Cleaning up ModelManager...")
        
        # Unload all models
        with self._load_lock:
            for model_name in list(self._loaded_models.keys()):
                self.unload_model(model_name)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("ModelManager cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Unit tests
if __name__ == "__main__":
    import unittest
    import tempfile
    import shutil
    
    class TestModelManager(unittest.TestCase):
        """Unit tests for ModelManager."""
        
        def setUp(self):
            """Set up test environment."""
            self.test_dir = tempfile.mkdtemp()
            self.manager = ModelManager(self.test_dir)
        
        def tearDown(self):
            """Clean up test environment."""
            self.manager.cleanup()
            shutil.rmtree(self.test_dir, ignore_errors=True)
        
        def test_model_discovery_empty_dir(self):
            """Test model discovery in empty directory."""
            models = self.manager.discover_models()
            self.assertEqual(len(models), 0)
        
        def test_memory_estimation(self):
            """Test memory requirement estimation."""
            # Test with 4GB model
            model_size = 4 * 1024 * 1024 * 1024  # 4GB
            estimated = ModelManager.estimate_inference_memory(model_size)
            self.assertGreater(estimated, 4.0)
            self.assertLess(estimated, 10.0)
        
        def test_system_memory(self):
            """Test system memory detection."""
            memory_info = ModelManager.get_system_memory()
            self.assertIn('total_gb', memory_info)
            self.assertIn('available_gb', memory_info)
            self.assertGreater(memory_info['total_gb'], 0)
        
        def test_quantization_extraction(self):
            """Test quantization extraction from filename."""
            test_cases = [
                ("model-7b-q4_0.gguf", "Q4_0"),
                ("llama-13b-q8_k_m.gguf", "Q8_K_M"),
                ("model-f16.gguf", "F16"),
                ("regular-model.gguf", None)
            ]
            
            for filename, expected in test_cases:
                result = self.manager._extract_quantization({}, filename)
                self.assertEqual(result, expected)
        
        def test_parameter_extraction(self):
            """Test parameter count extraction from filename."""
            test_cases = [
                ("model-7b-q4.gguf", "7B"),
                ("llama-13b.gguf", "13B"),
                ("small-125m.gguf", "125M"),
                ("model.gguf", None)
            ]
            
            for filename, expected in test_cases:
                result = self.manager._extract_parameter_count({}, filename)
                self.assertEqual(result, expected)
    
    # Run tests
    unittest.main(verbosity=2)
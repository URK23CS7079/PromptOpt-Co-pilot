"""
PromptOpt Co-Pilot - Llama.cpp Wrapper

A comprehensive wrapper for llama.cpp providing OpenAI-compatible REST interface
for prompt optimization using local GGUF models.

Author: PromptOpt Co-Pilot Development Team
License: MIT
"""

import asyncio
import logging
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import json

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

try:
    from backend.core.config import ModelConfig, get_model_config
except ImportError:
    # Fallback configuration if backend config is not available
    @dataclass
    class ModelConfig:
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 40
        repeat_penalty: float = 1.1
        n_ctx: int = 2048
        n_threads: int = -1
        n_gpu_layers: int = 0
        verbose: bool = False
    
    def get_model_config() -> ModelConfig:
        return ModelConfig()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "promptopt"

@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    tokens_generated: int
    time_taken: float
    finish_reason: str = "stop"

class LlamaWrapper:
    """
    Wrapper class for llama.cpp providing text generation capabilities.
    
    Handles GGUF model loading, memory management, and provides consistent
    inference capabilities for the optimization pipeline.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = -1,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the LlamaWrapper with a GGUF model.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads to use (-1 for auto)
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to Llama constructor
        
        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
        
        self.model_path = Path(model_path)
        self.model: Optional[Llama] = None
        self._loaded = False
        self._loading_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        
        # Configuration
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads != -1 else os.cpu_count()
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Statistics
        self._total_tokens_generated = 0
        self._total_requests = 0
        self._load_time = 0.0
        
        # Validate and load model
        self._validate_model_path()
        self._load_model()
    
    def _validate_model_path(self) -> None:
        """Validate that the model path exists and is a valid GGUF file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not validate_gguf_file(str(self.model_path)):
            raise ValueError(f"Invalid GGUF file: {self.model_path}")
    
    def _load_model(self) -> None:
        """Load the GGUF model into memory."""
        with self._loading_lock:
            if self._loaded:
                return
            
            start_time = time.time()
            logger.info(f"Loading model from {self.model_path}")
            
            try:
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    **self.kwargs
                )
                
                self._load_time = time.time() - start_time
                self._loaded = True
                
                logger.info(f"Model loaded successfully in {self._load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: Stop sequences
            **kwargs: Additional generation parameters
        
        Returns:
            GenerationResult containing generated text and metadata
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        start_time = time.time()
        
        try:
            with self._generation_lock:
                # Prepare stop sequences
                stop_sequences = []
                if stop:
                    if isinstance(stop, str):
                        stop_sequences = [stop]
                    else:
                        stop_sequences = stop
                
                # Generate text
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                    echo=False,
                    **kwargs
                )
                
                generated_text = output['choices'][0]['text']
                tokens_generated = output['usage']['completion_tokens']
                finish_reason = output['choices'][0]['finish_reason']
                
                # Update statistics
                self._total_tokens_generated += tokens_generated
                self._total_requests += 1
                
                generation_time = time.time() - start_time
                
                logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
                
                return GenerationResult(
                    text=generated_text,
                    tokens_generated=tokens_generated,
                    time_taken=generation_time,
                    finish_reason=finish_reason
                )
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_workers: int = 4,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            max_workers: Maximum number of concurrent workers
            **kwargs: Generation parameters passed to generate()
        
        Returns:
            List of GenerationResult objects
        """
        if not prompts:
            return []
        
        logger.info(f"Starting batch generation for {len(prompts)} prompts")
        
        async def generate_single(prompt: str) -> GenerationResult:
            """Generate text for a single prompt in thread pool."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.generate(prompt, **kwargs)
            )
        
        # Use semaphore to limit concurrent generations
        semaphore = asyncio.Semaphore(max_workers)
        
        async def generate_with_semaphore(prompt: str) -> GenerationResult:
            async with semaphore:
                return await generate_single(prompt)
        
        # Execute all generations concurrently
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate for prompt {i}: {result}")
                final_results.append(GenerationResult(
                    text="",
                    tokens_generated=0,
                    time_taken=0.0,
                    finish_reason="error"
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._loaded and self.model is not None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage information
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        usage = {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024,
        }
        
        if self.is_loaded():
            # Estimate model memory usage
            model_size_mb = estimate_memory_requirements(str(self.model_path)) / 1024 / 1024
            usage["estimated_model_mb"] = model_size_mb
        
        return usage
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "model_load_time": self._load_time,
            "is_loaded": self.is_loaded(),
            "model_path": str(self.model_path),
            "memory_usage": self.get_memory_usage()
        }
    
    def unload(self) -> None:
        """Unload the model from memory."""
        with self._loading_lock:
            if self.model is not None:
                logger.info("Unloading model from memory")
                del self.model
                self.model = None
                self._loaded = False
                
                # Force garbage collection
                import gc
                gc.collect()


class ModelPool:
    """
    Pool manager for multiple GGUF models.
    
    Handles loading, unloading, and selection of models based on availability
    and memory constraints.
    """
    
    def __init__(self, max_models: int = 3):
        """
        Initialize the model pool.
        
        Args:
            max_models: Maximum number of models to keep loaded simultaneously
        """
        self.max_models = max_models
        self.models: Dict[str, LlamaWrapper] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._pool_lock = threading.Lock()
    
    def add_model(
        self,
        model_id: str,
        model_path: str,
        **config
    ) -> None:
        """
        Add a model to the pool.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the GGUF model file
            **config: Model configuration parameters
        """
        with self._pool_lock:
            self.model_configs[model_id] = {
                "path": model_path,
                **config
            }
            logger.info(f"Added model '{model_id}' to pool")
    
    def get_model(self, model_id: str) -> LlamaWrapper:
        """
        Get a model from the pool, loading if necessary.
        
        Args:
            model_id: Model identifier
        
        Returns:
            LlamaWrapper instance
        
        Raises:
            ValueError: If model_id is not registered
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Model '{model_id}' not registered in pool")
        
        with self._pool_lock:
            # If model is already loaded, return it
            if model_id in self.models and self.models[model_id].is_loaded():
                self.access_times[model_id] = time.time()
                return self.models[model_id]
            
            # Check if we need to unload models due to limit
            if len(self.models) >= self.max_models:
                self._evict_least_recent()
            
            # Load the model
            config = self.model_configs[model_id]
            model_path = config.pop("path")
            
            try:
                wrapper = LlamaWrapper(model_path, **config)
                self.models[model_id] = wrapper
                self.access_times[model_id] = time.time()
                
                logger.info(f"Loaded model '{model_id}' from {model_path}")
                return wrapper
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_id}': {e}")
                raise
    
    def _evict_least_recent(self) -> None:
        """Evict the least recently used model."""
        if not self.access_times:
            return
        
        lru_model = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        if lru_model in self.models:
            self.models[lru_model].unload()
            del self.models[lru_model]
            del self.access_times[lru_model]
            logger.info(f"Evicted model '{lru_model}' from pool")
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.model_configs.keys())
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        loaded_models = [mid for mid, model in self.models.items() if model.is_loaded()]
        
        return {
            "total_registered": len(self.model_configs),
            "currently_loaded": len(loaded_models),
            "max_models": self.max_models,
            "loaded_models": loaded_models,
            "memory_usage": sum(
                model.get_memory_usage().get("estimated_model_mb", 0)
                for model in self.models.values()
                if model.is_loaded()
            )
        }
    
    def unload_all(self) -> None:
        """Unload all models from the pool."""
        with self._pool_lock:
            for model in self.models.values():
                model.unload()
            
            self.models.clear()
            self.access_times.clear()
            logger.info("Unloaded all models from pool")


class OpenAICompatibleServer:
    """
    OpenAI-compatible REST API server for llama.cpp models.
    
    Provides chat completions and completions endpoints that mirror
    the OpenAI API structure for easy integration.
    """
    
    def __init__(self, model_pool: ModelPool):
        """
        Initialize the server with a model pool.
        
        Args:
            model_pool: ModelPool instance for model management
        """
        self.model_pool = model_pool
        self.app = FastAPI(
            title="PromptOpt Co-Pilot LLM Server",
            description="OpenAI-compatible API for local GGUF models",
            version="1.0.0"
        )
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self) -> None:
        """Setup middleware for request logging and error handling."""
        
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.3f}s"
            )
            return response
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            models = []
            for model_id in self.model_pool.list_models():
                models.append(ModelInfo(id=model_id))
            
            return {"object": "list", "data": [asdict(model) for model in models]}
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint."""
            try:
                # Get model
                model = self.model_pool.get_model(request.model)
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(request.messages)
                
                # Generate response
                result = model.generate(
                    prompt=prompt,
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 0.9,
                    stop=request.stop
                )
                
                # Format OpenAI response
                response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.text.strip()
                        },
                        "finish_reason": result.finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": 0,  # Would need tokenizer to calculate
                        "completion_tokens": result.tokens_generated,
                        "total_tokens": result.tokens_generated
                    }
                }
                
                return response
                
            except Exception as e:
                logger.error(f"Chat completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """OpenAI-compatible completions endpoint."""
            try:
                # Get model
                model = self.model_pool.get_model(request.model)
                
                # Handle single or multiple prompts
                prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
                
                # Generate responses
                if len(prompts) == 1:
                    result = model.generate(
                        prompt=prompts[0],
                        max_tokens=request.max_tokens or 512,
                        temperature=request.temperature or 0.7,
                        top_p=request.top_p or 0.9,
                        stop=request.stop
                    )
                    results = [result]
                else:
                    results = await model.generate_batch(
                        prompts=prompts,
                        max_tokens=request.max_tokens or 512,
                        temperature=request.temperature or 0.7,
                        top_p=request.top_p or 0.9,
                        stop=request.stop
                    )
                
                # Format OpenAI response
                choices = []
                total_tokens = 0
                
                for i, result in enumerate(results):
                    choices.append({
                        "text": result.text,
                        "index": i,
                        "logprobs": None,
                        "finish_reason": result.finish_reason
                    })
                    total_tokens += result.tokens_generated
                
                response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": choices,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": total_tokens,
                        "total_tokens": total_tokens
                    }
                }
                
                return response
                
            except Exception as e:
                logger.error(f"Completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "models_loaded": len([
                    m for m in self.model_pool.models.values() 
                    if m.is_loaded()
                ])
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            return {
                "pool_stats": self.model_pool.get_pool_stats(),
                "memory": psutil.virtual_memory()._asdict(),
                "cpu_percent": psutil.cpu_percent()
            }
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info"
    ) -> None:
        """
        Start the FastAPI server.
        
        Args:
            host: Host address to bind to
            port: Port number to bind to
            log_level: Logging level
        """
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Starting server on {host}:{port}")
        await server.serve()


# Helper Functions

def validate_gguf_file(path: str) -> bool:
    """
    Validate that a file is a valid GGUF format.
    
    Args:
        path: Path to the file to validate
    
    Returns:
        True if file appears to be valid GGUF format
    """
    try:
        with open(path, 'rb') as f:
            # Read GGUF magic number (first 4 bytes should be 'GGUF')
            magic = f.read(4)
            return magic == b'GGUF'
    except (IOError, OSError):
        return False

def estimate_memory_requirements(model_path: str) -> int:
    """
    Estimate memory requirements for a GGUF model.
    
    Args:
        model_path: Path to the GGUF model file
    
    Returns:
        Estimated memory requirement in bytes
    """
    try:
        file_size = Path(model_path).stat().st_size
        # Rough estimate: model size + 20% overhead for context and operations
        return int(file_size * 1.2)
    except (OSError, IOError):
        return 0

def get_model_info(path: str) -> Dict[str, Any]:
    """
    Extract basic information about a GGUF model.
    
    Args:
        path: Path to the GGUF model file
    
    Returns:
        Dictionary containing model information
    """
    model_path = Path(path)
    
    info = {
        "path": str(model_path),
        "filename": model_path.name,
        "size_bytes": 0,
        "size_mb": 0,
        "is_valid": False,
        "estimated_memory_mb": 0,
    }
    
    try:
        if model_path.exists():
            stat = model_path.stat()
            info["size_bytes"] = stat.st_size
            info["size_mb"] = stat.st_size / 1024 / 1024
            info["is_valid"] = validate_gguf_file(path)
            info["estimated_memory_mb"] = estimate_memory_requirements(path) / 1024 / 1024
            info["modified_time"] = stat.st_mtime
    except (OSError, IOError) as e:
        logger.error(f"Error reading model info for {path}: {e}")
    
    return info


# Main execution and examples
if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="PromptOpt Co-Pilot LLM Server")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--model-id", default="default", help="Model ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--max-models", type=int, default=3, help="Max models in pool")
    
    args = parser.parse_args()
    
    # Validate model file
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        exit(1)
    
    if not validate_gguf_file(args.model_path):
        logger.error(f"Invalid GGUF file: {args.model_path}")
        exit(1)
    
    # Create model pool and add model
    pool = ModelPool(max_models=args.max_models)
    pool.add_model(
        model_id=args.model_id,
        model_path=args.model_path,
        n_ctx=2048,
        n_threads=-1,
        n_gpu_layers=0
    )
    
    # Create and start server
    server = OpenAICompatibleServer(pool)
    
    try:
        asyncio.run(server.start_server(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        pool.unload_all()
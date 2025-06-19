"""
PromptOpt Co-Pilot API Middleware

This module provides comprehensive middleware stack for the FastAPI application,
including authentication, logging, error handling, rate limiting, and request/response processing.

Components:
- LoggingMiddleware: Request/response logging with performance metrics
- AuthenticationMiddleware: API key validation and session management
- RateLimitMiddleware: Request rate limiting with sliding window
- ErrorHandlingMiddleware: Global exception handling
- RequestValidationMiddleware: Request validation and sanitization
- CORS configuration for frontend integration

Author: PromptOpt Co-Pilot Development Team
Version: 1.0.0
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set, Callable, List, Tuple
import traceback
import re
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib
import hmac

# Import from project modules
try:
    from backend.app_logging.sqlite_logger import SQLiteLogger
    from backend.core.config import MiddlewareConfig
except ImportError:
    # Fallback for development/testing
    class SQLiteLogger:
        def log_request(self, *args, **kwargs): pass
        def log_error(self, *args, **kwargs): pass
    
    class MiddlewareConfig:
        def __init__(self):
            self.api_keys = {"dev-key-123": "development"}
            self.rate_limit_requests = 100
            self.rate_limit_window = 3600
            self.max_request_size = 10 * 1024 * 1024  # 10MB
            self.request_timeout = 30
            self.cors_origins = ["http://localhost:3000"]
            self.log_level = "INFO"
            self.enable_auth = True


class RateLimitStore:
    """Thread-safe rate limiting store using sliding window algorithm."""
    
    def __init__(self):
        self._store: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        async with self._lock:
            now = time.time()
            requests = self._store[key]
            
            # Remove expired entries
            while requests and requests[0] <= now - window:
                requests.popleft()
            
            # Check if under limit
            if len(requests) < limit:
                requests.append(now)
                return True
            
            return False
    
    async def get_stats(self, key: str, window: int) -> Dict[str, Any]:
        """Get rate limit statistics for a key."""
        async with self._lock:
            now = time.time()
            requests = self._store[key]
            
            # Clean expired entries
            while requests and requests[0] <= now - window:
                requests.popleft()
            
            return {
                "requests_made": len(requests),
                "window_start": now - window,
                "next_reset": requests[0] + window if requests else now
            }


def generate_request_id() -> str:
    """Generate unique request identifier."""
    return str(uuid.uuid4())


def sanitize_request_data(data: dict) -> dict:
    """Sanitize request data by removing sensitive information."""
    sensitive_keys = {
        'password', 'token', 'api_key', 'secret', 'auth',
        'authorization', 'x-api-key', 'cookie'
    }
    
    def _sanitize_value(key: str, value: Any) -> Any:
        if isinstance(key, str) and key.lower() in sensitive_keys:
            return "[REDACTED]"
        elif isinstance(value, dict):
            return {k: _sanitize_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_sanitize_value("", item) for item in value]
        else:
            return value
    
    return {k: _sanitize_value(k, v) for k, v in data.items()}


def validate_api_key(api_key: str, valid_keys: Dict[str, str]) -> Optional[str]:
    """Validate API key and return associated user/service name."""
    if not api_key:
        return None
    
    # Remove 'Bearer ' prefix if present
    if api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    return valid_keys.get(api_key)


def format_error_response(error: Exception, request_id: str, include_details: bool = False) -> dict:
    """Format error response with consistent structure."""
    error_type = type(error).__name__
    
    base_response = {
        "error": True,
        "error_type": error_type,
        "message": str(error),
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if include_details and hasattr(error, '__traceback__'):
        base_response["traceback"] = traceback.format_exception(
            type(error), error, error.__traceback__
        )
    
    return base_response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app: ASGIApp, logger: SQLiteLogger, config: MiddlewareConfig):
        super().__init__(app)
        self.logger = logger
        self.config = config
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = generate_request_id()
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, duration, request_id)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            await self._log_error(request, e, duration, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details."""
        try:
            # Get request data
            url = str(request.url)
            method = request.method
            headers = dict(request.headers)
            client_ip = request.client.host if request.client else "unknown"
            
            # Sanitize headers
            sanitized_headers = sanitize_request_data(headers)
            
            log_data = {
                "request_id": request_id,
                "method": method,
                "url": url,
                "client_ip": client_ip,
                "headers": sanitized_headers,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.log_request(
                request_id=request_id,
                method=method,
                url=url,
                headers=json.dumps(sanitized_headers),
                client_ip=client_ip
            )
            
        except Exception as e:
            # Log error in logging itself
            print(f"Error logging request: {e}")
    
    async def _log_response(self, request: Request, response: Response, 
                          duration: float, request_id: str):
        """Log response details and performance metrics."""
        try:
            log_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": len(response.body) if hasattr(response, 'body') else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log performance metrics
            if duration > 1.0:  # Log slow requests
                print(f"SLOW REQUEST: {request.method} {request.url} - {duration:.2f}s")
            
        except Exception as e:
            print(f"Error logging response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, 
                        duration: float, request_id: str):
        """Log error details."""
        try:
            error_data = {
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "url": str(request.url),
                "method": request.method,
                "duration_ms": round(duration * 1000, 2),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.log_error(
                request_id=request_id,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc()
            )
            
        except Exception as e:
            print(f"Error logging error: {e}")


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API authentication and session management."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig):
        super().__init__(app)
        self.config = config
        self.public_paths = {'/docs', '/openapi.json', '/health', '/favicon.ico'}
        self.api_keys = getattr(config, 'api_keys', {})
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for public paths
        if not self.config.enable_auth or self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Extract API key
        api_key = self._extract_api_key(request)
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": True,
                    "message": "API key required",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
        
        # Validate API key
        user_context = validate_api_key(api_key, self.api_keys)
        if not user_context:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": True,
                    "message": "Invalid API key",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
        
        # Add user context to request
        request.state.user_context = user_context
        request.state.api_key = api_key
        
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require authentication."""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers or query parameters."""
        # Check Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            return auth_header
        
        # Check X-API-Key header
        api_key_header = request.headers.get('X-API-Key')
        if api_key_header:
            return api_key_header
        
        # Check query parameter
        api_key_param = request.query_params.get('api_key')
        if api_key_param:
            return api_key_param
        
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for request rate limiting with sliding window algorithm."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig):
        super().__init__(app)
        self.config = config
        self.rate_store = RateLimitStore()
        self.default_limit = getattr(config, 'rate_limit_requests', 100)
        self.default_window = getattr(config, 'rate_limit_window', 3600)
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            '/api/v1/optimize': (10, 60),  # 10 requests per minute
            '/api/v1/analyze': (20, 60),   # 20 requests per minute
            '/api/v1/generate': (5, 60),   # 5 requests per minute
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit for endpoint
        limit, window = self._get_rate_limit(request.url.path)
        
        # Check rate limit
        rate_key = f"{client_id}:{request.url.path}"
        is_allowed = await self.rate_store.is_allowed(rate_key, limit, window)
        
        if not is_allowed:
            # Get rate limit stats
            stats = await self.rate_store.get_stats(rate_key, window)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": True,
                    "message": "Rate limit exceeded",
                    "limit": limit,
                    "window_seconds": window,
                    "retry_after": int(stats["next_reset"] - time.time()),
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Window": str(window),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(int(stats["next_reset"] - time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        stats = await self.rate_store.get_stats(rate_key, window)
        remaining = max(0, limit - stats["requests_made"])
        
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(window)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if available
        api_key = getattr(request.state, 'api_key', None)
        if api_key:
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Use IP address as fallback
        client_ip = request.client.host if request.client else "unknown"
        return client_ip
    
    def _get_rate_limit(self, path: str) -> Tuple[int, int]:
        """Get rate limit and window for specific endpoint."""
        for endpoint_path, (limit, window) in self.endpoint_limits.items():
            if path.startswith(endpoint_path):
                return limit, window
        
        return self.default_limit, self.default_window


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global exception handling and error formatting."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig, logger: SQLiteLogger):
        super().__init__(app)
        self.config = config
        self.logger = logger
        self.include_details = getattr(config, 'debug', False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": True,
                    "message": e.detail,
                    "status_code": e.status_code,
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        except ValueError as e:
            # Handle validation errors
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            error_response = format_error_response(e, request_id, self.include_details)
            error_response["status_code"] = 400
            
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_response
            )
        
        except PermissionError as e:
            # Handle permission errors
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            error_response = format_error_response(e, request_id, False)  # Never include details for permission errors
            error_response["status_code"] = 403
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=error_response
            )
        
        except Exception as e:
            # Handle all other exceptions
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # Log the error
            try:
                self.logger.log_error(
                    request_id=request_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc()
                )
            except:
                pass  # Don't let logging errors crash the app
            
            error_response = format_error_response(e, request_id, self.include_details)
            error_response["status_code"] = 500
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""
    
    def __init__(self, app: ASGIApp, config: MiddlewareConfig):
        super().__init__(app)
        self.config = config
        self.max_request_size = getattr(config, 'max_request_size', 10 * 1024 * 1024)  # 10MB
        self.request_timeout = getattr(config, 'request_timeout', 30)
        self.allowed_content_types = {
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'text/plain'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Validate request size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": True,
                    "message": f"Request too large. Maximum size: {self.max_request_size} bytes",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('content-type', '').split(';')[0].strip()
            if content_type and not any(allowed in content_type for allowed in self.allowed_content_types):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={
                        "error": True,
                        "message": f"Unsupported content type: {content_type}",
                        "supported_types": list(self.allowed_content_types),
                        "request_id": getattr(request.state, 'request_id', 'unknown')
                    }
                )
        
        # Process request with timeout
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.request_timeout
            )
            return response
        
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                content={
                    "error": True,
                    "message": f"Request timeout after {self.request_timeout} seconds",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )


def setup_middleware(app: FastAPI, config: MiddlewareConfig, logger: SQLiteLogger) -> None:
    """
    Setup comprehensive middleware stack for FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Middleware configuration
        logger: SQLite logger instance
    """
    
    # Add CORS middleware (should be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(config, 'cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add custom middleware in reverse order (last added = first executed)
    
    # Request validation (outermost)
    app.add_middleware(RequestValidationMiddleware, config=config)
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware, config=config, logger=logger)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware, config=config)
    
    # Authentication
    app.add_middleware(AuthenticationMiddleware, config=config)
    
    # Logging (innermost, closest to application)
    app.add_middleware(LoggingMiddleware, logger=logger, config=config)


def check_rate_limit(client_id: str, endpoint: str, rate_store: RateLimitStore, 
                    limit: int = 100, window: int = 3600) -> bool:
    """
    Check if client is within rate limit for endpoint.
    
    Args:
        client_id: Client identifier
        endpoint: Endpoint path
        rate_store: Rate limiting store
        limit: Request limit
        window: Time window in seconds
    
    Returns:
        True if within limit, False otherwise
    """
    import asyncio
    
    async def _check():
        rate_key = f"{client_id}:{endpoint}"
        return await rate_store.is_allowed(rate_key, limit, window)
    
    # For synchronous usage
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_check())
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(_check())


def log_request_response(request: Request, response: Response, duration: float, 
                        logger: SQLiteLogger) -> None:
    """
    Log HTTP request and response details.
    
    Args:
        request: HTTP request
        response: HTTP response
        duration: Request duration in seconds
        logger: SQLite logger instance
    """
    try:
        request_id = getattr(request.state, 'request_id', generate_request_id())
        
        # Log request details
        logger.log_request(
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            headers=json.dumps(sanitize_request_data(dict(request.headers))),
            client_ip=request.client.host if request.client else "unknown"
        )
        
        # Log performance if slow
        if duration > 1.0:
            print(f"SLOW REQUEST: {request.method} {request.url} - {duration:.2f}s")
            
    except Exception as e:
        print(f"Error in log_request_response: {e}")


# Health check endpoint for middleware monitoring
async def middleware_health_check() -> dict:
    """Health check for middleware components."""
    return {
        "status": "healthy",
        "middleware": {
            "logging": "active",
            "authentication": "active",
            "rate_limiting": "active",
            "error_handling": "active",
            "request_validation": "active"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Example usage and testing
    print("PromptOpt Co-Pilot API Middleware")
    print("=================================")
    
    # Test utility functions
    print("\nTesting utility functions:")
    
    # Test request ID generation
    request_id = generate_request_id()
    print(f"Generated request ID: {request_id}")
    
    # Test data sanitization
    test_data = {
        "username": "user123",
        "password": "secret123",
        "data": {"api_key": "key123", "value": "safe"}
    }
    sanitized = sanitize_request_data(test_data)
    print(f"Sanitized data: {sanitized}")
    
    # Test API key validation
    test_keys = {"valid-key": "test-user", "another-key": "another-user"}
    result = validate_api_key("valid-key", test_keys)
    print(f"API key validation result: {result}")
    
    # Test error formatting
    test_error = ValueError("Test error message")
    error_response = format_error_response(test_error, request_id, True)
    print(f"Error response: {error_response}")
    
    print("\nMiddleware module loaded successfully!")
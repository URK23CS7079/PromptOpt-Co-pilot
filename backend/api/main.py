"""
PromptOpt Co-Pilot - Main FastAPI Server

This module serves as the central API gateway for PromptOpt Co-Pilot,
orchestrating all backend services including LLM inference, optimization,
evaluation, and data management.

Author: PromptOpt Co-Pilot Development Team
License: MIT
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set
import uuid

import uvicorn
from fastapi import (
    FastAPI, 
    HTTPException, 
    Request, 
    Response,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Depends,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import websockets

# Core imports
from backend.core.config import settings
from backend.core.database import Database, get_database
from backend.core.logging import setup_logging, get_logger
from backend.llm.model_manager import ModelManager, get_model_manager

# API route imports
from backend.api.routes.prompts import router as prompts_router
from backend.api.routes.optimization import router as optimization_router
from backend.api.routes.evaluation import router as evaluation_router

# Middleware imports
from backend.api.middleware.auth import AuthMiddleware
from backend.api.middleware.rate_limit import RateLimitMiddleware
from backend.api.middleware.request_logging import RequestLoggingMiddleware

# Exception and response models
from backend.api.models.responses import (
    ErrorResponse,
    HealthResponse,
    SystemStatusResponse
)

# Initialize logging
logger = get_logger(__name__)

# Global state management
class ApplicationState:
    """Global application state management"""
    
    def __init__(self):
        self.database: Optional[Database] = None
        self.model_manager: Optional[ModelManager] = None
        self.websocket_connections: Set[WebSocket] = set()
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.is_shutting_down = False
        self.startup_time = None
        self.health_status = "starting"
    
    async def initialize(self):
        """Initialize all application components"""
        try:
            logger.info("Initializing application state...")
            
            # Initialize database
            self.database = Database()
            await self.database.initialize()
            logger.info("Database initialized successfully")
            
            # Initialize model manager
            self.model_manager = ModelManager()
            await self.model_manager.initialize()
            logger.info("Model manager initialized successfully")
            
            # Start background monitoring tasks
            await self._start_background_tasks()
            
            self.startup_time = time.time()
            self.health_status = "healthy"
            logger.info("Application initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all application components"""
        try:
            logger.info("Starting application shutdown...")
            self.is_shutting_down = True
            self.health_status = "shutting_down"
            
            # Cancel background tasks
            for task_id, task in self.background_tasks.items():
                if not task.done():
                    logger.info(f"Cancelling background task: {task_id}")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close WebSocket connections
            if self.websocket_connections:
                logger.info(f"Closing {len(self.websocket_connections)} WebSocket connections")
                disconnect_tasks = []
                for websocket in self.websocket_connections.copy():
                    disconnect_tasks.append(self._close_websocket(websocket))
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
            # Shutdown model manager
            if self.model_manager:
                await self.model_manager.shutdown()
                logger.info("Model manager shutdown completed")
            
            # Shutdown database
            if self.database:
                await self.database.close()
                logger.info("Database shutdown completed")
            
            logger.info("Application shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # System monitoring task
        self.background_tasks["system_monitor"] = asyncio.create_task(
            self._system_monitor_task()
        )
        
        # WebSocket heartbeat task
        self.background_tasks["websocket_heartbeat"] = asyncio.create_task(
            self._websocket_heartbeat_task()
        )
        
        # Resource cleanup task
        self.background_tasks["resource_cleanup"] = asyncio.create_task(
            self._resource_cleanup_task()
        )
    
    async def _system_monitor_task(self):
        """Background task for system monitoring"""
        try:
            while not self.is_shutting_down:
                # Monitor system resources and broadcast to WebSocket clients
                if self.websocket_connections and self.model_manager:
                    status = await self.get_system_status()
                    await self.broadcast_to_websockets({
                        "type": "system_status",
                        "data": status.dict()
                    })
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except asyncio.CancelledError:
            logger.info("System monitor task cancelled")
        except Exception as e:
            logger.error(f"Error in system monitor task: {e}")
    
    async def _websocket_heartbeat_task(self):
        """Background task for WebSocket heartbeat"""
        try:
            while not self.is_shutting_down:
                # Send heartbeat to all connected WebSocket clients
                if self.websocket_connections:
                    await self.broadcast_to_websockets({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    })
                
                await asyncio.sleep(60)  # Heartbeat every minute
                
        except asyncio.CancelledError:
            logger.info("WebSocket heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Error in WebSocket heartbeat task: {e}")
    
    async def _resource_cleanup_task(self):
        """Background task for resource cleanup"""
        try:
            while not self.is_shutting_down:
                # Clean up completed background tasks
                completed_tasks = [
                    task_id for task_id, task in self.background_tasks.items()
                    if task.done() and task_id not in ["system_monitor", "websocket_heartbeat", "resource_cleanup"]
                ]
                
                for task_id in completed_tasks:
                    del self.background_tasks[task_id]
                    logger.debug(f"Cleaned up completed task: {task_id}")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
        except asyncio.CancelledError:
            logger.info("Resource cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in resource cleanup task: {e}")
    
    async def get_system_status(self) -> SystemStatusResponse:
        """Get current system status"""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get model manager status
        model_status = {}
        if self.model_manager:
            model_status = await self.model_manager.get_status()
        
        # Get database status
        db_status = "disconnected"
        if self.database:
            db_status = "connected" if await self.database.is_healthy() else "unhealthy"
        
        return SystemStatusResponse(
            status=self.health_status,
            uptime=time.time() - self.startup_time if self.startup_time else 0,
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_total=memory.total,
            memory_available=memory.available,
            disk_usage=disk.percent,
            disk_total=disk.total,
            disk_free=disk.free,
            active_connections=len(self.websocket_connections),
            background_tasks=len(self.background_tasks),
            database_status=db_status,
            model_status=model_status
        )
    
    async def broadcast_to_websockets(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def _close_websocket(self, websocket: WebSocket):
        """Safely close a WebSocket connection"""
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")
        finally:
            self.websocket_connections.discard(websocket)

# Global application state
app_state = ApplicationState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        logger.info("Starting PromptOpt Co-Pilot server...")
        await app_state.initialize()
        logger.info("Server startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
    finally:
        # Shutdown
        await app_state.shutdown()
        logger.info("Server shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="PromptOpt Co-Pilot API",
    description="Central API gateway for PromptOpt Co-Pilot offline prompt optimization system",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTP Exception",
                message=str(exc.detail),
                status_code=exc.status_code
            ).dict()
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            status_code=500
        ).dict()
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_healthy = False
        if app_state.database:
            db_healthy = await app_state.database.is_healthy()
        
        # Check model manager status
        models_healthy = False
        if app_state.model_manager:
            models_healthy = await app_state.model_manager.is_healthy()
        
        overall_healthy = db_healthy and models_healthy and not app_state.is_shutting_down
        
        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=time.time(),
            database=db_healthy,
            models=models_healthy,
            uptime=time.time() - app_state.startup_time if app_state.startup_time else 0
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            database=False,
            models=False,
            uptime=0
        )

# System status endpoint
@app.get("/status", response_model=SystemStatusResponse)
async def system_status():
    """Get detailed system status"""
    try:
        return await app_state.get_system_status()
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time optimization progress and system updates"""
    client_id = str(uuid.uuid4())
    logger.info(f"New WebSocket connection: {client_id}")
    
    await websocket.accept()
    app_state.websocket_connections.add(websocket)
    
    try:
        # Send initial status
        initial_status = await app_state.get_system_status()
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "system_status": initial_status.dict()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Set a timeout for receiving messages
                message = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific events
                    logger.info(f"Client {client_id} subscribed to: {message.get('events', [])}")
                elif message.get("type") == "get_status":
                    # Send current system status
                    status = await app_state.get_system_status()
                    await websocket.send_json({
                        "type": "system_status",
                        "data": status.dict()
                    })
                
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to process message"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        app_state.websocket_connections.discard(websocket)
        logger.info(f"WebSocket connection {client_id} cleaned up")

# Background task management endpoint
@app.post("/tasks/{task_id}/cancel")
async def cancel_background_task(task_id: str):
    """Cancel a background task"""
    if task_id not in app_state.background_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task = app_state.background_tasks[task_id]
    if not task.done():
        task.cancel()
        logger.info(f"Cancelled background task: {task_id}")
    
    return {"message": f"Task {task_id} cancellation requested"}

@app.get("/tasks")
async def list_background_tasks():
    """List all background tasks"""
    tasks = {}
    for task_id, task in app_state.background_tasks.items():
        tasks[task_id] = {
            "done": task.done(),
            "cancelled": task.cancelled(),
            "exception": str(task.exception()) if task.done() and task.exception() else None
        }
    
    return {"tasks": tasks}

# API route registration
app.include_router(prompts_router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(optimization_router, prefix="/api/v1/optimization", tags=["optimization"])
app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["evaluation"])

# Mount static files for model downloads and exports
app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="PromptOpt Co-Pilot API",
        version="1.0.0",
        description="Central API gateway for PromptOpt Co-Pilot offline prompt optimization system",
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    app_state.is_shutting_down = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Dependency providers
async def get_app_state() -> ApplicationState:
    """Get application state dependency"""
    return app_state

async def get_websocket_manager():
    """Get WebSocket manager dependency"""
    return app_state

# Main entry point
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Server configuration based on environment
    if settings.ENVIRONMENT == "development":
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            log_level="debug",
            access_log=True
        )
    else:
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            workers=settings.WORKERS,
            log_level="info",
            access_log=True,
            loop="uvloop"  # Use uvloop for better performance in production
        )
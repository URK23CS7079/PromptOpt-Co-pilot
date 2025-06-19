"""
PromptLayer Logging Adapter for PromptOpt Co-Pilot

This module provides PromptLayer-compatible logging and tracking functionality
for offline prompt optimization workflows. It implements the PromptLayer API
patterns while storing data locally in SQLite.

Author: PromptOpt Co-Pilot
Date: 2025-06-19
"""

import csv
import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from contextlib import contextmanager

# Import from backend modules
try:
    from backend.core.database import DatabaseManager
    from backend.core.models import BaseModel
except ImportError:
    # Fallback for standalone usage
    class DatabaseManager:
        pass
    class BaseModel:
        pass


class LogLevel(Enum):
    """Log level enumeration for PromptLayer compatibility."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExportFormat(Enum):
    """Export format enumeration."""
    JSON = "json"
    CSV = "csv"
    PROMPTLAYER = "promptlayer"


@dataclass
class PromptLayerConfig:
    """Configuration for PromptLayer adapter."""
    log_level: LogLevel = LogLevel.INFO
    batch_size: int = 100
    retention_days: int = 30
    analytics_enabled: bool = True
    database_path: str = "promptlayer_logs.db"
    auto_flush: bool = True
    flush_interval: int = 60  # seconds
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_enabled: bool = True
    backup_interval: int = 24  # hours


@dataclass
class PromptLogEntry:
    """Data structure for prompt log entries compatible with PromptLayer."""
    request_id: str
    timestamp: datetime
    prompt: str
    response: Optional[str] = None
    model: str = ""
    latency: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    optimization_run_id: Optional[str] = None
    evaluation_run_id: Optional[str] = None
    dataset_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptLogEntry':
        """Create from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class PromptLayerAdapter:
    """
    PromptLayer-compatible logging adapter for offline operation.
    
    This adapter provides comprehensive logging capabilities similar to
    PromptLayer's cloud-based service while operating entirely offline
    with local SQLite storage.
    """
    
    def __init__(self, config: PromptLayerConfig):
        """
        Initialize the PromptLayer adapter.
        
        Args:
            config: Configuration object for the adapter
        """
        self.config = config
        self.logger = self._setup_logger()
        self._batch_buffer: List[PromptLogEntry] = []
        self._setup_database()
        
        # Statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_expiry: Optional[datetime] = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup internal logger."""
        logger = logging.getLogger("promptlayer_adapter")
        logger.setLevel(getattr(logging, self.config.log_level.value.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_database(self):
        """Setup SQLite database for log storage."""
        self.db_path = Path(self.config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_logs (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT,
                    model TEXT,
                    latency REAL,
                    metadata TEXT,
                    session_id TEXT,
                    optimization_run_id TEXT,
                    evaluation_run_id TEXT,
                    dataset_name TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL,
                    error TEXT,
                    tags TEXT,
                    user_id TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_timestamp ON prompt_logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_session ON prompt_logs(session_id);
                CREATE INDEX IF NOT EXISTS idx_optimization_run ON prompt_logs(optimization_run_id);
                CREATE INDEX IF NOT EXISTS idx_evaluation_run ON prompt_logs(evaluation_run_id);
                CREATE INDEX IF NOT EXISTS idx_model ON prompt_logs(model);
                
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    prompts TEXT NOT NULL,
                    results TEXT,
                    status TEXT DEFAULT 'running'
                );
                
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    results TEXT,
                    status TEXT DEFAULT 'running'
                );
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def log_prompt_request(
        self, 
        prompt: str, 
        model: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a prompt request in PromptLayer format.
        
        Args:
            prompt: The prompt text
            model: Model identifier
            metadata: Additional metadata
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        log_entry = PromptLogEntry(
            request_id=request_id,
            timestamp=datetime.now(),
            prompt=prompt,
            model=model,
            metadata=metadata or {}
        )
        
        self._add_to_batch(log_entry)
        self.logger.debug(f"Logged prompt request: {request_id}")
        
        return request_id
    
    def log_prompt_response(
        self, 
        request_id: str, 
        response: str, 
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log a prompt response and update the existing request.
        
        Args:
            request_id: The request ID from log_prompt_request
            response: The model response
            metrics: Performance metrics (latency, tokens, cost, etc.)
        """
        metrics = metrics or {}
        
        # Update existing entry or create new one
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM prompt_logs WHERE request_id = ?",
                    (request_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry
                    cursor.execute("""
                        UPDATE prompt_logs 
                        SET response = ?, latency = ?, input_tokens = ?, 
                            output_tokens = ?, cost = ?, error = ?
                        WHERE request_id = ?
                    """, (
                        response,
                        metrics.get('latency'),
                        metrics.get('input_tokens'),
                        metrics.get('output_tokens'),
                        metrics.get('cost'),
                        metrics.get('error'),
                        request_id
                    ))
                else:
                    # Create new entry if not found
                    log_entry = PromptLogEntry(
                        request_id=request_id,
                        timestamp=datetime.now(),
                        prompt="",  # Not available in response-only log
                        response=response,
                        latency=metrics.get('latency'),
                        input_tokens=metrics.get('input_tokens'),
                        output_tokens=metrics.get('output_tokens'),
                        cost=metrics.get('cost'),
                        error=metrics.get('error'),
                        metadata=metrics
                    )
                    self._add_to_batch(log_entry)
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to log response for {request_id}: {e}")
            raise
        
        self.logger.debug(f"Logged response for request: {request_id}")
    
    def track_optimization_run(
        self, 
        session_id: str, 
        prompts: List[str], 
        results: Optional[Dict[str, Any]] = None
    ):
        """
        Track an optimization run session.
        
        Args:
            session_id: Unique session identifier
            prompts: List of prompts being optimized
            results: Optimization results and metrics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if run exists
                cursor.execute(
                    "SELECT session_id FROM optimization_runs WHERE session_id = ?",
                    (session_id,)
                )
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing run
                    cursor.execute("""
                        UPDATE optimization_runs 
                        SET end_time = ?, results = ?, status = ?
                        WHERE session_id = ?
                    """, (
                        datetime.now().isoformat(),
                        json.dumps(results) if results else None,
                        'completed' if results else 'running',
                        session_id
                    ))
                else:
                    # Create new run
                    cursor.execute("""
                        INSERT INTO optimization_runs 
                        (session_id, start_time, prompts, results, status)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        datetime.now().isoformat(),
                        json.dumps(prompts),
                        json.dumps(results) if results else None,
                        'completed' if results else 'running'
                    ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to track optimization run {session_id}: {e}")
            raise
        
        self.logger.info(f"Tracked optimization run: {session_id}")
    
    def log_evaluation_run(
        self, 
        run_id: str, 
        dataset: str, 
        results: Optional[Dict[str, Any]] = None
    ):
        """
        Log an evaluation run.
        
        Args:
            run_id: Unique run identifier
            dataset: Dataset name or identifier
            results: Evaluation results and metrics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if run exists
                cursor.execute(
                    "SELECT run_id FROM evaluation_runs WHERE run_id = ?",
                    (run_id,)
                )
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing run
                    cursor.execute("""
                        UPDATE evaluation_runs 
                        SET end_time = ?, results = ?, status = ?
                        WHERE run_id = ?
                    """, (
                        datetime.now().isoformat(),
                        json.dumps(results) if results else None,
                        'completed' if results else 'running',
                        run_id
                    ))
                else:
                    # Create new run
                    cursor.execute("""
                        INSERT INTO evaluation_runs 
                        (run_id, dataset, start_time, results, status)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        dataset,
                        datetime.now().isoformat(),
                        json.dumps(results) if results else None,
                        'completed' if results else 'running'
                    ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to log evaluation run {run_id}: {e}")
            raise
        
        self.logger.info(f"Logged evaluation run: {run_id}")
    
    def get_prompt_analytics(
        self, 
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get prompt analytics data for the specified time range.
        
        Args:
            time_range: (start_time, end_time) tuple, defaults to last 24 hours
            
        Returns:
            Analytics data dictionary
        """
        if not time_range:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            time_range = (start_time, end_time)
        
        start_time, end_time = time_range
        
        # Check cache
        cache_key = f"{start_time.isoformat()}_{end_time.isoformat()}"
        if (self._cache_expiry and datetime.now() < self._cache_expiry and 
            cache_key in self._stats_cache):
            return self._stats_cache[cache_key]
        
        logs = self._get_logs_in_range(time_range)
        
        analytics = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'usage_statistics': calculate_usage_statistics(logs),
            'performance_metrics': generate_performance_metrics(logs),
            'model_breakdown': self._get_model_breakdown(logs),
            'cost_analysis': self._get_cost_analysis(logs),
            'error_analysis': self._get_error_analysis(logs)
        }
        
        # Cache results
        self._stats_cache[cache_key] = analytics
        self._cache_expiry = datetime.now() + timedelta(minutes=15)
        
        return analytics
    
    def export_logs(
        self, 
        format: str, 
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Export logs in the specified format.
        
        Args:
            format: Export format ('json', 'csv', 'promptlayer')
            time_range: Time range for export
            
        Returns:
            Exported data in the specified format
        """
        if not time_range:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Default to last week
            time_range = (start_time, end_time)
        
        logs = self._get_logs_in_range(time_range)
        
        export_format = ExportFormat(format.lower())
        
        if export_format == ExportFormat.JSON:
            return export_to_json(logs)
        elif export_format == ExportFormat.CSV:
            return export_to_csv(logs)
        elif export_format == ExportFormat.PROMPTLAYER:
            return export_to_promptlayer_format(logs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _add_to_batch(self, log_entry: PromptLogEntry):
        """Add log entry to batch buffer."""
        self._batch_buffer.append(log_entry)
        
        if len(self._batch_buffer) >= self.config.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Flush batch buffer to database."""
        if not self._batch_buffer:
            return
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for entry in self._batch_buffer:
                    cursor.execute("""
                        INSERT OR REPLACE INTO prompt_logs 
                        (request_id, timestamp, prompt, response, model, latency,
                         metadata, session_id, optimization_run_id, evaluation_run_id,
                         dataset_name, input_tokens, output_tokens, cost, error, tags, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.request_id,
                        entry.timestamp.isoformat(),
                        entry.prompt,
                        entry.response,
                        entry.model,
                        entry.latency,
                        json.dumps(entry.metadata) if entry.metadata else None,
                        entry.session_id,
                        entry.optimization_run_id,
                        entry.evaluation_run_id,
                        entry.dataset_name,
                        entry.input_tokens,
                        entry.output_tokens,
                        entry.cost,
                        entry.error,
                        json.dumps(entry.tags) if entry.tags else None,
                        entry.user_id
                    ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to flush batch: {e}")
            raise
        
        self._batch_buffer.clear()
        self.logger.debug(f"Flushed batch of {len(self._batch_buffer)} entries")
    
    def _get_logs_in_range(
        self, 
        time_range: Tuple[datetime, datetime]
    ) -> List[PromptLogEntry]:
        """Get logs within the specified time range."""
        start_time, end_time = time_range
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM prompt_logs 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """, (start_time.isoformat(), end_time.isoformat()))
                
                logs = []
                for row in cursor.fetchall():
                    log_data = dict(row)
                    log_data['timestamp'] = datetime.fromisoformat(log_data['timestamp'])
                    log_data['metadata'] = json.loads(log_data['metadata']) if log_data['metadata'] else {}
                    log_data['tags'] = json.loads(log_data['tags']) if log_data['tags'] else []
                    logs.append(PromptLogEntry(**log_data))
                
                return logs
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get logs in range: {e}")
            raise
    
    def _get_model_breakdown(self, logs: List[PromptLogEntry]) -> Dict[str, Any]:
        """Get model usage breakdown."""
        model_stats = {}
        
        for log in logs:
            if log.model:
                if log.model not in model_stats:
                    model_stats[log.model] = {
                        'request_count': 0,
                        'total_tokens': 0,
                        'total_cost': 0.0,
                        'avg_latency': 0.0,
                        'latencies': []
                    }
                
                stats = model_stats[log.model]
                stats['request_count'] += 1
                
                if log.input_tokens:
                    stats['total_tokens'] += log.input_tokens
                if log.output_tokens:
                    stats['total_tokens'] += log.output_tokens
                if log.cost:
                    stats['total_cost'] += log.cost
                if log.latency:
                    stats['latencies'].append(log.latency)
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['latencies']:
                stats['avg_latency'] = sum(stats['latencies']) / len(stats['latencies'])
            del stats['latencies']  # Remove raw data
        
        return model_stats
    
    def _get_cost_analysis(self, logs: List[PromptLogEntry]) -> Dict[str, Any]:
        """Get cost analysis."""
        total_cost = sum(log.cost for log in logs if log.cost)
        costs_by_model = {}
        
        for log in logs:
            if log.model and log.cost:
                costs_by_model[log.model] = costs_by_model.get(log.model, 0) + log.cost
        
        return {
            'total_cost': total_cost,
            'cost_by_model': costs_by_model,
            'avg_cost_per_request': total_cost / len(logs) if logs else 0
        }
    
    def _get_error_analysis(self, logs: List[PromptLogEntry]) -> Dict[str, Any]:
        """Get error analysis."""
        error_logs = [log for log in logs if log.error]
        error_types = {}
        
        for log in error_logs:
            error_type = log.error.split(':')[0] if ':' in log.error else log.error
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'error_count': len(error_logs),
            'error_rate': len(error_logs) / len(logs) if logs else 0,
            'error_types': error_types
        }
    
    def cleanup_old_logs(self):
        """Clean up logs older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM prompt_logs WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old log entries")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            raise


# Analytics Functions

def calculate_usage_statistics(logs: List[PromptLogEntry]) -> Dict[str, Any]:
    """Calculate usage statistics from log entries."""
    if not logs:
        return {
            'total_requests': 0,
            'unique_sessions': 0,
            'total_tokens': 0,
            'avg_latency': 0.0,
            'success_rate': 0.0
        }
    
    total_requests = len(logs)
    unique_sessions = len(set(log.session_id for log in logs if log.session_id))
    total_input_tokens = sum(log.input_tokens for log in logs if log.input_tokens)
    total_output_tokens = sum(log.output_tokens for log in logs if log.output_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    
    latencies = [log.latency for log in logs if log.latency]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    successful_requests = len([log for log in logs if not log.error])
    success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
    
    return {
        'total_requests': total_requests,
        'unique_sessions': unique_sessions,
        'total_tokens': total_tokens,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'avg_latency': avg_latency,
        'success_rate': success_rate
    }


def generate_performance_metrics(logs: List[PromptLogEntry]) -> Dict[str, Any]:
    """Generate performance metrics from log entries."""
    if not logs:
        return {
            'avg_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0
        }
    
    latencies = sorted([log.latency for log in logs if log.latency])
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_index = int(0.95 * len(latencies))
        p99_index = int(0.99 * len(latencies))
        p95_latency = latencies[p95_index] if p95_index < len(latencies) else latencies[-1]
        p99_latency = latencies[p99_index] if p99_index < len(latencies) else latencies[-1]
    else:
        avg_latency = p95_latency = p99_latency = 0.0
    
    # Calculate throughput (requests per minute)
    if len(logs) > 1:
        time_span = (logs[0].timestamp - logs[-1].timestamp).total_seconds() / 60  # minutes
        throughput = len(logs) / time_span if time_span > 0 else 0.0
    else:
        throughput = 0.0
    
    error_count = len([log for log in logs if log.error])
    error_rate = error_count / len(logs) if logs else 0.0
    
    return {
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'throughput': throughput,
        'error_rate': error_rate
    }


def create_usage_report(time_range: Tuple[datetime, datetime]) -> str:
    """Create a formatted usage report."""
    start_time, end_time = time_range
    
    report = f"""
PromptLayer Usage Report
========================
Time Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}
Duration: {(end_time - start_time).total_seconds() / 3600:.1f} hours

This report would contain detailed usage statistics.
Note: This is a placeholder. In a real implementation, you would:
1. Query the logs for the specified time range
2. Calculate comprehensive statistics
3. Format the data into a readable report
"""
    
    return report


# Data Export Utilities

def export_to_promptlayer_format(logs: List[PromptLogEntry]) -> Dict[str, Any]:
    """Export logs in PromptLayer-compatible format."""
    return {
        'version': '1.0',
        'export_timestamp': datetime.now().isoformat(),
        'logs': [
            {
                'request_id': log.request_id,
                'timestamp': log.timestamp.isoformat(),
                'prompt_template': log.prompt,
                'prompt_inputs': log.metadata.get('inputs', {}),
                'llm_kwargs': {
                    'model': log.model,
                    'temperature': log.metadata.get('temperature'),
                    'max_tokens': log.metadata.get('max_tokens')
                },
                'response': log.response,
                'response_time': log.latency,
                'prompt_tokens': log.input_tokens,
                'completion_tokens': log.output_tokens,
                'cost': log.cost,
                'tags': log.tags,
                'metadata': log.metadata
            }
            for log in logs
        ]
    }


def export_to_csv(logs: List[PromptLogEntry]) -> str:
    """Export logs to CSV format."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'request_id', 'timestamp', 'prompt', 'response', 'model',
        'latency', 'input_tokens', 'output_tokens', 'cost', 'error',
        'session_id', 'tags'
    ])
    
    # Write data
    for log in logs:
        writer.writerow([
            log.request_id,
            log.timestamp.isoformat(),
            log.prompt[:100] + '...' if len(log.prompt) > 100 else log.prompt,
            log.response[:100] + '...' if log.response and len(log.response) > 100 else log.response,
            log.model,
            log.latency,
            log.input_tokens,
            log.output_tokens,
            log.cost,
            log.error,
            log.session_id,
            ','.join(log.tags) if log.tags else ''
        ])
    
    return output.getvalue()


def export_to_json(logs: List[PromptLogEntry]) -> Dict[str, Any]:
    """Export logs to JSON format."""
    return {
        'export_timestamp': datetime.now().isoformat(),
        'total_logs': len(logs),
        'logs': [log.to_dict() for log in logs]
    }


# Unit Tests

def test_promptlayer_adapter():
    """Basic unit tests for PromptLayer adapter."""
    import tempfile
    import os
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        config = PromptLayerConfig(database_path=tmp.name)
        adapter = PromptLayerAdapter(config)
        
        try:
            # Test logging
            request_id = adapter.log_prompt_request(
                prompt="Test prompt",
                model="gpt-3.5-turbo",
                metadata={'test': True}
            )
            
            adapter.log_prompt_response(
                request_id=request_id,
                response="Test response",
                metrics={'latency': 1.5, 'input_tokens': 10, 'output_tokens': 20}
            )
            
            # Test analytics
            analytics = adapter.get_prompt_analytics()
            assert analytics['usage_statistics']['total_requests'] >= 1
            
            # Test export
            json_export = adapter.export_logs('json')
            assert 'logs' in json_export
            assert len(json_export['logs']) >= 1
            
            csv_export = adapter.export_logs('csv')
            assert 'request_id' in csv_export
            
            # Test optimization tracking
            session_id = str(uuid.uuid4())
            adapter.track_optimization_run(
                session_id=session_id,
                prompts=["prompt1", "prompt2"],
                results={'best_score': 0.95}
            )
            
            # Test evaluation tracking
            run_id = str(uuid.uuid4())
            adapter.log_evaluation_run(
                run_id=run_id,
                dataset="test_dataset",
                results={'accuracy': 0.85}
            )
            
            print("All tests passed!")
            
        finally:
            # Cleanup
            adapter.flush_batch()
            os.unlink(tmp.name)


def test_analytics_functions():
    """Test analytics functions."""
    # Create sample log entries
    logs = [
        PromptLogEntry(
            request_id="test1",
            timestamp=datetime.now(),
            prompt="Test prompt 1",
            response="Test response 1",
            model="gpt-3.5-turbo",
            latency=1.5,
            input_tokens=10,
            output_tokens=20,
            cost=0.001
        ),
        PromptLogEntry(
            request_id="test2",
            timestamp=datetime.now(),
            prompt="Test prompt 2",
            response="Test response 2",
            model="gpt-4",
            latency=2.0,
            input_tokens=15,
            output_tokens=25,
            cost=0.002,
            error="Test error"
        )
    ]
    
    # Test usage statistics
    stats = calculate_usage_statistics(logs)
    assert stats['total_requests'] == 2
    assert stats['total_tokens'] == 70
    assert stats['success_rate'] == 0.5
    
    # Test performance metrics
    metrics = generate_performance_metrics(logs)
    assert metrics['avg_latency'] == 1.75
    assert metrics['error_rate'] == 0.5
    
    # Test export functions
    json_export = export_to_json(logs)
    assert len(json_export['logs']) == 2
    
    csv_export = export_to_csv(logs)
    assert 'request_id' in csv_export
    assert 'test1' in csv_export
    
    promptlayer_export = export_to_promptlayer_format(logs)
    assert len(promptlayer_export['logs']) == 2
    assert 'version' in promptlayer_export
    
    print("Analytics tests passed!")


if __name__ == "__main__":
    """Run basic tests when module is executed directly."""
    test_promptlayer_adapter()
    test_analytics_functions()
    print("All PromptLayer adapter tests completed successfully!")
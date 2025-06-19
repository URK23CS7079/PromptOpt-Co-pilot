"""
SQLite logging system for PromptOpt Co-Pilot.

This module provides a high-performance SQLite-based logging system with structured
storage, efficient querying capabilities, and comprehensive log management features.
"""

import sqlite3
import json
import csv
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import os

from backend.core.config import get_logger_config


@dataclass
class LoggerConfig:
    """Configuration for SQLite logger."""
    log_level: str = "INFO"
    batch_size: int = 1000
    auto_commit: bool = True
    retention_days: int = 30
    max_log_size: int = 100_000_000  # 100MB
    enable_compression: bool = True
    thread_pool_size: int = 4
    vacuum_interval_hours: int = 24


@dataclass
class LogEntry:
    """Structured log entry."""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    level: str = "INFO"
    category: str = "general"
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    error_code: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class LogBatch:
    """Thread-safe log batch for efficient insertions."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: List[LogEntry] = []
        self.lock = threading.Lock()
    
    def add(self, entry: LogEntry) -> bool:
        """Add entry to batch. Returns True if batch is full."""
        with self.lock:
            self.entries.append(entry)
            return len(self.entries) >= self.max_size
    
    def get_and_clear(self) -> List[LogEntry]:
        """Get all entries and clear the batch."""
        with self.lock:
            entries = self.entries.copy()
            self.entries.clear()
            return entries
    
    def size(self) -> int:
        """Get current batch size."""
        with self.lock:
            return len(self.entries)


class DatabaseManager:
    """Thread-safe database manager for SQLite operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self.lock = threading.Lock()
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            self.local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self.local.connection.execute("PRAGMA journal_mode=WAL")
            self.local.connection.execute("PRAGMA synchronous=NORMAL")
            self.local.connection.execute("PRAGMA cache_size=10000")
            self.local.connection.execute("PRAGMA temp_store=MEMORY")
        return self.local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def close_connection(self):
        """Close thread-local connection."""
        if hasattr(self.local, 'connection'):
            self.local.connection.close()
            delattr(self.local, 'connection')


class SQLiteLogger:
    """High-performance SQLite-based logging system."""
    
    def __init__(self, db_path: str, config: Optional[LoggerConfig] = None):
        """Initialize SQLite logger.
        
        Args:
            db_path: Path to SQLite database file
            config: Logger configuration
        """
        self.db_path = db_path
        self.config = config or LoggerConfig()
        self.db_manager = DatabaseManager(db_path)
        
        # Initialize batch processing
        self.log_batch = LogBatch(self.config.batch_size)
        self.batch_executor = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size,
            thread_name_prefix="log_batch"
        )
        
        # Background maintenance
        self.last_vacuum = datetime.utcnow()
        self.maintenance_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="log_maintenance"
        )
        
        # Statistics tracking
        self.stats = {
            'total_logs': 0,
            'logs_per_level': {},
            'logs_per_category': {},
            'batch_writes': 0,
            'last_maintenance': None
        }
        self.stats_lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        # Start background maintenance
        self._schedule_maintenance()
    
    def _initialize_database(self):
        """Initialize database tables and indexes."""
        self.create_log_tables()
        self._create_indexes()
        self._load_statistics()
    
    def create_log_tables(self):
        """Create log tables if they don't exist."""
        with self.db_manager.transaction() as conn:
            # Main logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    session_id TEXT,
                    duration_ms REAL,
                    error_code TEXT,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # LLM requests table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    session_id TEXT,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metrics TEXT,
                    duration_ms REAL,
                    tokens_used INTEGER,
                    cost REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Optimization steps table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    session_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    metrics TEXT,
                    parameters TEXT,
                    score REAL,
                    duration_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Evaluation results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    run_id TEXT NOT NULL,
                    result TEXT NOT NULL,
                    metrics TEXT,
                    dataset_name TEXT,
                    model_name TEXT,
                    score REAL,
                    duration_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_indexes(self):
        """Create database indexes for efficient querying."""
        with self.db_manager.transaction() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)",
                "CREATE INDEX IF NOT EXISTS idx_logs_category ON logs(category)",
                "CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_logs_user ON logs(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_requests(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_llm_session ON llm_requests(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_llm_model ON llm_requests(model)",
                "CREATE INDEX IF NOT EXISTS idx_opt_timestamp ON optimization_steps(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_opt_session ON optimization_steps(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_eval_timestamp ON evaluation_results(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.Error as e:
                    logging.warning(f"Failed to create index: {e}")
    
    def _load_statistics(self):
        """Load existing statistics from database."""
        try:
            with self.db_manager.transaction() as conn:
                # Count total logs
                result = conn.execute("SELECT COUNT(*) FROM logs").fetchone()
                self.stats['total_logs'] = result[0] if result else 0
                
                # Count by level
                results = conn.execute("""
                    SELECT level, COUNT(*) 
                    FROM logs 
                    GROUP BY level
                """).fetchall()
                self.stats['logs_per_level'] = {row[0]: row[1] for row in results}
                
                # Count by category
                results = conn.execute("""
                    SELECT category, COUNT(*) 
                    FROM logs 
                    GROUP BY category
                """).fetchall()
                self.stats['logs_per_category'] = {row[0]: row[1] for row in results}
                
        except sqlite3.Error as e:
            logging.warning(f"Failed to load statistics: {e}")
    
    def log(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None, 
            category: str = "general", session_id: Optional[str] = None,
            duration_ms: Optional[float] = None, error_code: Optional[str] = None,
            user_id: Optional[str] = None):
        """Main logging method.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            metadata: Additional metadata dictionary
            category: Log category
            session_id: Session identifier
            duration_ms: Operation duration in milliseconds
            error_code: Error code if applicable
            user_id: User identifier
        """
        # Check log level
        if not self._should_log(level):
            return
        
        entry = LogEntry(
            level=level.upper(),
            message=message,
            metadata=metadata or {},
            category=category,
            session_id=session_id,
            duration_ms=duration_ms,
            error_code=error_code,
            user_id=user_id
        )
        
        # Update statistics
        self._update_stats(level.upper(), category)
        
        # Add to batch or write immediately
        if self.config.auto_commit and self.log_batch.add(entry):
            self.batch_executor.submit(self._flush_batch)
        elif not self.config.auto_commit:
            self._write_log_entry(entry)
    
    def log_llm_request(self, prompt: str, model: str, response: str, 
                       metrics: Dict[str, Any], session_id: Optional[str] = None,
                       duration_ms: Optional[float] = None):
        """Log LLM request and response.
        
        Args:
            prompt: Input prompt
            model: Model name
            response: Model response
            metrics: Performance metrics
            session_id: Session identifier
            duration_ms: Request duration
        """
        with self.db_manager.transaction() as conn:
            conn.execute("""
                INSERT INTO llm_requests 
                (timestamp, session_id, prompt, model, response, metrics, 
                 duration_ms, tokens_used, cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow(),
                session_id,
                prompt,
                model,
                response,
                json.dumps(metrics),
                duration_ms,
                metrics.get('tokens_used'),
                metrics.get('cost')
            ))
        
        # Also log as regular entry
        self.log(
            "INFO",
            f"LLM request to {model}",
            {**metrics, "prompt_length": len(prompt), "response_length": len(response)},
            "llm_request",
            session_id,
            duration_ms
        )
    
    def log_optimization_step(self, session_id: str, step: int, 
                            metrics: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None,
                            duration_ms: Optional[float] = None):
        """Log optimization step.
        
        Args:
            session_id: Optimization session ID
            step: Step number
            metrics: Step metrics
            parameters: Optimization parameters
            duration_ms: Step duration
        """
        with self.db_manager.transaction() as conn:
            conn.execute("""
                INSERT INTO optimization_steps 
                (timestamp, session_id, step, metrics, parameters, score, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow(),
                session_id,
                step,
                json.dumps(metrics),
                json.dumps(parameters or {}),
                metrics.get('score'),
                duration_ms
            ))
        
        # Also log as regular entry
        self.log(
            "INFO",
            f"Optimization step {step}",
            {**metrics, "step": step, "parameters": parameters or {}},
            "optimization",
            session_id,
            duration_ms
        )
    
    def log_evaluation_result(self, run_id: str, result: Dict[str, Any],
                            dataset_name: Optional[str] = None,
                            model_name: Optional[str] = None,
                            duration_ms: Optional[float] = None):
        """Log evaluation result.
        
        Args:
            run_id: Evaluation run ID
            result: Evaluation results
            dataset_name: Dataset name
            model_name: Model name
            duration_ms: Evaluation duration
        """
        with self.db_manager.transaction() as conn:
            conn.execute("""
                INSERT INTO evaluation_results 
                (timestamp, run_id, result, metrics, dataset_name, model_name, score, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow(),
                run_id,
                json.dumps(result),
                json.dumps(result.get('metrics', {})),
                dataset_name,
                model_name,
                result.get('score'),
                duration_ms
            ))
        
        # Also log as regular entry
        self.log(
            "INFO",
            f"Evaluation completed for run {run_id}",
            {**result, "dataset": dataset_name, "model": model_name},
            "evaluation",
            None,
            duration_ms
        )
    
    def query_logs(self, filters: Optional[Dict[str, Any]] = None, 
                  limit: int = 1000, offset: int = 0) -> List[LogEntry]:
        """Query logs with filters.
        
        Args:
            filters: Query filters
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            List of log entries
        """
        filters = filters or {}
        
        # Build query
        where_clauses = []
        params = []
        
        if 'level' in filters:
            where_clauses.append("level = ?")
            params.append(filters['level'])
        
        if 'category' in filters:
            where_clauses.append("category = ?")
            params.append(filters['category'])
        
        if 'session_id' in filters:
            where_clauses.append("session_id = ?")
            params.append(filters['session_id'])
        
        if 'user_id' in filters:
            where_clauses.append("user_id = ?")
            params.append(filters['user_id'])
        
        if 'start_time' in filters:
            where_clauses.append("timestamp >= ?")
            params.append(filters['start_time'])
        
        if 'end_time' in filters:
            where_clauses.append("timestamp <= ?")
            params.append(filters['end_time'])
        
        if 'message_contains' in filters:
            where_clauses.append("message LIKE ?")
            params.append(f"%{filters['message_contains']}%")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT * FROM logs 
            WHERE {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        with self.db_manager.transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            
            return [
                LogEntry(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
                    level=row['level'],
                    category=row['category'],
                    message=row['message'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    session_id=row['session_id'],
                    duration_ms=row['duration_ms'],
                    error_code=row['error_code'],
                    user_id=row['user_id']
                )
                for row in rows
            ]
    
    def search_logs(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[LogEntry]:
        """Search logs by text query.
        
        Args:
            query: Search query
            filters: Additional filters
            
        Returns:
            List of matching log entries
        """
        filters = filters or {}
        filters['message_contains'] = query
        return self.query_logs(filters)
    
    def get_logs_by_category(self, category: str, 
                           time_range: Optional[Tuple[datetime, datetime]] = None) -> List[LogEntry]:
        """Get logs by category.
        
        Args:
            category: Log category
            time_range: Optional time range tuple (start, end)
            
        Returns:
            List of log entries
        """
        filters = {'category': category}
        if time_range:
            filters['start_time'] = time_range[0]
            filters['end_time'] = time_range[1]
        
        return self.query_logs(filters)
    
    def get_session_logs(self, session_id: str) -> List[LogEntry]:
        """Get all logs for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of log entries
        """
        return self.query_logs({'session_id': session_id})
    
    def get_log_statistics(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get log statistics.
        
        Args:
            time_range: Optional time range
            
        Returns:
            Statistics dictionary
        """
        with self.db_manager.transaction() as conn:
            base_query = "SELECT COUNT(*), level FROM logs"
            params = []
            
            if time_range:
                base_query += " WHERE timestamp BETWEEN ? AND ?"
                params.extend([time_range[0], time_range[1]])
            
            # Get counts by level
            level_query = base_query + " GROUP BY level"
            level_results = conn.execute(level_query, params).fetchall()
            levels_stats = {row[1]: row[0] for row in level_results}
            
            # Get counts by category
            category_query = base_query.replace("level", "category") + " GROUP BY category"
            category_results = conn.execute(category_query, params).fetchall()
            category_stats = {row[1]: row[0] for row in category_results}
            
            # Get total count
            total_query = "SELECT COUNT(*) FROM logs"
            if time_range:
                total_query += " WHERE timestamp BETWEEN ? AND ?"
            total_count = conn.execute(total_query, params).fetchone()[0]
            
            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_logs': total_count,
                'logs_by_level': levels_stats,
                'logs_by_category': category_stats,
                'database_size_mb': db_size / 1024 / 1024,
                'time_range': time_range,
                'generated_at': datetime.utcnow().isoformat()
            }
    
    def export_logs_csv(self, filters: Optional[Dict[str, Any]] = None, 
                       output_path: str = "logs_export.csv"):
        """Export logs to CSV file.
        
        Args:
            filters: Query filters
            output_path: Output CSV file path
        """
        logs = self.query_logs(filters, limit=100000)  # Large limit for export
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'timestamp', 'level', 'category', 'message', 
                         'session_id', 'duration_ms', 'error_code', 'user_id', 'metadata']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for log in logs:
                row = asdict(log)
                row['metadata'] = json.dumps(row['metadata']) if row['metadata'] else ''
                writer.writerow(row)
    
    def rotate_logs(self, retention_days: int):
        """Rotate and cleanup old logs.
        
        Args:
            retention_days: Number of days to retain logs
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        with self.db_manager.transaction() as conn:
            # Delete old logs
            conn.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_date,))
            conn.execute("DELETE FROM llm_requests WHERE timestamp < ?", (cutoff_date,))
            conn.execute("DELETE FROM optimization_steps WHERE timestamp < ?", (cutoff_date,))
            conn.execute("DELETE FROM evaluation_results WHERE timestamp < ?", (cutoff_date,))
            
            deleted_count = conn.total_changes
        
        # Vacuum database to reclaim space
        self.optimize_log_database()
        
        self.log("INFO", f"Rotated logs: deleted {deleted_count} old entries", 
                category="maintenance")
    
    def optimize_log_database(self):
        """Optimize database performance."""
        with self.db_manager.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        
        self.log("INFO", "Database optimization completed", category="maintenance")
    
    def batch_insert_logs(self, entries: List[LogEntry]):
        """Batch insert log entries.
        
        Args:
            entries: List of log entries to insert
        """
        if not entries:
            return
        
        with self.db_manager.transaction() as conn:
            data = [
                (
                    entry.timestamp,
                    entry.level,
                    entry.category,
                    entry.message,
                    json.dumps(entry.metadata) if entry.metadata else None,
                    entry.session_id,
                    entry.duration_ms,
                    entry.error_code,
                    entry.user_id
                )
                for entry in entries
            ]
            
            conn.executemany("""
                INSERT INTO logs 
                (timestamp, level, category, message, metadata, session_id, 
                 duration_ms, error_code, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
        
        with self.stats_lock:
            self.stats['batch_writes'] += 1
    
    def cleanup_old_logs(self, retention_days: int):
        """Clean up old logs beyond retention period.
        
        Args:
            retention_days: Days to retain
        """
        self.rotate_logs(retention_days)
    
    def _should_log(self, level: str) -> bool:
        """Check if log level should be recorded."""
        level_hierarchy = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
        
        min_level = level_hierarchy.get(self.config.log_level.upper(), 1)
        current_level = level_hierarchy.get(level.upper(), 1)
        
        return current_level >= min_level
    
    def _update_stats(self, level: str, category: str):
        """Update logging statistics."""
        with self.stats_lock:
            self.stats['total_logs'] += 1
            self.stats['logs_per_level'][level] = self.stats['logs_per_level'].get(level, 0) + 1
            self.stats['logs_per_category'][category] = self.stats['logs_per_category'].get(category, 0) + 1
    
    def _write_log_entry(self, entry: LogEntry):
        """Write single log entry to database."""
        with self.db_manager.transaction() as conn:
            conn.execute("""
                INSERT INTO logs 
                (timestamp, level, category, message, metadata, session_id, 
                 duration_ms, error_code, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp,
                entry.level,
                entry.category,
                entry.message,
                json.dumps(entry.metadata) if entry.metadata else None,
                entry.session_id,
                entry.duration_ms,
                entry.error_code,
                entry.user_id
            ))
    
    def _flush_batch(self):
        """Flush current log batch to database."""
        entries = self.log_batch.get_and_clear()
        if entries:
            self.batch_insert_logs(entries)
    
    def _schedule_maintenance(self):
        """Schedule background maintenance tasks."""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    
                    # Check if vacuum is needed
                    if (datetime.utcnow() - self.last_vacuum).total_seconds() > \
                       self.config.vacuum_interval_hours * 3600:
                        self.optimize_log_database()
                        self.last_vacuum = datetime.utcnow()
                    
                    # Auto-rotate logs if needed
                    if self.config.retention_days > 0:
                        self.rotate_logs(self.config.retention_days)
                    
                    # Update maintenance timestamp
                    with self.stats_lock:
                        self.stats['last_maintenance'] = datetime.utcnow().isoformat()
                
                except Exception as e:
                    logging.error(f"Maintenance worker error: {e}")
        
        self.maintenance_executor.submit(maintenance_worker)
    
    def flush(self):
        """Flush any pending log entries."""
        self._flush_batch()
    
    def close(self):
        """Close logger and cleanup resources."""
        # Flush any remaining logs
        self.flush()
        
        # Shutdown executors
        self.batch_executor.shutdown(wait=True)
        self.maintenance_executor.shutdown(wait=False)
        
        # Close database connections
        self.db_manager.close_connection()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Utility functions for external use
def create_logger(db_path: str = "logs/promptopt.db", 
                 config: Optional[LoggerConfig] = None) -> SQLiteLogger:
    """Create a configured SQLite logger instance.
    
    Args:
        db_path: Database file path
        config: Logger configuration
        
    Returns:
        Configured SQLite logger
    """
    if config is None:
        try:
            config = get_logger_config()
        except ImportError:
            config = LoggerConfig()
    
    return SQLiteLogger(db_path, config)


def get_log_level_from_string(level_str: str) -> str:
    """Convert string to valid log level.
    
    Args:
        level_str: Log level string
        
    Returns:
        Valid log level
    """
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    level = level_str.upper()
    return level if level in valid_levels else 'INFO'


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import unittest
    import shutil
    
    class TestSQLiteLogger(unittest.TestCase):
        """Unit tests for SQLite logger."""
        
        def setUp(self):
            self.temp_dir = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "test.db")
            self.logger = SQLiteLogger(self.db_path)
        
        def tearDown(self):
            self.logger.close()
            shutil.rmtree(self.temp_dir)
        
        def test_basic_logging(self):
            """Test basic logging functionality."""
            self.logger.log("INFO", "Test message", {"key": "value"}, "test")
            
            logs = self.logger.query_logs({"level": "INFO"}, limit=1)
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].message, "Test message")
            self.assertEqual(logs[0].metadata["key"], "value")
        
        def test_llm_logging(self):
            """Test LLM request logging."""
            self.logger.log_llm_request(
                "Test prompt",
                "gpt-4",
                "Test response",
                {"tokens_used": 100, "cost": 0.01}
            )
            
            logs = self.logger.query_logs({"category": "llm_request"})
            self.assertEqual(len(logs), 1)
        
        def test_optimization_logging(self):
            """Test optimization step logging."""
            self.logger.log_optimization_step(
                "session_123",
                1,
                {"score": 0.85, "loss": 0.15},
                {"learning_rate": 0.001}
            )
            
            logs = self.logger.query_logs({"category": "optimization"})
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].metadata["score"], 0.85)
        
        def test_evaluation_logging(self):
            """Test evaluation result logging."""
            self.logger.log_evaluation_result(
                "run_456",
                {"score": 0.92, "accuracy": 0.89},
                "test_dataset",
                "gpt-4"
            )
            
            logs = self.logger.query_logs({"category": "evaluation"})
            self.assertEqual(len(logs), 1)
        
        def test_query_filtering(self):
            """Test log query filtering."""
            # Add multiple logs
            self.logger.log("INFO", "Info message", category="test")
            self.logger.log("ERROR", "Error message", category="test")
            self.logger.log("INFO", "Another info", category="other")
            
            # Test level filtering
            error_logs = self.logger.query_logs({"level": "ERROR"})
            self.assertEqual(len(error_logs), 1)
            self.assertEqual(error_logs[0].level, "ERROR")
            
            # Test category filtering
            test_logs = self.logger.query_logs({"category": "test"})
            self.assertEqual(len(test_logs), 2)
        
        def test_session_logs(self):
            """Test session-based log retrieval."""
            session_id = "test_session_789"
            
            self.logger.log("INFO", "Session start", session_id=session_id)
            self.logger.log("INFO", "Session activity", session_id=session_id)
            self.logger.log("INFO", "Other session", session_id="other_session")
            
            session_logs = self.logger.get_session_logs(session_id)
            self.assertEqual(len(session_logs), 2)
            
            for log in session_logs:
                self.assertEqual(log.session_id, session_id)
        
        def test_search_logs(self):
            """Test log search functionality."""
            self.logger.log("INFO", "Database connection established")
            self.logger.log("INFO", "User authentication successful")
            self.logger.log("ERROR", "Database connection failed")
            
            # Search for database-related logs
            db_logs = self.logger.search_logs("database")
            self.assertEqual(len(db_logs), 2)
            
            # Search for specific terms
            auth_logs = self.logger.search_logs("authentication")
            self.assertEqual(len(auth_logs), 1)
        
        def test_statistics(self):
            """Test log statistics generation."""
            # Add various logs
            self.logger.log("INFO", "Info 1")
            self.logger.log("INFO", "Info 2")
            self.logger.log("ERROR", "Error 1")
            self.logger.log("WARNING", "Warning 1")
            
            stats = self.logger.get_log_statistics()
            
            self.assertEqual(stats['total_logs'], 4)
            self.assertEqual(stats['logs_by_level']['INFO'], 2)
            self.assertEqual(stats['logs_by_level']['ERROR'], 1)
            self.assertEqual(stats['logs_by_level']['WARNING'], 1)
        
        def test_batch_operations(self):
            """Test batch insert operations."""
            entries = [
                LogEntry(level="INFO", message=f"Batch message {i}", category="batch")
                for i in range(10)
            ]
            
            self.logger.batch_insert_logs(entries)
            
            batch_logs = self.logger.query_logs({"category": "batch"})
            self.assertEqual(len(batch_logs), 10)
        
        def test_csv_export(self):
            """Test CSV export functionality."""
            # Add some logs
            self.logger.log("INFO", "Export test 1", {"data": "value1"})
            self.logger.log("ERROR", "Export test 2", {"data": "value2"})
            
            export_path = os.path.join(self.temp_dir, "export.csv")
            self.logger.export_logs_csv(output_path=export_path)
            
            # Verify file exists and has content
            self.assertTrue(os.path.exists(export_path))
            
            with open(export_path, 'r') as f:
                content = f.read()
                self.assertIn("Export test 1", content)
                self.assertIn("Export test 2", content)
        
        def test_log_rotation(self):
            """Test log rotation and cleanup."""
            # Add old logs (simulate by setting old timestamp)
            old_time = datetime.utcnow() - timedelta(days=35)
            
            with self.logger.db_manager.transaction() as conn:
                conn.execute("""
                    INSERT INTO logs (timestamp, level, category, message)
                    VALUES (?, 'INFO', 'test', 'Old log')
                """, (old_time,))
            
            # Add recent log
            self.logger.log("INFO", "Recent log")
            
            # Rotate with 30-day retention
            self.logger.rotate_logs(30)
            
            # Should only have recent log
            remaining_logs = self.logger.query_logs()
            self.assertEqual(len(remaining_logs), 1)
            self.assertEqual(remaining_logs[0].message, "Recent log")
        
        def test_time_range_filtering(self):
            """Test time range filtering."""
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            # Add logs at different times
            with self.logger.db_manager.transaction() as conn:
                conn.execute("""
                    INSERT INTO logs (timestamp, level, category, message)
                    VALUES (?, 'INFO', 'test', 'Day ago log')
                """, (day_ago,))
                
                conn.execute("""
                    INSERT INTO logs (timestamp, level, category, message)
                    VALUES (?, 'INFO', 'test', 'Hour ago log')
                """, (hour_ago,))
            
            # Current log
            self.logger.log("INFO", "Current log")
            
            # Query last hour
            recent_logs = self.logger.query_logs({
                'start_time': hour_ago - timedelta(minutes=30),
                'end_time': now + timedelta(minutes=5)
            })
            
            # Should get hour ago and current logs
            self.assertEqual(len(recent_logs), 2)
        
        def test_concurrent_logging(self):
            """Test thread-safe concurrent logging."""
            import threading
            
            def log_worker(worker_id):
                for i in range(10):
                    self.logger.log(
                        "INFO", 
                        f"Worker {worker_id} message {i}",
                        {"worker_id": worker_id, "message_id": i}
                    )
            
            # Create multiple threads
            threads = []
            for worker_id in range(5):
                thread = threading.Thread(target=log_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Should have 50 total logs (5 workers × 10 messages)
            all_logs = self.logger.query_logs(limit=100)
            self.assertEqual(len(all_logs), 50)
        
        def test_metadata_serialization(self):
            """Test complex metadata serialization."""
            complex_metadata = {
                "nested": {"key": "value", "number": 42},
                "list": [1, 2, 3, "string"],
                "boolean": True,
                "null_value": None
            }
            
            self.logger.log("INFO", "Complex metadata test", complex_metadata)
            
            logs = self.logger.query_logs(limit=1)
            retrieved_metadata = logs[0].metadata
            
            self.assertEqual(retrieved_metadata["nested"]["key"], "value")
            self.assertEqual(retrieved_metadata["nested"]["number"], 42)
            self.assertEqual(retrieved_metadata["list"], [1, 2, 3, "string"])
            self.assertEqual(retrieved_metadata["boolean"], True)
            self.assertIsNone(retrieved_metadata["null_value"])
        
        def test_database_optimization(self):
            """Test database optimization operations."""
            # Add many logs to create fragmentation
            for i in range(100):
                self.logger.log("INFO", f"Test message {i}")
            
            # Delete some logs to create gaps
            with self.logger.db_manager.transaction() as conn:
                conn.execute("DELETE FROM logs WHERE id % 2 = 0")
            
            # Get initial database size
            initial_size = os.path.getsize(self.db_path)
            
            # Optimize database
            self.logger.optimize_log_database()
            
            # Database should still exist and be functional
            self.assertTrue(os.path.exists(self.db_path))
            
            # Should still be able to query
            remaining_logs = self.logger.query_logs(limit=100)
            self.assertGreater(len(remaining_logs), 0)
    
    # Run the tests
    unittest.main(argv=[''], exit=False, verbosity=2)
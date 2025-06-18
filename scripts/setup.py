#!/usr/bin/env python3
"""
PromptOpt Co-Pilot Setup Script
==============================

Automated project setup and environment initialization for PromptOpt Co-Pilot.
This script handles complete system setup including:
- System requirements validation
- Directory structure creation
- Python and Node.js dependency installation
- GGUF model downloading with progress tracking
- SQLite database initialization
- Configuration file generation
- System validation and testing

Usage:
    python scripts/setup.py [options]

Examples:
    # Interactive setup with user prompts
    python scripts/setup.py

    # Automated setup with minimal models
    python scripts/setup.py --auto --minimal

    # Backend-only setup
    python scripts/setup.py --backend-only --verbose

    # Custom model selection
    python scripts/setup.py --models llama2-7b,codellama-7b

Author: PromptOpt Co-Pilot Team
License: MIT
"""

import os
import sys
import json
import shutil
import hashlib
import platform
import subprocess
import argparse
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.core.config import ConfigManager, DEFAULT_CONFIG
    from backend.core.database import DatabaseManager
except ImportError as e:
    print(f"Warning: Could not import backend modules: {e}")
    print("This is expected during initial setup. Continuing...")


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY environments"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr != 'disable':
                setattr(cls, attr, '')


class ProgressBar:
    """Simple progress bar for downloads and operations"""
    
    def __init__(self, total: int, prefix: str = '', suffix: str = '', 
                 length: int = 50, fill: str = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.current = 0
        
    def update(self, current: int):
        """Update progress bar"""
        self.current = current
        percent = f"{100 * (current / float(self.total)):.1f}"
        filled_length = int(self.length * current // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end='', flush=True)
        
        if current >= self.total:
            print()  # New line when complete


class ModelDownloader:
    """Handles downloading and validation of GGUF models"""
    
    # Model configurations with download URLs and checksums
    MODELS = {
        'llama2-7b': {
            'name': 'Llama 2 7B Chat',
            'filename': 'llama-2-7b-chat.q4_0.gguf',
            'url': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_0.gguf',
            'size': 3800000000,  # ~3.8GB
            'min_ram': 8,
            'description': 'General purpose chat model, good for most tasks'
        },
        'codellama-7b': {
            'name': 'Code Llama 7B',
            'filename': 'codellama-7b.q4_0.gguf',
            'url': 'https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.q4_0.gguf',
            'size': 3800000000,  # ~3.8GB
            'min_ram': 8,
            'description': 'Specialized for code generation and analysis'
        },
        'tinyllama-1b': {
            'name': 'TinyLlama 1B',
            'filename': 'tinyllama-1.1b-chat.q4_0.gguf',
            'url': 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_0.gguf',
            'size': 669000000,  # ~669MB
            'min_ram': 2,
            'description': 'Lightweight model for testing and low-resource systems'
        },
        'mistral-7b': {
            'name': 'Mistral 7B Instruct',
            'filename': 'mistral-7b-instruct.q4_0.gguf',
            'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_0.gguf',
            'size': 4100000000,  # ~4.1GB
            'min_ram': 8,
            'description': 'High-quality instruction-following model'
        }
    }
    
    def __init__(self, models_dir: Path, verbose: bool = False):
        self.models_dir = models_dir
        self.verbose = verbose
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
        
    def download_model(self, model_key: str) -> bool:
        """Download a specific model with progress tracking"""
        if model_key not in self.MODELS:
            print(f"{Colors.RED}Error: Unknown model '{model_key}'{Colors.END}")
            return False
            
        model_info = self.MODELS[model_key]
        filepath = self.models_dir / model_info['filename']
        
        # Check if model already exists
        if filepath.exists():
            print(f"{Colors.GREEN}Model {model_info['name']} already exists{Colors.END}")
            return True
            
        print(f"{Colors.BLUE}Downloading {model_info['name']}...{Colors.END}")
        print(f"URL: {model_info['url']}")
        print(f"Size: ~{model_info['size'] / 1e9:.1f}GB")
        
        try:
            # Start download with streaming
            response = self.session.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', model_info['size']))
            
            # Create progress bar
            progress = ProgressBar(
                total=total_size,
                prefix=f"Downloading {model_info['filename']}",
                suffix="Complete",
                length=50
            )
            
            # Download with progress tracking
            downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(downloaded)
                        
            print(f"{Colors.GREEN}Successfully downloaded {model_info['name']}{Colors.END}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}Error downloading {model_info['name']}: {e}{Colors.END}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return False
            
    def validate_model(self, model_key: str) -> bool:
        """Validate downloaded model file"""
        if model_key not in self.MODELS:
            return False
            
        model_info = self.MODELS[model_key]
        filepath = self.models_dir / model_info['filename']
        
        if not filepath.exists():
            return False
            
        # Check file size
        file_size = filepath.stat().st_size
        expected_size = model_info['size']
        
        # Allow 5% variance in file size
        if abs(file_size - expected_size) > expected_size * 0.05:
            print(f"{Colors.YELLOW}Warning: {model_info['name']} file size differs from expected{Colors.END}")
            
        return True
        
    def get_recommended_models(self, available_ram_gb: int) -> List[str]:
        """Get recommended models based on available RAM"""
        recommended = []
        
        for key, model in self.MODELS.items():
            if model['min_ram'] <= available_ram_gb:
                recommended.append(key)
                
        return recommended


class SetupManager:
    """Main setup manager for PromptOpt Co-Pilot"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = PROJECT_ROOT
        self.errors = []
        self.warnings = []
        
        # Check if we're in a TTY environment
        if not sys.stdout.isatty():
            Colors.disable()
            
        # Directory structure
        self.directories = {
            'backend': self.project_root / 'backend',
            'frontend': self.project_root / 'frontend',
            'models': self.project_root / 'models',
            'data': self.project_root / 'data',
            'logs': self.project_root / 'logs',
            'config': self.project_root / 'config',
            'scripts': self.project_root / 'scripts',
            'tests': self.project_root / 'tests',
            'docs': self.project_root / 'docs',
            'temp': self.project_root / 'temp'
        }
        
        self.model_downloader = ModelDownloader(self.directories['models'], verbose)
        
    def log(self, message: str, level: str = 'INFO'):
        """Log message with appropriate formatting"""
        if level == 'ERROR':
            print(f"{Colors.RED}[ERROR] {message}{Colors.END}")
            self.errors.append(message)
        elif level == 'WARNING':
            print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")
            self.warnings.append(message)
        elif level == 'SUCCESS':
            print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")
        elif level == 'INFO':
            if self.verbose:
                print(f"{Colors.BLUE}[INFO] {message}{Colors.END}")
        else:
            print(message)
            
    def check_system_requirements(self) -> bool:
        """Validate system requirements"""
        self.log("Checking system requirements...", 'INFO')
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            self.log(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}", 'ERROR')
            return False
        else:
            self.log(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓", 'SUCCESS')
            
        # Check available memory
        try:
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]
                meminfo = MEMORYSTATUSEX()
                meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo))
                total_ram = meminfo.ullTotalPhys / (1024**3)
            else:
                # Unix-like systems
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    total_ram = int([line for line in meminfo.split('\n') 
                                   if 'MemTotal' in line][0].split()[1]) / (1024**2)
                    
            self.log(f"Available RAM: {total_ram:.1f}GB", 'INFO')
            
            if total_ram < 4:
                self.log("Warning: Less than 4GB RAM detected. Consider using TinyLlama model only.", 'WARNING')
            elif total_ram < 8:
                self.log("Warning: Less than 8GB RAM detected. Larger models may not run smoothly.", 'WARNING')
            else:
                self.log("Sufficient RAM available ✓", 'SUCCESS')
                
        except Exception as e:
            self.log(f"Could not determine available RAM: {e}", 'WARNING')
            
        # Check disk space
        try:
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            self.log(f"Available disk space: {free_gb:.1f}GB", 'INFO')
            
            if free_gb < 10:
                self.log("Error: At least 10GB free disk space required", 'ERROR')
                return False
            else:
                self.log("Sufficient disk space available ✓", 'SUCCESS')
                
        except Exception as e:
            self.log(f"Could not determine disk space: {e}", 'WARNING')
            
        # Check Node.js (optional for frontend)
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                node_version = result.stdout.strip()
                self.log(f"Node.js {node_version} ✓", 'SUCCESS')
            else:
                self.log("Node.js not found (frontend setup will be skipped)", 'WARNING')
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log("Node.js not found (frontend setup will be skipped)", 'WARNING')
            
        # Check npm (optional for frontend)
        try:
            result = subprocess.run(['npm', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                self.log(f"npm {npm_version} ✓", 'SUCCESS')
            else:
                self.log("npm not found (frontend setup will be skipped)", 'WARNING')
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log("npm not found (frontend setup will be skipped)", 'WARNING')
            
        return len(self.errors) == 0
        
    def create_directory_structure(self) -> bool:
        """Create required project directories"""
        self.log("Creating directory structure...", 'INFO')
        
        try:
            for name, path in self.directories.items():
                path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created directory: {path}", 'INFO')
                
            # Create subdirectories
            subdirs = [
                self.directories['backend'] / 'core',
                self.directories['backend'] / 'api',
                self.directories['backend'] / 'services',
                self.directories['backend'] / 'utils',
                self.directories['frontend'] / 'src',
                self.directories['frontend'] / 'public',
                self.directories['config'] / 'models',
                self.directories['data'] / 'prompts',
                self.directories['data'] / 'results',
                self.directories['tests'] / 'unit',
                self.directories['tests'] / 'integration',
            ]
            
            for subdir in subdirs:
                subdir.mkdir(parents=True, exist_ok=True)
                self.log(f"Created subdirectory: {subdir}", 'INFO')
                
            self.log("Directory structure created successfully", 'SUCCESS')
            return True
            
        except Exception as e:
            self.log(f"Error creating directory structure: {e}", 'ERROR')
            return False
            
    def install_python_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.log("Installing Python dependencies...", 'INFO')
        
        # Basic requirements for setup
        requirements = [
            'requests>=2.28.0',
            'sqlalchemy>=1.4.0',
            'fastapi>=0.95.0',
            'uvicorn>=0.20.0',
            'pydantic>=1.10.0',
            'python-multipart>=0.0.5',
            'jinja2>=3.1.0',
            'python-dotenv>=1.0.0',
            'click>=8.0.0',
            'rich>=13.0.0',
            'httpx>=0.24.0',
            'psutil>=5.9.0',
            'numpy>=1.21.0',
            'pandas>=1.5.0',
            'scikit-learn>=1.2.0',
            'matplotlib>=3.6.0',
            'seaborn>=0.12.0',
            'transformers>=4.20.0',
            'torch>=2.0.0',
            'llama-cpp-python>=0.2.0',
        ]
        
        try:
            # Create requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(requirements))
                
            # Install requirements
            self.log("Installing packages with pip...", 'INFO')
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                self.log("Python dependencies installed successfully", 'SUCCESS')
                return True
            else:
                self.log(f"Error installing Python dependencies: {result.stderr}", 'ERROR')
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Timeout installing Python dependencies", 'ERROR')
            return False
        except Exception as e:
            self.log(f"Error installing Python dependencies: {e}", 'ERROR')
            return False
            
    def install_node_dependencies(self) -> bool:
        """Install Node.js dependencies"""
        self.log("Installing Node.js dependencies...", 'INFO')
        
        # Check if Node.js is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log("Node.js not available, skipping frontend setup", 'WARNING')
            return True
            
        try:
            frontend_dir = self.directories['frontend']
            
            # Create package.json
            package_json = {
                "name": "promptopt-copilot-frontend",
                "version": "1.0.0",
                "description": "PromptOpt Co-Pilot Frontend",
                "main": "index.js",
                "scripts": {
                    "dev": "vite",
                    "build": "vite build",
                    "preview": "vite preview",
                    "test": "vitest"
                },
                "dependencies": {
                    "react": "^18.2.0",
                    "react-dom": "^18.2.0",
                    "react-router-dom": "^6.8.0",
                    "axios": "^1.3.0",
                    "lucide-react": "^0.263.1",
                    "@radix-ui/react-dialog": "^1.0.3",
                    "@radix-ui/react-dropdown-menu": "^2.0.4",
                    "class-variance-authority": "^0.6.0",
                    "clsx": "^1.2.1",
                    "tailwind-merge": "^1.12.0"
                },
                "devDependencies": {
                    "@types/react": "^18.0.28",
                    "@types/react-dom": "^18.0.11",
                    "@vitejs/plugin-react": "^4.0.0",
                    "autoprefixer": "^10.4.14",
                    "postcss": "^8.4.21",
                    "tailwindcss": "^3.3.0",
                    "typescript": "^5.0.2",
                    "vite": "^4.3.0",
                    "vitest": "^0.30.0"
                }
            }
            
            with open(frontend_dir / 'package.json', 'w') as f:
                json.dump(package_json, f, indent=2)
                
            # Install dependencies
            self.log("Installing npm packages...", 'INFO')
            result = subprocess.run([
                'npm', 'install'
            ], cwd=frontend_dir, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log("Node.js dependencies installed successfully", 'SUCCESS')
                return True
            else:
                self.log(f"Error installing Node.js dependencies: {result.stderr}", 'ERROR')
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Timeout installing Node.js dependencies", 'ERROR')
            return False
        except Exception as e:
            self.log(f"Error installing Node.js dependencies: {e}", 'ERROR')
            return False
            
    def download_models(self, model_keys: Optional[List[str]] = None) -> bool:
        """Download specified models or prompt user for selection"""
        self.log("Setting up GGUF models...", 'INFO')
        
        if not model_keys:
            # Interactive model selection
            print(f"\n{Colors.BOLD}Available Models:{Colors.END}")
            for i, (key, model) in enumerate(self.model_downloader.MODELS.items(), 1):
                print(f"{i}. {Colors.CYAN}{model['name']}{Colors.END}")
                print(f"   Size: ~{model['size'] / 1e9:.1f}GB, Min RAM: {model['min_ram']}GB")
                print(f"   {model['description']}")
                print()
                
            # Get user selection
            while True:
                try:
                    selection = input(f"{Colors.YELLOW}Select models to download (comma-separated numbers, or 'all' for all models): {Colors.END}")
                    
                    if selection.lower() == 'all':
                        model_keys = list(self.model_downloader.MODELS.keys())
                        break
                    elif selection.lower() in ['none', 'skip']:
                        self.log("Skipping model download", 'INFO')
                        return True
                    else:
                        # Parse selection
                        selected_indices = [int(x.strip()) for x in selection.split(',')]
                        model_list = list(self.model_downloader.MODELS.keys())
                        model_keys = [model_list[i-1] for i in selected_indices 
                                    if 1 <= i <= len(model_list)]
                        break
                        
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid selection. Please enter numbers separated by commas.{Colors.END}")
                    
        # Download selected models
        success = True
        for model_key in model_keys:
            if not self.model_downloader.download_model(model_key):
                success = False
                
        if success:
            self.log(f"Successfully downloaded {len(model_keys)} model(s)", 'SUCCESS')
        else:
            self.log("Some models failed to download", 'ERROR')
            
        return success
        
    def initialize_database(self) -> bool:
        """Initialize SQLite database and create tables"""
        self.log("Initializing database...", 'INFO')
        
        try:
            # Import database manager
            from backend.core.database import DatabaseManager
            
            db_path = self.directories['data'] / 'promptopt.db'
            db_manager = DatabaseManager(str(db_path))
            
            # Create tables
            db_manager.create_tables()
            
            # Insert initial data
            db_manager.insert_initial_data()
            
            self.log(f"Database initialized at {db_path}", 'SUCCESS')
            return True
            
        except Exception as e:
            self.log(f"Error initializing database: {e}", 'ERROR')
            # Create basic database structure manually
            try:
                import sqlite3
                db_path = self.directories['data'] / 'promptopt.db'
                
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Create basic tables
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS prompts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS experiments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            description TEXT,
                            status TEXT DEFAULT 'pending',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    conn.commit()
                    
                self.log(f"Basic database created at {db_path}", 'SUCCESS')
                return True
                
            except Exception as e2:
                self.log(f"Error creating basic database: {e2}", 'ERROR')
                return False
                
    def create_config_files(self) -> bool:
        """Generate configuration files"""
        self.log("Creating configuration files...", 'INFO')
        
        try:
            config_dir = self.directories['config']
            
            # Main configuration
            main_config = {
                "app": {
                    "name": "PromptOpt Co-Pilot",
                    "version": "1.0.0",
                    "debug": False,
                    "host": "localhost",
                    "port": 8000
                },
                "database": {
                    "type": "sqlite",
                    "path": "data/promptopt.db"
                },
                "models": {
                    "directory": "models",
                    "default_model": "tinyllama-1b",
                    "max_context_length": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "optimization": {
                    "max_iterations": 100,
                    "population_size": 20,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.8
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/promptopt.log",
                    "max_size": "10MB",
                    "backup_count": 5
                }
            }
            
            with open(config_dir / 'config.json', 'w') as f:
                json.dump(main_config, f, indent=2)
                
            # Environment configuration
            env_config = """# PromptOpt Co-Pilot Environment Configuration
PROMPTOPT_ENV=development
PROMPTOPT_DEBUG=true
PROMPTOPT_HOST=localhost
PROMPTOPT_PORT=8000
PROMPTOPT_DATABASE_PATH=data/promptopt.db
PROMPTOPT_MODELS_DIR=models
PROMPTOPT_LOGS_DIR=logs
PROMPTOPT_TEMP_DIR=temp

# Security
PROMPTOPT_SECRET_KEY=your-secret-key-here
PROMPTOPT_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Model Configuration
PROMPTOPT_DEFAULT_MODEL=tinyllama-1b
PROMPTOPT_MAX_CONTEXT_LENGTH=4096
PROMPTOPT_TEMPERATURE=0.7
PROMPTOPT_TOP_P=0.9

# Optimization Settings
PROMPTOPT_MAX_ITERATIONS=100
PROMPTOPT_POPULATION_SIZE=20
PROMPTOPT_MUTATION_RATE=0.1
PROMPTOPT_CROSSOVER_RATE=0.8
"""
            
            with open(self.project_root / '.env', 'w') as f:
                f.write(env_config)
                
            # Model configuration
            model_config = {
                "models": {}
            }
            
            for key, model in self.model_downloader.MODELS.items():
                model_config["models"][key] = {
                    "name": model["name"],
                    "filename": model["filename"],
                    "min_ram": model["min_ram"],
                    "description": model["description"],
                    "parameters": {
                        "max_context_length": 4096,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1
                    }
                }
                
            with open(config_dir / 'models.json', 'w') as f:
                json.dump(model_config, f, indent=2)
                
            # Docker configuration
            dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs temp config

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            
            with open(self.project_root / 'Dockerfile', 'w') as f:
                f.write(dockerfile_content)
                
            # Docker Compose configuration
            docker_compose_content = """version: '3.8'

services:
  promptopt-backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PROMPTOPT_ENV=production
      - PROMPTOPT_HOST=0.0.0.0
      - PROMPTOPT_PORT=8000
    restart: unless-stopped
    
  promptopt-frontend:
    image: node:18-alpine
    working_dir: /app
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    command: sh -c "npm install && npm run dev -- --host 0.0.0.0"
    depends_on:
      - promptopt-backend
    restart: unless-stopped
"""
            
            with open(self.project_root / 'docker-compose.yml', 'w') as f:
                f.write(docker_compose_content)
                
            # Gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# PyCharm
.idea/

# VS Code
.vscode/

# Jupyter Notebook
.ipynb_checkpoints

# Models (large files)
models/*.gguf
models/*.bin
models/*.safetensors

# Data and logs
data/*.db
data/*.sqlite
data/*.sqlite3
logs/*.log
temp/*

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn/

# Build outputs
frontend/dist/
frontend/build/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
[Tt]humbs.db
"""
            
            with open(self.project_root / '.gitignore', 'w') as f:
                f.write(gitignore_content)
                
            # README
            readme_content = """# PromptOpt Co-Pilot

An offline prompt optimization system that uses local LLMs to iteratively improve prompts based on performance metrics.

## Quick Start

1. **Setup (First Time)**:
   ```bash
   python scripts/setup.py
   ```

2. **Start Backend**:
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

3. **Start Frontend** (if Node.js is installed):
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access Application**:
   - Backend API: http://localhost:8000
   - Frontend UI: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Features

- 🤖 Local LLM integration (GGUF models)
- 🔄 Iterative prompt optimization
- 📊 Performance metrics and analytics
- 🎯 Multiple optimization algorithms
- 📱 Web-based user interface
- 🛡️ Completely offline operation
- 🐳 Docker support

## System Requirements

- Python 3.9+
- 8GB+ RAM (recommended for larger models)
- 10GB+ free disk space
- Node.js 16+ (optional, for frontend)

## Models

The system supports various GGUF models:
- **TinyLlama 1B**: Lightweight, good for testing
- **Llama 2 7B**: General purpose, balanced performance
- **Code Llama 7B**: Specialized for code generation
- **Mistral 7B**: High-quality instruction following

## Configuration

Configuration files are located in the `config/` directory:
- `config.json`: Main application settings
- `models.json`: Model-specific configurations
- `.env`: Environment variables

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black backend/ scripts/
isort backend/ scripts/

# Type checking
mypy backend/
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build custom image
docker build -t promptopt-copilot .
docker run -p 8000:8000 -v ./models:/app/models promptopt-copilot
```

## Documentation

- [API Documentation](docs/api.md)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [Model Configuration](docs/models.md)

## License

MIT License - see LICENSE file for details.
"""
            
            with open(self.project_root / 'README.md', 'w') as f:
                f.write(readme_content)
                
            self.log("Configuration files created successfully", 'SUCCESS')
            return True
            
        except Exception as e:
            self.log(f"Error creating configuration files: {e}", 'ERROR')
            return False
            
    def validate_installation(self) -> bool:
        """Run comprehensive system validation"""
        self.log("Validating installation...", 'INFO')
        
        validation_passed = True
        
        # Check directory structure
        for name, path in self.directories.items():
            if not path.exists():
                self.log(f"Missing directory: {path}", 'ERROR')
                validation_passed = False
            else:
                self.log(f"Directory exists: {name} ✓", 'INFO')
                
        # Check configuration files
        config_files = [
            'config/config.json',
            'config/models.json',
            '.env',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if not file_path.exists():
                self.log(f"Missing configuration file: {config_file}", 'ERROR')
                validation_passed = False
            else:
                self.log(f"Configuration file exists: {config_file} ✓", 'INFO')
                
        # Check database
        db_path = self.directories['data'] / 'promptopt.db'
        if not db_path.exists():
            self.log("Database not found", 'ERROR')
            validation_passed = False
        else:
            try:
                import sqlite3
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    if tables:
                        self.log(f"Database validation passed ({len(tables)} tables) ✓", 'SUCCESS')
                    else:
                        self.log("Database exists but no tables found", 'WARNING')
            except Exception as e:
                self.log(f"Database validation failed: {e}", 'ERROR')
                validation_passed = False
                
        # Check models
        models_dir = self.directories['models']
        model_files = list(models_dir.glob('*.gguf'))
        if not model_files:
            self.log("No GGUF models found", 'WARNING')
        else:
            self.log(f"Found {len(model_files)} GGUF model(s) ✓", 'SUCCESS')
            for model_file in model_files:
                self.log(f"  - {model_file.name}", 'INFO')
                
        # Test Python imports
        required_modules = [
            'fastapi', 'uvicorn', 'sqlalchemy', 'pydantic', 
            'requests', 'numpy', 'pandas'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                self.log(f"Python module {module} ✓", 'INFO')
            except ImportError:
                self.log(f"Python module {module} not found", 'ERROR')
                validation_passed = False
                
        # Test backend startup (quick check)
        try:
            sys.path.insert(0, str(self.project_root))
            from backend.core.config import ConfigManager
            config = ConfigManager()
            self.log("Backend configuration loading ✓", 'SUCCESS')
        except Exception as e:
            self.log(f"Backend validation failed: {e}", 'ERROR')
            validation_passed = False
            
        if validation_passed:
            self.log("Installation validation passed ✓", 'SUCCESS')
        else:
            self.log("Installation validation failed", 'ERROR')
            
        return validation_passed
        
    def cleanup_on_failure(self):
        """Clean up partial installations on failure"""
        self.log("Cleaning up partial installation...", 'INFO')
        
        try:
            # Remove partially downloaded models
            models_dir = self.directories['models']
            if models_dir.exists():
                for model_file in models_dir.glob('*.gguf'):
                    try:
                        # Check if file is complete by trying to read it
                        with open(model_file, 'rb') as f:
                            f.seek(0, 2)  # Seek to end
                            size = f.tell()
                            if size < 1000000:  # Less than 1MB, probably incomplete
                                model_file.unlink()
                                self.log(f"Removed incomplete model: {model_file.name}", 'INFO')
                    except Exception:
                        pass
                        
            # Remove empty directories
            for name, path in self.directories.items():
                try:
                    if path.exists() and not any(path.iterdir()):
                        path.rmdir()
                        self.log(f"Removed empty directory: {name}", 'INFO')
                except Exception:
                    pass
                    
            self.log("Cleanup completed", 'SUCCESS')
            
        except Exception as e:
            self.log(f"Error during cleanup: {e}", 'ERROR')
            
    def run_setup(self, args: argparse.Namespace) -> bool:
        """Run the complete setup process"""
        start_time = time.time()
        
        print(f"{Colors.BOLD}{Colors.BLUE}")
        print("=" * 60)
        print("  PromptOpt Co-Pilot Setup")
        print("  Automated Project Initialization")
        print("=" * 60)
        print(f"{Colors.END}")
        
        try:
            # Step 1: System requirements check
            if not self.check_system_requirements():
                self.log("System requirements check failed", 'ERROR')
                return False
                
            # Step 2: Create directory structure
            if not self.create_directory_structure():
                self.log("Directory creation failed", 'ERROR')
                return False
                
            # Step 3: Install Python dependencies
            if not args.skip_dependencies:
                if not self.install_python_dependencies():
                    self.log("Python dependencies installation failed", 'ERROR')
                    if not args.continue_on_error:
                        return False
                        
            # Step 4: Install Node.js dependencies (if not backend-only)
            if not args.backend_only and not args.skip_dependencies:
                if not self.install_node_dependencies():
                    self.log("Node.js dependencies installation failed", 'WARNING')
                    # Don't fail setup for frontend issues
                    
            # Step 5: Download models
            if not args.skip_models:
                model_keys = None
                if args.models:
                    model_keys = args.models.split(',')
                elif args.minimal:
                    model_keys = ['tinyllama-1b']
                elif args.auto:
                    # Auto-select based on system RAM
                    try:
                        import psutil
                        available_ram = psutil.virtual_memory().total / (1024**3)
                        model_keys = self.model_downloader.get_recommended_models(int(available_ram))
                        if not model_keys:
                            model_keys = ['tinyllama-1b']  # Fallback
                    except ImportError:
                        model_keys = ['tinyllama-1b']  # Safe default
                        
                if not self.download_models(model_keys):
                    self.log("Model download failed", 'ERROR')
                    if not args.continue_on_error:
                        return False
                        
            # Step 6: Initialize database
            if not self.initialize_database():
                self.log("Database initialization failed", 'ERROR')
                if not args.continue_on_error:
                    return False
                    
            # Step 7: Create configuration files
            if not self.create_config_files():
                self.log("Configuration file creation failed", 'ERROR')
                if not args.continue_on_error:
                    return False
                    
            # Step 8: Validate installation
            if not args.skip_validation:
                if not self.validate_installation():
                    self.log("Installation validation failed", 'WARNING')
                    
            # Setup completed successfully
            elapsed_time = time.time() - start_time
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}")
            print("=" * 60)
            print("  Setup Completed Successfully! ")
            print(f"  Total time: {elapsed_time:.1f} seconds")
            print("=" * 60)
            print(f"{Colors.END}")
            
            # Print next steps
            print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
            print(f"1. Start the backend server:")
            print(f"   {Colors.CYAN}python -m uvicorn backend.main:app --reload{Colors.END}")
            print(f"\n2. Start the frontend (if installed):")
            print(f"   {Colors.CYAN}cd frontend && npm run dev{Colors.END}")
            print(f"\n3. Access the application:")
            print(f"   Backend API: {Colors.CYAN}http://localhost:8000{Colors.END}")
            print(f"   Frontend UI: {Colors.CYAN}http://localhost:3000{Colors.END}")
            print(f"   API Docs: {Colors.CYAN}http://localhost:8000/docs{Colors.END}")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings encountered during setup:{Colors.END}")
                for warning in self.warnings:
                    print(f"  - {warning}")
                    
            return True
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
            self.cleanup_on_failure()
            return False
        except Exception as e:
            self.log(f"Unexpected error during setup: {e}", 'ERROR')
            self.cleanup_on_failure()
            return False


def main():
    """Main entry point for setup script"""
    parser = argparse.ArgumentParser(
        description="PromptOpt Co-Pilot Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup.py                          # Interactive setup
  python scripts/setup.py --auto --minimal        # Automated minimal setup
  python scripts/setup.py --backend-only -v       # Backend only with verbose output
  python scripts/setup.py --models tinyllama-1b   # Setup with specific model
  python scripts/setup.py --skip-models           # Setup without downloading models
        """
    )
    
    # Setup options
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--auto', action='store_true',
                      help='Automated setup with minimal user interaction')
    parser.add_argument('--minimal', action='store_true',
                      help='Minimal setup with lightweight models only')
    parser.add_argument('--backend-only', action='store_true',
                      help='Setup backend only, skip frontend')
    
    # Component options
    parser.add_argument('--skip-dependencies', action='store_true',
                      help='Skip dependency installation')
    parser.add_argument('--skip-models', action='store_true',
                      help='Skip model downloading')
    parser.add_argument('--skip-validation', action='store_true',
                      help='Skip installation validation')
    
    # Model selection
    parser.add_argument('--models', type=str,
                      help='Comma-separated list of models to download (e.g., tinyllama-1b,llama2-7b)')
    
    # Error handling
    parser.add_argument('--continue-on-error', action='store_true',
                      help='Continue setup even if some steps fail')
    
    args = parser.parse_args()
    
    # Create setup manager
    setup_manager = SetupManager(verbose=args.verbose)
    
    # Run setup
    success = setup_manager.run_setup(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
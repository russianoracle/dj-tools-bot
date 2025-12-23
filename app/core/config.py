"""
Application configuration with environment variable support.

Separates code (app/) from data (DATA_DIR) for safe deployments.
"""

import os
from pathlib import Path
from typing import Optional

from app.common.logging import get_logger

logger = get_logger(__name__)


def get_data_dir() -> Path:
    """
    Get data directory for persistent storage.

    Reads from DATA_DIR environment variable, fallback to project cache/.

    Usage:
        Local dev: DATA_DIR not set → uses ./cache/
        Production: DATA_DIR=/data → uses /data/

    This ensures database and caches survive application updates.
    """
    data_dir_env = os.getenv('DATA_DIR')

    if data_dir_env:
        # Production: use external data directory
        data_dir = Path(data_dir_env)
    else:
        # Local dev: use project cache directory
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'cache'

    # Ensure directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def get_cache_dir() -> Path:
    """Get cache directory (inside data dir)."""
    return get_data_dir()


def get_db_path() -> Path:
    """Get path to SQLite database."""
    return get_data_dir() / 'predictions.db'


# Module-level constants (lazy-evaluated)
DATA_DIR: Optional[Path] = None
CACHE_DIR: Optional[Path] = None
DB_PATH: Optional[Path] = None


def init_config():
    """Initialize configuration (call once at startup)."""
    global DATA_DIR, CACHE_DIR, DB_PATH

    DATA_DIR = get_data_dir()
    CACHE_DIR = get_cache_dir()
    DB_PATH = get_db_path()

    logger.info("Configuration initialized", data={
        "data_dir": str(DATA_DIR),
        "cache_dir": str(CACHE_DIR),
        "db_path": str(DB_PATH)
    })

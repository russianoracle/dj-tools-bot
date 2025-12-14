"""
Settings - Application configuration using dataclasses.

Environment variables:
- CACHE_BACKEND: sqlite, redis, memory
- REDIS_URL: Redis connection URL
- DB_PATH: SQLite database path
- LOG_LEVEL: DEBUG, INFO, WARNING, ERROR
- TELEGRAM_BOT_TOKEN: Bot token
- ADMIN_USER_ID: Admin Telegram ID
"""

import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


class CacheBackend(str, Enum):
    """Cache backend options."""
    SQLITE = "sqlite"
    REDIS = "redis"
    MEMORY = "memory"


class LogLevel(str, Enum):
    """Log level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Settings:
    """Application settings from environment."""

    # Cache
    cache_backend: CacheBackend = field(
        default_factory=lambda: CacheBackend(os.getenv("CACHE_BACKEND", "sqlite"))
    )
    redis_url: Optional[str] = field(
        default_factory=lambda: os.getenv("REDIS_URL")
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "cache/predictions.db")
    )

    # Logging
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "INFO"))
    )
    log_json: bool = field(
        default_factory=lambda: os.getenv("LOG_JSON", "false").lower() == "true"
    )

    # Telegram
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    admin_user_id: int = field(
        default_factory=lambda: int(os.getenv("ADMIN_USER_ID", "0"))
    )

    # Audio
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("SAMPLE_RATE", "22050"))
    )
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    )

    # Downloads
    downloads_dir: str = field(
        default_factory=lambda: os.getenv("DOWNLOADS_DIR", "/app/downloads")
    )

    # Performance
    workers: int = field(
        default_factory=lambda: int(os.getenv("WORKERS", "4"))
    )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

"""Centralized logging configuration with YC Cloud Logging support."""

import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from .formatters import JSONFormatter, StructuredLogAdapter
from .correlation import CorrelationLogFilter
from .logging_config import get_logging_config


# Global flag to track if logging is configured
_logging_configured = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: Optional[bool] = None,
    enable_yc_logging: bool = True,
    yc_log_group_id: Optional[str] = None,
    yc_folder_id: Optional[str] = None,
    yc_resource_type: str = "bot",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    component: str = "default",
) -> None:
    """
    Configure application-wide logging with optional YC Cloud Logging.

    Uses centralized logging-config.yaml for component-specific configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). If None, uses centralized config
        log_file: Path to local log file (optional)
        json_format: Use JSON formatter for structured logs. If None, uses centralized config
        enable_yc_logging: Enable Yandex Cloud Logging handler
        yc_log_group_id: YC Log Group ID
        yc_folder_id: YC Folder ID (alternative to log_group_id)
        yc_resource_type: Resource type label for YC logs
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
        component: Component name for centralized config (bot, worker, fluent-bit, etc)
    """
    global _logging_configured
    if _logging_configured:
        return

    # Get centralized logging config
    log_config = get_logging_config()

    # Use centralized config if not explicitly provided
    if level is None:
        level = log_config.get_level(component)
    if json_format is None:
        json_format = log_config.get_json_format(component)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add correlation filter to inject context vars
    correlation_filter = CorrelationLogFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.addFilter(correlation_filter)

    if json_format:
        console_handler.setFormatter(JSONFormatter(include_path=False))
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(correlation_id)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.addFilter(correlation_filter)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Yandex Cloud Logging handler (optional)
    yc_group_id = yc_log_group_id or os.getenv("YC_LOG_GROUP_ID")
    yc_fid = yc_folder_id or os.getenv("YC_FOLDER_ID")

    if enable_yc_logging and (yc_group_id or yc_fid):
        try:
            from .yc_handler import YandexCloudLoggingHandler

            yc_handler = YandexCloudLoggingHandler(
                log_group_id=yc_group_id,
                folder_id=yc_fid,
                resource_type=yc_resource_type,
                level=getattr(logging, level.upper()),
            )
            yc_handler.addFilter(correlation_filter)
            root_logger.addHandler(yc_handler)
            root_logger.info("YC Cloud Logging enabled")
        except ImportError:
            root_logger.warning("yandexcloud SDK not installed, YC logging disabled")
        except Exception as e:
            root_logger.warning(f"Failed to init YC logging: {e}")

    _logging_configured = True


def get_logger(name: str) -> StructuredLogAdapter:
    """
    Get structured logger adapter for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogAdapter with correlation ID support
    """
    return StructuredLogAdapter(logging.getLogger(name))


# Legacy compatibility
def setup_logger(
    name: str = "mood_classifier",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3
) -> logging.Logger:
    """Legacy setup_logger for backward compatibility."""
    setup_logging(
        level=level,
        log_file=log_file,
        json_format=False,
        enable_yc_logging=False,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    return logging.getLogger(name)

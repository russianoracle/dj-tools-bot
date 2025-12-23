"""Utility modules for mood classifier."""

from .config import Config, get_config
from .logger import setup_logger, setup_logging, get_logger
from .logging_config import LoggingConfig, get_logging_config
from .formatters import JSONFormatter, StructuredLogAdapter
from .correlation import (
    CorrelationMiddleware,
    CorrelationLogFilter,
    get_correlation_id,
    set_correlation_id,
    get_user_id,
    set_user_id,
    get_job_id,
    set_job_id,
)
from .format import (
    format_time,
    format_time_range,
    format_duration,
    format_bpm,
    format_percent,
    format_confidence,
)
from .file_discovery import (
    AUDIO_EXTENSIONS,
    find_audio_files,
    find_audio_files_with_filter,
    get_relative_path,
)
from .warnings_config import (
    suppress_audio_warnings,
    suppress_all_warnings,
    restore_warnings,
    WarningContext,
)
from .utils import (
    truncate_for_display,
    truncate_for_metrics,
    capture_output,
    setup_exception_handler,
)

__all__ = [
    # Config
    'Config',
    'get_config',
    # Logger
    'setup_logger',
    'setup_logging',
    'get_logger',
    # Logging config
    'LoggingConfig',
    'get_logging_config',
    # Structured logging
    'JSONFormatter',
    'StructuredLogAdapter',
    # Correlation
    'CorrelationMiddleware',
    'CorrelationLogFilter',
    'get_correlation_id',
    'set_correlation_id',
    'get_user_id',
    'set_user_id',
    'get_job_id',
    'set_job_id',
    # Format
    'format_time',
    'format_time_range',
    'format_duration',
    'format_bpm',
    'format_percent',
    'format_confidence',
    # File discovery
    'AUDIO_EXTENSIONS',
    'find_audio_files',
    'find_audio_files_with_filter',
    'get_relative_path',
    # Warnings
    'suppress_audio_warnings',
    'suppress_all_warnings',
    'restore_warnings',
    'WarningContext',
    # Logging utils
    'truncate_for_display',
    'truncate_for_metrics',
    'capture_output',
    'setup_exception_handler',
]

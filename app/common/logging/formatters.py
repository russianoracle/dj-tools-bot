"""Structured JSON logging formatter with correlation ID support."""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces log entries in JSON format suitable for log aggregation systems.
    Includes correlation_id, structured_data, and exception info.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_path: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ):
        """
        Initialize JSON formatter.

        Args:
            include_timestamp: Include ISO timestamp
            include_level: Include log level
            include_logger: Include logger name
            include_path: Include file:line info
            extra_fields: Static fields to add to every log entry
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_path = include_path
        self.extra_fields = extra_fields or {}

    @staticmethod
    def _extract_component(logger_name: str) -> str:
        """
        Extract component name from logger hierarchy.

        Examples:
            app.modules.bot.handlers.analyze -> bot.handlers.analyze
            app.common.logging.logger -> common.logging
            app.modules.analysis.tasks.key_analysis -> analysis.tasks.key_analysis
            __main__ -> main
        """
        if logger_name == "__main__":
            return "main"

        parts = logger_name.split(".")
        # Remove "app" and "modules" prefixes
        if parts[0] == "app":
            parts = parts[1:]
        if parts and parts[0] == "modules":
            parts = parts[1:]

        return ".".join(parts) if parts else logger_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry: dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat()

        if self.include_level:
            log_entry["level"] = record.levelname

        # Component extraction (e.g., "bot.handlers.analyze" from "app.modules.bot.handlers.analyze")
        component = self._extract_component(record.name)
        log_entry["component"] = component

        if self.include_logger:
            log_entry["logger"] = record.name

        if self.include_path:
            log_entry["path"] = f"{record.pathname}:{record.lineno}"

        # Message
        message = record.getMessage()
        log_entry["message"] = message.strip() if message else ""

        # Correlation ID
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id

        # User ID (for bot context)
        if hasattr(record, "user_id") and record.user_id:
            log_entry["user_id"] = record.user_id

        # Job ID (for task tracking)
        if hasattr(record, "job_id") and record.job_id:
            log_entry["job_id"] = record.job_id

        # Structured data
        if hasattr(record, "structured_data") and record.structured_data:
            log_entry["data"] = record.structured_data

        # Exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Static extra fields
        log_entry.update(self.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class StructuredLogAdapter(logging.LoggerAdapter):
    """
    Log adapter that adds correlation_id and structured data to logs.

    Usage:
        logger = StructuredLogAdapter(logging.getLogger(__name__))
        logger.set_correlation_id("req-123")
        logger.info("Processing", data={"user_id": 42, "action": "analyze"})
    """

    def __init__(self, logger: logging.Logger, extra: dict | None = None):
        super().__init__(logger, extra or {})
        self._correlation_id: str | None = None
        self._user_id: int | None = None
        self._job_id: str | None = None

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracing."""
        self._correlation_id = correlation_id

    def set_user_id(self, user_id: int):
        """Set user ID for user context."""
        self._user_id = user_id

    def set_job_id(self, job_id: str):
        """Set job ID for task tracking."""
        self._job_id = job_id

    def clear_context(self):
        """Clear all context fields."""
        self._correlation_id = None
        self._user_id = None
        self._job_id = None

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Add context fields to log record."""
        extra = kwargs.get("extra", {})

        if self._correlation_id:
            extra["correlation_id"] = self._correlation_id
        if self._user_id:
            extra["user_id"] = self._user_id
        if self._job_id:
            extra["job_id"] = self._job_id

        # Support structured data via 'data' kwarg
        if "data" in kwargs:
            extra["structured_data"] = kwargs.pop("data")

        kwargs["extra"] = extra
        return msg, kwargs

    def _validate_message(self, msg: str) -> str:
        """
        Validate and sanitize log message.

        Args:
            msg: Log message to validate

        Returns:
            Validated message (empty string if no message)
        """
        if not msg or not msg.strip():
            # Return empty string - Fluent Bit filter will drop these records
            return ""
        return msg

    def info(self, msg: str, *args, data: dict | None = None, **kwargs):
        """Log info with optional structured data."""
        msg = self._validate_message(msg)
        if data:
            kwargs["data"] = data
        super().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, data: dict | None = None, **kwargs):
        """Log warning with optional structured data."""
        msg = self._validate_message(msg)
        if data:
            kwargs["data"] = data
        super().warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, data: dict | None = None, **kwargs):
        """Log error with optional structured data."""
        msg = self._validate_message(msg)
        if data:
            kwargs["data"] = data
        super().error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, data: dict | None = None, **kwargs):
        """Log debug with optional structured data."""
        msg = self._validate_message(msg)
        if data:
            kwargs["data"] = data
        super().debug(msg, *args, **kwargs)

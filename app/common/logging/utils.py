"""Logging utility functions."""

import sys
import logging
from io import StringIO
from contextlib import contextmanager
from typing import Optional


def truncate_for_display(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text for display in UI (Telegram messages, status updates).

    DO NOT use for logs - logs should contain full messages.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated

    Returns:
        Truncated text with suffix if needed

    Examples:
        >>> truncate_for_display("Short message", 100)
        "Short message"
        >>> truncate_for_display("A" * 150, 100)
        "AAAA...AAAA... (150 chars)"
    """
    if not text:
        return text

    if len(text) <= max_length:
        return text

    # Show beginning and indicate total length
    truncated = text[:max_length - len(suffix)]
    return f"{truncated}{suffix} ({len(text)} chars)"


def truncate_for_metrics(text: str, max_length: int = 500) -> str:
    """
    Truncate text for metrics storage (Prometheus, YC Monitoring).

    More generous than display truncation, still bounded for storage.

    Args:
        text: Text to truncate
        max_length: Maximum length (default 500)

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length] + f"... (truncated from {len(text)} chars)"


@contextmanager
def capture_output(logger: logging.Logger, level: int = logging.WARNING):
    """
    Capture stdout/stderr and redirect to logger.

    Useful for wrapping external processes (yt-dlp, librosa) that write to stdout/stderr.

    Usage:
        with capture_output(logger):
            subprocess.run(["yt-dlp", url])  # Output goes to logger

    Args:
        logger: Logger to send captured output to
        level: Log level for captured output
    """
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Log captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        if stdout_content.strip():
            for line in stdout_content.strip().split('\n'):
                if line.strip():
                    logger.log(level, f"[stdout] {line}", data={"source": "captured_stdout"})

        if stderr_content.strip():
            for line in stderr_content.strip().split('\n'):
                if line.strip():
                    logger.log(level, f"[stderr] {line}", data={"source": "captured_stderr"})


def setup_exception_handler(logger: logging.Logger):
    """
    Setup global exception handler to log uncaught exceptions.

    Ensures all exceptions are logged even if not explicitly caught.

    Args:
        logger: Logger to use for uncaught exceptions
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Log uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
            data={
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
            }
        )

    sys.excepthook = exception_handler

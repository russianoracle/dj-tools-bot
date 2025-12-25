"""
Custom error classes with structured logging and error propagation.

All errors include correlation context and structured data for observability.
"""

from typing import Optional, Dict, Any
from app.common.logging import get_logger
from app.common.logging.correlation import get_correlation_id, get_user_id, get_job_id

logger = get_logger(__name__)


class MoodClassifierError(Exception):
    """
    Base error class for all application errors.

    Automatically logs errors with correlation context when raised.
    """

    def __init__(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize error with structured context.

        Args:
            message: Human-readable error message
            data: Structured data for observability
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.data = data or {}
        self.cause = cause

        # Add correlation context
        self.correlation_id = get_correlation_id()
        self.user_id = get_user_id()
        self.job_id = get_job_id()

        # Log error with full context
        self._log_error()

    def _log_error(self):
        """Log error with structured data."""
        log_data = {
            "error_type": self.__class__.__name__,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            **self.data,
        }

        if self.cause:
            log_data["cause"] = str(self.cause)

        logger.error(self.message, data=log_data, exc_info=self.cause is not None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "cause": str(self.cause) if self.cause else None,
        }


# Audio processing errors
class AudioProcessingError(MoodClassifierError):
    """Error during audio processing (loading, STFT, etc.)."""
    pass


class AudioLoadError(AudioProcessingError):
    """Error loading audio file."""
    pass


class STFTError(AudioProcessingError):
    """Error computing STFT."""
    pass


# Analysis errors
class AnalysisError(MoodClassifierError):
    """Error during analysis pipeline."""
    pass


class TaskExecutionError(AnalysisError):
    """Error executing analysis task."""
    pass


class SegmentationError(AnalysisError):
    """Error during segmentation."""
    pass


class TransitionDetectionError(AnalysisError):
    """Error during transition detection."""
    pass


class DropDetectionError(AnalysisError):
    """Error during drop detection."""
    pass


class GenreAnalysisError(AnalysisError):
    """Error during genre analysis."""
    pass


# Cache errors
class CacheError(MoodClassifierError):
    """Error accessing cache."""
    pass


# Configuration errors
class ConfigurationError(MoodClassifierError):
    """Error in configuration."""
    pass


# External service errors
class ExternalServiceError(MoodClassifierError):
    """Error communicating with external service."""
    pass


class DownloadError(ExternalServiceError):
    """Error downloading audio from URL."""
    pass


class RedisError(ExternalServiceError):
    """Error communicating with Redis."""
    pass


# Feature extraction errors
class FeatureExtractionError(MoodClassifierError):
    """Error during feature extraction."""
    pass


# Metadata errors
class MetadataError(MoodClassifierError):
    """Error reading/writing metadata."""
    pass


# File system errors
class FileSystemError(MoodClassifierError):
    """Error accessing file system."""
    pass


# Validation errors
class ValidationError(MoodClassifierError):
    """Error validating input data."""
    pass

"""Warning suppression configuration.

This module centralizes warning suppression that was duplicated
across many scripts. Call suppress_audio_warnings() at script start.
"""

import os
import warnings
from typing import Optional


def suppress_audio_warnings(
    librosa: bool = True,
    tensorflow: bool = True,
    numpy: bool = True,
    pysoundfile: bool = True,
) -> None:
    """Suppress common warnings from audio/ML libraries.

    This should be called at the start of scripts to prevent
    noisy warnings from cluttering output.

    Args:
        librosa: Suppress librosa FutureWarnings
        tensorflow: Suppress TensorFlow/ABSL logging
        numpy: Suppress numpy deprecation warnings
        pysoundfile: Suppress PySoundFile fallback warnings
    """
    if librosa:
        warnings.filterwarnings(
            'ignore',
            category=FutureWarning,
            module='librosa',
        )
        warnings.filterwarnings(
            'ignore',
            message='.*librosa.*',
            category=FutureWarning,
        )

    if pysoundfile:
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            message='PySoundFile failed.*',
        )
        warnings.filterwarnings(
            'ignore',
            message='.*audioread.*',
        )

    if tensorflow:
        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

        # Suppress TF deprecation warnings
        warnings.filterwarnings(
            'ignore',
            category=DeprecationWarning,
            module='tensorflow',
        )

    if numpy:
        warnings.filterwarnings(
            'ignore',
            category=DeprecationWarning,
            module='numpy',
        )
        warnings.filterwarnings(
            'ignore',
            message='.*numpy.*',
            category=FutureWarning,
        )


def suppress_all_warnings() -> None:
    """Suppress all warnings. Use sparingly."""
    warnings.filterwarnings('ignore')


def restore_warnings() -> None:
    """Restore default warning behavior."""
    warnings.resetwarnings()


class WarningContext:
    """Context manager for temporary warning suppression.

    Example:
        with WarningContext(suppress_all=True):
            # Code that generates warnings
            result = noisy_function()
        # Warnings restored here
    """

    def __init__(
        self,
        suppress_all: bool = False,
        categories: Optional[list] = None,
    ):
        self.suppress_all = suppress_all
        self.categories = categories or []
        self._filters = None

    def __enter__(self):
        self._filters = warnings.filters.copy()
        if self.suppress_all:
            warnings.filterwarnings('ignore')
        else:
            for cat in self.categories:
                warnings.filterwarnings('ignore', category=cat)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters = self._filters
        return False

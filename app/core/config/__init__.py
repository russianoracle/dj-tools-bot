"""
Config - Application configuration.

- settings.py: Pydantic settings from environment
- cache.py: Cache client factory
"""

from .settings import Settings, CacheBackend, LogLevel, get_settings
from .cache import create_cache_client, create_set_cache
from .mixing_styles import (
    MixingStyle,
    TransitionConfig,
    DropDetectionConfig,
    SegmentationConfig,
    SetAnalysisConfig,
    DEFAULT_TRANSITION,
    DEFAULT_SEGMENTATION,
    DEFAULT_SET_ANALYSIS,
    SMOOTH_MIXING,
    STANDARD_MIXING,
    HARD_MIXING,
)

__all__ = [
    # Settings
    "Settings",
    "CacheBackend",
    "LogLevel",
    "get_settings",
    # Cache
    "create_cache_client",
    "create_set_cache",
    # Mixing Styles
    "MixingStyle",
    "TransitionConfig",
    "DropDetectionConfig",
    "SegmentationConfig",
    "SetAnalysisConfig",
    "DEFAULT_TRANSITION",
    "DEFAULT_SEGMENTATION",
    "DEFAULT_SET_ANALYSIS",
    "SMOOTH_MIXING",
    "STANDARD_MIXING",
    "HARD_MIXING",
]

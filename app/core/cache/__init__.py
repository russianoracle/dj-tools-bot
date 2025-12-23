"""
Unified Cache System for DJ Analysis.

Provides domain-aware caching with abstractions for:
- Set (DJ set analysis)
- Track (individual track analysis)
- Features (raw audio features)
- DJProfile (aggregated DJ style)

Architecture:
    CacheRepository (public API, Unit of Work)
        └── CacheManager (internal implementation, NOT exported)

    CacheRepository implements ICacheStatusProvider for read-only queries.

    Services should use ICacheStatusProvider for UI status display.
    Pipelines should use CacheRepository for full read/write access.
"""

from .interfaces import (
    ICacheStatusProvider,
    CacheStats,
)
from .models import (
    CachedSetAnalysis,
    CachedTrackAnalysis,
    CachedDJProfile,
    CachedFeatures,
    CachedDrop,
    CachedTransition,
    CachedSegment,
)
from .repository import CacheRepository

__all__ = [
    # Interfaces
    'ICacheStatusProvider',
    'CacheStats',
    # Repository
    'CacheRepository',
    # Models
    'CachedSetAnalysis',
    'CachedTrackAnalysis',
    'CachedDJProfile',
    'CachedFeatures',
    'CachedDrop',
    'CachedTransition',
    'CachedSegment',
]

"""
Service Layer - Unified API for all audio analysis operations.

This layer provides a clean interface between CLI/UI and the underlying
Pipelines/Tasks architecture. All scripts and interactive menus should
use these services instead of calling Tasks directly.

Architecture:
    CLI/UI → Services → Pipelines → Tasks → Primitives

    Services use ICacheStatusProvider for READ-ONLY cache status queries.
    Pipelines handle cache-aside pattern internally (read + write).

Services:
    - AnalysisService: Track and DJ set analysis
    - ProfilingService: DJ style profiling and metrics
    - GenerationService: Set generation from profiles (future)

Factory Functions:
    - create_analysis_service: Create AnalysisService with dependencies
    - create_profiling_service: Create ProfilingService with dependencies

Usage:
    from src.services import create_analysis_service, create_profiling_service

    # Create services (recommended)
    analysis = create_analysis_service()
    profiling = create_profiling_service()

    # Check cache status (for UI)
    if analysis.is_set_cached("set.mp3"):
        print("✓ Cached")

    # Analyze a DJ set (Pipeline handles caching internally)
    result = analysis.analyze_set("path/to/set.mp3")

    # Profile a DJ
    profile = profiling.profile_batch("Josh Baker", "path/to/sets/")
"""

from ..core.cache import CacheRepository, ICacheStatusProvider

from .analysis import AnalysisService, BatchAnalysisResult
from .profiling import ProfilingService


def create_analysis_service(cache_dir: str = "cache", sr: int = 22050) -> AnalysisService:
    """
    Factory function to create AnalysisService with dependencies.

    Args:
        cache_dir: Cache directory path
        sr: Sample rate for analysis

    Returns:
        Configured AnalysisService
    """
    cache_status: ICacheStatusProvider = CacheRepository(cache_dir)
    return AnalysisService(cache_status=cache_status, cache_dir=cache_dir, sr=sr)


def create_profiling_service(cache_dir: str = "cache", sr: int = 22050) -> ProfilingService:
    """
    Factory function to create ProfilingService with dependencies.

    Args:
        cache_dir: Cache directory path
        sr: Sample rate for analysis

    Returns:
        Configured ProfilingService
    """
    cache_status: ICacheStatusProvider = CacheRepository(cache_dir)
    return ProfilingService(cache_status=cache_status, cache_dir=cache_dir, sr=sr)


__all__ = [
    # Services
    'AnalysisService',
    'ProfilingService',
    'BatchAnalysisResult',
    # Factory functions (recommended)
    'create_analysis_service',
    'create_profiling_service',
]

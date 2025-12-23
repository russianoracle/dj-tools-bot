"""
Cache Interfaces - Abstract contracts for cache operations.

Separates read-only status queries from write operations:
- ICacheStatusProvider: Read-only interface for UI status display
- Full cache operations remain in CacheRepository

This allows Services to query cache status without write access,
while Pipelines retain full read/write control.

Architecture:
    Service → ICacheStatusProvider (read-only)
    Pipeline → CacheRepository (full access)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CacheStats:
    """Cache statistics for UI display."""
    set_count: int
    track_count: int
    profile_count: int
    total_size_mb: float

    def to_dict(self) -> Dict:
        return {
            'set_count': self.set_count,
            'track_count': self.track_count,
            'profile_count': self.profile_count,
            'total_size_mb': self.total_size_mb,
        }


class ICacheStatusProvider(ABC):
    """
    Read-only interface for querying cache status.

    Used by Services ONLY for displaying status in UI.
    Does NOT allow writing to cache - that's Pipeline's responsibility.

    This separation ensures:
    - Services don't accidentally modify cache
    - Pipelines control cache-aside pattern internally
    - UI can display cache status without side effects

    Usage:
        class AnalysisService:
            def __init__(self, cache_status: ICacheStatusProvider):
                self._cache_status = cache_status

            def is_set_cached(self, path: str) -> bool:
                return self._cache_status.exists_set(path)
    """

    @abstractmethod
    def exists_set(self, path: str) -> bool:
        """Check if set analysis is cached."""
        pass

    @abstractmethod
    def exists_track(self, path: str) -> bool:
        """Check if track analysis is cached."""
        pass

    @abstractmethod
    def exists_profile(self, dj_name: str) -> bool:
        """Check if DJ profile is cached."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get overall cache statistics."""
        pass

    @abstractmethod
    def get_folder_status(
        self,
        folder: str,
        mode: str = 'set',
        recursive: bool = True
    ) -> Dict[str, bool]:
        """
        Get cache status for all audio files in a folder.

        Args:
            folder: Path to folder
            mode: 'set' or 'track'
            recursive: Scan subfolders

        Returns:
            Dict mapping absolute file path to cache status (True = cached)
        """
        pass

    @abstractmethod
    def list_cached_sets(self) -> List[str]:
        """Get paths of all cached sets."""
        pass

    @abstractmethod
    def list_profiles(self) -> List[str]:
        """Get names of all cached DJ profiles."""
        pass

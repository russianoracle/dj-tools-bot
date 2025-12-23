"""
Cache Protocol - Interface for cache implementations.

Implementations:
- SQLiteCache (app.core.connectors.sqlite_cache)
- RedisCache (app.core.connectors.redis_cache)
- InMemoryCache (app.core.connectors.inmemory_cache)
"""

from typing import Protocol, Optional, Any, Dict, List, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations (DI interface)."""

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL in seconds."""
        ...

    def delete(self, key: str) -> None:
        """Delete key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def clear(self) -> None:
        """Clear all cache."""
        ...


@runtime_checkable
class SetCacheProtocol(Protocol):
    """Extended protocol for DJ set analysis cache."""

    def get_set(self, file_path: str) -> Optional[Any]:
        """Get cached set analysis result."""
        ...

    def save_set(self, result: Any) -> None:
        """Save set analysis result."""
        ...

    def is_set_cached(self, file_path: str) -> bool:
        """Check if set is cached."""
        ...

    def invalidate_set(self, file_path: str) -> None:
        """Invalidate cached set."""
        ...

    def get_features(self, file_hash: str) -> Optional[Dict]:
        """Get cached features for training."""
        ...

    def save_features(self, file_hash: str, features: Dict) -> None:
        """Save features for training."""
        ...

    def compute_file_hash(self, file_path: str) -> str:
        """Compute file hash for cache key."""
        ...


@runtime_checkable
class DJProfileCacheProtocol(Protocol):
    """Protocol for DJ profile caching."""

    def get_profile(self, dj_name: str) -> Optional[Any]:
        """Get cached DJ profile."""
        ...

    def save_profile(self, profile: Any) -> None:
        """Save DJ profile."""
        ...

    def list_profiles(self) -> List[str]:
        """List all cached DJ names."""
        ...

    def invalidate_profile(self, dj_name: str) -> None:
        """Invalidate DJ profile."""
        ...

"""
InMemoryCache - In-memory cache implementation for unit tests.

Simple dict-based cache without persistence.
"""

from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with optional expiration."""
    value: Any
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class InMemoryCache:
    """
    In-memory cache implementation.

    Implements CacheProtocol for unit testing.
    No persistence - data lost on restart.
    """

    def __init__(self):
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL in seconds."""
        expires_at = None
        if ttl is not None:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        """Delete key."""
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        """Check if key exists and not expired."""
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            del self._store[key]
            return False
        return True

    def clear(self) -> None:
        """Clear all cache."""
        self._store.clear()

    def keys(self) -> list:
        """Get all non-expired keys (vectorized cleanup)."""
        import numpy as np
        # Vectorized expired detection
        keys_arr = np.array(list(self._store.keys()))
        if len(keys_arr) == 0:
            return []
        expired_mask = np.array([self._store[k].is_expired() for k in keys_arr])
        # Delete expired entries
        expired_keys = keys_arr[expired_mask]
        if len(expired_keys) > 0:
            # Vectorized deletion via dict comprehension
            self._store = {k: v for k, v in self._store.items() if k not in expired_keys}
        return keys_arr[~expired_mask].tolist()

    def size(self) -> int:
        """Get number of entries."""
        return len(self.keys())

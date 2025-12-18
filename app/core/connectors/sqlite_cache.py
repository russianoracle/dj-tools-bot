"""
SQLiteCache - SQLite-based cache implementation.

Uses CacheRepository from app.core.cache.
Implements CacheProtocol, SetCacheProtocol, and DJProfileCacheProtocol.
"""

from typing import Optional, Any, Dict, List

from app.common.logging import get_logger

logger = get_logger(__name__)


class SQLiteCache:
    """
    SQLite-based cache implementation.

    Implements CacheProtocol and SetCacheProtocol.
    Uses existing CacheRepository internally for full functionality.
    """

    def __init__(self, db_path: str = "cache/predictions.db"):
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Import CacheRepository from app
        try:
            from app.core.cache import CacheRepository
            import os
            cache_dir = os.path.dirname(db_path) or "cache"
            self._repo = CacheRepository(cache_dir)
            self._direct_mode = False
            logger.info(f"SQLiteCache using CacheRepository: {cache_dir}")
        except ImportError:
            # Fallback to simple dict cache if src not available
            self._repo = None
            self._direct_mode = True
            self._store: Dict[str, Any] = {}
            logger.warning("CacheRepository not available, using in-memory fallback")

    # ============== CacheProtocol Implementation ==============

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if self._direct_mode:
            return self._store.get(key)
        # Use features dict for generic key-value
        return self._repo.get_features_dict(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value (TTL ignored for SQLite)."""
        if self._direct_mode:
            self._store[key] = value
            return
        if isinstance(value, dict):
            self._repo.save_features_dict(key, value)

    def delete(self, key: str) -> None:
        """Delete key."""
        if self._direct_mode:
            self._store.pop(key, None)
            return
        # No direct delete in CacheRepository, invalidate by type
        pass

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clear all cache."""
        if self._direct_mode:
            self._store.clear()
            return
        self._repo.clear_all()

    # ============== SetCacheProtocol Implementation ==============

    def get_set(self, file_path: str) -> Optional[Any]:
        """Get cached set analysis result."""
        if self._direct_mode:
            return self._store.get(f"set:{file_path}")
        return self._repo.get_set(file_path)

    def save_set(self, result: Any) -> None:
        """Save set analysis result."""
        if self._direct_mode:
            if hasattr(result, 'file_path'):
                self._store[f"set:{result.file_path}"] = result
            return
        self._repo.save_set(result)

    def is_set_cached(self, file_path: str) -> bool:
        """Check if set is cached."""
        if self._direct_mode:
            return f"set:{file_path}" in self._store
        return self._repo.exists_set(file_path)

    def invalidate_set(self, file_path: str) -> None:
        """Invalidate cached set."""
        if self._direct_mode:
            self._store.pop(f"set:{file_path}", None)
            return
        self._repo.invalidate_set(file_path)

    def get_features(self, file_hash: str) -> Optional[Dict]:
        """Get cached features for training."""
        if self._direct_mode:
            return self._store.get(f"features:{file_hash}")
        return self._repo.get_features_dict(file_hash)

    def save_features(self, file_hash: str, features: Dict) -> None:
        """Save features for training."""
        if self._direct_mode:
            self._store[f"features:{file_hash}"] = features
            return
        self._repo.save_features_dict(file_hash, features)

    def compute_file_hash(self, file_path: str) -> str:
        """Compute file hash for cache key."""
        if self._direct_mode:
            import hashlib
            import os
            stat = os.stat(file_path)
            return hashlib.md5(f"{file_path}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()
        return self._repo.compute_file_hash(file_path)

    # ============== DJProfileCacheProtocol Implementation ==============

    def get_profile(self, dj_name: str) -> Optional[Any]:
        """Get cached DJ profile."""
        if self._direct_mode:
            return self._store.get(f"profile:{dj_name.lower()}")
        return self._repo.get_dj_profile(dj_name)

    def save_profile(self, profile: Any) -> None:
        """Save DJ profile."""
        if self._direct_mode:
            if hasattr(profile, 'dj_name'):
                self._store[f"profile:{profile.dj_name.lower()}"] = profile
            return
        self._repo.save_dj_profile(profile)

    def list_profiles(self) -> List[str]:
        """List all cached DJ names (vectorized)."""
        if self._direct_mode:
            import numpy as np
            keys = np.array(list(self._store.keys()))
            if len(keys) == 0:
                return []
            profile_mask = np.char.startswith(keys, "profile:")
            profile_keys = keys[profile_mask]
            return np.char.replace(profile_keys, "profile:", "").tolist()
        return self._repo.get_all_dj_profiles()

    def invalidate_profile(self, dj_name: str) -> None:
        """Invalidate DJ profile."""
        if self._direct_mode:
            self._store.pop(f"profile:{dj_name.lower()}", None)
            return
        self._repo.invalidate_by_dj(dj_name)

    # ============== Additional Methods ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (vectorized)."""
        if self._direct_mode:
            import numpy as np
            keys = np.array(list(self._store.keys()))
            if len(keys) == 0:
                return {'entries': 0, 'sets': 0, 'profiles': 0}
            sets_count = int(np.sum(np.char.startswith(keys, 'set:')))
            profiles_count = int(np.sum(np.char.startswith(keys, 'profile:')))
            return {
                'entries': len(keys),
                'sets': sets_count,
                'profiles': profiles_count,
            }
        return self._repo.get_stats()

    @property
    def underlying_repo(self):
        """Get underlying CacheRepository (for advanced operations)."""
        return self._repo

"""
Cache Factory - Create cache client based on configuration.

Uses factory pattern for dependency injection.
"""

from typing import Optional
from ..interfaces import CacheProtocol, SetCacheProtocol
from .settings import CacheBackend, get_settings


def create_cache_client(
    backend: Optional[CacheBackend] = None,
    **kwargs
) -> CacheProtocol:
    """
    Factory for cache clients.

    Args:
        backend: Cache backend (default from settings)
        **kwargs: Backend-specific arguments

    Returns:
        CacheProtocol implementation

    Example:
        cache = create_cache_client()  # Uses settings
        cache = create_cache_client(CacheBackend.REDIS, url="redis://...")
    """
    settings = get_settings()
    backend = backend or settings.cache_backend

    if backend == CacheBackend.SQLITE:
        from ..connectors.sqlite_cache import SQLiteCache
        db_path = kwargs.get('db_path', settings.db_path)
        return SQLiteCache(db_path=db_path)

    elif backend == CacheBackend.REDIS:
        from ..connectors.redis_cache import RedisCache
        url = kwargs.get('url', settings.redis_url)
        if not url:
            raise ValueError("Redis URL required for redis backend")
        return RedisCache(url=url)

    elif backend == CacheBackend.MEMORY:
        from ..connectors.inmemory_cache import InMemoryCache
        return InMemoryCache()

    raise ValueError(f"Unknown cache backend: {backend}")


def create_set_cache() -> SetCacheProtocol:
    """
    Create cache client with SetCacheProtocol interface.

    Returns:
        SetCacheProtocol implementation (SQLiteCache)
    """
    settings = get_settings()

    # Only SQLite supports full SetCacheProtocol
    if settings.cache_backend != CacheBackend.SQLITE:
        # Fall back to SQLite for set caching
        from ..connectors.sqlite_cache import SQLiteCache
        return SQLiteCache(db_path=settings.db_path)

    return create_cache_client()

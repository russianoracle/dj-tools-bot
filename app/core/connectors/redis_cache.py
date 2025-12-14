"""
RedisCache - Redis-based cache implementation for production.

Requires redis package: pip install redis
"""

from typing import Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based cache implementation.

    Implements CacheProtocol for production use.
    Requires Redis server.
    """

    def __init__(self, url: str, prefix: str = "djtools:"):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL (redis://host:port/db)
            prefix: Key prefix for namespacing
        """
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required: pip install redis")

        self.client = redis.from_url(url, decode_responses=True)
        self.prefix = prefix
        logger.info(f"Redis cache initialized: {url}")

    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            value = self.client.get(self._key(key))
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL in seconds."""
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                self.client.setex(self._key(key), ttl, serialized)
            else:
                self.client.set(self._key(key), serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> None:
        """Delete key."""
        try:
            self.client.delete(self._key(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.client.exists(self._key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache with prefix."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def keys(self) -> list:
        """Get all keys with prefix."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            return [k.replace(self.prefix, "") for k in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []

    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self.client.ping()
        except Exception:
            return False

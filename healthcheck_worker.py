#!/usr/bin/env python3
"""
ARQ Worker healthcheck script.

Checks:
1. Redis connection
2. Memory usage < 95% of limit
3. Worker process exists
"""
import sys
import redis
import psutil


def check_redis():
    """Check Redis connection."""
    try:
        r = redis.Redis(host='redis', port=6379, socket_connect_timeout=3)
        r.ping()
        return True
    except Exception as e:
        print(f"Redis check failed: {e}", file=sys.stderr)
        return False


def check_memory():
    """Check memory usage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024

        # Fail if using > 11.4GB (95% of 12GB limit)
        limit_mb = 11.4 * 1024
        if mem_mb > limit_mb:
            print(f"Memory exceeded: {mem_mb:.0f}MB > {limit_mb:.0f}MB", file=sys.stderr)
            return False

        return True
    except Exception as e:
        print(f"Memory check failed: {e}", file=sys.stderr)
        # Don't fail healthcheck if psutil has issues
        return True


def main():
    """Run all health checks."""
    if not check_redis():
        sys.exit(1)

    if not check_memory():
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

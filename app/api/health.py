"""
Health check and metrics endpoints for observability.

Provides comprehensive system health monitoring:
- /health - Basic liveness check
- /health/ready - Readiness check (includes dependencies)
- /metrics - Prometheus-compatible metrics
"""

import os
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.common.logging import get_logger
from app.core.cache import CacheRepository

logger = get_logger(__name__)
router = APIRouter()

# Track startup time for uptime calculation
_start_time = time.time()


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    details: Optional[Dict[str, Any]] = None


class ReadinessStatus(BaseModel):
    """Readiness check response model."""
    ready: bool
    timestamp: str
    checks: Dict[str, Dict[str, Any]]


def get_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)
        r.ping()

        info = r.info("stats")
        return {
            "status": "healthy",
            "connected_clients": info.get("connected_clients", 0),
            "total_commands": info.get("total_commands_processed", 0),
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def get_disk_health() -> Dict[str, Any]:
    """Check disk space availability."""
    try:
        disk = psutil.disk_usage("/")
        downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")

        downloads_size = 0
        if os.path.exists(downloads_dir):
            downloads_size = sum(
                os.path.getsize(os.path.join(downloads_dir, f))
                for f in os.listdir(downloads_dir)
                if os.path.isfile(os.path.join(downloads_dir, f))
            )

        used_percent = (disk.used / disk.total) * 100
        status = "healthy" if used_percent < 90 else "degraded" if used_percent < 95 else "critical"

        return {
            "status": status,
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_percent": round(used_percent, 1),
            "downloads_mb": round(downloads_size / (1024**2), 2),
        }
    except Exception as e:
        logger.error(f"Disk health check failed: {e}")
        return {
            "status": "unknown",
            "error": str(e),
        }


def get_memory_health() -> Dict[str, Any]:
    """Check memory usage."""
    try:
        mem = psutil.virtual_memory()
        status = "healthy" if mem.percent < 85 else "degraded" if mem.percent < 95 else "critical"

        return {
            "status": status,
            "total_mb": round(mem.total / (1024**2), 2),
            "available_mb": round(mem.available / (1024**2), 2),
            "used_percent": round(mem.percent, 1),
        }
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        return {
            "status": "unknown",
            "error": str(e),
        }


def get_arq_health() -> Dict[str, Any]:
    """Check ARQ worker status."""
    try:
        from app.services.arq_worker import get_redis_pool
        import asyncio

        # Try to get pool info
        pool = asyncio.run(get_redis_pool())
        return {
            "status": "healthy",
            "queue": "arq:queue",
        }
    except Exception as e:
        logger.error(f"ARQ health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def get_cache_health() -> Dict[str, Any]:
    """Check cache repository health."""
    try:
        repo = CacheRepository.get_instance()
        stats = repo.get_cache_stats()

        # Check if cache directories are accessible
        stft_dir_exists = repo.stft_dir.exists()
        features_dir_exists = repo.features_dir.exists()

        return {
            "status": "healthy",
            "sets_cached": stats.set_count,
            "tracks_cached": stats.track_count,
            "profiles_cached": stats.profile_count,
            "total_size_mb": stats.total_size_mb,
            "stft_dir_accessible": stft_dir_exists,
            "features_dir_accessible": features_dir_exists,
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint (liveness probe).

    Returns 200 if service is running.
    """
    uptime = time.time() - _start_time

    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=round(uptime, 2),
        details={
            "service": "mood-classifier-bot",
            "environment": os.getenv("ENVIRONMENT", "production"),
        }
    )


@router.get("/health/ready", response_model=ReadinessStatus)
async def readiness_check(response: Response):
    """
    Readiness check endpoint (readiness probe).

    Checks all dependencies (Redis, disk, memory, ARQ, cache repository).
    Returns 200 if ready, 503 if not ready.
    """
    checks = {
        "redis": get_redis_health(),
        "disk": get_disk_health(),
        "memory": get_memory_health(),
        "arq": get_arq_health(),
        "cache": get_cache_health(),
    }

    # Determine overall readiness
    all_healthy = all(
        check.get("status") in ("healthy", "degraded")
        for check in checks.values()
    )

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessStatus(
        ready=all_healthy,
        timestamp=datetime.utcnow().isoformat(),
        checks=checks,
    )


@router.get("/metrics", response_class=Response)
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Exposes system metrics and cache statistics in Prometheus exposition format.
    """
    uptime = time.time() - _start_time

    # Get system metrics
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Get ARQ metrics
    arq_queued = 0
    arq_active = 0
    try:
        import redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=2)

        # ARQ uses Redis lists for queue
        arq_queued = r.llen("arq:queue") or 0
    except Exception:
        pass

    # Get cache metrics
    cache_sets = 0
    cache_tracks = 0
    cache_profiles = 0
    cache_size_mb = 0.0
    try:
        repo = CacheRepository.get_instance()
        stats = repo.get_cache_stats()
        cache_sets = stats.set_count
        cache_tracks = stats.track_count
        cache_profiles = stats.profile_count
        cache_size_mb = stats.total_size_mb
    except Exception as e:
        logger.warning(f"Failed to get cache metrics: {e}")

    # Get downloads directory metrics
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
    downloads_count = 0
    downloads_size = 0
    if os.path.exists(downloads_dir):
        files = [f for f in os.listdir(downloads_dir) if os.path.isfile(os.path.join(downloads_dir, f))]
        downloads_count = len(files)
        downloads_size = sum(
            os.path.getsize(os.path.join(downloads_dir, f))
            for f in files
        )

    # Build Prometheus format
    metrics_lines = [
        "# HELP mood_classifier_uptime_seconds Time since service started",
        "# TYPE mood_classifier_uptime_seconds gauge",
        f"mood_classifier_uptime_seconds {uptime:.2f}",
        "",
        "# HELP mood_classifier_memory_used_percent Memory usage percentage",
        "# TYPE mood_classifier_memory_used_percent gauge",
        f"mood_classifier_memory_used_percent {mem.percent:.1f}",
        "",
        "# HELP mood_classifier_disk_used_percent Disk usage percentage",
        "# TYPE mood_classifier_disk_used_percent gauge",
        f"mood_classifier_disk_used_percent {(disk.used / disk.total * 100):.1f}",
        "",
        "# HELP mood_classifier_cpu_used_percent CPU usage percentage",
        "# TYPE mood_classifier_cpu_used_percent gauge",
        f"mood_classifier_cpu_used_percent {cpu_percent:.1f}",
        "",
        "# HELP mood_classifier_arq_queued_jobs Number of jobs in ARQ queue",
        "# TYPE mood_classifier_arq_queued_jobs gauge",
        f"mood_classifier_arq_queued_jobs {arq_queued}",
        "",
        "# HELP mood_classifier_arq_active_jobs Number of active ARQ jobs",
        "# TYPE mood_classifier_arq_active_jobs gauge",
        f"mood_classifier_arq_active_jobs {arq_active}",
        "",
        "# HELP mood_classifier_cache_sets_count Number of cached set analyses",
        "# TYPE mood_classifier_cache_sets_count gauge",
        f"mood_classifier_cache_sets_count {cache_sets}",
        "",
        "# HELP mood_classifier_cache_tracks_count Number of cached track analyses",
        "# TYPE mood_classifier_cache_tracks_count gauge",
        f"mood_classifier_cache_tracks_count {cache_tracks}",
        "",
        "# HELP mood_classifier_cache_profiles_count Number of cached DJ profiles",
        "# TYPE mood_classifier_cache_profiles_count gauge",
        f"mood_classifier_cache_profiles_count {cache_profiles}",
        "",
        "# HELP mood_classifier_cache_size_mb Total size of cache in MB",
        "# TYPE mood_classifier_cache_size_mb gauge",
        f"mood_classifier_cache_size_mb {cache_size_mb:.2f}",
        "",
        "# HELP mood_classifier_downloads_count Number of files in downloads directory",
        "# TYPE mood_classifier_downloads_count gauge",
        f"mood_classifier_downloads_count {downloads_count}",
        "",
        "# HELP mood_classifier_downloads_size_bytes Total size of downloads directory",
        "# TYPE mood_classifier_downloads_size_bytes gauge",
        f"mood_classifier_downloads_size_bytes {downloads_size}",
        "",
    ]

    return Response(
        content="\n".join(metrics_lines),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

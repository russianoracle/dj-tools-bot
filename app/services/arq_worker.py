"""
ARQ Worker - Async task queue for DJ set analysis.

ARQ is asyncio-native and compatible with Redis 8.x/Valkey.
Reports progress every 30% with estimated time remaining.
"""

import os
import time
import random
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List

from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

logger = logging.getLogger(__name__)


# ============================================================================
# Proxy Rotation for geo-blocked services (SoundCloud, YouTube)
# ============================================================================

# Free proxy list - rotated on failure
# Format: protocol://host:port or protocol://user:pass@host:port
FREE_PROXY_SOURCES = [
    # Direct (no proxy) - try first
    None,
    # Public SOCKS5 proxies (often unreliable, but free)
    # These will be replaced with working ones at runtime
]

# Cache of working proxies
_working_proxies: List[Optional[str]] = [None]  # Start with direct
_failed_proxies: set = set()
_proxy_index = 0
_last_proxy_fetch = 0.0


async def fetch_free_proxies() -> List[str]:
    """
    Fetch free proxy list from public sources.

    Returns list of proxy URLs in format: socks5://host:port or http://host:port
    """
    import aiohttp

    proxies = []

    # Try to fetch from free proxy APIs
    sources = [
        # Free proxy list API (SOCKS5)
        ("https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=socks5&timeout=5000&country=all", "socks5"),
        # HTTP proxies as fallback
        ("https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=http&timeout=5000&country=all", "http"),
    ]

    for url, protocol in sources:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.strip().split("\n"):
                            line = line.strip()
                            if ":" in line and len(line) < 50:
                                proxies.append(f"{protocol}://{line}")
                        logger.info(f"Fetched {len(proxies)} proxies from {url[:40]}...")
                        if len(proxies) >= 10:
                            break
        except Exception as e:
            logger.debug(f"Failed to fetch proxies from {url[:40]}: {e}")

    return proxies[:20]  # Limit to 20 proxies


async def refresh_proxy_list() -> None:
    """Refresh proxy list if stale (older than 1 hour)."""
    global _working_proxies, _last_proxy_fetch

    # Refresh every hour
    if time.time() - _last_proxy_fetch < 3600:
        return

    try:
        new_proxies = await fetch_free_proxies()
        if new_proxies:
            # Keep direct (None) as first option, add new proxies
            _working_proxies = [None] + new_proxies
            _failed_proxies.clear()
            _last_proxy_fetch = time.time()
            logger.info(f"Refreshed proxy list: {len(new_proxies)} proxies available")
    except Exception as e:
        logger.warning(f"Failed to refresh proxy list: {e}")


def get_next_proxy() -> Optional[str]:
    """Get next proxy from rotation, skipping failed ones."""
    global _proxy_index

    # First try custom proxy from env
    custom_proxy = os.getenv("YTDLP_PROXY")
    if custom_proxy and custom_proxy not in _failed_proxies:
        return custom_proxy

    # Rotate through available proxies
    attempts = 0
    while attempts < len(_working_proxies):
        proxy = _working_proxies[_proxy_index % len(_working_proxies)]
        _proxy_index += 1

        if proxy not in _failed_proxies:
            return proxy
        attempts += 1

    # All proxies failed, reset and try direct
    logger.warning("All proxies failed, resetting to direct connection")
    _failed_proxies.clear()
    return None


def mark_proxy_failed(proxy: Optional[str]) -> None:
    """Mark a proxy as failed."""
    if proxy:
        _failed_proxies.add(proxy)
        logger.info(f"Marked proxy as failed: {proxy[:30]}...")


def download_with_failover(url: str, output_template: str, max_retries: int = 3) -> str:
    """
    Download with automatic proxy failover.

    Tries direct connection first, then rotates through proxies on failure.

    Returns:
        Path to downloaded file

    Raises:
        Exception if all attempts fail
    """
    last_error = None

    for attempt in range(max_retries):
        proxy = get_next_proxy()
        proxy_desc = proxy[:30] + "..." if proxy else "direct"

        logger.info(f"Download attempt {attempt + 1}/{max_retries} via {proxy_desc}")

        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", output_template,
            "--max-filesize", "500M",
            "--no-warnings",
            "--socket-timeout", "30",
        ]

        if proxy:
            cmd.extend(["--proxy", proxy])

        cmd.append(url)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 min timeout per attempt
            )

            if result.returncode == 0:
                logger.info(f"Download successful via {proxy_desc}")
                return output_template.replace(".%(ext)s", "")

            # Check for geo-block or rate limit
            stderr = result.stderr.lower()
            if "403" in stderr or "forbidden" in stderr or "geo" in stderr:
                logger.warning(f"Geo-blocked via {proxy_desc}, trying next proxy")
                mark_proxy_failed(proxy)
                continue

            # Other error
            last_error = result.stderr[:200]

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout via {proxy_desc}")
            mark_proxy_failed(proxy)
            last_error = "Download timeout"
            continue

        except Exception as e:
            last_error = str(e)
            mark_proxy_failed(proxy)

    raise Exception(f"Download failed after {max_retries} attempts: {last_error}")

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def get_redis_settings() -> RedisSettings:
    """Get Redis connection settings."""
    return RedisSettings(
        host=REDIS_HOST,
        port=REDIS_PORT,
    )


# In-memory job results (for standalone mode, use Redis in production)
_job_results: Dict[str, Dict[str, Any]] = {}

# Job timing for ETA calculation
_job_start_times: Dict[str, float] = {}


def estimate_remaining_time(job_id: str, current_progress: int) -> str:
    """Estimate remaining time based on current progress."""
    if job_id not in _job_start_times or current_progress <= 0:
        return "calculating..."

    elapsed = time.time() - _job_start_times[job_id]
    if current_progress >= 100:
        return "done"

    # Linear estimate
    total_estimated = elapsed / (current_progress / 100)
    remaining = total_estimated - elapsed

    if remaining < 60:
        return f"~{int(remaining)}s"
    elif remaining < 3600:
        return f"~{int(remaining / 60)}m"
    else:
        return f"~{int(remaining / 3600)}h"


def update_job_progress(job_id: str, progress: int, status: str):
    """Update job progress with ETA."""
    eta = estimate_remaining_time(job_id, progress)
    _job_results[job_id] = {
        "state": "PROGRESS",
        "progress": progress,
        "status": status,
        "eta": eta,
        "updated_at": datetime.now().isoformat(),
    }


async def analyze_set_task(ctx: dict, file_path: str, user_id: int) -> Dict[str, Any]:
    """
    Analyze DJ set in background.

    Reports progress at 30%, 60%, 90% with ETA.

    Args:
        ctx: ARQ context with redis connection
        file_path: Path to audio file
        user_id: Telegram user ID

    Returns:
        Analysis result dict
    """
    job_id = ctx.get("job_id", "unknown")
    logger.info(f"[{job_id}] Starting set analysis: {file_path}")

    # Start timing
    _job_start_times[job_id] = time.time()

    try:
        # 10% - Starting
        update_job_progress(job_id, 10, "🎵 Loading audio...")

        # Import analysis pipeline
        from app.modules.analysis.pipelines.set_analysis import SetAnalysisPipeline

        # Progress callback - reports at ~30%, 60%, 90%
        def on_stage_complete(stage_name: str, context):
            # Map stages to progress milestones (30%, 60%, 90%)
            progress_map = {
                "LoadAudioStage": 15,
                "ComputeSTFTStage": 30,         # 30% milestone
                "DetectTransitionsStage": 45,
                "SegmentTracksStage": 60,       # 60% milestone
                "DetectAllDropsStage": 75,
                "BuildTimelineStage": 90,       # 90% milestone
            }
            progress = progress_map.get(stage_name, 50)

            # Human-readable stage names
            stage_labels = {
                "LoadAudioStage": "🎵 Audio loaded",
                "ComputeSTFTStage": "📊 30% — Spectral analysis...",
                "DetectTransitionsStage": "🔀 Detecting transitions...",
                "SegmentTracksStage": "🎯 60% — Segmenting tracks...",
                "DetectAllDropsStage": "💥 Detecting drops...",
                "BuildTimelineStage": "📋 90% — Building timeline...",
            }
            status = stage_labels.get(stage_name, f"Processing {stage_name}...")

            update_job_progress(job_id, progress, status)
            logger.info(f"[{job_id}] {progress}% - {stage_name}")

        # Run analysis
        pipeline = SetAnalysisPipeline(
            sr=22050,
            analyze_genres=False,
            verbose=False,
        )
        pipeline.on_stage_complete = on_stage_complete

        result = pipeline.analyze(file_path)
        result_dict = result.to_dict() if hasattr(result, "to_dict") else {}

        elapsed = time.time() - _job_start_times.get(job_id, time.time())
        logger.info(f"[{job_id}] Analysis completed in {elapsed:.1f}s: {result.n_segments} segments, {result.total_drops} drops")

        # Cleanup file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{job_id}] Cleaned up: {file_path}")

        # Store result
        _job_results[job_id] = {
            "state": "SUCCESS",
            "progress": 100,
            "status": f"✅ Completed in {int(elapsed)}s",
            "result": result_dict,
            "elapsed_sec": int(elapsed),
        }

        return {
            "status": "completed",
            "result": result_dict,
            "user_id": user_id,
            "elapsed_sec": int(elapsed),
        }

    except Exception as e:
        logger.error(f"[{job_id}] Analysis failed: {e}")

        # Cleanup on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        _job_results[job_id] = {
            "state": "FAILURE",
            "progress": 0,
            "status": f"❌ Failed: {str(e)[:100]}",
        }

        return {
            "status": "failed",
            "error": str(e),
            "user_id": user_id,
        }


async def download_and_analyze_task(ctx: dict, url: str, user_id: int) -> Dict[str, Any]:
    """
    Download audio from URL and analyze.

    Reports progress: 5% downloading, then analysis stages.

    Args:
        ctx: ARQ context
        url: Audio URL (SoundCloud, YouTube, etc.)
        user_id: Telegram user ID

    Returns:
        Analysis result dict
    """
    import uuid

    job_id = ctx.get("job_id", "unknown")
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    file_id = str(uuid.uuid4())[:8]
    output_template = os.path.join(downloads_dir, f"{file_id}.%(ext)s")

    logger.info(f"[{job_id}] Downloading: {url}")

    # Start timing
    _job_start_times[job_id] = time.time()

    update_job_progress(job_id, 5, "📥 Downloading audio...")

    try:
        # Refresh proxy list if needed (async)
        await refresh_proxy_list()

        # Download with failover (tries direct, then proxies)
        download_with_failover(url, output_template, max_retries=3)

        # Find downloaded file
        file_path = None
        for f in os.listdir(downloads_dir):
            if f.startswith(file_id):
                file_path = os.path.join(downloads_dir, f)
                break

        if not file_path:
            raise Exception("Downloaded file not found")

        download_time = time.time() - _job_start_times[job_id]
        logger.info(f"[{job_id}] Downloaded in {download_time:.1f}s: {file_path}")

        # Run analysis (will continue using same job timing)
        return await analyze_set_task(ctx, file_path, user_id)

    except Exception as e:
        logger.error(f"[{job_id}] Download failed: {e}")

        _job_results[job_id] = {
            "state": "FAILURE",
            "progress": 0,
            "status": f"❌ Download failed: {str(e)[:80]}",
        }

        return {
            "status": "failed",
            "error": str(e),
            "user_id": user_id,
        }


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from in-memory storage."""
    return _job_results.get(job_id, {
        "state": "PENDING",
        "progress": 0,
        "status": "Waiting in queue...",
    })


# ARQ Worker class
class WorkerSettings:
    """ARQ Worker settings for long-running DJ set analysis."""
    functions = [analyze_set_task, download_and_analyze_task]
    redis_settings = get_redis_settings()

    # Allow 2 concurrent jobs
    max_jobs = 2

    # Long timeouts for 2+ hour DJ sets
    job_timeout = 3600          # 1 hour max per job
    keep_result = 7200          # Keep results for 2 hours

    # Health check settings - don't mark jobs as failed during long operations
    health_check_interval = 300  # 5 minutes between health checks
    max_tries = 1                # Don't retry failed jobs automatically


# Redis pool for enqueueing jobs
_redis_pool: Optional[ArqRedis] = None


async def get_redis_pool() -> ArqRedis:
    """Get or create Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await create_pool(get_redis_settings())
    return _redis_pool


async def enqueue_analyze_set(file_path: str, user_id: int) -> str:
    """Enqueue set analysis task. Returns job ID."""
    pool = await get_redis_pool()
    job = await pool.enqueue_job("analyze_set_task", file_path, user_id)
    return job.job_id


async def enqueue_download_and_analyze(url: str, user_id: int) -> str:
    """Enqueue download and analyze task. Returns job ID."""
    pool = await get_redis_pool()
    job = await pool.enqueue_job("download_and_analyze_task", url, user_id)
    return job.job_id

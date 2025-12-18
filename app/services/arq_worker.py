"""
ARQ Worker - Async task queue for DJ set analysis.

ARQ is asyncio-native and compatible with Redis 8.x/Valkey.
Reports progress every 30% with estimated time remaining.
"""

import os
import time
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any

from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from app.common.logging import get_logger, setup_logging
from app.common.logging.correlation import set_job_id, set_user_id
from app.common.monitoring import get_metrics_collector

# Configure JSON logging for ARQ worker and framework
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=bool(os.getenv("LOG_JSON_FORMAT", "true").lower() in ("true", "1", "yes")),
    enable_yc_logging=False,  # YC logging via fluent-bit
)

# Configure ARQ framework loggers explicitly
for arq_logger_name in ["arq.worker", "arq.jobs", "arq"]:
    arq_logger = logging.getLogger(arq_logger_name)
    arq_logger.setLevel(logging.INFO)
    # ARQ loggers will inherit root logger's handlers (JSON format)

logger = get_logger(__name__)


# ============================================================================
# Download with yt-dlp
# ============================================================================


def download_audio(url: str, output_template: str) -> str:
    """
    Download audio using yt-dlp via NAT Gateway.

    Args:
        url: Audio URL (SoundCloud, YouTube, etc.)
        output_template: Output path template with %(ext)s

    Returns:
        Path to downloaded file (without extension placeholder)

    Raises:
        Exception if download fails
    """
    logger.info(f"Download attempt via NAT Gateway")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_template,
        "--max-filesize", "500M",
        "--no-warnings",
        "--socket-timeout", "60",
        # Geo-bypass options
        "--geo-bypass",
        "--geo-bypass-country", "US",
        # User-agent to avoid bot detection
        "--user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        # Retry on network errors (yt-dlp internal retries)
        "--retries", "3",
        "--fragment-retries", "3",
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout for large files
        )

        if result.returncode == 0:
            logger.info("Download successful via NAT Gateway")
            return output_template.replace(".%(ext)s", "")

        # Download failed
        stderr = result.stderr
        stdout = result.stdout
        error_msg = stderr[:300] if stderr else stdout[:300]
        logger.warning(f"Download failed: {error_msg}")
        raise Exception(f"yt-dlp failed: {error_msg}")

    except subprocess.TimeoutExpired:
        logger.warning("Download timeout (300s)")
        raise Exception("Download timeout after 300s")

    except Exception as e:
        logger.error(f"Download exception: {e}")
        raise


def download_with_failover(url: str, output_template: str) -> str:
    """Alias for download_audio (backwards compatibility)."""
    return download_audio(url, output_template)

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
    result = {
        "state": "PROGRESS",
        "progress": progress,
        "status": status,
        "eta": eta,
        "updated_at": datetime.now().isoformat(),
    }
    _job_results[job_id] = result

    # Store in Redis for cross-process access
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, schedule task
            asyncio.create_task(_store_progress_in_redis(job_id, result))
        else:
            # Sync context - skip Redis (worker will handle)
            pass
    except Exception:
        pass  # Redis update is optional


async def _store_progress_in_redis(job_id: str, progress_data: dict):
    """Store progress in Redis for cross-process access."""
    try:
        import json
        pool = await get_redis_pool()
        key = f"arq:progress:{job_id}"
        await pool.set(key, json.dumps(progress_data), ex=3600)  # Expire in 1 hour
    except Exception as e:
        logger.warning(f"Failed to store progress in Redis: {e}")


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

    # Set context for logging
    set_job_id(job_id)
    set_user_id(user_id)

    logger.info("Starting set analysis", data={
        "job_id": job_id,
        "user_id": user_id,
        "file_path": file_path,
    })

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

        # Collect metrics
        metrics = get_metrics_collector()
        metrics.record_task_metrics(
            task_name="analyze_set_task",
            duration_sec=elapsed,
            success=result.success,
            file_duration_sec=result.duration_sec,
            peak_memory_mb=result.peak_memory_mb,
        )

        logger.info("Analysis completed", data={
            "job_id": job_id,
            "user_id": user_id,
            "elapsed_sec": round(elapsed, 1),
            "n_segments": result.n_segments,
            "total_drops": result.total_drops,
            "success": result.success,
        })

        # Cleanup file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Cleaned up temp file", data={
                "job_id": job_id,
                "file_path": file_path,
            })

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
        elapsed = time.time() - _job_start_times.get(job_id, time.time())

        # Record failure metrics
        metrics = get_metrics_collector()
        metrics.record_task_metrics(
            task_name="analyze_set_task",
            duration_sec=elapsed,
            success=False,
            error=str(e)[:200],
        )

        logger.error("Analysis failed", data={
            "job_id": job_id,
            "user_id": user_id,
            "file_path": file_path,
            "error": str(e),
        }, exc_info=True)

        # Cleanup on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.warning("Failed to cleanup temp file", data={
                    "job_id": job_id,
                    "file_path": file_path,
                    "error": str(cleanup_error),
                })

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

    # Set context for logging
    set_job_id(job_id)
    set_user_id(user_id)

    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    file_id = str(uuid.uuid4())[:8]
    output_template = os.path.join(downloads_dir, f"{file_id}.%(ext)s")

    logger.info("Starting download and analysis", data={
        "job_id": job_id,
        "user_id": user_id,
        "url": url[:100],
    })

    # Start timing
    _job_start_times[job_id] = time.time()

    update_job_progress(job_id, 5, "📥 Downloading audio...")

    try:
        # Download audio via NAT Gateway
        download_audio(url, output_template)

        # Find downloaded file
        file_path = None
        for f in os.listdir(downloads_dir):
            if f.startswith(file_id):
                file_path = os.path.join(downloads_dir, f)
                break

        if not file_path:
            raise Exception("Downloaded file not found")

        download_time = time.time() - _job_start_times[job_id]
        logger.info("Download completed", data={
            "job_id": job_id,
            "user_id": user_id,
            "download_time_sec": round(download_time, 1),
            "file_path": file_path,
        })

        # Run analysis (will continue using same job timing)
        return await analyze_set_task(ctx, file_path, user_id)

    except Exception as e:
        elapsed = time.time() - _job_start_times.get(job_id, time.time())

        # Record download failure metrics
        metrics = get_metrics_collector()
        metrics.record_task_metrics(
            task_name="download_and_analyze_task",
            duration_sec=elapsed,
            success=False,
            error=str(e)[:200],
        )

        logger.error("Download failed", data={
            "job_id": job_id,
            "user_id": user_id,
            "url": url[:100],
            "error": str(e),
        }, exc_info=True)

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


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from Redis (ARQ result)."""
    try:
        from arq.jobs import Job, JobStatus

        pool = await get_redis_pool()
        job = Job(job_id, pool)

        # Get job status
        status = await job.status()

        if status == JobStatus.not_found:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "Waiting in queue...",
            }

        # Map ARQ status to our format
        if status == JobStatus.complete:
            result = await job.result()
            if isinstance(result, dict):
                if result.get("status") == "failed":
                    return {
                        "state": "FAILURE",
                        "progress": 0,
                        "status": f"❌ {result.get('error', 'Unknown error')[:100]}",
                    }
                return {
                    "state": "SUCCESS",
                    "progress": 100,
                    "status": "✅ Completed",
                    "result": result.get("result", {}),
                }
            return {
                "state": "SUCCESS",
                "progress": 100,
                "status": "✅ Completed",
            }
        elif status == JobStatus.in_progress:
            # Try to get progress from Redis (cross-process)
            try:
                import json
                key = f"arq:progress:{job_id}"
                progress_str = await pool.get(key)
                if progress_str:
                    return json.loads(progress_str.decode() if isinstance(progress_str, bytes) else progress_str)
            except Exception:
                pass

            # Fallback to in-memory progress tracking
            if job_id in _job_results:
                return _job_results[job_id]
            return {
                "state": "PROGRESS",
                "progress": 50,
                "status": "🔄 Processing...",
            }
        elif status == JobStatus.deferred:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "⏳ Scheduled...",
            }
        elif status == JobStatus.queued:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "Waiting in queue...",
            }
        else:
            return {
                "state": "UNKNOWN",
                "progress": 0,
                "status": f"Unknown status: {status}",
            }
    except Exception as e:
        logger.warning(f"Failed to get job status: {e}")
        return {
            "state": "UNKNOWN",
            "progress": 0,
            "status": f"Status unavailable: {str(e)[:50]}",
        }


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

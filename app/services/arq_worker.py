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

from app.common.logging import get_logger
from app.common.logging.correlation import set_job_id, set_user_id

logger = get_logger(__name__)


# ============================================================================
# Proxy Rotation with xray-core for VLESS/SS
# ============================================================================

import json
import socket
import tempfile
import shutil
from typing import List

# Proxy configs from Happ subscription (tested working for SoundCloud, YT, Discord)
_PROXY_CONFIGS: List[dict] = [
    # VLESS gRPC - RU (WORKS for SoundCloud, YT, Discord, Instagram)
    {
        "name": "RU-VLESS-gRPC",
        "local_port": 10808,
        "outbound": {
            "protocol": "vless",
            "settings": {
                "vnext": [{
                    "address": "193.233.231.221",
                    "port": 8443,
                    "users": [{
                        "id": "362229f7-9a9c-4810-8986-a2377f7d3bf9",
                        "encryption": "none",
                        "level": 8
                    }]
                }]
            },
            "streamSettings": {
                "network": "grpc",
                "security": "none",
                "grpcSettings": {
                    "serviceName": "",
                    "multiMode": False
                }
            }
        }
    },
    # Shadowsocks - UK (backup, works for YT but not SoundCloud)
    {
        "name": "UK-SS",
        "local_port": 10809,
        "outbound": {
            "protocol": "shadowsocks",
            "settings": {
                "servers": [{
                    "address": "144.31.178.150",
                    "port": 8080,
                    "method": "chacha20-ietf-poly1305",
                    "password": "UPm3HNg8x4uAzAxThpaSnABG9uRoJa3j"
                }]
            },
            "streamSettings": {
                "network": "tcp",
                "security": "none"
            }
        }
    },
]

_current_proxy_idx = 0
_failed_proxies: set = set()
_xray_process: Optional[subprocess.Popen] = None
_xray_config_file: Optional[str] = None


def _is_xray_available() -> bool:
    """Check if xray-core is installed."""
    return shutil.which("xray") is not None


def _is_port_open(port: int) -> bool:
    """Check if local port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def _start_xray_proxy(config: dict) -> bool:
    """Start xray with given config. Returns True if successful."""
    global _xray_process, _xray_config_file

    if not _is_xray_available():
        logger.warning("xray-core not installed, skipping proxy")
        return False

    # Stop existing xray
    _stop_xray_proxy()

    # Build xray config
    local_port = config["local_port"]
    xray_config = {
        "log": {"loglevel": "warning"},
        "inbounds": [{
            "port": local_port,
            "listen": "127.0.0.1",
            "protocol": "socks",
            "settings": {"udp": True}
        }],
        "outbounds": [config["outbound"]]
    }

    # Write config to temp file
    try:
        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="xray_")
        with os.fdopen(fd, "w") as f:
            json.dump(xray_config, f)
        _xray_config_file = config_path

        # Start xray
        _xray_process = subprocess.Popen(
            ["xray", "run", "-c", config_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Wait for proxy to start
        for _ in range(10):
            time.sleep(0.5)
            if _is_port_open(local_port):
                logger.info(f"Started xray proxy on port {local_port} ({config['name']})")
                return True

        # Failed to start
        logger.warning(f"xray failed to start for {config['name']}")
        _stop_xray_proxy()
        return False

    except Exception as e:
        logger.error(f"Failed to start xray: {e}")
        _stop_xray_proxy()
        return False


def _stop_xray_proxy() -> None:
    """Stop xray process and cleanup."""
    global _xray_process, _xray_config_file

    if _xray_process:
        try:
            _xray_process.terminate()
            _xray_process.wait(timeout=5)
        except Exception:
            try:
                _xray_process.kill()
            except Exception:
                pass
        _xray_process = None

    if _xray_config_file and os.path.exists(_xray_config_file):
        try:
            os.remove(_xray_config_file)
        except Exception:
            pass
        _xray_config_file = None


def get_next_proxy() -> Optional[str]:
    """Get next working proxy URL. Starts xray if needed."""
    global _current_proxy_idx

    # First try env var (custom proxy)
    env_proxy = os.getenv("YTDLP_PROXY")
    if env_proxy:
        return env_proxy

    # No configs - direct connection
    if not _PROXY_CONFIGS:
        return None

    # Try to start next proxy
    attempts = 0
    while attempts < len(_PROXY_CONFIGS):
        idx = _current_proxy_idx % len(_PROXY_CONFIGS)
        config = _PROXY_CONFIGS[idx]
        _current_proxy_idx += 1

        if config["name"] in _failed_proxies:
            attempts += 1
            continue

        # Try to start this proxy
        if _start_xray_proxy(config):
            return f"socks5://127.0.0.1:{config['local_port']}"

        # Failed, mark and try next
        _failed_proxies.add(config["name"])
        attempts += 1

    # All failed, reset and try direct
    logger.warning("All proxies failed, using direct connection")
    _failed_proxies.clear()
    return None


def mark_proxy_failed(proxy: Optional[str]) -> None:
    """Mark current proxy as failed."""
    global _current_proxy_idx
    if proxy and _PROXY_CONFIGS and _current_proxy_idx > 0:
        idx = (_current_proxy_idx - 1) % len(_PROXY_CONFIGS)
        name = _PROXY_CONFIGS[idx]["name"]
        _failed_proxies.add(name)
        logger.info(f"Marked proxy as failed: {name}")
        _stop_xray_proxy()


# ============================================================================
# Download with yt-dlp (supports geo-bypass via extractor args)
# ============================================================================


def download_audio(url: str, output_template: str, max_retries: int = 3) -> str:
    """
    Download audio using yt-dlp with proxy rotation.

    Tries direct first, then rotates through VLESS/SS proxies via xray.

    Args:
        url: Audio URL (SoundCloud, YouTube, etc.)
        output_template: Output path template with %(ext)s
        max_retries: Number of retry attempts

    Returns:
        Path to downloaded file (without extension placeholder)

    Raises:
        Exception if all attempts fail
    """
    last_error = None
    current_proxy = None

    for attempt in range(max_retries):
        # Get proxy (starts xray if needed)
        current_proxy = get_next_proxy()
        proxy_desc = current_proxy[:30] + "..." if current_proxy else "direct"

        logger.info(f"Download attempt {attempt + 1}/{max_retries} via {proxy_desc}")

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
            # Retry on network errors
            "--retries", "3",
            "--fragment-retries", "3",
        ]

        # Add proxy if available
        if current_proxy:
            cmd.extend(["--proxy", current_proxy])
            logger.info(f"Using proxy: {proxy_desc}")

        cmd.append(url)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout for large files
            )

            if result.returncode == 0:
                logger.info(f"Download successful via {proxy_desc}")
                _stop_xray_proxy()  # Cleanup after success
                return output_template.replace(".%(ext)s", "")

            # Log error details
            stderr = result.stderr
            stdout = result.stdout

            # Check specific errors
            if "403" in stderr or "forbidden" in stderr.lower():
                last_error = f"Geo-blocked via {proxy_desc}"
                logger.warning(f"Geo-blocked via {proxy_desc}: {stderr[:150]}")
                mark_proxy_failed(current_proxy)
                continue  # Try next proxy
            elif "Sign in" in stderr or "age" in stderr.lower():
                last_error = "Age-restricted content. Requires authentication."
                logger.warning(f"Age-restricted: {stderr[:200]}")
            elif "not available" in stderr.lower():
                last_error = "Content not available in this region."
                logger.warning(f"Not available: {stderr[:200]}")
                mark_proxy_failed(current_proxy)
                continue  # Try next proxy
            elif "proxy" in stderr.lower() or "tunnel" in stderr.lower():
                last_error = f"Proxy connection failed: {proxy_desc}"
                logger.warning(f"Proxy error: {stderr[:150]}")
                mark_proxy_failed(current_proxy)
                continue  # Try next proxy
            else:
                last_error = stderr[:300] if stderr else stdout[:300]
                logger.warning(f"Download error: {last_error}")

            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        except subprocess.TimeoutExpired:
            last_error = f"Download timeout via {proxy_desc}"
            logger.warning(f"Download timeout via {proxy_desc}")
            mark_proxy_failed(current_proxy)

        except Exception as e:
            last_error = str(e)
            logger.error(f"Download exception: {e}")
            mark_proxy_failed(current_proxy)

    _stop_xray_proxy()  # Cleanup on failure
    raise Exception(f"Download failed after {max_retries} attempts: {last_error}")


def download_with_failover(url: str, output_template: str, max_retries: int = 3) -> str:
    """Alias for download_audio (backwards compatibility)."""
    return download_audio(url, output_template, max_retries)

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
        # Download audio (with geo-bypass options)
        download_audio(url, output_template, max_retries=3)

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
        pool = await get_redis_pool()
        job = await pool.job(job_id)
        if job is None:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "Waiting in queue...",
            }

        # Get job info
        info = await job.info()
        if info is None:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "Waiting in queue...",
            }

        # Map ARQ status to our format
        from arq.jobs import JobStatus
        if info.status == JobStatus.complete:
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
        elif info.status == JobStatus.in_progress:
            return {
                "state": "PROGRESS",
                "progress": 50,
                "status": "🔄 Processing...",
            }
        elif info.status == JobStatus.deferred:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "⏳ Scheduled...",
            }
        else:
            return {
                "state": "PENDING",
                "progress": 0,
                "status": "Waiting in queue...",
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

"""
Celery application for background task processing.

Tasks:
- analyze_set_task - Analyze DJ set
- analyze_track_task - Analyze single track
- cleanup_downloads_task - Periodic cleanup
"""

import os
import logging
from datetime import datetime, timedelta

from celery import Celery
from celery.schedules import crontab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery
celery_app = Celery(
    "mood_classifier",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes max
    task_soft_time_limit=1500,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=2,  # 2 concurrent workers
    result_expires=86400,  # Results expire after 24 hours
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-downloads-every-5-min": {
        "task": "src.services.celery_app.cleanup_downloads_task",
        "schedule": crontab(minute="*/5"),
    },
    "cleanup-old-results-hourly": {
        "task": "src.services.celery_app.cleanup_old_results_task",
        "schedule": crontab(minute=0),
    },
}


# ============== Tasks ==============

@celery_app.task(bind=True, name="analyze_set")
def analyze_set_task(self, file_path: str, user_id: int, callback_data: dict = None):
    """
    Analyze DJ set in background.

    Args:
        file_path: Path to audio file
        user_id: Telegram user ID for notifications
        callback_data: Optional data for callback

    Returns:
        Analysis result dict
    """
    from .analysis import AnalysisService

    logger.info(f"Starting set analysis: {file_path}")

    try:
        # Update progress
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading audio..."})

        service = AnalysisService()

        # Progress callback
        def progress_callback(percent: int, message: str):
            self.update_state(state="PROGRESS", meta={"progress": percent, "status": message})

        # Run analysis
        result = service.analyze_set(file_path, progress_callback=progress_callback)

        # Convert result to dict
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result

        logger.info(f"Analysis completed: {file_path}")

        # Cleanup file after analysis
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")

        return {
            "status": "completed",
            "result": result_dict,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")

        # Cleanup on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        return {
            "status": "failed",
            "error": str(e),
            "user_id": user_id,
        }


@celery_app.task(bind=True, name="analyze_track")
def analyze_track_task(self, file_path: str, user_id: int):
    """
    Analyze single track in background.

    Args:
        file_path: Path to audio file
        user_id: Telegram user ID

    Returns:
        Analysis result dict
    """
    from .analysis import AnalysisService

    logger.info(f"Starting track analysis: {file_path}")

    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading audio..."})

        service = AnalysisService()
        result = service.analyze_track(file_path)

        result_dict = result.to_dict() if hasattr(result, "to_dict") else result

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "completed",
            "result": result_dict,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Track analysis failed: {e}")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        return {
            "status": "failed",
            "error": str(e),
            "user_id": user_id,
        }


@celery_app.task(name="download_and_analyze")
def download_and_analyze_task(url: str, user_id: int):
    """
    Download audio from URL and analyze.

    Args:
        url: SoundCloud or other audio URL
        user_id: Telegram user ID

    Returns:
        Analysis result dict
    """
    import subprocess
    import uuid

    downloads_dir = os.getenv("DOWNLOADS_DIR", "/app/downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    file_id = str(uuid.uuid4())[:8]
    output_template = os.path.join(downloads_dir, f"{file_id}.%(ext)s")

    logger.info(f"Downloading: {url}")

    try:
        # Download with yt-dlp
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "-o", output_template,
            "--max-filesize", "500M",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise Exception(f"Download failed: {result.stderr}")

        # Find downloaded file
        for f in os.listdir(downloads_dir):
            if f.startswith(file_id):
                file_path = os.path.join(downloads_dir, f)
                break
        else:
            raise Exception("Downloaded file not found")

        logger.info(f"Downloaded: {file_path}")

        # Queue analysis
        return analyze_set_task.delay(file_path, user_id).id

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "user_id": user_id,
        }


@celery_app.task(name="cleanup_downloads")
def cleanup_downloads_task():
    """
    Periodic task to clean up old downloads.

    Removes files older than 1 hour.
    """
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/app/downloads")

    if not os.path.exists(downloads_dir):
        return {"cleaned": 0}

    now = datetime.now()
    max_age = timedelta(hours=1)
    cleaned = 0

    for filename in os.listdir(downloads_dir):
        file_path = os.path.join(downloads_dir, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_time > max_age:
                try:
                    os.remove(file_path)
                    cleaned += 1
                    logger.info(f"Cleaned up old file: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {filename}: {e}")

    logger.info(f"Cleanup complete: {cleaned} files removed")
    return {"cleaned": cleaned}


@celery_app.task(name="cleanup_old_results")
def cleanup_old_results_task():
    """
    Periodic task to clean up old task results from Redis.
    """
    # Celery handles this via result_expires setting
    logger.info("Old results cleanup triggered")
    return {"status": "ok"}

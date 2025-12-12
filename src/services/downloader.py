"""
Audio downloader for SoundCloud and other platforms.

Uses yt-dlp for downloading.
"""

import os
import asyncio
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


async def download_audio(
    url: str,
    output_dir: str,
    file_id: str,
    max_size_mb: int = 500,
) -> str:
    """
    Download audio from URL using yt-dlp.

    Args:
        url: Audio URL (SoundCloud, etc.)
        output_dir: Directory to save file
        file_id: Unique identifier for the file
        max_size_mb: Maximum file size in MB

    Returns:
        Path to downloaded file

    Raises:
        Exception: If download fails
    """
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, f"{file_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--max-filesize", f"{max_size_mb}M",
        "--no-playlist",  # Don't download playlists
        "--no-warnings",
        url,
    ]

    logger.info(f"Downloading: {url}")

    # Run yt-dlp in thread pool to not block async
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    )

    if result.returncode != 0:
        error_msg = result.stderr or "Unknown error"
        logger.error(f"Download failed: {error_msg}")
        raise Exception(f"Download failed: {error_msg}")

    # Find downloaded file
    for filename in os.listdir(output_dir):
        if filename.startswith(file_id):
            file_path = os.path.join(output_dir, filename)
            logger.info(f"Downloaded: {file_path}")
            return file_path

    raise Exception("Downloaded file not found")


def get_audio_info(url: str) -> Optional[dict]:
    """
    Get audio metadata without downloading.

    Args:
        url: Audio URL

    Returns:
        Metadata dict or None
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Failed to get audio info: {e}")

    return None


def is_supported_url(url: str) -> bool:
    """
    Check if URL is supported by yt-dlp.

    Args:
        url: URL to check

    Returns:
        True if supported
    """
    supported_domains = [
        "soundcloud.com",
        "mixcloud.com",
        "youtube.com",
        "youtu.be",
        "bandcamp.com",
    ]

    return any(domain in url.lower() for domain in supported_domains)

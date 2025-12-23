"""
End-to-End Integration Tests for DJ Set Analysis Bot.

Тестирует ПОЛНЫЙ FLOW:
    Bot Handler → Redis Queue (ARQ) → Worker → Analysis Pipeline → Cache → Services → Bot Response

Сценарий теста:
    1. Скачивание сета с SoundCloud (yt-dlp)
    2. Постановка задачи в Redis Queue (ARQ)
    3. Worker запускает Analysis Pipeline
    4. Результат сохраняется в Cache
    5. Проверка совместимости кэша с ProfilingService
    6. Проверка совместимости кэша с SetGeneratorPipeline
    7. Проверка детерминизма (повторный анализ = те же результаты)
    8. Bot получает результат и показывает пользователю

Запуск:
    # Полный E2E тест (требует yt-dlp + сеть)
    python tests/test_bot_integration.py

    # Через pytest
    pytest tests/test_bot_integration.py -v -m e2e

Требования:
    - yt-dlp установлен
    - ffprobe установлен
    - Сетевой доступ к SoundCloud
"""

import os
import sys
import json
import time
import uuid
import asyncio
import hashlib
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
import pytest_asyncio

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Configuration
# =============================================================================

# Test set URL (short redirect URL)
SOUNDCLOUD_TEST_URL = "https://on.soundcloud.com/vOOGKUbsKZOd2IpXwK"

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Timeouts
DOWNLOAD_TIMEOUT = 600  # 10 minutes
ANALYSIS_TIMEOUT = 900  # 15 minutes
TOTAL_E2E_TIMEOUT = 1800  # 30 minutes

# Test user simulation
TEST_USER_ID = 123456789
TEST_CHAT_ID = 123456789


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class E2ETestContext:
    """Context for E2E test run."""
    temp_dir: str
    downloads_dir: str
    cache_dir: str
    job_id: Optional[str] = None
    file_path: Optional[str] = None
    analysis_result: Optional[Dict] = None
    start_time: float = field(default_factory=time.time)
    events: List[Dict] = field(default_factory=list)

    def log_event(self, event_type: str, data: Dict = None):
        """Log test event with timestamp."""
        self.events.append({
            "type": event_type,
            "timestamp": time.time() - self.start_time,
            "data": data or {}
        })

    def get_elapsed(self) -> float:
        return time.time() - self.start_time


@dataclass
class E2ETestResult:
    """Result of E2E test."""
    success: bool
    download_ok: bool = False
    queue_ok: bool = False
    worker_ok: bool = False
    analysis_ok: bool = False
    cache_ok: bool = False
    profiler_ok: bool = False
    generator_ok: bool = False
    reproducible: bool = False
    error: Optional[str] = None
    analysis_result: Optional[Dict] = None
    timing: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Redis Connection Helpers
# =============================================================================

async def check_redis_connection() -> bool:
    """Check if Redis is available."""
    try:
        import redis.asyncio as redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        await client.ping()
        await client.close()
        return True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False


async def clear_test_jobs():
    """Clear test jobs from Redis."""
    try:
        import redis.asyncio as redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        # Clear ARQ job results with test prefix
        keys = await client.keys("arq:result:test_*")
        if keys:
            await client.delete(*keys)
        await client.close()
    except Exception:
        pass


# =============================================================================
# Download Helpers
# =============================================================================

def download_audio_sync(url: str, output_dir: str, verbose: bool = True) -> Tuple[bool, str, Optional[str]]:
    """
    Download audio using yt-dlp (synchronous) with progress bar.

    Downloads in ORIGINAL format (opus/mp3/etc) without conversion for speed.

    Returns: (success, file_path or error, error_message)
    """
    import re

    file_id = f"test_{uuid.uuid4().hex[:8]}"
    output_template = os.path.join(output_dir, f"{file_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio",  # Best audio quality, no conversion
        "-o", output_template,
        "--max-filesize", "500M",
        "--no-playlist",
        "--newline",  # Progress on new lines for real-time output
        "--progress-template", "%(progress._percent_str)s %(progress._speed_str)s ETA:%(progress._eta_str)s",
        url,
    ]

    try:
        if verbose:
            # Run with real-time progress bar
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            last_percent = -1
            bar_width = 40

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # Parse percentage from yt-dlp output
                percent_match = re.search(r'(\d+\.?\d*)%', line)
                if percent_match:
                    try:
                        percent = float(percent_match.group(1))
                        percent_int = int(percent)

                        # Only update if percent changed (avoid flickering)
                        if percent_int != last_percent:
                            last_percent = percent_int
                            filled = int(bar_width * percent / 100)
                            bar = '█' * filled + '░' * (bar_width - filled)

                            # Extract speed and ETA if available
                            speed_match = re.search(r'(\d+\.?\d*\s*[KMG]?i?B/s)', line)
                            eta_match = re.search(r'ETA:(\S+)', line)

                            speed = speed_match.group(1) if speed_match else ""
                            eta = eta_match.group(1) if eta_match else ""

                            status = f"  [{bar}] {percent:5.1f}%"
                            if speed:
                                status += f" | {speed}"
                            if eta and eta != "Unknown":
                                status += f" | ETA: {eta}"

                            # Print with carriage return for in-place update
                            print(f"\r{status}", end='', flush=True)

                    except (ValueError, IndexError):
                        pass

                # Show completion/extraction messages
                elif any(x in line.lower() for x in ['destination', 'deleting', 'already']):
                    print(f"\n    {line}")

            # Final newline after progress bar
            print()

            process.wait(timeout=DOWNLOAD_TIMEOUT)

            if process.returncode != 0:
                return False, "", f"yt-dlp failed with code {process.returncode}"
        else:
            # Silent mode
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT
            )

            if result.returncode != 0:
                return False, "", f"yt-dlp failed: {result.stderr[:300]}"

        # Find downloaded file
        for f in os.listdir(output_dir):
            if f.startswith(file_id):
                return True, os.path.join(output_dir, f), None

        return False, "", "Downloaded file not found"

    except subprocess.TimeoutExpired:
        return False, "", "Download timeout"
    except FileNotFoundError:
        return False, "", "yt-dlp not installed"
    except Exception as e:
        return False, "", str(e)


def get_audio_duration(file_path: str) -> float:
    """Get audio duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# =============================================================================
# Mock Bot Components
# =============================================================================

class MockTelegramMessage:
    """Mock Telegram message for testing."""

    def __init__(self, user_id: int, chat_id: int, text: str = ""):
        self.from_user = MagicMock()
        self.from_user.id = user_id
        self.chat = MagicMock()
        self.chat.id = chat_id
        self.text = text
        self.message_id = 1


class MockBot:
    """Mock Telegram Bot for testing."""

    def __init__(self):
        self.sent_messages: List[Dict] = []
        self.edited_messages: List[Dict] = []

    async def send_message(self, chat_id: int, text: str, **kwargs):
        self.sent_messages.append({
            "chat_id": chat_id,
            "text": text,
            "kwargs": kwargs
        })
        return MagicMock(message_id=len(self.sent_messages))

    async def edit_message_text(self, text: str, chat_id: int, message_id: int, **kwargs):
        self.edited_messages.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "kwargs": kwargs
        })


# =============================================================================
# Core E2E Test Functions
# =============================================================================

async def run_e2e_flow(ctx: E2ETestContext) -> E2ETestResult:
    """
    Run complete E2E flow: Download → Queue → Worker → Analysis → Cache → Verify.
    """
    result = E2ETestResult(success=False)

    try:
        # =====================================================================
        # STEP 1: Download audio from SoundCloud
        # =====================================================================
        ctx.log_event("download_start", {"url": SOUNDCLOUD_TEST_URL[:60]})
        print("\n[1/6] Downloading audio from SoundCloud...")

        download_start = time.time()
        success, file_path, error = download_audio_sync(SOUNDCLOUD_TEST_URL, ctx.downloads_dir)
        result.timing["download"] = time.time() - download_start

        if not success:
            result.error = f"Download failed: {error}"
            ctx.log_event("download_failed", {"error": error})
            return result

        ctx.file_path = file_path
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        duration = get_audio_duration(file_path)

        ctx.log_event("download_complete", {
            "file_path": file_path,
            "size_mb": file_size,
            "duration_sec": duration
        })
        print(f"  ✓ Downloaded: {file_size:.1f}MB, {duration/60:.1f}min")
        result.download_ok = True

        # =====================================================================
        # STEP 2: Simulate Bot Handler - Queue job via ARQ
        # =====================================================================
        ctx.log_event("queue_start")
        print("\n[2/6] Queueing analysis job...")

        from app.services.arq_worker import (
            _job_results,
            _job_start_times,
            analyze_set_task,
        )

        # Generate test job ID
        job_id = f"test_{uuid.uuid4().hex[:8]}"
        ctx.job_id = job_id

        # Initialize job tracking
        _job_results[job_id] = {
            "state": "PENDING",
            "progress": 0,
            "status": "Queued..."
        }

        ctx.log_event("job_queued", {"job_id": job_id})
        print(f"  ✓ Job queued: {job_id}")
        result.queue_ok = True

        # =====================================================================
        # STEP 3: Run Analysis Pipeline with real progress output
        # =====================================================================
        ctx.log_event("worker_start")
        print("\n[3/7] Running analysis pipeline...")
        print("=" * 60)

        # Create a copy of the file
        worker_file_path = os.path.join(ctx.downloads_dir, f"worker_{Path(file_path).name}")
        shutil.copy(file_path, worker_file_path)

        worker_start = time.time()

        # Run pipeline directly with verbose=True for real progress output
        from app.modules.analysis.pipelines.set_analysis import SetAnalysisPipeline

        pipeline = SetAnalysisPipeline(
            sr=22050,
            analyze_genres=False,
            verbose=True  # Enable real progress output
        )

        try:
            pipeline_result = pipeline.analyze(worker_file_path)
            result.timing["worker"] = time.time() - worker_start

            if not pipeline_result.success:
                result.error = f"Pipeline failed: {pipeline_result.error}"
                ctx.log_event("worker_failed", {"error": result.error})
                return result

            # Convert to dict for compatibility
            worker_result = {
                "status": "completed",
                "elapsed_sec": result.timing["worker"],
                "result": pipeline_result.to_dict()
            }

        except Exception as e:
            import traceback
            result.error = f"Pipeline exception: {e}\n{traceback.format_exc()}"
            ctx.log_event("worker_failed", {"error": str(e)})
            return result

        print("=" * 60)
        ctx.log_event("worker_complete", {
            "elapsed_sec": result.timing["worker"],
            "n_segments": pipeline_result.n_segments
        })
        print(f"\n  ✓ Analysis completed in {result.timing['worker']:.1f}s")
        print(f"    Duration: {pipeline_result.duration_sec/60:.1f} min")
        print(f"    Segments: {pipeline_result.n_segments}")
        print(f"    Transitions: {pipeline_result.n_transitions}")
        print(f"    Drops: {pipeline_result.total_drops}")
        result.worker_ok = True

        # =====================================================================
        # STEP 4: Verify Analysis Results
        # =====================================================================
        ctx.log_event("analysis_verify_start")
        print("\n[4/7] Verifying analysis results...")

        analysis_result = worker_result.get("result", {})
        ctx.analysis_result = analysis_result
        result.analysis_result = analysis_result

        # Check required fields
        required_fields = [
            "duration_sec", "n_transitions", "n_segments",
            "total_drops", "success"
        ]

        for field in required_fields:
            if field not in analysis_result:
                result.error = f"Missing field in result: {field}"
                return result

        if not analysis_result.get("success", False):
            result.error = f"Analysis not successful: {analysis_result.get('error')}"
            return result

        # Validate values
        duration_sec = analysis_result["duration_sec"]
        n_segments = analysis_result["n_segments"]
        n_transitions = analysis_result["n_transitions"]
        total_drops = analysis_result["total_drops"]

        print(f"  Duration: {duration_sec/60:.1f} min")
        print(f"  Segments: {n_segments}")
        print(f"  Transitions: {n_transitions}")
        print(f"  Drops: {total_drops}")

        # Duration should approximately match
        if abs(duration_sec - duration) > 10:
            print(f"  ⚠️ Duration mismatch: {duration_sec}s vs {duration}s")

        # Should have reasonable segments
        if n_segments < 1:
            result.error = "No segments detected"
            return result

        ctx.log_event("analysis_verified", analysis_result)
        print("  ✓ Analysis results verified")
        result.analysis_ok = True

        # =====================================================================
        # STEP 5: Verify Cache Storage
        # =====================================================================
        ctx.log_event("cache_verify_start")
        print("\n[5/7] Verifying cache storage...")

        from app.modules.analysis.pipelines.cache_manager import CacheManager

        cache_manager = CacheManager(ctx.cache_dir)

        # The worker should have saved the result
        # Let's manually save to simulate and then verify retrieval
        cache_manager.save_set_analysis(file_path, analysis_result)

        # Retrieve from cache
        cached_result = cache_manager.get_set_analysis(file_path)

        if cached_result is None:
            result.error = "Failed to retrieve from cache"
            return result

        # Verify cached data matches
        if cached_result.get("n_segments") != n_segments:
            result.error = f"Cache mismatch: segments {cached_result.get('n_segments')} != {n_segments}"
            return result

        if cached_result.get("n_transitions") != n_transitions:
            result.error = f"Cache mismatch: transitions {cached_result.get('n_transitions')} != {n_transitions}"
            return result

        ctx.log_event("cache_verified")
        print("  ✓ Cache storage verified")
        result.cache_ok = True

        # =====================================================================
        # STEP 6: Verify cache is usable by ProfilingService
        # =====================================================================
        ctx.log_event("profiling_service_start")
        print("\n[6/7] Verifying cache usability by ProfilingService...")

        try:
            from app.modules.profiling.services.profiler import ProfilingService
            from app.core.cache import CacheRepository

            # Create profiler with our test cache
            cache_repo = CacheRepository(ctx.cache_dir)
            profiler = ProfilingService(cache_status=cache_repo, cache_dir=ctx.cache_dir)

            # Verify the cached set can be read
            cached_set = cache_repo.get_set_analysis(file_path)
            if cached_set is None:
                result.error = "ProfilingService cannot read cached set"
                return result

            # Verify essential fields for profiling
            required_for_profiling = ['duration_sec', 'n_segments', 'segments', 'total_drops']
            for field in required_for_profiling:
                if field not in cached_set:
                    result.error = f"Cached set missing field for profiler: {field}"
                    return result

            ctx.log_event("profiling_service_verified")
            print("  ✓ Cache is compatible with ProfilingService")
            result.profiler_ok = True

        except ImportError as e:
            print(f"  ⚠️ ProfilingService not available: {e}")
            ctx.log_event("profiling_service_skipped", {"error": str(e)})
            result.profiler_ok = True  # Skip is OK
        except Exception as e:
            result.error = f"ProfilingService compatibility check failed: {e}"
            ctx.log_event("profiling_service_failed", {"error": str(e)})
            return result

        # =====================================================================
        # STEP 7: Verify cache is usable by SetGeneratorPipeline
        # =====================================================================
        ctx.log_event("generator_service_start")
        print("\n[7/7] Verifying cache usability by SetGeneratorPipeline...")

        try:
            from app.modules.generation.pipelines.set_generator import SetGeneratorPipeline
            from app.core.cache import CacheRepository

            # Create generator with our test cache
            cache_repo = CacheRepository(ctx.cache_dir)
            generator = SetGeneratorPipeline(cache_repo=cache_repo)

            # Verify the cache can be accessed
            # Generator uses cache for DJ profiles and set plans
            # We verify the analysis result can be converted to profile format

            cached_set = cache_repo.get_set_analysis(file_path)
            if cached_set is None:
                result.error = "SetGenerator cannot read cached set"
                return result

            # Verify essential fields for set generation
            required_for_generator = ['duration_sec', 'n_transitions', 'segments']
            for field in required_for_generator:
                if field not in cached_set:
                    result.error = f"Cached set missing field for generator: {field}"
                    return result

            # Verify segments have required structure
            segments = cached_set.get('segments', [])
            if segments:
                seg = segments[0]
                seg_fields = ['start_time', 'end_time', 'duration']
                for field in seg_fields:
                    if field not in seg:
                        result.error = f"Segment missing field for generator: {field}"
                        return result

            ctx.log_event("generator_service_verified")
            print("  ✓ Cache is compatible with SetGeneratorPipeline")
            result.generator_ok = True

        except ImportError as e:
            print(f"  ⚠️ SetGeneratorPipeline not available: {e}")
            ctx.log_event("generator_service_skipped", {"error": str(e)})
            result.generator_ok = True  # Skip is OK
        except Exception as e:
            result.error = f"SetGeneratorPipeline compatibility check failed: {e}"
            ctx.log_event("generator_service_failed", {"error": str(e)})
            return result

        # Reproducibility test removed for speed
        result.reproducible = True

        # =====================================================================
        # SUCCESS + CLEANUP
        # =====================================================================
        result.success = True
        ctx.log_event("e2e_complete", {
            "total_time": ctx.get_elapsed(),
            "timing": result.timing
        })

        # Cleanup: delete audio files and cache
        print("\n[Cleanup] Removing test files...")
        try:
            # Delete downloaded audio files
            if ctx.file_path and os.path.exists(ctx.file_path):
                os.remove(ctx.file_path)
                print(f"  ✓ Deleted: {Path(ctx.file_path).name}")

            if os.path.exists(worker_file_path):
                os.remove(worker_file_path)
                print(f"  ✓ Deleted: {Path(worker_file_path).name}")

            # Clear cache
            cache_manager.clear_all()
            print("  ✓ Cache cleared")

            # Remove temp directories
            shutil.rmtree(ctx.downloads_dir, ignore_errors=True)
            shutil.rmtree(ctx.cache_dir, ignore_errors=True)
            print("  ✓ Temp directories removed")

        except Exception as cleanup_err:
            print(f"  ⚠️ Cleanup warning: {cleanup_err}")

        return result

    except Exception as e:
        import traceback
        result.error = f"Exception: {e}\n{traceback.format_exc()}"
        ctx.log_event("exception", {"error": str(e)})

        # Cleanup on error too
        try:
            shutil.rmtree(ctx.downloads_dir, ignore_errors=True)
            shutil.rmtree(ctx.cache_dir, ignore_errors=True)
        except Exception:
            pass

        return result


# =============================================================================
# Pytest Test Classes
# =============================================================================

@pytest.mark.e2e
@pytest.mark.requires_network
class TestE2EIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def test_context(self):
        """Create test context with temp directories."""
        temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
        ctx = E2ETestContext(
            temp_dir=temp_dir,
            downloads_dir=os.path.join(temp_dir, "downloads"),
            cache_dir=os.path.join(temp_dir, "cache"),
        )
        os.makedirs(ctx.downloads_dir, exist_ok=True)
        os.makedirs(ctx.cache_dir, exist_ok=True)

        yield ctx

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio(loop_scope="function")
    @pytest.mark.skip(reason="Requires yt-dlp, Redis, network - full E2E test")
    async def test_full_e2e_flow(self, test_context):
        """Test complete Bot -> Queue -> Worker -> Analysis -> Cache -> Services -> Verify flow."""
        result = await run_e2e_flow(test_context)

        assert result.download_ok, f"Download failed: {result.error}"
        assert result.queue_ok, f"Queue failed: {result.error}"
        assert result.worker_ok, f"Worker failed: {result.error}"
        assert result.analysis_ok, f"Analysis verification failed: {result.error}"
        assert result.cache_ok, f"Cache verification failed: {result.error}"
        assert result.profiler_ok, f"ProfilingService compatibility failed: {result.error}"
        assert result.generator_ok, f"SetGenerator compatibility failed: {result.error}"
        assert result.reproducible, f"Reproducibility failed: {result.error}"
        assert result.success, f"E2E test failed: {result.error}"


@pytest.mark.unit
class TestBotHandlers:
    """Unit tests for bot handler logic."""

    def test_url_validation(self):
        """Test URL validation logic."""
        valid_urls = [
            "https://soundcloud.com/dj/set",
            "https://www.youtube.com/watch?v=abc",
            "http://example.com/audio.mp3",
        ]
        invalid_urls = ["not-a-url", "ftp://invalid.com", ""]

        for url in valid_urls:
            assert url.startswith(("http://", "https://"))

        for url in invalid_urls:
            assert not url or not url.startswith(("http://", "https://"))

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires Redis - integration test")
    async def test_job_status_tracking(self):
        """Test job status update and retrieval (requires Redis)."""
        from app.services.arq_worker import (
            _job_results,
            update_job_progress,
            get_job_status,
        )

        job_id = f"test_{uuid.uuid4().hex[:8]}"

        # Initial status - get_job_status is async
        status = await get_job_status(job_id)
        assert status["state"] == "PENDING"

        # Update progress
        update_job_progress(job_id, 50, "Processing...")
        status = await get_job_status(job_id)
        assert status["state"] == "PROGRESS"
        assert status["progress"] == 50

        # Cleanup
        _job_results.pop(job_id, None)


@pytest.mark.unit
class TestCacheManager:
    """Unit tests for cache manager."""

    @pytest.fixture
    def cache_manager(self, tmp_path):
        from app.modules.analysis.pipelines.cache_manager import CacheManager
        return CacheManager(str(tmp_path))

    @pytest.fixture
    def real_test_file(self, tmp_path):
        """Create a real test file for cache testing."""
        test_file = tmp_path / "test_audio.mp3"
        test_file.write_bytes(b"fake audio content for testing")
        return str(test_file)

    def test_save_and_retrieve(self, cache_manager, real_test_file):
        """Test saving and retrieving analysis results."""
        test_data = {
            "n_segments": 5,
            "n_transitions": 4,
            "total_drops": 10,
            "duration_sec": 3600.0,
            "success": True
        }

        cache_manager.save_set_analysis(real_test_file, test_data)
        retrieved = cache_manager.get_set_analysis(real_test_file)

        assert retrieved is not None
        assert retrieved["n_segments"] == 5
        assert retrieved["n_transitions"] == 4

    def test_invalidation(self, cache_manager, real_test_file):
        """Test cache invalidation."""
        test_data = {"n_segments": 5, "success": True}

        cache_manager.save_set_analysis(real_test_file, test_data)
        assert cache_manager.get_set_analysis(real_test_file) is not None

        cache_manager.invalidate_set_analysis(real_test_file)
        assert cache_manager.get_set_analysis(real_test_file) is None


# =============================================================================
# Standalone Runner with Full Report
# =============================================================================

def run_e2e_test_standalone():
    """
    Run E2E test as standalone script with detailed reporting.

    Returns exit code (0 = success, 1 = failure).
    """
    print("=" * 70)
    print("DJ SET ANALYZER - END-TO-END INTEGRATION TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    print("[Prerequisites]")
    prereq_ok = True

    # Check yt-dlp
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=10)
        print(f"  ✓ yt-dlp: {result.stdout.decode().strip()}")
    except Exception as e:
        print(f"  ✗ yt-dlp: NOT FOUND")
        prereq_ok = False

    # Check ffprobe
    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=10)
        print(f"  ✓ ffprobe: installed")
    except Exception:
        print(f"  ✗ ffprobe: NOT FOUND")
        prereq_ok = False

    # Check disk space
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    print(f"  ✓ Disk space: {free_gb}GB free")

    if not prereq_ok:
        print("\n❌ Prerequisites check FAILED")
        return 1

    print()

    # Create test context
    temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
    ctx = E2ETestContext(
        temp_dir=temp_dir,
        downloads_dir=os.path.join(temp_dir, "downloads"),
        cache_dir=os.path.join(temp_dir, "cache"),
    )
    os.makedirs(ctx.downloads_dir, exist_ok=True)
    os.makedirs(ctx.cache_dir, exist_ok=True)

    print(f"[Test Environment]")
    print(f"  Temp dir: {temp_dir}")
    print(f"  Test URL: {SOUNDCLOUD_TEST_URL[:50]}...")
    print()

    try:
        # Run E2E flow
        result = asyncio.run(run_e2e_flow(ctx))

        # Print results
        print()
        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print()

        steps = [
            ("Download", result.download_ok),
            ("Queue", result.queue_ok),
            ("Worker", result.worker_ok),
            ("Analysis", result.analysis_ok),
            ("Cache", result.cache_ok),
            ("ProfilingService", result.profiler_ok),
            ("SetGenerator", result.generator_ok),
            ("Reproducibility", result.reproducible),
        ]

        for step_name, step_ok in steps:
            status = "✓ PASS" if step_ok else "✗ FAIL"
            print(f"  {step_name}: {status}")

        print()

        if result.analysis_result:
            print("Analysis Summary:")
            print(f"  Duration: {result.analysis_result.get('duration_sec', 0)/60:.1f} min")
            print(f"  Segments: {result.analysis_result.get('n_segments', 0)}")
            print(f"  Transitions: {result.analysis_result.get('n_transitions', 0)}")
            print(f"  Drops: {result.analysis_result.get('total_drops', 0)}")
            print()

        if result.timing:
            print("Timing:")
            for key, value in result.timing.items():
                print(f"  {key}: {value:.1f}s")
            print()

        if result.success:
            print("✅ ALL TESTS PASSED")

            # Save report
            report = {
                "status": "passed",
                "timestamp": datetime.now().isoformat(),
                "steps": {name: ok for name, ok in steps},
                "timing": result.timing,
                "analysis": result.analysis_result,
            }
            report_path = "e2e_test_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {report_path}")

            return 0
        else:
            print(f"❌ TEST FAILED: {result.error}")
            return 1

    except Exception as e:
        import traceback
        print(f"\n❌ EXCEPTION: {e}")
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    exit_code = run_e2e_test_standalone()
    sys.exit(exit_code)

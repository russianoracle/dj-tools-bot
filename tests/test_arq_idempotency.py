"""
Idempotency tests for ARQ worker tasks.

Ensures same URL → same job_id → no duplicate analysis.
"""

import pytest
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch


class TestEnqueueIdempotency:
    """Test idempotent job enqueueing for URL analysis."""

    @pytest.mark.asyncio
    async def test_same_url_generates_same_job_id(self):
        """Same URL should generate identical job_id (idempotent)."""
        from app.services.arq_worker import enqueue_download_and_analyze

        url = "https://soundcloud.com/dj/techno-set-2024"
        user_id = 12345

        # Mock Redis pool and job
        mock_job = MagicMock()
        expected_job_id = f"url-{hashlib.md5(url.encode()).hexdigest()}"
        mock_job.job_id = expected_job_id

        mock_pool = AsyncMock()
        mock_pool.enqueue_job = AsyncMock(return_value=mock_job)

        with patch("app.services.arq_worker.get_redis_pool", return_value=mock_pool):
            # First enqueue
            job_id_1 = await enqueue_download_and_analyze(url, user_id)

            # Second enqueue (same URL)
            job_id_2 = await enqueue_download_and_analyze(url, user_id)

            # CRITICAL: Same URL must produce same job_id
            assert job_id_1 == job_id_2
            assert job_id_1 == expected_job_id

            # Verify enqueue_job was called with deterministic _job_id
            calls = mock_pool.enqueue_job.call_args_list
            assert len(calls) == 2

            # Check first call
            assert calls[0].args == ("download_and_analyze_task", url, user_id)
            assert calls[0].kwargs["_job_id"] == expected_job_id

            # Check second call (same parameters)
            assert calls[1].args == ("download_and_analyze_task", url, user_id)
            assert calls[1].kwargs["_job_id"] == expected_job_id

    @pytest.mark.asyncio
    async def test_different_urls_generate_different_job_ids(self):
        """Different URLs should generate different job_ids."""
        from app.services.arq_worker import enqueue_download_and_analyze

        url1 = "https://soundcloud.com/dj/set-1"
        url2 = "https://soundcloud.com/dj/set-2"
        user_id = 12345

        # Mock Redis pool
        def create_mock_job(job_id):
            mock_job = MagicMock()
            mock_job.job_id = job_id
            return mock_job

        mock_pool = AsyncMock()

        # Mock enqueue_job to return job with provided _job_id
        async def mock_enqueue(*args, **kwargs):
            return create_mock_job(kwargs["_job_id"])

        mock_pool.enqueue_job = mock_enqueue

        with patch("app.services.arq_worker.get_redis_pool", return_value=mock_pool):
            job_id_1 = await enqueue_download_and_analyze(url1, user_id)
            job_id_2 = await enqueue_download_and_analyze(url2, user_id)

            # Different URLs must produce different job_ids
            assert job_id_1 != job_id_2

            # Verify expected format
            expected_1 = f"url-{hashlib.md5(url1.encode()).hexdigest()}"
            expected_2 = f"url-{hashlib.md5(url2.encode()).hexdigest()}"

            assert job_id_1 == expected_1
            assert job_id_2 == expected_2

    @pytest.mark.asyncio
    async def test_url_normalization_before_hashing(self):
        """URLs should be hashed as-is (no normalization yet)."""
        from app.services.arq_worker import enqueue_download_and_analyze

        # These URLs are different (trailing slash, query params)
        url1 = "https://soundcloud.com/dj/set"
        url2 = "https://soundcloud.com/dj/set/"
        url3 = "https://soundcloud.com/dj/set?utm_source=copy"

        user_id = 12345

        mock_pool = AsyncMock()

        def create_mock_job(job_id):
            mock_job = MagicMock()
            mock_job.job_id = job_id
            return mock_job

        async def mock_enqueue(*args, **kwargs):
            return create_mock_job(kwargs["_job_id"])

        mock_pool.enqueue_job = mock_enqueue

        with patch("app.services.arq_worker.get_redis_pool", return_value=mock_pool):
            job_id_1 = await enqueue_download_and_analyze(url1, user_id)
            job_id_2 = await enqueue_download_and_analyze(url2, user_id)
            job_id_3 = await enqueue_download_and_analyze(url3, user_id)

            # Currently, no normalization → different job_ids
            # (Future enhancement: normalize URLs before hashing)
            assert job_id_1 != job_id_2
            assert job_id_1 != job_id_3
            assert job_id_2 != job_id_3

    @pytest.mark.asyncio
    async def test_job_id_format(self):
        """Job ID should follow url-{hash} format."""
        from app.services.arq_worker import enqueue_download_and_analyze

        url = "https://soundcloud.com/test/set"
        user_id = 123

        mock_pool = AsyncMock()
        mock_job = MagicMock()
        mock_job.job_id = f"url-{hashlib.md5(url.encode()).hexdigest()}"
        mock_pool.enqueue_job = AsyncMock(return_value=mock_job)

        with patch("app.services.arq_worker.get_redis_pool", return_value=mock_pool):
            job_id = await enqueue_download_and_analyze(url, user_id)

            # Verify format: url-{32-char-hex}
            assert job_id.startswith("url-")
            assert len(job_id) == 4 + 32  # "url-" + MD5 hex
            assert all(c in "0123456789abcdef-" for c in job_id[4:])

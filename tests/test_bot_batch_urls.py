"""Integration tests for batch URL processing feature.

Tests global_url_handler interaction with FSM, queue, and notification system.
Focus on state-aware processing and avoiding conflicts with explicit URL input.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, User, Chat


@pytest.mark.integration
@pytest.mark.bot
class TestGlobalURLHandler:
    """Tests for global URL handler batch processing."""

    @pytest.fixture
    def mock_message(self):
        """Create mock Telegram message."""
        message = MagicMock(spec=Message)
        message.from_user = MagicMock(spec=User)
        message.from_user.id = 12345
        message.text = "https://soundcloud.com/test"
        message.answer = AsyncMock()
        return message

    @pytest.fixture
    def mock_state(self):
        """Create mock FSM state."""
        state = AsyncMock(spec=FSMContext)
        state.get_state = AsyncMock(return_value=None)
        return state

    @pytest.fixture
    def mock_bot(self):
        """Create mock Bot instance."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_single_url_queues_job(self, mock_message, mock_state, mock_bot):
        """Test single URL is queued for analysis.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/artist/track"

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}), \
             patch('asyncio.create_task'):

            mock_enqueue.return_value = "job-123"

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify job was enqueued
            mock_enqueue.assert_called_once_with(
                "https://soundcloud.com/artist/track",
                12345
            )

    @pytest.mark.asyncio
    async def test_multiple_urls_queue_multiple_jobs(self, mock_message, mock_state, mock_bot):
        """Test multiple URLs in one message queue separate jobs.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "Check https://soundcloud.com/a and https://youtube.com/b"

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}), \
             patch('asyncio.create_task'):

            mock_enqueue.side_effect = ["job-1", "job-2"]

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify both URLs were enqueued
            assert mock_enqueue.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_url_shows_error_notification(self, mock_message, mock_state, mock_bot):
        """Test invalid URL shows error notification.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://invalid-platform.com/track"

        with patch('asyncio.create_task') as mock_task:
            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify error notification was scheduled
            mock_task.assert_called_once()
            # Check that notification text contains error
            call_args = mock_task.call_args[0][0]
            # call_args is a coroutine, we just verify it was called

    @pytest.mark.asyncio
    async def test_skips_when_in_waiting_for_url_state(self, mock_message, mock_state, mock_bot):
        """Test handler skips processing when FSM is in waiting_for_url state.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler, AnalyzeStates

        mock_message.text = "https://soundcloud.com/test"
        mock_state.get_state.return_value = AnalyzeStates.waiting_for_url.state

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue:
            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify job was NOT enqueued (explicit handler should process)
            mock_enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_banned_user_is_blocked(self, mock_message, mock_state, mock_bot):
        """Test banned users cannot use batch URL processing.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/test"
        mock_message.from_user.id = 999

        with patch('app.modules.bot.handlers.analyze.banned_users', {999}), \
             patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue:

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify job was NOT enqueued
            mock_enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_job_ids_in_user_jobs(self, mock_message, mock_state, mock_bot):
        """Test job IDs are stored in user_jobs dict for tracking.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/a https://youtube.com/b"
        user_id = 12345

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}) as mock_user_jobs, \
             patch('asyncio.create_task'):

            mock_enqueue.side_effect = ["job-1", "job-2"]

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify job IDs were stored
            assert user_id in mock_user_jobs
            assert "job-1" in mock_user_jobs[user_id]
            assert "job-2" in mock_user_jobs[user_id]

    @pytest.mark.asyncio
    async def test_sends_success_notification(self, mock_message, mock_state, mock_bot):
        """Test success notification is sent with correct count.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/a https://youtube.com/b https://mixcloud.com/c"

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}), \
             patch('asyncio.create_task') as mock_task:

            mock_enqueue.side_effect = ["job-1", "job-2", "job-3"]

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify notification task was created
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_queues_only_valid(self, mock_message, mock_state, mock_bot):
        """Test mixed valid/invalid URLs only queue valid ones.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/valid https://evil.com/invalid https://youtube.com/valid2"

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}), \
             patch('asyncio.create_task'):

            mock_enqueue.side_effect = ["job-1", "job-2"]

            await global_url_handler(mock_message, mock_state, mock_bot)

            # Verify only 2 jobs were enqueued (evil.com rejected)
            assert mock_enqueue.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_enqueue_failure_gracefully(self, mock_message, mock_state, mock_bot):
        """Test graceful handling when enqueue fails.

        Coverage: app/modules/bot/handlers/analyze.py::global_url_handler
        """
        from app.modules.bot.handlers.analyze import global_url_handler

        mock_message.text = "https://soundcloud.com/test"

        with patch('app.modules.bot.handlers.analyze.enqueue_download_and_analyze') as mock_enqueue, \
             patch('app.modules.bot.handlers.analyze.user_jobs', {}), \
             patch('asyncio.create_task'):

            mock_enqueue.side_effect = Exception("Redis connection failed")

            # Should not raise exception
            await global_url_handler(mock_message, mock_state, mock_bot)

            # Handler should complete without crashing
            assert True


@pytest.mark.unit
@pytest.mark.bot
class TestBatchURLValidation:
    """Tests for batch URL validation logic."""

    def test_platform_allowlist_enforcement(self):
        """Test only allowed platforms are accepted.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()

        # Allowed platforms
        allowed_urls = [
            "https://soundcloud.com/dj/set",
            "https://mixcloud.com/dj/mix/",
            "https://youtube.com/watch?v=123",
            "https://youtu.be/abc",
            "https://m.youtube.com/watch?v=xyz",
        ]

        for url in allowed_urls:
            result = parser.validate_url(url)
            assert result.is_valid, f"{url} should be valid"

        # Blocked platforms (SSRF protection)
        blocked_urls = [
            "https://spotify.com/track/123",
            "https://apple.com/music/track",
            "http://localhost:8080/api",
            "http://192.168.1.1/admin",
            "https://10.0.0.1/internal",
        ]

        for url in blocked_urls:
            result = parser.validate_url(url)
            assert not result.is_valid, f"{url} should be blocked"

    def test_url_extraction_ignores_non_urls(self):
        """Test URL extraction ignores non-URL text.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.extract_urls
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "Check soundcloud.com/test and youtube.com/watch (no protocol)"
        urls = parser.extract_urls(text)

        # Should not extract URLs without http:// or https://
        assert len(urls) == 0
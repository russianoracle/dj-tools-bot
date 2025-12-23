"""Tests for correlation ID middleware and context management."""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Update, Message, User, CallbackQuery, InlineQuery

from app.common.logging.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    get_user_id,
    set_user_id,
    get_job_id,
    set_job_id,
    correlation_id_var,
    user_id_var,
    job_id_var,
    CorrelationMiddleware,
    CorrelationLogFilter,
)


class TestCorrelationIDGeneration:
    """Test correlation ID generation and validation."""

    def test_generate_correlation_id_format(self):
        """Test correlation ID format (8 chars from UUID)."""
        cid = generate_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 8
        assert all(c in "0123456789abcdef-" for c in cid)

    def test_generate_correlation_id_unique(self):
        """Test that generated IDs are unique."""
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100  # All unique


class TestContextManagement:
    """Test context variable management."""

    def test_correlation_id_get_set(self):
        """Test correlation ID get/set operations."""
        assert get_correlation_id() is None
        set_correlation_id("test123")
        assert get_correlation_id() == "test123"
        correlation_id_var.set(None)

    def test_user_id_get_set(self):
        """Test user ID get/set operations."""
        assert get_user_id() is None
        set_user_id(12345)
        assert get_user_id() == 12345
        user_id_var.set(None)

    def test_job_id_get_set(self):
        """Test job ID get/set operations."""
        assert get_job_id() is None
        set_job_id("job-abc")
        assert get_job_id() == "job-abc"
        job_id_var.set(None)

    def test_context_isolation(self):
        """Test context vars are isolated between tasks."""
        async def task1():
            set_correlation_id("task1")
            await asyncio.sleep(0.01)
            assert get_correlation_id() == "task1"

        async def task2():
            set_correlation_id("task2")
            await asyncio.sleep(0.01)
            assert get_correlation_id() == "task2"

        async def run():
            await asyncio.gather(task1(), task2())

        asyncio.run(run())


class TestCorrelationMiddleware:
    """Test CorrelationMiddleware integration."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        return CorrelationMiddleware()

    @pytest.fixture
    def mock_handler(self):
        """Create mock async handler."""
        return AsyncMock(return_value=None)

    @pytest.mark.asyncio
    async def test_middleware_sets_correlation_id(self, middleware, mock_handler):
        """Test middleware sets correlation ID in context."""
        update = MagicMock(spec=Update, message=None, callback_query=None, inline_query=None)
        data = {}

        await middleware(mock_handler, update, data)

        # Verify correlation ID was set in data
        assert "correlation_id" in data
        assert isinstance(data["correlation_id"], str)
        assert len(data["correlation_id"]) == 8

    @pytest.mark.asyncio
    async def test_middleware_extracts_user_from_message(self, middleware, mock_handler):
        """Test middleware extracts user ID from message."""
        user = MagicMock(spec=User, id=123456)
        message = MagicMock(spec=Message, from_user=user)
        update = MagicMock(spec=Update, message=message, callback_query=None, inline_query=None)
        data = {}

        # Set up context to track changes
        original_user_id = get_user_id()

        await middleware(mock_handler, update, data)

        # Handler should have been called
        mock_handler.assert_called_once_with(update, data)

    @pytest.mark.asyncio
    async def test_middleware_extracts_user_from_callback_query(self, middleware, mock_handler):
        """Test middleware extracts user ID from callback query."""
        user = MagicMock(spec=User, id=654321)
        callback = MagicMock(spec=CallbackQuery, from_user=user)
        update = MagicMock(spec=Update, message=None, callback_query=callback, inline_query=None)
        data = {}

        await middleware(mock_handler, update, data)

        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_extracts_user_from_inline_query(self, middleware, mock_handler):
        """Test middleware extracts user ID from inline query."""
        user = MagicMock(spec=User, id=789012)
        inline = MagicMock(spec=InlineQuery, from_user=user)
        update = MagicMock(spec=Update, message=None, callback_query=None, inline_query=inline)
        data = {}

        await middleware(mock_handler, update, data)

        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_clears_context_after_handler(self, middleware, mock_handler):
        """Test middleware clears context variables after handler."""
        update = MagicMock(spec=Update, message=None, callback_query=None, inline_query=None)
        data = {}

        await middleware(mock_handler, update, data)

        # Context should be cleared
        assert get_correlation_id() is None
        assert get_user_id() is None
        assert get_job_id() is None

    @pytest.mark.asyncio
    async def test_middleware_clears_context_on_exception(self, middleware):
        """Test middleware clears context even when handler raises."""
        handler = AsyncMock(side_effect=ValueError("test error"))
        update = MagicMock(spec=Update, message=None, callback_query=None, inline_query=None)
        data = {}

        with pytest.raises(ValueError, match="test error"):
            await middleware(handler, update, data)

        # Context should still be cleared
        assert get_correlation_id() is None
        assert get_user_id() is None
        assert get_job_id() is None

    @pytest.mark.asyncio
    async def test_middleware_handles_non_update_events(self, middleware, mock_handler):
        """Test middleware handles non-Update TelegramObject gracefully."""
        event = MagicMock()  # Not an Update
        data = {}

        await middleware(mock_handler, event, data)

        # Should still set correlation ID
        assert "correlation_id" in data
        mock_handler.assert_called_once()


class TestCorrelationLogFilter:
    """Test CorrelationLogFilter for logging integration."""

    @pytest.fixture
    def log_filter(self):
        """Create log filter instance."""
        return CorrelationLogFilter()

    @pytest.fixture
    def log_record(self):
        """Create mock log record."""
        return MagicMock(spec=logging.LogRecord)

    def test_filter_adds_correlation_id(self, log_filter, log_record):
        """Test filter adds correlation_id to log record."""
        set_correlation_id("abc123")

        result = log_filter.filter(log_record)

        assert result is True
        assert log_record.correlation_id == "abc123"

        correlation_id_var.set(None)

    def test_filter_adds_user_id(self, log_filter, log_record):
        """Test filter adds user_id to log record."""
        set_user_id(99999)

        result = log_filter.filter(log_record)

        assert result is True
        assert log_record.user_id == 99999

        user_id_var.set(None)

    def test_filter_adds_job_id(self, log_filter, log_record):
        """Test filter adds job_id to log record."""
        set_job_id("job-xyz")

        result = log_filter.filter(log_record)

        assert result is True
        assert log_record.job_id == "job-xyz"

        job_id_var.set(None)

    def test_filter_handles_none_values(self, log_filter, log_record):
        """Test filter handles None context values."""
        correlation_id_var.set(None)
        user_id_var.set(None)
        job_id_var.set(None)

        result = log_filter.filter(log_record)

        assert result is True
        assert log_record.correlation_id is None
        assert log_record.user_id is None
        assert log_record.job_id is None

    def test_filter_with_all_context_set(self, log_filter, log_record):
        """Test filter with all context variables set."""
        set_correlation_id("full-test")
        set_user_id(11111)
        set_job_id("job-full")

        result = log_filter.filter(log_record)

        assert result is True
        assert log_record.correlation_id == "full-test"
        assert log_record.user_id == 11111
        assert log_record.job_id == "job-full"

        # Cleanup
        correlation_id_var.set(None)
        user_id_var.set(None)
        job_id_var.set(None)

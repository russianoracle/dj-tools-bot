"""Unit tests for bot services - URLParser and notification helpers.

Tests batch URL processing services without Telegram API dependencies.
Focus on URL extraction, validation, platform detection, and notification logic.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.unit
@pytest.mark.bot
class TestURLParser:
    """Tests for URLParser service."""

    def test_extract_single_url(self):
        """Test extracting single URL from text.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.extract_urls
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "Check out https://soundcloud.com/dj/set-name"
        urls = parser.extract_urls(text)

        assert len(urls) == 1
        assert urls[0] == "https://soundcloud.com/dj/set-name"

    def test_extract_multiple_urls(self):
        """Test extracting multiple URLs from single message.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.extract_urls
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "Listen: https://soundcloud.com/a and https://youtube.com/b also http://mixcloud.com/c"
        urls = parser.extract_urls(text)

        assert len(urls) == 3
        assert "https://soundcloud.com/a" in urls
        assert "https://youtube.com/b" in urls
        assert "http://mixcloud.com/c" in urls

    def test_extract_no_urls(self):
        """Test text without URLs returns empty list.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.extract_urls
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "Just regular text without any links"
        urls = parser.extract_urls(text)

        assert len(urls) == 0

    def test_validate_soundcloud_url(self):
        """Test SoundCloud URL validation.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://soundcloud.com/artist/track-name")

        assert result.is_valid is True
        assert result.platform == "SoundCloud"
        assert result.error is None

    def test_validate_youtube_url(self):
        """Test YouTube URL validation.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://youtube.com/watch?v=abc123")

        assert result.is_valid is True
        assert result.platform == "YouTube"
        assert result.error is None

    def test_validate_youtu_be_short_url(self):
        """Test YouTube short URL (youtu.be) validation.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://youtu.be/abc123")

        assert result.is_valid is True
        assert result.platform == "YouTube"

    def test_validate_mixcloud_url(self):
        """Test Mixcloud URL validation.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://mixcloud.com/dj/mix-name/")

        assert result.is_valid is True
        assert result.platform == "Mixcloud"

    def test_validate_www_prefix_removed(self):
        """Test www. prefix is correctly stripped.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://www.soundcloud.com/artist/track")

        assert result.is_valid is True
        assert result.platform == "SoundCloud"

    def test_validate_invalid_domain(self):
        """Test invalid domain is rejected (SSRF protection).

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("https://evil.com/malicious")

        assert result.is_valid is False
        assert result.platform == "Unknown"
        assert "Unsupported platform" in result.error

    def test_validate_localhost_rejected(self):
        """Test localhost/internal IPs are rejected (SSRF protection).

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()

        # Localhost
        result = parser.validate_url("http://localhost:8080/api")
        assert result.is_valid is False

        # Internal IP
        result = parser.validate_url("http://192.168.1.1/admin")
        assert result.is_valid is False

    def test_validate_malformed_url(self):
        """Test malformed URL returns error.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.validate_url
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        result = parser.validate_url("not-a-valid-url")

        assert result.is_valid is False
        # URL without protocol is treated as unsupported platform
        assert result.error is not None

    def test_parse_batch_mixed_valid_invalid(self):
        """Test parsing batch with mix of valid and invalid URLs.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.parse_batch
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "Check https://soundcloud.com/a and https://evil.com/b and https://youtube.com/c"
        results = parser.parse_batch(text)

        assert len(results) == 3

        valid_results = [r for r in results if r.is_valid]
        invalid_results = [r for r in results if not r.is_valid]

        assert len(valid_results) == 2
        assert len(invalid_results) == 1

    def test_parse_batch_all_valid(self):
        """Test parsing batch with all valid URLs.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.parse_batch
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        text = "https://soundcloud.com/a https://youtube.com/b https://mixcloud.com/c"
        results = parser.parse_batch(text)

        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_parse_batch_empty_text(self):
        """Test parsing empty text returns empty list.

        Coverage: app/modules/bot/services/url_parser.py::URLParser.parse_batch
        """
        from app.modules.bot.services.url_parser import URLParser

        parser = URLParser()
        results = parser.parse_batch("")

        assert len(results) == 0


@pytest.mark.unit
@pytest.mark.bot
class TestNotificationHelper:
    """Tests for auto-delete notification helper."""

    @pytest.mark.asyncio
    async def test_send_notification_creates_message(self):
        """Test notification is sent via message.answer().

        Coverage: app/modules/bot/services/notifications.py::send_auto_delete_notification
        """
        from app.modules.bot.services.notifications import send_auto_delete_notification

        # Mock message
        mock_message = AsyncMock()
        mock_notification = AsyncMock()
        mock_message.answer.return_value = mock_notification

        # Don't wait for delete
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await send_auto_delete_notification(
                mock_message,
                "Test notification",
                delete_after=0
            )

        # Verify answer was called
        mock_message.answer.assert_called_once_with(
            "Test notification",
            parse_mode="HTML"
        )

    @pytest.mark.asyncio
    async def test_send_notification_deletes_after_delay(self):
        """Test notification is deleted after specified delay.

        Coverage: app/modules/bot/services/notifications.py::send_auto_delete_notification
        """
        from app.modules.bot.services.notifications import send_auto_delete_notification

        # Mock message and notification
        mock_message = AsyncMock()
        mock_notification = AsyncMock()
        mock_message.answer.return_value = mock_notification

        # Track sleep call
        sleep_mock = AsyncMock()

        with patch('asyncio.sleep', sleep_mock):
            await send_auto_delete_notification(
                mock_message,
                "Test",
                delete_after=5
            )

        # Verify sleep with correct delay
        sleep_mock.assert_called_once_with(5)

        # Verify delete was called
        mock_notification.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_handles_delete_error(self):
        """Test notification handles delete errors gracefully.

        Coverage: app/modules/bot/services/notifications.py::send_auto_delete_notification
        """
        from app.modules.bot.services.notifications import send_auto_delete_notification

        # Mock message with delete error
        mock_message = AsyncMock()
        mock_notification = AsyncMock()
        mock_notification.delete.side_effect = Exception("Already deleted")
        mock_message.answer.return_value = mock_notification

        # Should not raise exception
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await send_auto_delete_notification(
                mock_message,
                "Test",
                delete_after=0
            )

        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_send_notification_custom_parse_mode(self):
        """Test notification respects custom parse mode.

        Coverage: app/modules/bot/services/notifications.py::send_auto_delete_notification
        """
        from app.modules.bot.services.notifications import send_auto_delete_notification

        mock_message = AsyncMock()
        mock_notification = AsyncMock()
        mock_message.answer.return_value = mock_notification

        with patch('asyncio.sleep', new_callable=AsyncMock):
            await send_auto_delete_notification(
                mock_message,
                "Test",
                delete_after=0,
                parse_mode="Markdown"
            )

        # Verify parse_mode was passed
        mock_message.answer.assert_called_once_with(
            "Test",
            parse_mode="Markdown"
        )
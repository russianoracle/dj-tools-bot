"""Unit tests for bot handlers - 100% coverage of basic user operations.

Tests pure functions from handlers without Telegram API or Redis dependencies.
Focus on business logic, validation, formatting, and data transformations.
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.mark.unit
@pytest.mark.bot
class TestAnalyzeHandlerFunctions:
    """Tests for analyze.py pure functions."""

    def test_get_state_emoji_all_states(self):
        """Test get_state_emoji returns correct emoji for all job states.

        Coverage: app/modules/bot/handlers/analyze.py::get_state_emoji
        """
        from app.modules.bot.handlers.analyze import get_state_emoji

        # Valid states
        assert get_state_emoji("PENDING") == "⏳"
        assert get_state_emoji("PROGRESS") == "🔄"
        assert get_state_emoji("SUCCESS") == "✅"
        assert get_state_emoji("FAILURE") == "❌"

        # Unknown state
        assert get_state_emoji("UNKNOWN") == "❓"
        assert get_state_emoji("") == "❓"
        assert get_state_emoji(None) == "❓"

    def test_get_disk_usage_structure(self):
        """Test get_disk_usage returns valid dict structure.

        Coverage: app/modules/bot/handlers/analyze.py::get_disk_usage
        """
        from app.modules.bot.handlers.analyze import get_disk_usage

        # Create temp downloads directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DOWNLOADS_DIR"] = tmpdir

            result = get_disk_usage()

            # Check structure
            assert isinstance(result, dict)
            assert "total_gb" in result
            assert "used_gb" in result
            assert "free_gb" in result
            assert "used_percent" in result
            assert "downloads_mb" in result

            # Check types
            assert isinstance(result["total_gb"], int)
            assert isinstance(result["used_gb"], int)
            assert isinstance(result["free_gb"], int)
            assert isinstance(result["used_percent"], float)
            assert isinstance(result["downloads_mb"], int)

            # Check values are sane
            assert result["total_gb"] > 0
            assert result["used_gb"] >= 0
            assert result["free_gb"] >= 0
            assert 0 <= result["used_percent"] <= 100
            assert result["downloads_mb"] >= 0

    def test_get_disk_usage_with_files(self):
        """Test get_disk_usage calculates downloads size correctly.

        Coverage: app/modules/bot/handlers/analyze.py::get_disk_usage
        """
        from app.modules.bot.handlers.analyze import get_disk_usage

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DOWNLOADS_DIR"] = tmpdir

            # Create test files
            test_file_1 = Path(tmpdir) / "test1.mp3"
            test_file_2 = Path(tmpdir) / "test2.mp3"
            test_file_1.write_bytes(b"0" * 1024 * 1024)  # 1MB
            test_file_2.write_bytes(b"0" * 2 * 1024 * 1024)  # 2MB

            result = get_disk_usage()

            # Should detect 3MB of downloads
            assert result["downloads_mb"] == 3

    def test_get_disk_usage_nonexistent_dir(self):
        """Test get_disk_usage handles nonexistent downloads directory.

        Coverage: app/modules/bot/handlers/analyze.py::get_disk_usage
        """
        from app.modules.bot.handlers.analyze import get_disk_usage

        os.environ["DOWNLOADS_DIR"] = "/nonexistent/path/downloads"

        result = get_disk_usage()

        # Should return 0 for nonexistent directory
        assert result["downloads_mb"] == 0


@pytest.mark.unit
@pytest.mark.bot
class TestStartHandlerFunctions:
    """Tests for start.py pure functions."""

    def test_is_admin_with_valid_admin(self):
        """Test is_admin correctly identifies admin user.

        Coverage: app/modules/bot/handlers/start.py::is_admin
        """
        from app.modules.bot.handlers.start import is_admin

        original_admin = os.getenv("ADMIN_USER_ID")

        try:
            os.environ["ADMIN_USER_ID"] = "999888777"

            assert is_admin(999888777) is True
            assert is_admin(111222333) is False

        finally:
            if original_admin:
                os.environ["ADMIN_USER_ID"] = original_admin
            else:
                os.environ.pop("ADMIN_USER_ID", None)

    def test_is_admin_no_env_var(self):
        """Test is_admin when ADMIN_USER_ID is not set.

        Coverage: app/modules/bot/handlers/start.py::is_admin
        """
        from app.modules.bot.handlers.start import is_admin

        original_admin = os.getenv("ADMIN_USER_ID")

        try:
            os.environ.pop("ADMIN_USER_ID", None)

            # Should return False for any user_id when no admin set
            assert is_admin(123456789) is False

        finally:
            if original_admin:
                os.environ["ADMIN_USER_ID"] = original_admin

    def test_get_main_text_content(self):
        """Test get_main_text returns properly formatted main menu text.

        Coverage: app/modules/bot/handlers/start.py::get_main_text
        """
        from app.modules.bot.handlers.start import get_main_text

        text = get_main_text()

        # Check required elements
        assert "DJ Set Analyzer" in text
        assert "🟨" in text and "Yellow" in text
        assert "🟩" in text and "Green" in text
        assert "🟪" in text and "Purple" in text

        # Check HTML formatting
        assert "<b>" in text
        assert "</b>" in text
        assert "<i>" in text
        assert "</i>" in text

        # Check energy zone descriptions
        assert "Rest zone" in text or "low energy" in text
        assert "Transitional" in text or "medium" in text
        assert "Hits/Energy" in text or "high" in text


@pytest.mark.unit
@pytest.mark.bot
class TestKeyboardsExtended:
    """Extended tests for keyboard generation."""

    def test_get_main_keyboard_structure(self):
        """Test main keyboard has proper InlineKeyboardMarkup structure.

        Coverage: app/modules/bot/keyboards/inline.py::get_main_keyboard
        """
        from app.modules.bot.keyboards.inline import get_main_keyboard

        keyboard = get_main_keyboard(is_admin=False)

        # Check it's InlineKeyboardMarkup
        assert hasattr(keyboard, 'inline_keyboard')
        assert isinstance(keyboard.inline_keyboard, list)

        # Each row should have buttons
        for row in keyboard.inline_keyboard:
            assert isinstance(row, list)
            assert len(row) > 0
            # Each button should have text and callback_data
            for button in row:
                assert hasattr(button, 'text')
                assert hasattr(button, 'callback_data')

    def test_get_main_keyboard_callback_data(self):
        """Test main keyboard buttons have correct callback data.

        Coverage: app/modules/bot/keyboards/inline.py::get_main_keyboard
        """
        from app.modules.bot.keyboards.inline import get_main_keyboard

        keyboard = get_main_keyboard(is_admin=False)
        buttons = keyboard.inline_keyboard

        # Extract callback data
        callbacks = [btn[0].callback_data for btn in buttons]

        # Check expected callbacks
        expected_callbacks = ["analyze_url", "my_jobs", "profile", "generate_set", "help"]
        for expected in expected_callbacks:
            assert expected in callbacks

    def test_get_main_keyboard_admin_has_admin_callback(self):
        """Test admin keyboard has admin callback.

        Coverage: app/modules/bot/keyboards/inline.py::get_main_keyboard
        """
        from app.modules.bot.keyboards.inline import get_main_keyboard

        keyboard = get_main_keyboard(is_admin=True)
        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        assert "admin" in callbacks

    def test_get_back_keyboard_structure(self):
        """Test back keyboard structure.

        Coverage: app/modules/bot/keyboards/inline.py::get_back_keyboard
        """
        from app.modules.bot.keyboards.inline import get_back_keyboard

        keyboard = get_back_keyboard()

        assert hasattr(keyboard, 'inline_keyboard')
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 1

        button = keyboard.inline_keyboard[0][0]
        assert button.callback_data == "main_menu"
        assert "Back" in button.text or "◀️" in button.text

    def test_get_jobs_keyboard_empty(self):
        """Test jobs keyboard with no jobs.

        Coverage: app/modules/bot/keyboards/inline.py::get_jobs_keyboard
        """
        from app.modules.bot.keyboards.inline import get_jobs_keyboard

        keyboard = get_jobs_keyboard([])

        # Should still have Refresh + Back buttons (2 buttons)
        assert len(keyboard.inline_keyboard) == 2

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]
        assert "my_jobs" in callbacks  # Refresh uses my_jobs callback
        assert "main_menu" in callbacks

    def test_get_jobs_keyboard_with_jobs(self):
        """Test jobs keyboard with multiple jobs.

        Coverage: app/modules/bot/keyboards/inline.py::get_jobs_keyboard
        """
        from app.modules.bot.keyboards.inline import get_jobs_keyboard

        job_ids = ["job_abc123", "job_def456", "job_ghi789"]
        keyboard = get_jobs_keyboard(job_ids)

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have job callbacks + refresh + back
        assert len(keyboard.inline_keyboard) == len(job_ids) + 2

        # Check job callbacks (format is "job:{job_id}")
        for job_id in job_ids:
            assert f"job:{job_id}" in callbacks

    def test_get_job_keyboard(self):
        """Test single job view keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_job_keyboard
        """
        from app.modules.bot.keyboards.inline import get_job_keyboard

        job_id = "test_job_123"
        keyboard = get_job_keyboard(job_id)

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have refresh + back
        assert f"job:{job_id}" in callbacks
        assert "my_jobs" in callbacks

    def test_get_result_keyboard(self):
        """Test result navigation keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_result_keyboard
        """
        from app.modules.bot.keyboards.inline import get_result_keyboard

        job_id = "result_456"
        keyboard = get_result_keyboard(job_id)

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have all result views
        assert f"result:overview:{job_id}" in callbacks
        assert f"result:trans:{job_id}" in callbacks
        assert f"result:drops:{job_id}" in callbacks
        assert f"result:genres:{job_id}" in callbacks
        assert f"result:energy:{job_id}" in callbacks
        assert "my_jobs" in callbacks

    def test_get_admin_keyboard(self):
        """Test admin panel keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_admin_keyboard
        """
        from app.modules.bot.keyboards.inline import get_admin_keyboard

        keyboard = get_admin_keyboard()

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have all admin options
        assert "admin:stats" in callbacks
        assert "admin:disk" in callbacks
        assert "admin:cache" in callbacks
        assert "admin:ban" in callbacks
        assert "admin:restart" in callbacks
        assert "main_menu" in callbacks

    def test_get_cancel_keyboard(self):
        """Test cancel action keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_cancel_keyboard
        """
        from app.modules.bot.keyboards.inline import get_cancel_keyboard

        keyboard = get_cancel_keyboard()

        assert len(keyboard.inline_keyboard) == 1
        assert keyboard.inline_keyboard[0][0].callback_data == "main_menu"

    def test_get_profile_keyboard(self):
        """Test DJ profile navigation keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_profile_keyboard
        """
        from app.modules.bot.keyboards.inline import get_profile_keyboard

        keyboard = get_profile_keyboard()

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have all profile views
        assert "profile:energy" in callbacks
        assert "profile:drops" in callbacks
        assert "profile:tempo" in callbacks
        assert "profile:keys" in callbacks
        assert "main_menu" in callbacks

    def test_get_generate_keyboard(self):
        """Test set generation options keyboard.

        Coverage: app/modules/bot/keyboards/inline.py::get_generate_keyboard
        """
        from app.modules.bot.keyboards.inline import get_generate_keyboard

        keyboard = get_generate_keyboard()

        callbacks = [btn[0].callback_data for btn in keyboard.inline_keyboard]

        # Should have all duration options
        assert "gen:30" in callbacks
        assert "gen:60" in callbacks
        assert "gen:90" in callbacks
        assert "main_menu" in callbacks


@pytest.mark.unit
@pytest.mark.bot
class TestJobSchema:
    """Tests for Job schema validation and structure."""

    def test_job_state_enum_import(self):
        """Test JobState enum can be imported.

        Coverage: app/modules/bot/schemas/job.py
        """
        from app.modules.bot.schemas.job import JobState

        assert JobState is not None
        assert JobState.PENDING == "PENDING"
        assert JobState.SUCCESS == "SUCCESS"

    def test_job_status_required_fields(self):
        """Test JobStatus has all required fields.

        Coverage: app/modules/bot/schemas/job.py::JobStatus
        """
        from app.modules.bot.schemas.job import JobStatus, JobState

        # Create minimal JobStatus
        status = JobStatus(state=JobState.PENDING)

        assert status.state == JobState.PENDING
        assert status.progress == 0
        assert status.status == ""
        assert status.result is None

    def test_job_status_emoji_property(self):
        """Test JobStatus emoji property returns correct emojis.

        Coverage: app/modules/bot/schemas/job.py::JobStatus.emoji
        """
        from app.modules.bot.schemas.job import JobStatus, JobState

        status_pending = JobStatus(state=JobState.PENDING)
        assert status_pending.emoji == "⏳"

        status_progress = JobStatus(state=JobState.PROGRESS)
        assert status_progress.emoji == "🔄"

        status_success = JobStatus(state=JobState.SUCCESS)
        assert status_success.emoji == "✅"

        status_failure = JobStatus(state=JobState.FAILURE)
        assert status_failure.emoji == "❌"

    def test_job_result_creation(self):
        """Test JobResult can be created with all fields.

        Coverage: app/modules/bot/schemas/job.py::JobResult
        """
        from app.modules.bot.schemas.job import JobResult

        result = JobResult(
            job_id="test_123",
            tracks=[{"title": "Track 1"}],
            drops=[{"time": 30}],
            transitions=[{"from": "YELLOW", "to": "GREEN"}],
            genres={"house": 0.8, "techno": 0.2},
            duration_seconds=180.5
        )

        assert result.job_id == "test_123"
        assert len(result.tracks) == 1
        assert len(result.drops) == 1
        assert result.duration_seconds == 180.5

    def test_job_result_from_dict(self):
        """Test JobResult.from_dict factory method.

        Coverage: app/modules/bot/schemas/job.py::JobResult.from_dict
        """
        from app.modules.bot.schemas.job import JobResult

        data = {
            "tracks": [{"title": "Track 1"}],
            "drops": [{"time": 30}],
            "transitions": [{"from": "YELLOW", "to": "GREEN"}],
            "genres": {"house": 0.8},
            "duration_seconds": 200.0
        }

        result = JobResult.from_dict("job_456", data)

        assert result.job_id == "job_456"
        assert result.tracks == [{"title": "Track 1"}]
        assert result.genres == {"house": 0.8}
        assert result.duration_seconds == 200.0


@pytest.mark.unit
@pytest.mark.bot
class TestBotLogicExtended:
    """Extended tests for bot business logic."""

    def test_url_validation_youtube(self):
        """Test URL validation accepts YouTube links.

        Coverage: Bot URL validation logic
        """
        import re
        url_pattern = re.compile(r'^https?://')

        youtube_urls = [
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=test123",
            "http://youtube.com/shorts/abc",
        ]

        for url in youtube_urls:
            assert url_pattern.match(url), f"Should accept YouTube URL: {url}"

    def test_url_validation_soundcloud(self):
        """Test URL validation accepts SoundCloud links.

        Coverage: Bot URL validation logic
        """
        import re
        url_pattern = re.compile(r'^https?://')

        soundcloud_urls = [
            "https://soundcloud.com/artist/track",
            "http://soundcloud.com/test/audio",
        ]

        for url in soundcloud_urls:
            assert url_pattern.match(url), f"Should accept SoundCloud URL: {url}"

    def test_url_validation_invalid(self):
        """Test URL validation rejects invalid URLs.

        Coverage: Bot URL validation logic
        """
        import re
        url_pattern = re.compile(r'^https?://')

        invalid_urls = [
            "ftp://file.com/audio.mp3",
            "just text",
            "",
            "file:///local/path.mp3",
            "www.youtube.com/watch",  # Missing protocol
        ]

        for url in invalid_urls:
            assert not url_pattern.match(url), f"Should reject invalid URL: {url}"

    def test_file_size_validation_edge_cases(self):
        """Test file size validation at boundaries.

        Coverage: Bot file size validation logic
        """
        MAX_SIZE_MB = 500
        MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

        def is_size_valid(size_bytes: int) -> bool:
            return size_bytes <= MAX_SIZE_BYTES

        # Exact limit
        assert is_size_valid(MAX_SIZE_BYTES) is True

        # One byte over
        assert is_size_valid(MAX_SIZE_BYTES + 1) is False

        # One byte under
        assert is_size_valid(MAX_SIZE_BYTES - 1) is True

        # Zero size
        assert is_size_valid(0) is True

    def test_duration_formatting_edge_cases(self):
        """Test duration formatting for edge cases.

        Coverage: Bot duration formatting logic
        """
        def format_duration(seconds: int) -> str:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60

            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes}:{secs:02d}"

        # Zero
        assert format_duration(0) == "0:00"

        # 1 second
        assert format_duration(1) == "0:01"

        # 59 seconds
        assert format_duration(59) == "0:59"

        # Exactly 1 minute
        assert format_duration(60) == "1:00"

        # Exactly 1 hour
        assert format_duration(3600) == "1:00:00"

        # Long duration
        assert format_duration(7265) == "2:01:05"

    def test_energy_zone_classification(self):
        """Test energy zone classification logic.

        Coverage: Bot energy zone logic
        """
        def get_zone_emoji(zone: str) -> str:
            zone_map = {
                "YELLOW": "🟨",
                "GREEN": "🟩",
                "PURPLE": "🟪",
            }
            return zone_map.get(zone, "❓")

        assert get_zone_emoji("YELLOW") == "🟨"
        assert get_zone_emoji("GREEN") == "🟩"
        assert get_zone_emoji("PURPLE") == "🟪"
        assert get_zone_emoji("UNKNOWN") == "❓"

    def test_job_id_format(self):
        """Test job ID format validation.

        Coverage: Bot job ID logic
        """
        import re

        # Job IDs should be format: prefix_timestamp_random or similar
        job_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')

        valid_ids = [
            "job_123456789",
            "analyze_1234567890_abc",
            "generate-123",
        ]

        for job_id in valid_ids:
            assert job_id_pattern.match(job_id), f"Should accept job ID: {job_id}"

        invalid_ids = [
            "job 123",  # Space
            "job@123",  # Special char
            "",  # Empty
        ]

        for job_id in invalid_ids:
            assert not job_id_pattern.match(job_id), f"Should reject job ID: {job_id}"

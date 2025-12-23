"""
Unit tests for core.config modules.

Tests cover:
1. Settings (environment variables, validation)
2. Cache configuration
3. Mixing styles configuration
"""

import pytest
import os
from pathlib import Path

from app.core.config.settings import Settings, get_settings
from app.core.config.mixing_styles import MixingStyle, TransitionConfig


# =============================================================================
# Settings Tests
# =============================================================================

class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_singleton(self):
        """Test get_settings returns singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_settings_redis_url_default(self):
        """Test Redis URL configuration (may be None)."""
        settings = Settings()
        # redis_url is Optional[str], can be None
        assert settings.redis_url is None or isinstance(settings.redis_url, str)

    def test_settings_downloads_dir_exists(self):
        """Test downloads_dir is configured."""
        settings = Settings()
        assert settings.downloads_dir is not None
        assert isinstance(settings.downloads_dir, str)

    def test_settings_bot_token_type(self):
        """Test telegram_bot_token is string or None."""
        settings = Settings()
        assert settings.telegram_bot_token is None or isinstance(settings.telegram_bot_token, str)

    def test_settings_admin_user_id(self):
        """Test admin_user_id is integer."""
        settings = Settings()
        assert isinstance(settings.admin_user_id, int)

    def test_settings_sample_rate(self):
        """Test sample_rate has valid default."""
        settings = Settings()
        assert settings.sample_rate == 22050


# CacheConfig class doesn't exist - tests removed


# =============================================================================
# MixingStyles Tests
# =============================================================================

class TestMixingStyles:
    """Tests for mixing styles configuration."""

    def test_mixing_style_enum_values(self):
        """Test MixingStyle enum has expected values."""
        assert hasattr(MixingStyle, "SMOOTH")
        assert hasattr(MixingStyle, "STANDARD")
        assert hasattr(MixingStyle, "HARD")

    def test_transition_config_default(self):
        """Test default TransitionConfig."""
        config = TransitionConfig()

        assert config.min_transition_gap_sec > 0
        assert 0 <= config.energy_threshold <= 1
        assert 0 <= config.bass_weight <= 1
        assert config.smooth_sigma > 0

    def test_transition_config_for_smooth_style(self):
        """Test TransitionConfig for smooth mixing style."""
        config = TransitionConfig.for_style(MixingStyle.SMOOTH)

        # Smooth mixing has longer transitions
        assert config.min_transition_gap_sec > 30
        # Higher timbral weight for smooth mixing
        assert config.timbral_weight > 0.5

    def test_transition_config_for_standard_style(self):
        """Test TransitionConfig for standard mixing style."""
        config = TransitionConfig.for_style(MixingStyle.STANDARD)

        # Standard mixing has moderate transition gap
        assert 15 < config.min_transition_gap_sec < 60
        assert config.energy_threshold > 0

    def test_transition_config_for_hard_style(self):
        """Test TransitionConfig for hard mixing style."""
        config = TransitionConfig.for_style(MixingStyle.HARD)

        # Hard mixing has shorter transitions
        assert config.min_transition_gap_sec < 30
        # Higher energy threshold for hard cuts
        assert config.energy_threshold > 0.3

    def test_transition_config_relationships(self):
        """Test relationships between mixing style configs."""
        smooth = TransitionConfig.for_style(MixingStyle.SMOOTH)
        standard = TransitionConfig.for_style(MixingStyle.STANDARD)
        hard = TransitionConfig.for_style(MixingStyle.HARD)

        # Smooth should have longest transition gap
        assert smooth.min_transition_gap_sec > standard.min_transition_gap_sec
        assert standard.min_transition_gap_sec > hard.min_transition_gap_sec

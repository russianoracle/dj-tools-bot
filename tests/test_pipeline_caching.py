"""
Tests for pipeline-level caching in SetAnalysisPipeline.

Ensures idempotent analysis: same file → cached result.
"""

import pytest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.modules.analysis.pipelines.set_analysis import (
    SetAnalysisPipeline,
    SetAnalysisResult,
)


class TestPipelineCaching:
    """Test caching behavior in SetAnalysisPipeline."""

    @pytest.fixture
    def mock_cache_repo(self):
        """Mock CacheRepository for testing."""
        mock_repo = MagicMock()
        mock_repo.get_set = MagicMock(return_value=None)
        mock_repo.save_set = MagicMock()
        return mock_repo

    @pytest.fixture
    def pipeline(self, mock_cache_repo):
        """Create pipeline with mocked cache."""
        with patch('app.modules.analysis.pipelines.set_analysis.CacheRepository.get_instance', return_value=mock_cache_repo):
            pipeline = SetAnalysisPipeline(sr=22050, analyze_genres=False)
            return pipeline

    def test_analyze_saves_to_cache_on_success(self, pipeline, mock_cache_repo):
        """Successful analysis should save result to cache."""
        test_file = "/tmp/test_set.mp3"

        # Mock successful analysis
        with patch.object(pipeline, 'run') as mock_run:
            mock_context = MagicMock()
            mock_context.results = {
                '_duration': 300.0,
                '_total_time': 10.0,
                '_peak_memory_mb': 100.0,
            }
            mock_context.get_result = MagicMock(side_effect=lambda key, default=None: {
                'transitions': None,
                'segments': [],
                'drops': None,
                'genre_distribution': None,
                'energy_timeline': None,
            }.get(key, default))
            mock_run.return_value = mock_context

            result = pipeline.analyze(test_file)

            # Verify cache was saved
            assert mock_cache_repo.save_set.called
            call_args = mock_cache_repo.save_set.call_args
            saved_path = call_args[0][0]
            saved_data = call_args[0][1]

            assert os.path.abspath(test_file) == saved_path
            assert isinstance(saved_data, dict)
            assert saved_data['success'] is True

    def test_analyze_returns_cached_result(self, pipeline, mock_cache_repo):
        """Second analysis should return cached result without re-processing."""
        test_file = "/tmp/test_set.mp3"
        abs_path = os.path.abspath(test_file)

        # Mock cached data
        cached_data = {
            'file_path': abs_path,
            'file_name': 'test_set.mp3',
            'duration_sec': 300.0,
            'n_transitions': 5,
            'transition_times': [(30.0, 35.0), (60.0, 65.0)],
            'transition_density': 1.0,
            'n_segments': 6,
            'segments': [],
            'total_drops': 10,
            'drop_density': 0.033,
            'processing_time_sec': 10.0,
            'peak_memory_mb': 100.0,
            'success': True,
            'error': None,
        }
        mock_cache_repo.get_set.return_value = cached_data

        with patch.object(pipeline, 'run') as mock_run:
            result = pipeline.analyze(test_file)

            # Verify pipeline.run() was NOT called (cache hit)
            assert not mock_run.called

            # Verify result came from cache
            assert result.file_path == abs_path
            assert result.duration_sec == 300.0
            assert result.n_transitions == 5
            assert result.total_drops == 10
            assert result.success is True

    def test_force_flag_bypasses_cache(self, pipeline, mock_cache_repo):
        """force=True should skip cache and re-analyze."""
        test_file = "/tmp/test_set.mp3"

        # Mock cached data
        cached_data = {
            'file_path': os.path.abspath(test_file),
            'file_name': 'test_set.mp3',
            'duration_sec': 300.0,
            'n_transitions': 5,
            'transition_times': [],
            'transition_density': 0.0,
            'n_segments': 0,
            'segments': [],
            'total_drops': 0,
            'drop_density': 0.0,
            'success': True,
        }
        mock_cache_repo.get_set.return_value = cached_data

        # Mock successful re-analysis
        with patch.object(pipeline, 'run') as mock_run:
            mock_context = MagicMock()
            mock_context.results = {
                '_duration': 350.0,
                '_total_time': 12.0,
                '_peak_memory_mb': 110.0,
            }
            mock_context.get_result = MagicMock(side_effect=lambda key, default=None: {
                'transitions': None,
                'segments': [],
                'drops': None,
                'genre_distribution': None,
                'energy_timeline': None,
            }.get(key, default))
            mock_run.return_value = mock_context

            result = pipeline.analyze(test_file, force=True)

            # Verify pipeline.run() WAS called (cache bypassed)
            assert mock_run.called

            # Verify result is from fresh analysis, not cache
            assert result.duration_sec == 350.0  # New value

            # Verify cache was NOT checked (force mode)
            assert not mock_cache_repo.get_set.called

    def test_cache_corruption_falls_back_to_fresh_analysis(self, pipeline, mock_cache_repo):
        """Corrupted cache should trigger fresh analysis."""
        test_file = "/tmp/test_set.mp3"

        # Mock corrupted cache data (missing required fields)
        corrupted_data = {'file_path': '/tmp/test_set.mp3'}  # Incomplete
        mock_cache_repo.get_set.return_value = corrupted_data

        with patch.object(pipeline, 'run') as mock_run:
            mock_context = MagicMock()
            mock_context.results = {
                '_duration': 300.0,
                '_total_time': 10.0,
                '_peak_memory_mb': 100.0,
            }
            mock_context.get_result = MagicMock(side_effect=lambda key, default=None: {
                'transitions': None,
                'segments': [],
                'drops': None,
                'genre_distribution': None,
                'energy_timeline': None,
            }.get(key, default))
            mock_run.return_value = mock_context

            # Should fall back to fresh analysis
            result = pipeline.analyze(test_file)

            # Verify fresh analysis was performed
            assert mock_run.called

    def test_idempotent_analysis_same_path(self, pipeline, mock_cache_repo):
        """Multiple calls with same path should use cache after first successful save."""
        test_file = "/tmp/test_set.mp3"
        abs_path = os.path.abspath(test_file)

        # Track cache state
        cache_storage = {}

        def get_set_side_effect(path):
            return cache_storage.get(path)

        def save_set_side_effect(path, data):
            cache_storage[path] = data

        mock_cache_repo.get_set.side_effect = get_set_side_effect
        mock_cache_repo.save_set.side_effect = save_set_side_effect

        with patch.object(pipeline, 'run') as mock_run:
            mock_context = MagicMock()
            mock_context.results = {
                '_duration': 300.0,
                '_total_time': 10.0,
                '_peak_memory_mb': 100.0,
            }
            mock_context.get_result = MagicMock(side_effect=lambda key, default=None: {
                'transitions': None,
                'segments': [],
                'drops': None,
                'genre_distribution': None,
                'energy_timeline': None,
            }.get(key, default))
            mock_run.return_value = mock_context

            # First call: cache miss → fresh analysis → save to cache
            result1 = pipeline.analyze(test_file)
            assert mock_run.call_count == 1
            assert result1.success is True
            assert abs_path in cache_storage  # Verify saved

            # Second call: cache hit → no analysis
            result2 = pipeline.analyze(test_file)
            assert mock_run.call_count == 1  # Still 1 (not called again)
            assert result2.success is True

            # Results should be identical (idempotent)
            assert result1.duration_sec == result2.duration_sec
            assert result1.file_path == result2.file_path

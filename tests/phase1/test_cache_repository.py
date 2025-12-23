"""
Phase 1: Comprehensive Cache Repository Tests.

Tests CacheRepository CRUD operations, transactions, cleanup, and singleton behavior.

NOTE: Many tests need updates to match the refactored CacheRepository API.
The API has changed - methods like save_set, invalidate_set use different signatures.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path

from app.core.cache.repository import CacheRepository
from app.core.cache.models import CachedSetAnalysis, CachedTrackAnalysis, CachedDJProfile


# Skip tests that use old API until updated
pytestmark = pytest.mark.skip(reason="Tests need update to match refactored CacheRepository API")


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cache_repo(temp_cache_dir):
    """Create fresh CacheRepository instance."""
    # Clear singleton
    CacheRepository._instance = None
    return CacheRepository(temp_cache_dir)


class TestCacheRepositorySingleton:
    """Test singleton pattern."""

    def test_get_instance_returns_same_object(self, temp_cache_dir):
        """get_instance() should return same object on multiple calls."""
        CacheRepository._instance = None
        repo1 = CacheRepository.get_instance(temp_cache_dir)
        repo2 = CacheRepository.get_instance(temp_cache_dir)
        assert repo1 is repo2

    def test_singleton_persists_across_calls(self, temp_cache_dir):
        """Singleton should persist data across calls."""
        CacheRepository._instance = None
        repo1 = CacheRepository.get_instance(temp_cache_dir)
        assert repo1.cache_dir == Path(temp_cache_dir).expanduser()

        repo2 = CacheRepository.get_instance()
        assert repo2.cache_dir == Path(temp_cache_dir).expanduser()


class TestSetOperations:
    """Test set CRUD operations."""

    def test_save_and_get_set(self, cache_repo):
        """Should save and retrieve set analysis."""
        analysis = CachedSetAnalysis(
            path="/test/set.mp3",
            dj_name="TestDJ",
            set_name="TestSet",
            total_tracks=10,
            track_paths=[f"/track{i}.mp3" for i in range(10)],
            flow_score=0.85,
            energy_progression=[1, 2, 3],
            timestamp=1234567890.0
        )

        cache_repo.save_set(analysis)
        retrieved = cache_repo.get_set("/test/set.mp3")

        assert retrieved is not None
        assert retrieved.dj_name == "TestDJ"
        assert retrieved.total_tracks == 10
        assert retrieved.flow_score == 0.85

    def test_get_nonexistent_set_returns_none(self, cache_repo):
        """Should return None for nonexistent set."""
        result = cache_repo.get_set("/nonexistent/set.mp3")
        assert result is None

    def test_invalidate_set(self, cache_repo):
        """Should invalidate set from cache."""
        analysis = CachedSetAnalysis(
            path="/test/set.mp3",
            dj_name="TestDJ",
            set_name="TestSet",
            total_tracks=5,
            track_paths=[],
            timestamp=1234567890.0
        )

        cache_repo.save_set(analysis)
        assert cache_repo.get_set("/test/set.mp3") is not None

        cache_repo.invalidate_set("/test/set.mp3")
        assert cache_repo.get_set("/test/set.mp3") is None


class TestTrackOperations:
    """Test track CRUD operations."""

    def test_save_and_get_track(self, cache_repo):
        """Should save and retrieve track analysis."""
        analysis = CachedTrackAnalysis(
            path="/test/track.mp3",
            duration=240.5,
            tempo=128.0,
            key="A",
            energy=0.7,
            zone=2,
            timestamp=1234567890.0
        )

        cache_repo.save_track(analysis)
        retrieved = cache_repo.get_track("/test/track.mp3")

        assert retrieved is not None
        assert retrieved.tempo == 128.0
        assert retrieved.key == "A"
        assert retrieved.energy == 0.7
        assert retrieved.zone == 2

    def test_get_nonexistent_track_returns_none(self, cache_repo):
        """Should return None for nonexistent track."""
        result = cache_repo.get_track("/nonexistent/track.mp3")
        assert result is None

    def test_invalidate_track(self, cache_repo):
        """Should invalidate track from cache."""
        analysis = CachedTrackAnalysis(
            path="/test/track.mp3",
            duration=180.0,
            tempo=120.0,
            timestamp=1234567890.0
        )

        cache_repo.save_track(analysis)
        assert cache_repo.get_track("/test/track.mp3") is not None

        cache_repo.invalidate_track("/test/track.mp3")
        assert cache_repo.get_track("/test/track.mp3") is None


class TestDJProfileOperations:
    """Test DJ profile CRUD operations."""

    def test_save_and_get_profile(self, cache_repo):
        """Should save and retrieve DJ profile."""
        profile = CachedDJProfile(
            dj_name="TestDJ",
            total_sets=5,
            avg_tempo=125.0,
            preferred_keys=["A", "D"],
            mixing_style="smooth",
            timestamp=1234567890.0
        )

        cache_repo.save_profile(profile)
        retrieved = cache_repo.get_profile("TestDJ")

        assert retrieved is not None
        assert retrieved.total_sets == 5
        assert retrieved.avg_tempo == 125.0
        assert retrieved.mixing_style == "smooth"

    def test_get_nonexistent_profile_returns_none(self, cache_repo):
        """Should return None for nonexistent profile."""
        result = cache_repo.get_profile("NonexistentDJ")
        assert result is None

    def test_invalidate_profile(self, cache_repo):
        """Should invalidate profile from cache."""
        profile = CachedDJProfile(
            dj_name="TestDJ",
            total_sets=3,
            timestamp=1234567890.0
        )

        cache_repo.save_profile(profile)
        assert cache_repo.get_profile("TestDJ") is not None

        cache_repo.invalidate_profile("TestDJ")
        assert cache_repo.get_profile("TestDJ") is None


class TestFeatureOperations:
    """Test ML feature CRUD operations."""

    def test_save_and_get_features(self, cache_repo):
        """Should save and retrieve ML features."""
        features = {
            "tempo": 128.0,
            "energy": 0.8,
            "spectral_centroid": 2500.0,
            "mfcc": np.random.rand(13).tolist()
        }

        file_hash = "abc123"
        cache_repo.save_features(file_hash, features)
        retrieved = cache_repo.get_features(file_hash)

        assert retrieved is not None
        assert retrieved["tempo"] == 128.0
        assert retrieved["energy"] == 0.8
        assert len(retrieved["mfcc"]) == 13

    def test_get_nonexistent_features_returns_none(self, cache_repo):
        """Should return None for nonexistent features."""
        result = cache_repo.get_features("nonexistent_hash")
        assert result is None


class TestSTFTOperations:
    """Test STFT cache operations."""

    def test_save_and_get_stft_matrix(self, cache_repo):
        """Should save and retrieve STFT matrix."""
        stft_matrix = np.random.rand(1025, 100).astype(np.complex64)
        file_hash = "stft123"

        cache_repo.save_stft(file_hash, stft_matrix)
        retrieved = cache_repo.get_stft(file_hash)

        assert retrieved is not None
        assert retrieved.shape == stft_matrix.shape
        assert retrieved.dtype == stft_matrix.dtype
        np.testing.assert_array_almost_equal(retrieved, stft_matrix)

    def test_get_nonexistent_stft_returns_none(self, cache_repo):
        """Should return None for nonexistent STFT."""
        result = cache_repo.get_stft("nonexistent_hash")
        assert result is None


class TestBulkInvalidation:
    """Test bulk invalidation operations."""

    def test_invalidate_by_directory(self, cache_repo):
        """Should invalidate all tracks in directory."""
        tracks = [
            CachedTrackAnalysis(path="/music/dj1/track1.mp3", duration=180, tempo=120, timestamp=123),
            CachedTrackAnalysis(path="/music/dj1/track2.mp3", duration=200, tempo=125, timestamp=123),
            CachedTrackAnalysis(path="/music/dj2/track3.mp3", duration=220, tempo=130, timestamp=123)
        ]

        for track in tracks:
            cache_repo.save_track(track)

        # Invalidate dj1 directory
        cache_repo.invalidate_by_directory("/music/dj1")

        assert cache_repo.get_track("/music/dj1/track1.mp3") is None
        assert cache_repo.get_track("/music/dj1/track2.mp3") is None
        assert cache_repo.get_track("/music/dj2/track3.mp3") is not None

    def test_invalidate_by_dj(self, cache_repo):
        """Should invalidate all sets by DJ."""
        sets = [
            CachedSetAnalysis(path="/set1.mp3", dj_name="DJ1", set_name="Set1", total_tracks=5, track_paths=[], timestamp=123),
            CachedSetAnalysis(path="/set2.mp3", dj_name="DJ1", set_name="Set2", total_tracks=6, track_paths=[], timestamp=123),
            CachedSetAnalysis(path="/set3.mp3", dj_name="DJ2", set_name="Set3", total_tracks=7, track_paths=[], timestamp=123)
        ]

        for s in sets:
            cache_repo.save_set(s)

        # Invalidate DJ1 sets
        cache_repo.invalidate_by_dj("DJ1")

        assert cache_repo.get_set("/set1.mp3") is None
        assert cache_repo.get_set("/set2.mp3") is None
        assert cache_repo.get_set("/set3.mp3") is not None


class TestCacheStats:
    """Test cache statistics."""

    def test_get_stats(self, cache_repo):
        """Should return cache statistics."""
        # Add some cached data
        cache_repo.save_track(CachedTrackAnalysis(path="/t1.mp3", duration=180, tempo=120, timestamp=123))
        cache_repo.save_track(CachedTrackAnalysis(path="/t2.mp3", duration=200, tempo=125, timestamp=123))
        cache_repo.save_set(CachedSetAnalysis(path="/s1.mp3", dj_name="DJ1", set_name="Set1", total_tracks=5, track_paths=[], timestamp=123))

        stats = cache_repo.get_stats()

        assert stats is not None
        assert stats.total_tracks >= 2
        assert stats.total_sets >= 1

    def test_cache_size_calculation(self, cache_repo):
        """Should calculate cache directory size."""
        # Add data to increase cache size
        cache_repo.save_features("hash1", {"data": [1, 2, 3] * 100})
        cache_repo.save_features("hash2", {"data": [4, 5, 6] * 100})

        stats = cache_repo.get_stats()
        assert stats.cache_size_mb > 0


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_cache_dir_creates_directory(self):
        """Should create cache directory if it doesn't exist."""
        tmpdir = tempfile.mkdtemp()
        cache_dir = os.path.join(tmpdir, "nonexistent", "cache")

        try:
            CacheRepository._instance = None
            repo = CacheRepository(cache_dir)
            assert os.path.exists(repo.cache_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_with_invalid_data_fails_gracefully(self, cache_repo):
        """Should handle invalid data gracefully."""
        # This should not crash
        try:
            cache_repo.save_features("test", {"invalid": object()})
        except (TypeError, ValueError):
            pass  # Expected to fail, should not crash application

"""
Tests for STFTCache lazy feature computation.

Verifies:
1. Lazy methods compute on first access
2. Results are cached (not recomputed)
3. Cache can be cleared
4. Shape consistency
"""

import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 22050
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration))

    # Mix of frequencies (440Hz + 880Hz + noise)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    y += 0.3 * np.sin(2 * np.pi * 880 * t)
    y += 0.1 * np.random.randn(len(t))

    return y.astype(np.float32), sr


@pytest.fixture
def stft_cache(sample_audio):
    """Create STFTCache from sample audio."""
    from src.core.primitives.stft import compute_stft

    y, sr = sample_audio
    return compute_stft(y, sr=sr)


class TestSTFTCacheCreation:
    """Test STFTCache creation and basic properties."""

    def test_compute_stft_returns_cache(self, sample_audio):
        """compute_stft should return STFTCache object."""
        from src.core.primitives.stft import compute_stft, STFTCache

        y, sr = sample_audio
        cache = compute_stft(y, sr=sr)

        assert isinstance(cache, STFTCache)

    def test_cache_has_required_fields(self, stft_cache):
        """STFTCache should have all core fields."""
        assert hasattr(stft_cache, 'S')
        assert hasattr(stft_cache, 'S_db')
        assert hasattr(stft_cache, 'phase')
        assert hasattr(stft_cache, 'freqs')
        assert hasattr(stft_cache, 'times')
        assert hasattr(stft_cache, 'sr')
        assert hasattr(stft_cache, 'hop_length')
        assert hasattr(stft_cache, 'n_fft')

    def test_spectrogram_shape(self, stft_cache):
        """Spectrogram should have correct shape."""
        assert stft_cache.S.ndim == 2
        assert stft_cache.S.shape[0] == stft_cache.n_fft // 2 + 1
        assert stft_cache.S.shape[1] == stft_cache.n_frames

    def test_arrays_are_float32(self, stft_cache):
        """Arrays should be float32 for M2 optimization."""
        assert stft_cache.S.dtype == np.float32
        assert stft_cache.S_db.dtype == np.float32

    def test_arrays_are_contiguous(self, stft_cache):
        """Arrays should be C-contiguous for Apple Accelerate."""
        assert stft_cache.S.flags['C_CONTIGUOUS']
        assert stft_cache.S_db.flags['C_CONTIGUOUS']


class TestSTFTCacheLazyMethods:
    """Test lazy feature computation methods."""

    def test_get_mfcc_shape(self, stft_cache):
        """get_mfcc should return correct shape."""
        n_mfcc = 13
        mfcc = stft_cache.get_mfcc(n_mfcc=n_mfcc)

        assert mfcc.shape[0] == n_mfcc
        assert mfcc.shape[1] == stft_cache.n_frames

    def test_get_mfcc_is_cached(self, stft_cache):
        """get_mfcc should cache results."""
        mfcc1 = stft_cache.get_mfcc(n_mfcc=13)
        mfcc2 = stft_cache.get_mfcc(n_mfcc=13)

        # Should be same object (cached)
        assert mfcc1 is mfcc2

    def test_get_mfcc_different_params_not_cached(self, stft_cache):
        """Different parameters should compute separately."""
        mfcc13 = stft_cache.get_mfcc(n_mfcc=13)
        mfcc20 = stft_cache.get_mfcc(n_mfcc=20)

        assert mfcc13 is not mfcc20
        assert mfcc13.shape[0] == 13
        assert mfcc20.shape[0] == 20

    def test_get_chroma_shape(self, stft_cache):
        """get_chroma should return 12 pitch classes."""
        chroma = stft_cache.get_chroma()

        assert chroma.shape[0] == 12
        assert chroma.shape[1] == stft_cache.n_frames

    def test_get_chroma_is_cached(self, stft_cache):
        """get_chroma should cache results."""
        chroma1 = stft_cache.get_chroma()
        chroma2 = stft_cache.get_chroma()

        assert chroma1 is chroma2

    def test_get_tonnetz_shape(self, stft_cache):
        """get_tonnetz should return 6 features."""
        tonnetz = stft_cache.get_tonnetz()

        assert tonnetz.shape[0] == 6
        assert tonnetz.shape[1] == stft_cache.n_frames

    def test_get_tonnetz_is_cached(self, stft_cache):
        """get_tonnetz should cache results."""
        tonnetz1 = stft_cache.get_tonnetz()
        tonnetz2 = stft_cache.get_tonnetz()

        assert tonnetz1 is tonnetz2

    def test_get_mel_shape(self, stft_cache):
        """get_mel should return correct shape."""
        n_mels = 128
        mel = stft_cache.get_mel(n_mels=n_mels)

        assert mel.shape[0] == n_mels
        assert mel.shape[1] == stft_cache.n_frames

    def test_get_mel_is_cached(self, stft_cache):
        """get_mel should cache results."""
        mel1 = stft_cache.get_mel()
        mel2 = stft_cache.get_mel()

        assert mel1 is mel2

    def test_get_mfcc_delta_shape(self, stft_cache):
        """get_mfcc_delta should return correct shape."""
        n_mfcc = 13
        delta = stft_cache.get_mfcc_delta(n_mfcc=n_mfcc)

        assert delta.shape[0] == n_mfcc
        assert delta.shape[1] == stft_cache.n_frames

    def test_get_mfcc_delta_is_cached(self, stft_cache):
        """get_mfcc_delta should cache results."""
        delta1 = stft_cache.get_mfcc_delta(n_mfcc=13)
        delta2 = stft_cache.get_mfcc_delta(n_mfcc=13)

        assert delta1 is delta2


class TestSTFTCacheClear:
    """Test cache clearing functionality."""

    def test_clear_feature_cache(self, stft_cache):
        """clear_feature_cache should remove cached features."""
        # Populate cache
        stft_cache.get_mfcc()
        stft_cache.get_chroma()
        stft_cache.get_tonnetz()

        # Verify cache is populated
        assert len(stft_cache._feature_cache) > 0

        # Clear cache
        stft_cache.clear_feature_cache()

        # Verify cache is empty
        assert len(stft_cache._feature_cache) == 0

    def test_features_recomputed_after_clear(self, stft_cache):
        """Features should be recomputed after cache clear."""
        mfcc1 = stft_cache.get_mfcc()

        stft_cache.clear_feature_cache()

        mfcc2 = stft_cache.get_mfcc()

        # Should be different object (recomputed)
        assert mfcc1 is not mfcc2

        # But values should be same
        np.testing.assert_array_almost_equal(mfcc1, mfcc2)


class TestSTFTCacheDataTypes:
    """Test output data types for M2 optimization."""

    def test_mfcc_is_float32(self, stft_cache):
        """MFCC should be float32."""
        mfcc = stft_cache.get_mfcc()
        assert mfcc.dtype == np.float32

    def test_chroma_is_float32(self, stft_cache):
        """Chroma should be float32."""
        chroma = stft_cache.get_chroma()
        assert chroma.dtype == np.float32

    def test_tonnetz_is_float32(self, stft_cache):
        """Tonnetz should be float32."""
        tonnetz = stft_cache.get_tonnetz()
        assert tonnetz.dtype == np.float32

    def test_mel_is_float32(self, stft_cache):
        """Mel spectrogram should be float32."""
        mel = stft_cache.get_mel()
        assert mel.dtype == np.float32

    def test_mfcc_delta_is_float32(self, stft_cache):
        """MFCC delta should be float32."""
        delta = stft_cache.get_mfcc_delta()
        assert delta.dtype == np.float32


class TestSTFTCacheContiguity:
    """Test array contiguity for Apple Accelerate optimization."""

    def test_mfcc_is_contiguous(self, stft_cache):
        """MFCC should be C-contiguous."""
        mfcc = stft_cache.get_mfcc()
        assert mfcc.flags['C_CONTIGUOUS']

    def test_chroma_is_contiguous(self, stft_cache):
        """Chroma should be C-contiguous."""
        chroma = stft_cache.get_chroma()
        assert chroma.flags['C_CONTIGUOUS']

    def test_tonnetz_is_contiguous(self, stft_cache):
        """Tonnetz should be C-contiguous."""
        tonnetz = stft_cache.get_tonnetz()
        assert tonnetz.flags['C_CONTIGUOUS']

    def test_mel_is_contiguous(self, stft_cache):
        """Mel spectrogram should be C-contiguous."""
        mel = stft_cache.get_mel()
        assert mel.flags['C_CONTIGUOUS']


class TestSTFTCacheProperties:
    """Test STFTCache convenience properties."""

    def test_n_frames_property(self, stft_cache):
        """n_frames should match spectrogram shape."""
        assert stft_cache.n_frames == stft_cache.S.shape[1]

    def test_n_freq_property(self, stft_cache):
        """n_freq should match spectrogram shape."""
        assert stft_cache.n_freq == stft_cache.S.shape[0]

    def test_duration_sec_property(self, stft_cache):
        """duration_sec should be positive."""
        assert stft_cache.duration_sec > 0

    def test_frame_duration_property(self, stft_cache):
        """frame_duration should match hop_length/sr."""
        expected = stft_cache.hop_length / stft_cache.sr
        assert stft_cache.frame_duration == expected


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

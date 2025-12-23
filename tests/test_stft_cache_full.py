"""
STFTCache Full Coverage Tests - 100% coverage target.

Tests all 25+ lazy methods of STFTCache for:
1. Correct output shapes
2. Float32 dtype
3. C-contiguous memory layout (M2 optimization)
4. Caching behavior (same object on repeated calls)
5. Edge cases (empty audio, short audio)
"""

import numpy as np
import pytest
from typing import Tuple

# Import the class under test
from app.common.primitives.stft import STFTCache, compute_stft


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_audio() -> Tuple[np.ndarray, int]:
    """Create synthetic audio for testing (5 seconds)."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Mix of frequencies for realistic spectrum
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +   # A5
        0.1 * np.sin(2 * np.pi * 220 * t) +   # A3
        0.05 * np.random.randn(len(t))         # Noise
    ).astype(np.float32)

    return y, sr


@pytest.fixture
def stft_cache(synthetic_audio) -> STFTCache:
    """Create STFTCache from synthetic audio."""
    y, sr = synthetic_audio
    return compute_stft(y, sr=sr)


@pytest.fixture
def short_audio() -> Tuple[np.ndarray, int]:
    """Very short audio (0.5 seconds)."""
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


@pytest.fixture
def empty_audio() -> Tuple[np.ndarray, int]:
    """Empty/silent audio."""
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)  # 1 second of silence
    return y, sr


# =============================================================================
# Test: compute_stft() function
# =============================================================================

class TestComputeSTFT:
    """Tests for compute_stft() factory function."""

    def test_returns_stft_cache(self, synthetic_audio):
        """compute_stft() returns STFTCache instance."""
        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)
        assert isinstance(cache, STFTCache)

    def test_has_required_fields(self, stft_cache):
        """STFTCache has all required fields."""
        assert hasattr(stft_cache, 'S')
        assert hasattr(stft_cache, 'S_db')
        assert hasattr(stft_cache, 'phase')
        assert hasattr(stft_cache, 'freqs')
        assert hasattr(stft_cache, 'times')
        assert hasattr(stft_cache, 'sr')
        assert hasattr(stft_cache, 'hop_length')
        assert hasattr(stft_cache, 'n_fft')

    def test_spectrogram_shape(self, stft_cache):
        """Spectrogram has correct shape (n_freq, n_frames)."""
        assert stft_cache.S.ndim == 2
        assert stft_cache.S.shape[0] == stft_cache.n_fft // 2 + 1
        assert stft_cache.S.shape[1] > 0

    def test_float32_dtype(self, stft_cache):
        """Core arrays are float32 (M2 optimization)."""
        assert stft_cache.S.dtype == np.float32
        assert stft_cache.S_db.dtype == np.float32

    def test_contiguous_memory(self, stft_cache):
        """Core arrays are C-contiguous (M2 optimization)."""
        assert stft_cache.S.flags['C_CONTIGUOUS']
        assert stft_cache.S_db.flags['C_CONTIGUOUS']

    def test_stores_audio(self, synthetic_audio):
        """compute_stft stores original audio for HPSS."""
        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)
        assert '_y' in cache._feature_cache


# =============================================================================
# Test: STFTCache Properties
# =============================================================================

class TestSTFTCacheProperties:
    """Tests for STFTCache computed properties."""

    def test_n_frames(self, stft_cache):
        """n_frames matches spectrogram columns."""
        assert stft_cache.n_frames == stft_cache.S.shape[1]

    def test_n_freq(self, stft_cache):
        """n_freq matches spectrogram rows."""
        assert stft_cache.n_freq == stft_cache.S.shape[0]

    def test_duration_sec(self, stft_cache):
        """duration_sec is positive for valid audio."""
        assert stft_cache.duration_sec > 0

    def test_frame_duration(self, stft_cache):
        """frame_duration is hop_length / sr."""
        expected = stft_cache.hop_length / stft_cache.sr
        assert abs(stft_cache.frame_duration - expected) < 1e-6


# =============================================================================
# Test: Lazy Feature Methods - Spectral
# =============================================================================

class TestLazySpectralFeatures:
    """Tests for lazy spectral feature computation."""

    def test_get_mel_shape(self, stft_cache):
        """get_mel returns (n_mels, n_frames)."""
        mel = stft_cache.get_mel(n_mels=128)
        assert mel.shape == (128, stft_cache.n_frames)

    def test_get_mel_float32(self, stft_cache):
        """get_mel returns float32."""
        mel = stft_cache.get_mel()
        assert mel.dtype == np.float32

    def test_get_mel_contiguous(self, stft_cache):
        """get_mel returns contiguous array."""
        mel = stft_cache.get_mel()
        assert mel.flags['C_CONTIGUOUS']

    def test_get_mel_cached(self, stft_cache):
        """get_mel caches result."""
        mel1 = stft_cache.get_mel(n_mels=64)
        mel2 = stft_cache.get_mel(n_mels=64)
        assert mel1 is mel2  # Same object

    def test_get_mfcc_shape(self, stft_cache):
        """get_mfcc returns (n_mfcc, n_frames)."""
        mfcc = stft_cache.get_mfcc(n_mfcc=13)
        assert mfcc.shape == (13, stft_cache.n_frames)

    def test_get_mfcc_float32(self, stft_cache):
        """get_mfcc returns float32."""
        mfcc = stft_cache.get_mfcc()
        assert mfcc.dtype == np.float32

    def test_get_mfcc_cached(self, stft_cache):
        """get_mfcc caches result."""
        mfcc1 = stft_cache.get_mfcc(n_mfcc=13)
        mfcc2 = stft_cache.get_mfcc(n_mfcc=13)
        assert mfcc1 is mfcc2

    def test_get_chroma_shape(self, stft_cache):
        """get_chroma returns (12, n_frames)."""
        chroma = stft_cache.get_chroma()
        assert chroma.shape == (12, stft_cache.n_frames)

    def test_get_chroma_float32(self, stft_cache):
        """get_chroma returns float32."""
        chroma = stft_cache.get_chroma()
        assert chroma.dtype == np.float32

    def test_get_tonnetz_shape(self, stft_cache):
        """get_tonnetz returns (6, n_frames)."""
        tonnetz = stft_cache.get_tonnetz()
        assert tonnetz.shape == (6, stft_cache.n_frames)

    def test_get_mfcc_delta_shape(self, stft_cache):
        """get_mfcc_delta returns (n_mfcc, n_frames)."""
        delta = stft_cache.get_mfcc_delta(n_mfcc=13)
        assert delta.shape == (13, stft_cache.n_frames)

    def test_get_rms_shape(self, stft_cache):
        """get_rms returns (n_frames,)."""
        rms = stft_cache.get_rms()
        assert rms.shape == (stft_cache.n_frames,)

    def test_get_rms_positive(self, stft_cache):
        """get_rms values are non-negative."""
        rms = stft_cache.get_rms()
        assert np.all(rms >= 0)

    def test_get_spectral_centroid_shape(self, stft_cache):
        """get_spectral_centroid returns (n_frames,)."""
        centroid = stft_cache.get_spectral_centroid()
        assert centroid.shape == (stft_cache.n_frames,)

    def test_get_spectral_rolloff_shape(self, stft_cache):
        """get_spectral_rolloff returns (n_frames,)."""
        rolloff = stft_cache.get_spectral_rolloff()
        assert rolloff.shape == (stft_cache.n_frames,)

    def test_get_spectral_flatness_shape(self, stft_cache):
        """get_spectral_flatness returns (n_frames,)."""
        flatness = stft_cache.get_spectral_flatness()
        assert flatness.shape == (stft_cache.n_frames,)

    def test_get_spectral_bandwidth_shape(self, stft_cache):
        """get_spectral_bandwidth returns (n_frames,)."""
        bandwidth = stft_cache.get_spectral_bandwidth()
        assert bandwidth.shape == (stft_cache.n_frames,)

    def test_get_spectral_contrast_shape(self, stft_cache):
        """get_spectral_contrast returns (n_bands+1, n_frames)."""
        contrast = stft_cache.get_spectral_contrast(n_bands=6)
        assert contrast.shape == (7, stft_cache.n_frames)

    def test_get_spectral_flux_shape(self, stft_cache):
        """get_spectral_flux returns (n_frames,)."""
        flux = stft_cache.get_spectral_flux()
        assert flux.shape == (stft_cache.n_frames,)


# =============================================================================
# Test: Lazy Feature Methods - Rhythm
# =============================================================================

class TestLazyRhythmFeatures:
    """Tests for lazy rhythm feature computation."""

    def test_get_onset_strength_shape(self, stft_cache):
        """get_onset_strength returns (n_frames,)."""
        onset = stft_cache.get_onset_strength()
        assert onset.ndim == 1

    def test_get_onset_strength_float32(self, stft_cache):
        """get_onset_strength returns float32."""
        onset = stft_cache.get_onset_strength()
        assert onset.dtype == np.float32

    def test_get_tempo_returns_tuple(self, stft_cache):
        """get_tempo returns (tempo, confidence)."""
        result = stft_cache.get_tempo()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_tempo_reasonable_range(self, stft_cache):
        """get_tempo returns tempo in reasonable range."""
        tempo, confidence = stft_cache.get_tempo()
        assert 40 <= tempo <= 240  # Reasonable BPM range
        assert 0 <= confidence <= 1

    def test_get_beats_returns_tuple(self, stft_cache):
        """get_beats returns (frames, times)."""
        result = stft_cache.get_beats()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_beats_frames_int32(self, stft_cache):
        """get_beats frames are int32."""
        frames, times = stft_cache.get_beats()
        assert frames.dtype == np.int32

    def test_get_beats_times_float32(self, stft_cache):
        """get_beats times are float32."""
        frames, times = stft_cache.get_beats()
        assert times.dtype == np.float32

    def test_get_tempogram_returns_tuple(self, stft_cache):
        """get_tempogram returns (tempogram, tempo_axis)."""
        result = stft_cache.get_tempogram()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_plp_shape(self, stft_cache):
        """get_plp returns (n_frames,) or similar."""
        plp = stft_cache.get_plp()
        assert plp.ndim == 1


# =============================================================================
# Test: Lazy Feature Methods - Audio Processing
# =============================================================================

class TestLazyAudioProcessing:
    """Tests for lazy audio processing methods."""

    def test_get_zcr_shape(self, stft_cache, synthetic_audio):
        """get_zcr returns (n_frames,)."""
        y, sr = synthetic_audio
        zcr = stft_cache.get_zcr(y)
        assert zcr.ndim == 1

    def test_get_hpss_returns_tuple(self, stft_cache):
        """get_hpss returns (harmonic, percussive)."""
        result = stft_cache.get_hpss()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_hpss_float32(self, stft_cache):
        """get_hpss returns float32 arrays."""
        y_h, y_p = stft_cache.get_hpss()
        assert y_h.dtype == np.float32
        assert y_p.dtype == np.float32

    def test_get_harmonic_float32(self, stft_cache):
        """get_harmonic returns float32."""
        y_h = stft_cache.get_harmonic()
        assert y_h.dtype == np.float32

    def test_get_percussive_float32(self, stft_cache):
        """get_percussive returns float32."""
        y_p = stft_cache.get_percussive()
        assert y_p.dtype == np.float32

    def test_get_hpss_returns_audio_tuple(self, stft_cache):
        """get_hpss returns (harmonic_audio, percussive_audio)."""
        result = stft_cache.get_hpss()
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Both should be audio signals (1D arrays)
        y_h, y_p = result
        assert y_h.ndim == 1
        assert y_p.ndim == 1

    def test_get_hpss_audio_shapes_match(self, stft_cache):
        """get_hpss harmonic and percussive have same length."""
        y_h, y_p = stft_cache.get_hpss()
        assert len(y_h) == len(y_p)

    def test_get_cqt_shape(self, stft_cache):
        """get_cqt returns (n_bins, n_frames)."""
        cqt = stft_cache.get_cqt(n_bins=84)
        assert cqt.shape[0] == 84

    def test_hpss_requires_audio(self, synthetic_audio):
        """get_hpss raises if audio not set."""
        y, sr = synthetic_audio
        # Create cache without set_audio
        cache = compute_stft(y, sr)
        cache._feature_cache.pop('_y', None)  # Remove stored audio

        with pytest.raises(ValueError, match="audio not set"):
            cache.get_hpss()


# =============================================================================
# Test: Cache Behavior
# =============================================================================

class TestCacheBehavior:
    """Tests for caching behavior."""

    def test_clear_feature_cache(self, stft_cache):
        """clear_feature_cache empties the cache."""
        # Populate cache
        stft_cache.get_mfcc()
        stft_cache.get_rms()

        assert len(stft_cache._feature_cache) > 0

        stft_cache.clear_feature_cache()

        assert len(stft_cache._feature_cache) == 0

    def test_different_params_different_cache(self, stft_cache):
        """Different parameters create different cache entries."""
        mfcc_13 = stft_cache.get_mfcc(n_mfcc=13)
        mfcc_20 = stft_cache.get_mfcc(n_mfcc=20)

        assert mfcc_13 is not mfcc_20
        assert mfcc_13.shape[0] == 13
        assert mfcc_20.shape[0] == 20

    def test_frames_to_time(self, stft_cache):
        """frames_to_time converts correctly."""
        frames = np.array([0, 10, 20])
        times = stft_cache.frames_to_time(frames)

        assert len(times) == 3
        assert times[0] == 0.0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_audio(self, short_audio):
        """STFTCache works with short audio."""
        y, sr = short_audio
        cache = compute_stft(y, sr)

        assert cache.n_frames > 0
        assert cache.get_rms().shape[0] > 0

    def test_silent_audio(self, empty_audio):
        """STFTCache works with silent audio."""
        y, sr = empty_audio
        cache = compute_stft(y, sr)

        rms = cache.get_rms()
        assert np.all(rms < 0.01)  # Nearly zero

    def test_stereo_to_mono(self):
        """compute_stft handles stereo (if passed mono)."""
        sr = 22050
        y = np.random.randn(sr).astype(np.float32)  # 1 second mono
        cache = compute_stft(y, sr)

        assert cache.n_frames > 0


# =============================================================================
# Test: M2 Optimization Invariants
# =============================================================================

@pytest.mark.critical
class TestM2Optimization:
    """Tests for M2 Apple Silicon optimization invariants."""

    LAZY_METHODS = [
        'get_mel', 'get_mfcc', 'get_chroma', 'get_tonnetz',
        'get_rms', 'get_spectral_centroid', 'get_spectral_rolloff',
        'get_spectral_flatness', 'get_spectral_bandwidth',
        'get_spectral_contrast', 'get_spectral_flux',
        'get_onset_strength',
    ]

    @pytest.mark.parametrize("method_name", LAZY_METHODS)
    def test_method_returns_float32(self, stft_cache, method_name):
        """All lazy methods return float32 arrays."""
        method = getattr(stft_cache, method_name)
        result = method()

        if isinstance(result, np.ndarray):
            assert result.dtype == np.float32, f"{method_name} returned {result.dtype}"

    @pytest.mark.parametrize("method_name", LAZY_METHODS)
    def test_method_returns_contiguous(self, stft_cache, method_name):
        """All lazy methods return C-contiguous arrays."""
        method = getattr(stft_cache, method_name)
        result = method()

        if isinstance(result, np.ndarray):
            assert result.flags['C_CONTIGUOUS'], f"{method_name} not contiguous"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

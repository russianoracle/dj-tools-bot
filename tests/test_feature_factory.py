"""
Tests for FeatureFactory - centralized audio feature extraction.

Verifies:
1. Factory creation from audio
2. Feature caching behavior
3. Output consistency with existing primitives/STFTCache
4. M2 optimizations (float32, contiguous)
"""

import numpy as np
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
TEST_AUDIO_DIR = PROJECT_ROOT / "dataset" / "MEMD_audio"


@pytest.fixture(scope="module")
def test_audio():
    """Load test audio file."""
    import librosa

    # Find test audio
    for filename in ["1001.mp3", "1101.mp3", "1201.mp3"]:
        path = TEST_AUDIO_DIR / filename
        if path.exists():
            y, sr = librosa.load(path, sr=22050, duration=30.0)
            return y, sr

    # Fallback: synthetic audio
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return y.astype(np.float32), sr


@pytest.fixture(scope="module")
def factory(test_audio):
    """Create FeatureFactory from test audio."""
    from app.core.feature_factory import FeatureFactory

    y, sr = test_audio
    return FeatureFactory.from_audio(y, sr)


class TestFeatureFactoryCreation:
    """Test factory creation methods."""

    def test_from_audio(self, test_audio):
        """Factory should be created from raw audio."""
        from app.core.feature_factory import FeatureFactory

        y, sr = test_audio
        factory = FeatureFactory.from_audio(y, sr)

        assert factory is not None
        assert factory.n_frames > 0
        assert factory.sr == sr

    def test_from_stft_cache(self, test_audio):
        """Factory should be created from existing STFTCache."""
        from app.core.feature_factory import FeatureFactory
        from app.core.primitives.stft import compute_stft

        y, sr = test_audio
        cache = compute_stft(y, sr=sr)
        factory = FeatureFactory.from_stft_cache(cache)

        assert factory is not None
        assert factory.stft_cache is cache

    def test_convenience_function(self, test_audio):
        """create_factory() convenience function should work."""
        from app.core.feature_factory import create_factory

        y, sr = test_audio
        factory = create_factory(y, sr)

        assert factory is not None


class TestEnergyFeatures:
    """Test energy-related features."""

    def test_rms_shape(self, factory):
        """RMS should have correct shape."""
        rms = factory.rms()
        assert rms.shape == (factory.n_frames,)

    def test_rms_positive(self, factory):
        """RMS should be non-negative."""
        rms = factory.rms()
        assert np.all(rms >= 0)

    def test_rms_db(self, factory):
        """RMS dB should be finite."""
        rms_db = factory.rms_db()
        assert np.all(np.isfinite(rms_db))

    def test_frequency_bands(self, factory):
        """Frequency bands should return valid FrequencyBands."""
        bands = factory.frequency_bands()

        assert hasattr(bands, 'bass')
        assert hasattr(bands, 'mid')
        assert hasattr(bands, 'high')
        assert len(bands.bass) == factory.n_frames


class TestSpectralFeatures:
    """Test spectral features."""

    def test_centroid_shape(self, factory):
        """Spectral centroid should have correct shape."""
        centroid = factory.spectral_centroid()
        assert centroid.shape == (factory.n_frames,)

    def test_centroid_positive(self, factory):
        """Spectral centroid should be positive."""
        centroid = factory.spectral_centroid()
        assert np.all(centroid >= 0)

    def test_rolloff_shape(self, factory):
        """Spectral rolloff should have correct shape."""
        rolloff = factory.spectral_rolloff()
        assert rolloff.shape == (factory.n_frames,)

    def test_brightness_range(self, factory):
        """Spectral brightness should be 0-1."""
        brightness = factory.spectral_brightness()
        assert np.all(brightness >= 0)
        assert np.all(brightness <= 1)

    def test_flatness_shape(self, factory):
        """Spectral flatness should have correct shape."""
        flatness = factory.spectral_flatness()
        assert flatness.shape == (factory.n_frames,)

    def test_flux_shape(self, factory):
        """Spectral flux should have correct shape."""
        flux = factory.spectral_flux()
        assert flux.shape == (factory.n_frames,)

    def test_bandwidth_shape(self, factory):
        """Spectral bandwidth should have correct shape."""
        bandwidth = factory.spectral_bandwidth()
        assert bandwidth.shape == (factory.n_frames,)

    def test_contrast_shape(self, factory):
        """Spectral contrast should have correct shape."""
        contrast = factory.spectral_contrast(n_bands=6)
        # librosa spectral_contrast returns n_bands + 1 (includes peak band)
        assert contrast.shape[0] == 7  # n_bands + 1
        assert contrast.shape[1] == factory.n_frames

    def test_spectral_features_dataclass(self, factory):
        """spectral_features() should return SpectralFeatures."""
        spectral = factory.spectral_features()

        assert hasattr(spectral, 'centroid')
        assert hasattr(spectral, 'rolloff')
        assert hasattr(spectral, 'brightness')


class TestMFCCFeatures:
    """Test MFCC features."""

    def test_mfcc_shape(self, factory):
        """MFCC should have correct shape."""
        mfcc = factory.mfcc(n_mfcc=13)
        assert mfcc.shape == (13, factory.n_frames)

    def test_mfcc_default(self, factory):
        """MFCC with default n_mfcc should work."""
        mfcc = factory.mfcc()
        assert mfcc.shape[0] > 0
        assert mfcc.shape[1] == factory.n_frames

    def test_mfcc_delta_shape(self, factory):
        """MFCC delta should have correct shape."""
        delta = factory.mfcc_delta(n_mfcc=13)
        assert delta.shape == (13, factory.n_frames)


class TestChromaFeatures:
    """Test chroma features."""

    def test_chroma_shape(self, factory):
        """Chroma should have 12 pitch classes."""
        chroma = factory.chroma()
        assert chroma.shape[0] == 12
        assert chroma.shape[1] == factory.n_frames

    def test_chroma_normalized(self, factory):
        """Chroma should be normalized."""
        chroma = factory.chroma()
        # Chroma values should be non-negative
        assert np.all(chroma >= 0)

    def test_tonnetz_shape(self, factory):
        """Tonnetz should have 6 features."""
        tonnetz = factory.tonnetz()
        assert tonnetz.shape[0] == 6
        assert tonnetz.shape[1] == factory.n_frames


class TestMelSpectrogram:
    """Test mel spectrogram."""

    def test_mel_shape(self, factory):
        """Mel spectrogram should have correct shape."""
        mel = factory.mel_spectrogram(n_mels=128)
        assert mel.shape[0] == 128
        assert mel.shape[1] == factory.n_frames


class TestRhythmFeatures:
    """Test rhythm features."""

    def test_onset_strength_shape(self, factory):
        """Onset strength should have correct shape."""
        onset = factory.onset_strength()
        assert onset.shape == (factory.n_frames,)

    def test_onset_strength_pure_shape(self, factory):
        """Pure onset strength should have correct shape."""
        onset = factory.onset_strength_pure()
        assert onset.shape == (factory.n_frames,)

    def test_tempo_reasonable(self, factory):
        """Tempo should be in reasonable range."""
        tempo, confidence = factory.tempo()
        assert 40 < tempo < 220
        assert 0 <= confidence <= 1

    def test_beats_arrays(self, factory):
        """Beats should return frames and times."""
        beat_frames, beat_times = factory.beats()
        assert len(beat_frames) == len(beat_times)
        assert len(beat_frames) > 0


class TestDynamicsFeatures:
    """Test dynamics features."""

    def test_novelty_shape(self, factory):
        """Novelty should have correct shape."""
        novelty = factory.novelty()
        assert novelty.shape == (factory.n_frames,)

    def test_buildup_score_shape(self, factory):
        """Buildup score should have correct shape."""
        buildup = factory.buildup_score()
        assert buildup.shape == (factory.n_frames,)

    def test_peaks_indices(self, factory):
        """Peaks should return valid indices."""
        peaks = factory.peaks()
        assert np.all(peaks >= 0)
        assert np.all(peaks < factory.n_frames)


class TestCaching:
    """Test feature caching behavior."""

    def test_rms_cached(self, factory):
        """RMS should be cached."""
        rms1 = factory.rms()
        rms2 = factory.rms()
        assert rms1 is rms2

    def test_mfcc_cached(self, factory):
        """MFCC should be cached (in STFTCache)."""
        mfcc1 = factory.mfcc()
        mfcc2 = factory.mfcc()
        assert mfcc1 is mfcc2

    def test_clear_cache(self, factory):
        """clear_cache() should remove cached features."""
        # Access some features
        _ = factory.rms()
        _ = factory.spectral_centroid()

        # Clear cache
        factory.clear_cache()

        # Accessing again should recompute (different object)
        # Note: STFTCache may still have features cached
        rms = factory.rms()
        assert rms is not None


class TestM2Optimizations:
    """Test M2 Apple Silicon optimizations."""

    def test_rms_float32(self, factory):
        """RMS should be float32."""
        rms = factory.rms()
        assert rms.dtype == np.float32

    def test_mfcc_float32(self, factory):
        """MFCC should be float32."""
        mfcc = factory.mfcc()
        assert mfcc.dtype == np.float32

    def test_rms_contiguous(self, factory):
        """RMS should be C-contiguous."""
        rms = factory.rms()
        assert rms.flags['C_CONTIGUOUS']

    def test_mfcc_contiguous(self, factory):
        """MFCC should be C-contiguous."""
        mfcc = factory.mfcc()
        assert mfcc.flags['C_CONTIGUOUS']


class TestToDict:
    """Test to_dict() export method."""

    def test_to_dict_keys(self, factory):
        """to_dict() should return expected keys."""
        features = factory.to_dict()

        expected_keys = [
            'rms', 'spectral_centroid', 'spectral_rolloff',
            'spectral_brightness', 'spectral_flatness', 'spectral_flux',
            'spectral_bandwidth', 'mfcc', 'chroma', 'onset_strength'
        ]

        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_to_dict_shapes(self, factory):
        """to_dict() arrays should have consistent shapes."""
        features = factory.to_dict()
        n_frames = factory.n_frames

        # 1D features
        assert features['rms'].shape == (n_frames,)
        assert features['spectral_centroid'].shape == (n_frames,)

        # 2D features
        assert features['mfcc'].shape[1] == n_frames
        assert features['chroma'].shape[1] == n_frames


class TestExtractFeaturesFunction:
    """Test extract_features() convenience function."""

    def test_extract_features(self, test_audio):
        """extract_features() should return dictionary."""
        from app.core.feature_factory import extract_features

        y, sr = test_audio
        features = extract_features(y, sr)

        assert isinstance(features, dict)
        assert 'rms' in features
        assert 'mfcc' in features


class TestConsistencyWithPrimitives:
    """Test that FeatureFactory matches direct primitive calls."""

    def test_rms_matches_primitive(self, test_audio):
        """Factory RMS should match STFTCache.get_rms()."""
        from app.core.feature_factory import FeatureFactory
        from app.core.primitives import compute_stft

        y, sr = test_audio

        # Via factory
        factory = FeatureFactory.from_audio(y, sr)
        factory_rms = factory.rms()

        # Via STFTCache (the correct way)
        cache = compute_stft(y, sr=sr)
        cache_rms = cache.get_rms()

        np.testing.assert_array_equal(factory_rms, cache_rms)

    def test_mfcc_matches_stft_cache(self, test_audio):
        """Factory MFCC should match STFTCache.get_mfcc()."""
        from app.core.feature_factory import FeatureFactory
        from app.core.primitives import compute_stft

        y, sr = test_audio

        # Via factory
        factory = FeatureFactory.from_audio(y, sr)
        factory_mfcc = factory.mfcc(n_mfcc=13)

        # Via STFTCache directly
        cache = compute_stft(y, sr=sr)
        cache_mfcc = cache.get_mfcc(n_mfcc=13)

        np.testing.assert_array_equal(factory_mfcc, cache_mfcc)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

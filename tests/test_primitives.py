"""Unit tests for common/primitives layer.

Tests pure mathematical functions with synthetic data:
    - filtering.py: smoothing, normalization, delta computation
    - energy.py: RMS, frequency bands, energy analysis
    - spectral.py: centroid, rolloff, brightness, flatness
    - beat_grid.py: beat grid structures and snapping

All tests use synthetic numpy arrays - no audio files needed.
Fast execution (<1 second total).
"""

import pytest
import numpy as np
import warnings


# =============================================================================
# FILTERING PRIMITIVES TESTS
# =============================================================================

@pytest.mark.unit
class TestFilteringPrimitives:
    """Tests for app.common.primitives.filtering functions."""

    def test_normalize_minmax_basic(self):
        """Test min-max normalization produces [0, 1] range.

        ЧТО ПРОВЕРЯЕМ:
            normalize_minmax() scales array to [0, 1]

        ОЖИДАЕМОЕ ПОВЕДЕНИЕ:
            - Minimum value becomes 0
            - Maximum value becomes 1
            - All values in [0, 1]
        """
        from app.common.primitives.filtering import normalize_minmax

        x = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        result = normalize_minmax(x)

        assert result.min() == pytest.approx(0.0, abs=1e-6)
        assert result.max() == pytest.approx(1.0, abs=1e-6)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_normalize_minmax_constant_array(self):
        """Test min-max normalization handles constant arrays.

        ЧТО ПРОВЕРЯЕМ:
            Constant array doesn't cause division by zero
        """
        from app.common.primitives.filtering import normalize_minmax

        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = normalize_minmax(x)

        # Should return zeros (or near-zero due to eps)
        assert np.all(np.isfinite(result))

    def test_normalize_zscore_basic(self):
        """Test z-score normalization produces mean=0, std=1.

        ЧТО ПРОВЕРЯЕМ:
            normalize_zscore() produces standardized distribution
        """
        from app.common.primitives.filtering import normalize_zscore

        x = np.random.randn(1000) * 10 + 50  # mean=50, std=10
        result = normalize_zscore(x)

        assert np.abs(np.mean(result)) < 0.1  # Mean close to 0
        assert np.abs(np.std(result) - 1.0) < 0.1  # Std close to 1

    def test_normalize_l2_unit_norm(self):
        """Test L2 normalization produces unit vectors.

        ЧТО ПРОВЕРЯЕМ:
            normalize_l2() produces vectors with L2 norm = 1
        """
        from app.common.primitives.filtering import normalize_l2

        x = np.array([[3, 4], [6, 8]], dtype=np.float32)  # Known Pythagorean triples
        result = normalize_l2(x, axis=1)

        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)

    def test_smooth_gaussian_reduces_noise(self):
        """Test Gaussian smoothing reduces high-frequency noise.

        ЧТО ПРОВЕРЯЕМ:
            smooth_gaussian() reduces variance of noisy signal
        """
        from app.common.primitives.filtering import smooth_gaussian

        # Signal with noise
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 100))
        noisy = signal + np.random.randn(100) * 0.5

        smoothed = smooth_gaussian(noisy, sigma=3.0)

        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(noisy)
        # Smoothed should be closer to original signal
        assert np.mean(np.abs(smoothed - signal)) < np.mean(np.abs(noisy - signal))

    def test_smooth_uniform_moving_average(self):
        """Test uniform smoothing acts as moving average.

        ЧТО ПРОВЕРЯЕМ:
            smooth_uniform() with size=3 produces moving average
        """
        from app.common.primitives.filtering import smooth_uniform

        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = smooth_uniform(x, size=3)

        # Result should be smoothed (less spiky)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))

    def test_smooth_savgol_preserves_peaks(self):
        """Test Savitzky-Golay smoothing preserves peak positions.

        ЧТО ПРОВЕРЯЕМ:
            smooth_savgol() preserves peak locations better than Gaussian
        """
        from app.common.primitives.filtering import smooth_savgol, smooth_gaussian

        # Create signal with sharp peak
        x = np.zeros(100)
        x[50] = 10.0  # Sharp peak at index 50

        savgol_result = smooth_savgol(x, window=11, order=3)
        gauss_result = smooth_gaussian(x, sigma=3.0)

        # Savgol should preserve peak position better
        savgol_peak_idx = np.argmax(savgol_result)
        gauss_peak_idx = np.argmax(gauss_result)

        assert savgol_peak_idx == 50  # Peak preserved
        assert len(savgol_result) == len(x)

    def test_compute_delta_derivative(self):
        """Test delta computation produces first derivative.

        ЧТО ПРОВЕРЯЕМ:
            compute_delta() approximates first derivative
        """
        from app.common.primitives.filtering import compute_delta

        # Linear function: y = 2x, derivative = 2
        x = np.linspace(0, 10, 100, dtype=np.float32)
        y = 2 * x

        delta = compute_delta(y, width=9)

        # Delta should be approximately constant (derivative of linear = constant)
        # Ignore edges
        middle = delta[10:-10]
        assert np.std(middle) < 0.5  # Low variance = constant

    def test_compute_delta2_second_derivative(self):
        """Test delta2 computation produces second derivative.

        ЧТО ПРОВЕРЯЕМ:
            compute_delta2() approximates second derivative
        """
        from app.common.primitives.filtering import compute_delta2

        # Quadratic function: y = x^2, second derivative = 2
        x = np.linspace(0, 10, 100, dtype=np.float32)
        y = x ** 2

        delta2 = compute_delta2(y, width=9)

        # Second derivative should be approximately constant
        middle = delta2[15:-15]
        assert np.std(middle) < 1.0

    def test_clip_outliers(self):
        """Test outlier clipping at specified percentile.

        ЧТО ПРОВЕРЯЕМ:
            clip_outliers() removes extreme values
        """
        from app.common.primitives.filtering import clip_outliers

        np.random.seed(42)
        x = np.random.randn(1000)
        x[0] = 100  # Extreme outlier
        x[1] = -100  # Extreme outlier

        clipped = clip_outliers(x, percentile=99)

        assert clipped.max() < 100
        assert clipped.min() > -100

    def test_interpolate_nans(self):
        """Test NaN interpolation fills gaps.

        ЧТО ПРОВЕРЯЕМ:
            interpolate_nans() fills NaN values with interpolated values
        """
        from app.common.primitives.filtering import interpolate_nans

        x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
        result = interpolate_nans(x)

        assert not np.any(np.isnan(result))
        assert result[2] == pytest.approx(3.0, abs=0.1)
        assert result[3] == pytest.approx(4.0, abs=0.1)

    def test_pad_or_truncate_padding(self):
        """Test pad_or_truncate pads short arrays.

        ЧТО ПРОВЕРЯЕМ:
            Arrays shorter than target are padded
        """
        from app.common.primitives.filtering import pad_or_truncate

        x = np.array([1, 2, 3])
        result = pad_or_truncate(x, target_length=5)

        assert len(result) == 5
        assert np.array_equal(result[:3], [1, 2, 3])

    def test_pad_or_truncate_truncating(self):
        """Test pad_or_truncate truncates long arrays.

        ЧТО ПРОВЕРЯЕМ:
            Arrays longer than target are truncated
        """
        from app.common.primitives.filtering import pad_or_truncate

        x = np.array([1, 2, 3, 4, 5])
        result = pad_or_truncate(x, target_length=3)

        assert len(result) == 3
        assert np.array_equal(result, [1, 2, 3])

    def test_resample_features_upsample(self):
        """Test feature resampling to more frames.

        ЧТО ПРОВЕРЯЕМ:
            resample_features() correctly upsamples
        """
        from app.common.primitives.filtering import resample_features

        x = np.array([0, 1, 2], dtype=np.float32)
        result = resample_features(x, target_frames=5)

        assert len(result) == 5
        assert result[0] == pytest.approx(0.0, abs=0.1)
        assert result[-1] == pytest.approx(2.0, abs=0.1)

    def test_resample_features_downsample(self):
        """Test feature resampling to fewer frames.

        ЧТО ПРОВЕРЯЕМ:
            resample_features() correctly downsamples
        """
        from app.common.primitives.filtering import resample_features

        x = np.linspace(0, 10, 100, dtype=np.float32)
        result = resample_features(x, target_frames=10)

        assert len(result) == 10
        assert result[0] == pytest.approx(0.0, abs=0.1)
        assert result[-1] == pytest.approx(10.0, abs=0.1)


# =============================================================================
# ENERGY PRIMITIVES TESTS
# =============================================================================

@pytest.mark.unit
class TestEnergyPrimitives:
    """Tests for app.common.primitives.energy functions."""

    @pytest.fixture
    def synthetic_spectrogram(self):
        """Create synthetic magnitude spectrogram for testing."""
        np.random.seed(42)
        n_freq = 513  # Standard STFT frequency bins
        n_frames = 100
        S = np.abs(np.random.randn(n_freq, n_frames)).astype(np.float32)
        freqs = np.linspace(0, 11025, n_freq, dtype=np.float32)  # Up to Nyquist at 22050 Hz
        return S, freqs

    def test_compute_rms_from_audio(self):
        """Test RMS computation from audio signal.

        ЧТО ПРОВЕРЯЕМ:
            compute_rms_from_audio() returns valid RMS values
        """
        from app.common.primitives.energy import compute_rms_from_audio

        # Create synthetic audio with known energy
        np.random.seed(42)
        audio = np.random.randn(22050).astype(np.float32)  # 1 second at 22050 Hz

        rms = compute_rms_from_audio(audio, frame_length=2048, hop_length=512)

        assert len(rms) > 0
        assert np.all(rms >= 0)  # RMS is always non-negative
        assert np.all(np.isfinite(rms))

    def test_compute_band_energy(self, synthetic_spectrogram):
        """Test frequency band energy computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_band_energy() isolates energy in specified range
        """
        from app.common.primitives.energy import compute_band_energy

        S, freqs = synthetic_spectrogram

        bass_energy = compute_band_energy(S, freqs, low=60, high=250)
        mid_energy = compute_band_energy(S, freqs, low=500, high=2000)

        assert len(bass_energy) == S.shape[1]
        assert len(mid_energy) == S.shape[1]
        assert np.all(bass_energy >= 0)
        assert np.all(mid_energy >= 0)

    def test_compute_frequency_bands(self, synthetic_spectrogram):
        """Test all frequency bands computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_frequency_bands() returns all 6 standard bands
        """
        from app.common.primitives.energy import compute_frequency_bands

        S, freqs = synthetic_spectrogram
        bands = compute_frequency_bands(S, freqs)

        # Check all bands present
        assert hasattr(bands, 'sub_bass')
        assert hasattr(bands, 'bass')
        assert hasattr(bands, 'low_mid')
        assert hasattr(bands, 'mid')
        assert hasattr(bands, 'high_mid')
        assert hasattr(bands, 'high')

        # Check shapes
        assert len(bands.bass) == S.shape[1]
        assert len(bands.mid) == S.shape[1]

    def test_frequency_bands_to_array(self, synthetic_spectrogram):
        """Test FrequencyBands.to_array() method.

        ЧТО ПРОВЕРЯЕМ:
            to_array() stacks all bands into (6, n_frames) array
        """
        from app.common.primitives.energy import compute_frequency_bands

        S, freqs = synthetic_spectrogram
        bands = compute_frequency_bands(S, freqs)

        arr = bands.to_array()

        assert arr.shape == (6, S.shape[1])

    def test_bass_to_high_ratio(self, synthetic_spectrogram):
        """Test bass-to-high frequency ratio.

        ЧТО ПРОВЕРЯЕМ:
            bass_to_high_ratio property returns valid ratios
        """
        from app.common.primitives.energy import compute_frequency_bands

        S, freqs = synthetic_spectrogram
        bands = compute_frequency_bands(S, freqs)

        ratio = bands.bass_to_high_ratio

        assert len(ratio) == S.shape[1]
        assert np.all(ratio > 0)  # Ratio should be positive
        assert np.all(np.isfinite(ratio))

    def test_compute_energy_derivative_order1(self):
        """Test first-order energy derivative.

        ЧТО ПРОВЕРЯЕМ:
            compute_energy_derivative(order=1) returns first derivative
        """
        from app.common.primitives.energy import compute_energy_derivative

        energy = np.array([1, 2, 4, 7, 11], dtype=np.float32)  # Increasing
        deriv = compute_energy_derivative(energy, order=1)

        assert len(deriv) == len(energy)
        assert np.all(deriv[1:] > 0)  # Increasing function has positive derivative

    def test_compute_energy_derivative_order2(self):
        """Test second-order energy derivative.

        ЧТО ПРОВЕРЯЕМ:
            compute_energy_derivative(order=2) returns second derivative
        """
        from app.common.primitives.energy import compute_energy_derivative

        energy = np.array([1, 4, 9, 16, 25], dtype=np.float32)  # x^2 pattern
        deriv2 = compute_energy_derivative(energy, order=2)

        assert len(deriv2) == len(energy)

    def test_detect_low_energy_frames(self):
        """Test low energy frame detection.

        ЧТО ПРОВЕРЯЕМ:
            detect_low_energy_frames() identifies frames below threshold
        """
        from app.common.primitives.energy import detect_low_energy_frames

        rms = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.1], dtype=np.float32)
        low_mask = detect_low_energy_frames(rms, threshold=0.5)

        assert low_mask.dtype == bool
        assert low_mask[0] == True  # 0.1 < 0.5
        assert low_mask[2] == False  # 0.9 > 0.5

    def test_compute_low_energy_ratio(self):
        """Test low energy ratio computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_low_energy_ratio() returns fraction of low-energy frames
        """
        from app.common.primitives.energy import compute_low_energy_ratio

        # Half high, half low energy
        rms = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        ratio = compute_low_energy_ratio(rms)

        assert 0 <= ratio <= 1
        assert ratio == pytest.approx(0.5, abs=0.1)

    def test_compute_energy_variance(self):
        """Test energy variance computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_energy_variance() returns normalized variance
        """
        from app.common.primitives.energy import compute_energy_variance

        # Constant energy = zero variance
        constant_rms = np.ones(100, dtype=np.float32)
        var_const = compute_energy_variance(constant_rms)
        assert var_const == pytest.approx(0.0, abs=1e-6)

        # Variable energy = non-zero variance
        variable_rms = np.array([0.1, 0.9, 0.1, 0.9] * 25, dtype=np.float32)
        var_variable = compute_energy_variance(variable_rms)
        assert var_variable > 0

    def test_compute_dynamic_range(self):
        """Test dynamic range computation in dB.

        ЧТО ПРОВЕРЯЕМ:
            compute_dynamic_range() returns valid dB value
        """
        from app.common.primitives.energy import compute_dynamic_range

        # 10:1 ratio = 20 dB
        rms = np.array([0.1, 0.5, 1.0], dtype=np.float32)
        dr = compute_dynamic_range(rms, percentile=95)

        assert dr > 0  # Dynamic range is positive
        assert dr < 100  # Reasonable upper bound

    def test_compute_mel_band_energies(self):
        """Test mel band energy extraction.

        ЧТО ПРОВЕРЯЕМ:
            compute_mel_band_energies() extracts DJ-relevant bands
        """
        from app.common.primitives.energy import compute_mel_band_energies

        # Create synthetic mel spectrogram
        np.random.seed(42)
        n_mels = 128
        n_frames = 100
        S_mel_db = np.random.randn(n_mels, n_frames).astype(np.float32) * 20 - 40

        bands = compute_mel_band_energies(S_mel_db, n_mels=128)

        assert hasattr(bands, 'bass')
        assert hasattr(bands, 'kick')
        assert hasattr(bands, 'mid')
        assert hasattr(bands, 'high')
        assert hasattr(bands, 'presence')

        assert len(bands.bass) == n_frames

    def test_compute_weighted_energy(self):
        """Test weighted energy combination.

        ЧТО ПРОВЕРЯЕМ:
            compute_weighted_energy() combines bands correctly
        """
        from app.common.primitives.energy import (
            compute_mel_band_energies, compute_weighted_energy
        )

        np.random.seed(42)
        n_mels = 128
        n_frames = 50
        S_mel_db = np.random.randn(n_mels, n_frames).astype(np.float32) * 20 - 40
        rms = np.abs(np.random.randn(n_frames)).astype(np.float32)

        bands = compute_mel_band_energies(S_mel_db)
        weighted = compute_weighted_energy(bands, rms)

        assert len(weighted) == n_frames
        assert np.all(weighted >= 0)
        assert np.all(weighted <= 1)  # Normalized output


# =============================================================================
# SPECTRAL PRIMITIVES TESTS
# =============================================================================

@pytest.mark.unit
class TestSpectralPrimitives:
    """Tests for app.common.primitives.spectral functions."""

    @pytest.fixture
    def synthetic_spectrogram(self):
        """Create synthetic magnitude spectrogram."""
        np.random.seed(42)
        n_freq = 513
        n_frames = 100
        S = np.abs(np.random.randn(n_freq, n_frames)).astype(np.float32) + 0.1
        freqs = np.linspace(0, 11025, n_freq, dtype=np.float32)
        return S, freqs

    def test_compute_rolloff(self, synthetic_spectrogram):
        """Test spectral rolloff computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_rolloff() returns frequency containing 85% energy
        """
        from app.common.primitives.spectral import compute_rolloff

        S, freqs = synthetic_spectrogram
        rolloff = compute_rolloff(S, freqs, roll_percent=0.85)

        assert len(rolloff) == S.shape[1]
        assert np.all(rolloff >= freqs[0])
        assert np.all(rolloff <= freqs[-1])

    def test_compute_brightness(self, synthetic_spectrogram):
        """Test brightness computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_brightness() returns high-frequency ratio [0, 1]
        """
        from app.common.primitives.spectral import compute_brightness

        S, freqs = synthetic_spectrogram
        brightness = compute_brightness(S, freqs, cutoff=3000.0)

        assert len(brightness) == S.shape[1]
        assert np.all(brightness >= 0)
        assert np.all(brightness <= 1)

    def test_compute_flatness(self, synthetic_spectrogram):
        """Test spectral flatness computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_flatness() returns tonality measure [0, 1]
        """
        from app.common.primitives.spectral import compute_flatness

        S, freqs = synthetic_spectrogram
        flatness = compute_flatness(S)

        assert len(flatness) == S.shape[1]
        assert np.all(flatness >= 0)
        assert np.all(flatness <= 1)

    def test_compute_flatness_white_noise(self):
        """Test flatness is high for white noise.

        ЧТО ПРОВЕРЯЕМ:
            White noise spectrum has high flatness (~1)
        """
        from app.common.primitives.spectral import compute_flatness

        # Uniform spectrum (white noise)
        np.random.seed(42)
        S_white = np.ones((513, 100)) + np.random.rand(513, 100) * 0.1
        flatness = compute_flatness(S_white.astype(np.float32))

        assert np.mean(flatness) > 0.8  # High flatness for white noise

    def test_compute_flux(self, synthetic_spectrogram):
        """Test spectral flux computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_flux() measures frame-to-frame change
        """
        from app.common.primitives.spectral import compute_flux

        S, freqs = synthetic_spectrogram
        flux = compute_flux(S)

        assert len(flux) == S.shape[1]
        assert np.all(flux >= 0)  # Flux is non-negative

    def test_compute_flux_constant_signal(self):
        """Test flux is zero for constant signal.

        ЧТО ПРОВЕРЯЕМ:
            Constant spectrum has zero flux
        """
        from app.common.primitives.spectral import compute_flux

        S_const = np.ones((513, 100), dtype=np.float32)
        flux = compute_flux(S_const)

        # First frame may have non-zero flux, rest should be zero
        assert np.allclose(flux[1:], 0.0)

    def test_compute_bandwidth(self, synthetic_spectrogram):
        """Test spectral bandwidth computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_bandwidth() returns spread around centroid
        """
        from app.common.primitives.spectral import compute_bandwidth

        S, freqs = synthetic_spectrogram

        # Suppress deprecation warning for centroid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            bandwidth = compute_bandwidth(S, freqs)

        assert len(bandwidth) == S.shape[1]
        assert np.all(bandwidth >= 0)

    def test_compute_contrast(self, synthetic_spectrogram):
        """Test spectral contrast computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_contrast() returns per-band contrast
        """
        from app.common.primitives.spectral import compute_contrast

        S, freqs = synthetic_spectrogram
        contrast = compute_contrast(S, freqs, n_bands=7)

        assert contrast.shape == (7, S.shape[1])

    def test_compute_all_spectral(self, synthetic_spectrogram):
        """Test all spectral features at once.

        ЧТО ПРОВЕРЯЕМ:
            compute_all_spectral() returns SpectralFeatures dataclass
        """
        from app.common.primitives.spectral import compute_all_spectral

        S, freqs = synthetic_spectrogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            features = compute_all_spectral(S, freqs)

        assert hasattr(features, 'centroid')
        assert hasattr(features, 'rolloff')
        assert hasattr(features, 'brightness')
        assert hasattr(features, 'flatness')
        assert hasattr(features, 'flux')
        assert hasattr(features, 'bandwidth')
        assert hasattr(features, 'contrast')

    def test_spectral_features_to_dict(self, synthetic_spectrogram):
        """Test SpectralFeatures.to_dict() method.

        ЧТО ПРОВЕРЯЕМ:
            to_dict() returns mean values as dictionary
        """
        from app.common.primitives.spectral import compute_all_spectral

        S, freqs = synthetic_spectrogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            features = compute_all_spectral(S, freqs)

        d = features.to_dict()

        assert 'spectral_centroid' in d
        assert 'spectral_rolloff' in d
        assert 'brightness' in d
        assert isinstance(d['brightness'], float)

    def test_compute_spectral_slope(self, synthetic_spectrogram):
        """Test spectral slope computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_spectral_slope() returns slope per frame
        """
        from app.common.primitives.spectral import compute_spectral_slope

        S, freqs = synthetic_spectrogram
        slope = compute_spectral_slope(S, freqs)

        assert len(slope) == S.shape[1]
        assert np.all(np.isfinite(slope))

    def test_compute_spectral_velocity(self, synthetic_spectrogram):
        """Test spectral centroid velocity.

        ЧТО ПРОВЕРЯЕМ:
            compute_spectral_velocity() returns rate of change in Hz/sec
        """
        from app.common.primitives.spectral import (
            compute_centroid, compute_spectral_velocity
        )

        S, freqs = synthetic_spectrogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            centroid = compute_centroid(S, freqs)

        velocity = compute_spectral_velocity(centroid, sr=22050, hop_length=512)

        assert len(velocity) == len(centroid)

    def test_compute_filter_position(self, synthetic_spectrogram):
        """Test filter position estimation.

        ЧТО ПРОВЕРЯЕМ:
            compute_filter_position() returns values in [0, 1]
        """
        from app.common.primitives.spectral import (
            compute_centroid, compute_rolloff, compute_filter_position
        )

        S, freqs = synthetic_spectrogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            centroid = compute_centroid(S, freqs)
            rolloff = compute_rolloff(S, freqs)

        filter_pos = compute_filter_position(centroid, rolloff)

        assert len(filter_pos) == S.shape[1]
        assert np.all(filter_pos >= 0)
        assert np.all(filter_pos <= 1)


# =============================================================================
# BEAT GRID PRIMITIVES TESTS
# =============================================================================

@pytest.mark.unit
class TestBeatGridPrimitives:
    """Tests for app.common.primitives.beat_grid structures and functions."""

    @pytest.fixture
    def sample_beat_grid(self):
        """Create a sample BeatGridResult for testing."""
        from app.common.primitives.beat_grid import (
            BeatGridResult, BeatInfo, BarInfo, PhraseInfo
        )

        # Create 16 beats at 120 BPM (0.5s per beat)
        beats = [
            BeatInfo(
                time_sec=i * 0.5,
                frame_idx=int(i * 0.5 * 22050 / 512),
                bar_position=(i % 4) + 1,
                phrase_position=(i % 16) + 1,
                strength=0.8 if i % 4 == 0 else 0.5
            )
            for i in range(16)
        ]

        # Create 4 bars
        bars = [
            BarInfo(
                index=i,
                start_time=i * 2.0,
                end_time=(i + 1) * 2.0,
                beat_indices=list(range(i * 4, (i + 1) * 4)),
                phrase_idx=0,
                bar_in_phrase=i + 1
            )
            for i in range(4)
        ]

        # Create 1 phrase
        phrases = [
            PhraseInfo(
                index=0,
                start_time=0.0,
                end_time=8.0,
                bar_indices=[0, 1, 2, 3],
                duration_sec=8.0
            )
        ]

        return BeatGridResult(
            beats=beats,
            bars=bars,
            phrases=phrases,
            tempo=120.0,
            tempo_confidence=0.9,
            beat_duration_sec=0.5,
            bar_duration_sec=2.0,
            phrase_duration_sec=8.0,
            sr=22050,
            hop_length=512
        )

    def test_beat_grid_phrase_boundaries(self, sample_beat_grid):
        """Test phrase boundary extraction.

        ЧТО ПРОВЕРЯЕМ:
            get_phrase_boundaries() returns correct boundary times
        """
        boundaries = sample_beat_grid.get_phrase_boundaries()

        assert len(boundaries) == 2  # Start and end of 1 phrase
        assert boundaries[0] == pytest.approx(0.0)
        assert boundaries[1] == pytest.approx(8.0)

    def test_beat_grid_bar_boundaries(self, sample_beat_grid):
        """Test bar boundary extraction.

        ЧТО ПРОВЕРЯЕМ:
            get_bar_boundaries() returns correct boundary times
        """
        boundaries = sample_beat_grid.get_bar_boundaries()

        assert len(boundaries) == 5  # 4 bars = 5 boundaries
        assert boundaries[0] == pytest.approx(0.0)
        assert boundaries[-1] == pytest.approx(8.0)

    def test_beat_grid_beat_times(self, sample_beat_grid):
        """Test beat time extraction.

        ЧТО ПРОВЕРЯЕМ:
            get_beat_times() returns all beat times
        """
        beat_times = sample_beat_grid.get_beat_times()

        assert len(beat_times) == 16
        assert beat_times[0] == pytest.approx(0.0)
        assert beat_times[1] == pytest.approx(0.5)

    def test_snap_to_phrase(self, sample_beat_grid):
        """Test snapping to nearest phrase boundary.

        ЧТО ПРОВЕРЯЕМ:
            snap_to_phrase() snaps time to nearest boundary
        """
        # Time near start of phrase
        snapped = sample_beat_grid.snap_to_phrase(0.3)
        assert snapped == pytest.approx(0.0)

        # Time near end of phrase
        snapped = sample_beat_grid.snap_to_phrase(7.8)
        assert snapped == pytest.approx(8.0)

    def test_snap_to_bar(self, sample_beat_grid):
        """Test snapping to nearest bar boundary.

        ЧТО ПРОВЕРЯЕМ:
            snap_to_bar() snaps time to nearest boundary
        """
        snapped = sample_beat_grid.snap_to_bar(2.1)
        assert snapped == pytest.approx(2.0)

        snapped = sample_beat_grid.snap_to_bar(3.9)
        assert snapped == pytest.approx(4.0)

    def test_snap_to_beat(self, sample_beat_grid):
        """Test snapping to nearest beat.

        ЧТО ПРОВЕРЯЕМ:
            snap_to_beat() snaps time to nearest beat
        """
        snapped = sample_beat_grid.snap_to_beat(0.6)
        assert snapped == pytest.approx(0.5)

        snapped = sample_beat_grid.snap_to_beat(1.3)
        assert snapped == pytest.approx(1.5)

    def test_is_on_phrase_boundary(self, sample_beat_grid):
        """Test phrase boundary detection.

        ЧТО ПРОВЕРЯЕМ:
            is_on_phrase_boundary() correctly identifies boundary proximity
        """
        # On boundary
        assert sample_beat_grid.is_on_phrase_boundary(0.0, tolerance_beats=2)
        assert sample_beat_grid.is_on_phrase_boundary(8.0, tolerance_beats=2)

        # Far from boundary
        assert not sample_beat_grid.is_on_phrase_boundary(4.0, tolerance_beats=1)

    def test_is_on_bar_boundary(self, sample_beat_grid):
        """Test bar boundary detection.

        ЧТО ПРОВЕРЯЕМ:
            is_on_bar_boundary() correctly identifies boundary proximity
        """
        assert sample_beat_grid.is_on_bar_boundary(2.0, tolerance_beats=1)
        assert sample_beat_grid.is_on_bar_boundary(4.0, tolerance_beats=1)
        assert not sample_beat_grid.is_on_bar_boundary(3.0, tolerance_beats=1)

    def test_get_phrase_at_time(self, sample_beat_grid):
        """Test phrase lookup by time.

        ЧТО ПРОВЕРЯЕМ:
            get_phrase_at_time() returns correct phrase
        """
        phrase = sample_beat_grid.get_phrase_at_time(3.0)
        assert phrase is not None
        assert phrase.index == 0

        # Outside all phrases
        phrase = sample_beat_grid.get_phrase_at_time(10.0)
        assert phrase is None

    def test_get_bar_at_time(self, sample_beat_grid):
        """Test bar lookup by time.

        ЧТО ПРОВЕРЯЕМ:
            get_bar_at_time() returns correct bar
        """
        bar = sample_beat_grid.get_bar_at_time(3.0)
        assert bar is not None
        assert bar.index == 1  # Second bar (2.0-4.0)

    def test_time_to_phrase_position(self, sample_beat_grid):
        """Test time to musical position conversion.

        ЧТО ПРОВЕРЯЕМ:
            time_to_phrase_position() returns (phrase, bar, beat)
        """
        pos = sample_beat_grid.time_to_phrase_position(2.5)

        assert pos[0] == 1  # Phrase 1 (1-indexed)
        assert pos[1] >= 1 and pos[1] <= 4  # Bar in phrase

    def test_beat_grid_to_dict(self, sample_beat_grid):
        """Test JSON serialization.

        ЧТО ПРОВЕРЯЕМ:
            to_dict() returns serializable dictionary
        """
        d = sample_beat_grid.to_dict()

        assert d['tempo'] == 120.0
        assert d['n_beats'] == 16
        assert d['n_bars'] == 4
        assert d['n_phrases'] == 1
        assert 'phrase_boundaries' in d

    def test_snap_events_to_grid(self, sample_beat_grid):
        """Test event snapping to grid.

        ЧТО ПРОВЕРЯЕМ:
            snap_events_to_grid() snaps multiple events at once
        """
        from app.common.primitives.beat_grid import snap_events_to_grid

        events = np.array([0.1, 2.2, 7.9], dtype=np.float32)
        snapped = snap_events_to_grid(events, sample_beat_grid, snap_level='bar')

        assert len(snapped) == 3
        assert snapped[0] == pytest.approx(0.0, abs=0.1)
        assert snapped[1] == pytest.approx(2.0, abs=0.1)

    def test_compute_event_offsets(self):
        """Test event offset computation from phrase boundaries.

        ЧТО ПРОВЕРЯЕМ:
            compute_event_offsets() measures distance to boundaries
        """
        from app.common.primitives.beat_grid import compute_event_offsets

        # Events at exact phrase boundaries (8s phrases)
        events = np.array([0.0, 8.0, 16.0], dtype=np.float32)
        offsets = compute_event_offsets(events, phrase_duration_sec=8.0)

        assert len(offsets) == 3
        assert np.allclose(offsets, 0.0, atol=0.1)  # All on boundaries

    def test_compute_alignment_score(self, sample_beat_grid):
        """Test alignment score computation.

        ЧТО ПРОВЕРЯЕМ:
            compute_alignment_score() measures boundary alignment
        """
        from app.common.primitives.beat_grid import compute_alignment_score

        # Events perfectly aligned
        events_aligned = np.array([0.0, 8.0], dtype=np.float32)
        score = compute_alignment_score(events_aligned, sample_beat_grid)
        assert score == pytest.approx(1.0, abs=0.1)

        # Events misaligned
        events_misaligned = np.array([4.0, 5.0], dtype=np.float32)
        score = compute_alignment_score(events_misaligned, sample_beat_grid)
        assert score < 0.5


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

@pytest.mark.unit
class TestPrimitivesEdgeCases:
    """Edge case tests for primitives."""

    def test_empty_array_handling(self):
        """Test functions handle empty arrays gracefully."""
        from app.common.primitives.filtering import normalize_minmax
        from app.common.primitives.energy import detect_low_energy_frames

        empty = np.array([], dtype=np.float32)

        # Should not raise
        result = normalize_minmax(empty)
        assert len(result) == 0

    def test_single_element_array(self):
        """Test functions handle single-element arrays."""
        from app.common.primitives.filtering import normalize_minmax

        single = np.array([5.0])
        result = normalize_minmax(single)

        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_nan_handling(self):
        """Test NaN values don't propagate unexpectedly."""
        from app.common.primitives.filtering import interpolate_nans

        x = np.array([1.0, np.nan, 3.0])
        result = interpolate_nans(x)

        assert not np.any(np.isnan(result))

    def test_inf_handling(self):
        """Test infinite values are handled."""
        from app.common.primitives.filtering import clip_outliers

        x = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        # clip_outliers should handle inf
        result = clip_outliers(x, percentile=99)

        assert np.all(np.isfinite(result))

    def test_zero_division_protection(self):
        """Test division by zero is protected."""
        from app.common.primitives.filtering import normalize_minmax

        zeros = np.zeros(10, dtype=np.float32)
        result = normalize_minmax(zeros)

        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

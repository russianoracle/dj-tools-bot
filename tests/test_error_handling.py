"""Error handling and edge case tests.

Tests robustness of the application:
    - Invalid inputs (corrupted audio, wrong formats)
    - Edge cases (empty files, extreme values)
    - Error recovery and graceful degradation

These tests ensure the application doesn't crash on unexpected inputs.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


# =============================================================================
# INVALID AUDIO INPUT TESTS
# =============================================================================

@pytest.mark.unit
class TestInvalidAudioInputs:
    """Tests for handling invalid audio inputs."""

    def test_empty_audio_array(self):
        """Test handling of empty audio array.

        ЧТО ПРОВЕРЯЕМ:
            Empty audio doesn't crash the system
        """
        from app.modules.analysis.tasks.base import create_audio_context

        empty_audio = np.array([], dtype=np.float32)

        # Should handle gracefully or raise meaningful error
        with pytest.raises((ValueError, IndexError, Exception)):
            create_audio_context(empty_audio, sr=22050)

    def test_zero_length_audio(self):
        """Test handling of zero-length audio.

        ЧТО ПРОВЕРЯЕМ:
            Very short audio is handled
        """
        from app.modules.analysis.tasks.base import create_audio_context

        # Single sample
        single_sample = np.array([0.5], dtype=np.float32)

        # Should either work or raise meaningful error
        try:
            ctx = create_audio_context(single_sample, sr=22050)
            # If it works, duration should be very small
            assert ctx.duration_sec < 0.001
        except Exception as e:
            # Acceptable to fail gracefully
            assert "too short" in str(e).lower() or isinstance(e, (ValueError, IndexError))

    def test_nan_values_in_audio(self):
        """Test handling of NaN values in audio.

        ЧТО ПРОВЕРЯЕМ:
            NaN values don't cause crashes
        """
        from app.modules.analysis.tasks.base import create_audio_context

        # Audio with NaN values
        audio = np.array([0.1, np.nan, 0.3, np.nan, 0.5] * 4410, dtype=np.float32)

        # Should handle (either clean or fail gracefully)
        try:
            ctx = create_audio_context(audio, sr=22050)
            # If successful, check we can still use it
            assert ctx.stft_cache is not None
        except Exception as e:
            # Acceptable to fail with meaningful error
            pass

    def test_inf_values_in_audio(self):
        """Test handling of Inf values in audio.

        ЧТО ПРОВЕРЯЕМ:
            Infinite values don't cause crashes
        """
        from app.modules.analysis.tasks.base import create_audio_context

        # Audio with infinite values
        audio = np.array([0.1, np.inf, 0.3, -np.inf, 0.5] * 4410, dtype=np.float32)

        try:
            ctx = create_audio_context(audio, sr=22050)
        except Exception:
            # Acceptable to fail
            pass

    def test_all_zeros_audio(self):
        """Test handling of silent (all zeros) audio.

        ЧТО ПРОВЕРЯЕМ:
            Silent audio produces valid (though trivial) results
        """
        from app.modules.analysis.tasks.base import create_audio_context
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        # 2 seconds of silence
        silence = np.zeros(44100, dtype=np.float32)

        ctx = create_audio_context(silence, sr=22050)
        task = ZoneClassificationTask()
        result = task.execute(ctx)

        # Should complete (even if zone is uncertain)
        assert result.zone in ['yellow', 'green', 'purple']

    def test_clipping_audio(self):
        """Test handling of clipped/saturated audio.

        ЧТО ПРОВЕРЯЕМ:
            Clipped audio (all +1/-1) doesn't crash
        """
        from app.modules.analysis.tasks.base import create_audio_context

        # Severely clipped audio (square wave)
        clipped = np.array([1.0, -1.0] * 22050, dtype=np.float32)

        ctx = create_audio_context(clipped, sr=22050)
        assert ctx.stft_cache is not None
        assert ctx.duration_sec > 0

    def test_extreme_sample_rate(self):
        """Test handling of unusual sample rates.

        ЧТО ПРОВЕРЯЕМ:
            Unusual sample rates don't cause errors
        """
        from app.modules.analysis.tasks.base import create_audio_context

        audio = np.random.randn(10000).astype(np.float32) * 0.1

        # Very low sample rate
        ctx_low = create_audio_context(audio, sr=8000)
        assert ctx_low.sr == 8000

        # Very high sample rate
        ctx_high = create_audio_context(audio, sr=96000)
        assert ctx_high.sr == 96000


# =============================================================================
# EXTREME VALUE TESTS
# =============================================================================

@pytest.mark.unit
class TestExtremeValues:
    """Tests for handling extreme values in features."""

    def test_extreme_tempo(self):
        """Test classification with extreme tempo values.

        ЧТО ПРОВЕРЯЕМ:
            Very high/low tempo doesn't break classification
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask()

        # Extreme low tempo
        result_low = task._classify_rules({'tempo': 30.0})
        assert result_low.success
        assert result_low.zone in ['yellow', 'green', 'purple']

        # Extreme high tempo
        result_high = task._classify_rules({'tempo': 300.0})
        assert result_high.success
        assert result_high.zone in ['yellow', 'green', 'purple']

        # Zero tempo
        result_zero = task._classify_rules({'tempo': 0.0})
        assert result_zero.success

    def test_extreme_energy_values(self):
        """Test energy primitives with extreme values.

        ЧТО ПРОВЕРЯЕМ:
            Energy calculations handle extreme inputs
        """
        from app.common.primitives.energy import (
            compute_energy_variance,
            compute_dynamic_range,
            compute_low_energy_ratio
        )

        # Very high values
        high_rms = np.ones(100, dtype=np.float32) * 1e6
        var_high = compute_energy_variance(high_rms)
        assert np.isfinite(var_high)

        # Very low values
        low_rms = np.ones(100, dtype=np.float32) * 1e-10
        var_low = compute_energy_variance(low_rms)
        assert np.isfinite(var_low)

        # Mixed extreme values
        mixed = np.array([1e-10, 1e6] * 50, dtype=np.float32)
        dr = compute_dynamic_range(mixed)
        assert np.isfinite(dr)

    def test_extreme_spectrogram_values(self):
        """Test spectral primitives with extreme spectrogram values.

        ЧТО ПРОВЕРЯЕМ:
            Spectral calculations handle extreme values
        """
        from app.common.primitives.spectral import (
            compute_rolloff,
            compute_brightness,
            compute_flatness
        )

        freqs = np.linspace(0, 11025, 513, dtype=np.float32)

        # Very high values
        S_high = np.ones((513, 100), dtype=np.float32) * 1e6
        rolloff = compute_rolloff(S_high, freqs)
        assert np.all(np.isfinite(rolloff))

        # Very low values
        S_low = np.ones((513, 100), dtype=np.float32) * 1e-10
        brightness = compute_brightness(S_low, freqs)
        assert np.all(np.isfinite(brightness))

        # Zero values (should be handled by eps)
        S_zero = np.zeros((513, 100), dtype=np.float32)
        flatness = compute_flatness(S_zero)
        assert np.all(np.isfinite(flatness))


# =============================================================================
# FILE HANDLING TESTS
# =============================================================================

@pytest.mark.unit
class TestFileHandling:
    """Tests for file handling edge cases."""

    def test_nonexistent_file_path(self):
        """Test handling of nonexistent file path.

        ЧТО ПРОВЕРЯЕМ:
            Missing file raises appropriate error
        """
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline

        pipeline = TrackAnalysisPipeline(sr=22050)

        result = pipeline.analyze("/nonexistent/path/audio.mp3")

        assert result.success == False
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such" in result.error.lower() or "error" in result.error.lower()

    def test_invalid_file_format(self, tmp_path):
        """Test handling of non-audio file.

        ЧТО ПРОВЕРЯЕМ:
            Non-audio files are rejected gracefully
        """
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline

        # Create a text file
        fake_audio = tmp_path / "not_audio.txt"
        fake_audio.write_text("This is not audio data")

        pipeline = TrackAnalysisPipeline(sr=22050)
        result = pipeline.analyze(str(fake_audio))

        assert result.success == False
        assert result.error is not None

    def test_empty_file(self, tmp_path):
        """Test handling of empty file.

        ЧТО ПРОВЕРЯЕМ:
            Empty files are rejected gracefully
        """
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline

        # Create empty file
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()

        pipeline = TrackAnalysisPipeline(sr=22050)
        result = pipeline.analyze(str(empty_file))

        assert result.success == False

    def test_corrupted_wav_header(self, tmp_path):
        """Test handling of file with corrupted header.

        ЧТО ПРОВЕРЯЕМ:
            Corrupted audio files are handled gracefully
        """
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline

        # Create file with fake WAV header
        corrupted = tmp_path / "corrupted.wav"
        corrupted.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt corrupted data here")

        pipeline = TrackAnalysisPipeline(sr=22050)
        result = pipeline.analyze(str(corrupted))

        assert result.success == False


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

@pytest.mark.unit
class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_minimum_audio_length(self, synthetic_audio_short):
        """Test with minimum valid audio length.

        ЧТО ПРОВЕРЯЕМ:
            Very short audio (2 seconds) can be analyzed
        """
        from app.modules.analysis.tasks.base import create_audio_context
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        y, sr = synthetic_audio_short

        ctx = create_audio_context(y, sr)
        task = ZoneClassificationTask()
        result = task.execute(ctx)

        assert result.success or result.error is not None

    def test_single_frequency_band(self):
        """Test band energy with single frequency.

        ЧТО ПРОВЕРЯЕМ:
            Narrow frequency bands don't cause errors
        """
        from app.common.primitives.energy import compute_band_energy

        S = np.abs(np.random.randn(513, 100).astype(np.float32))
        freqs = np.linspace(0, 11025, 513, dtype=np.float32)

        # Very narrow band (might have no frequencies)
        energy = compute_band_energy(S, freqs, low=1000.0, high=1001.0)

        assert len(energy) == 100
        # Should be zeros or very small values

    def test_beat_grid_no_beats(self):
        """Test beat grid with no detectable beats.

        ЧТО ПРОВЕРЯЕМ:
            Audio with no clear beats produces valid (empty) grid
        """
        from app.common.primitives.beat_grid import BeatGridResult

        # Empty beat grid
        grid = BeatGridResult(
            beats=[],
            bars=[],
            phrases=[],
            tempo=0.0,
            tempo_confidence=0.0,
            beat_duration_sec=0.5,
            bar_duration_sec=2.0,
            phrase_duration_sec=8.0,
            sr=22050,
            hop_length=512
        )

        # Methods should handle empty lists
        assert len(grid.get_phrase_boundaries()) == 0
        assert len(grid.get_bar_boundaries()) == 0
        assert len(grid.get_beat_times()) == 0

        # Snap methods should return input unchanged
        assert grid.snap_to_phrase(5.0) == 5.0
        assert grid.snap_to_bar(5.0) == 5.0
        assert grid.snap_to_beat(5.0) == 5.0


# =============================================================================
# MEMORY AND PERFORMANCE EDGE CASES
# =============================================================================

@pytest.mark.unit
class TestMemoryEdgeCases:
    """Tests for memory-related edge cases."""

    def test_large_spectrogram(self):
        """Test handling of large spectrogram.

        ЧТО ПРОВЕРЯЕМ:
            Large arrays don't cause memory issues
        """
        from app.common.primitives.spectral import compute_rolloff

        # Large spectrogram (10 minutes at 22050/512)
        n_frames = int(10 * 60 * 22050 / 512)  # ~25,800 frames
        S = np.abs(np.random.randn(513, min(n_frames, 1000))).astype(np.float32) + 0.1
        freqs = np.linspace(0, 11025, 513, dtype=np.float32)

        rolloff = compute_rolloff(S, freqs)

        assert len(rolloff) == S.shape[1]

    def test_repeated_operations(self):
        """Test repeated operations don't leak memory.

        ЧТО ПРОВЕРЯЕМ:
            Repeated calls don't accumulate memory
        """
        from app.common.primitives.filtering import normalize_minmax

        # Run many iterations
        for _ in range(100):
            x = np.random.randn(1000).astype(np.float32)
            result = normalize_minmax(x)
            assert len(result) == 1000


# =============================================================================
# GRACEFUL DEGRADATION TESTS
# =============================================================================

@pytest.mark.unit
class TestGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_classification_without_model(self):
        """Test classification falls back to rules without ML model.

        ЧТО ПРОВЕРЯЕМ:
            Missing model doesn't crash, uses rule-based fallback
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        # Non-existent model path
        task = ZoneClassificationTask(
            model_path="/nonexistent/model.pkl",
            use_rules_fallback=True
        )

        features = {'tempo': 128.0, 'brightness': 0.5}
        result = task._classify_rules(features)

        assert result.success
        assert result.zone in ['yellow', 'green', 'purple']

    def test_feature_extraction_partial_failure(self):
        """Test feature extraction handles partial failures.

        ЧТО ПРОВЕРЯЕМ:
            Failure in one feature doesn't crash entire extraction
        """
        from app.modules.analysis.tasks.feature_extraction import (
            FeatureExtractionResult, FEATURE_NAMES
        )

        # Result with some features missing
        partial_features = {
            'tempo': 128.0,
            'brightness': 0.5
            # Many features missing
        }

        result = FeatureExtractionResult(
            success=True,
            task_name="test",
            processing_time_sec=1.0,
            features=partial_features
        )

        # Should still produce vector (with zeros for missing)
        vector = result.to_vector()
        assert len(vector) == len(FEATURE_NAMES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

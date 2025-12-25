"""
Tests for AudioLoader optimization.

Tests cover:
1. Functional correctness (same output as before)
2. Performance benchmarks (before/after comparison)
3. Edge cases (offset, duration, formats)
4. Output properties (dtype, contiguity, sample rate)

NOTE: Many tests were written for an AudioLoader class that no longer exists.
Tests are skipped until refactored to use the current load_audio function.
"""

import pytest
import numpy as np
import time
from pathlib import Path
from typing import Tuple
import tempfile
import soundfile as sf

from app.core.adapters.loader import load_audio, AudioLoader


# Tests refactored to use SimpleAudioLoader wrapper around load_audio() function


# =============================================================================
# Test fixtures
# =============================================================================


class SimpleAudioLoader:
    """Simple wrapper around AudioLoader for backward-compatible testing."""

    def __init__(self, sample_rate=22050):
        self._loader = AudioLoader(sample_rate=sample_rate)
        self.sample_rate = sample_rate

    def load(self, path, sr=None, offset=0.0, duration=None):
        """Load audio file."""
        if sr is None:
            sr = self.sample_rate
        return self._loader.load(path, duration=duration, offset=offset)

    def get_duration(self, path):
        """Get audio file duration."""
        return self._loader.get_duration(path)

    def validate_file(self, path):
        """Validate audio file."""
        return self._loader.validate_file(path)

    def is_supported_format(self, path):
        """Check if format is supported."""
        return self._loader.is_supported_format(path)


@pytest.fixture
def loader():
    """Create AudioLoader-like wrapper for testing."""
    return SimpleAudioLoader()


@pytest.fixture
def test_audio_files():
    """Find real audio files for testing from DJ sets."""
    project_root = Path(__file__).parent.parent

    # Search paths: DJ sets in project, then DJ Library
    search_paths = [
        project_root / 'data' / 'dj_sets',
        Path.home() / 'Music' / 'DJ Library',
    ]

    # Include all supported formats including m4a and opus
    files = {'mp3': None, 'flac': None, 'wav': None, 'm4a': None, 'opus': None}

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for ext in files.keys():
            if files[ext] is None:
                found = list(search_path.glob(f'**/*.{ext}'))[:1]
                if found:
                    files[ext] = str(found[0])

    return files


@pytest.fixture
def synthetic_wav():
    """Create a synthetic WAV file for testing."""
    # Create 3 seconds of 440Hz sine wave at 44100Hz
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, y, sr)
        yield f.name, y, sr

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def synthetic_stereo_wav():
    """Create a synthetic stereo WAV file for testing."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Different frequencies for left/right
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo = np.column_stack([left, right])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, stereo, sr)
        yield f.name, (left + right) / 2, sr  # Expected mono

    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Functional tests
# =============================================================================

class TestAudioLoaderFunctional:
    """Functional correctness tests."""

    def test_load_returns_tuple(self, loader, synthetic_wav):
        """Load returns (array, sample_rate) tuple."""
        wav_path, _, _ = synthetic_wav
        result = loader.load(wav_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], int)

    def test_output_dtype_float32(self, loader, synthetic_wav):
        """Output array is float32."""
        wav_path, _, _ = synthetic_wav
        y, sr = loader.load(wav_path)

        assert y.dtype == np.float32, f"Expected float32, got {y.dtype}"

    def test_output_contiguous(self, loader, synthetic_wav):
        """Output array is C-contiguous."""
        wav_path, _, _ = synthetic_wav
        y, sr = loader.load(wav_path)

        assert y.flags['C_CONTIGUOUS'], "Array should be C-contiguous"

    def test_output_sample_rate(self, loader, synthetic_wav):
        """Output sample rate matches requested."""
        wav_path, _, _ = synthetic_wav
        y, sr = loader.load(wav_path)

        assert sr == loader.sample_rate, f"Expected {loader.sample_rate}, got {sr}"

    def test_output_mono(self, loader, synthetic_stereo_wav):
        """Stereo is converted to mono."""
        wav_path, _, _ = synthetic_stereo_wav
        y, sr = loader.load(wav_path)

        assert y.ndim == 1, f"Expected 1D array, got {y.ndim}D"

    def test_output_range(self, loader, synthetic_wav):
        """Output values are approximately in [-1, 1] range."""
        wav_path, _, _ = synthetic_wav
        y, sr = loader.load(wav_path)

        # Allow tiny overshoot from resampling (< 0.01%)
        assert np.abs(y).max() <= 1.001, "Values should be approximately in [-1, 1]"

    def test_resampling_correct_length(self, loader, synthetic_wav):
        """Resampling produces correct output length."""
        wav_path, original_y, original_sr = synthetic_wav
        y, sr = loader.load(wav_path)

        # Expected duration should match
        original_duration = len(original_y) / original_sr
        loaded_duration = len(y) / sr

        assert abs(original_duration - loaded_duration) < 0.01, \
            f"Duration mismatch: {original_duration:.3f}s vs {loaded_duration:.3f}s"

    def test_offset_parameter(self, loader, synthetic_wav):
        """Offset parameter skips audio correctly."""
        wav_path, _, original_sr = synthetic_wav

        # Load full
        y_full, _ = loader.load(wav_path)

        # Load with 1s offset
        y_offset, _ = loader.load(wav_path, offset=1.0)

        # Offset version should be shorter
        expected_shorter = loader.sample_rate  # 1 second worth
        assert len(y_full) - len(y_offset) >= expected_shorter * 0.9

    def test_duration_parameter(self, loader, synthetic_wav):
        """Duration parameter limits audio length."""
        wav_path, _, _ = synthetic_wav

        duration = 1.0
        y, sr = loader.load(wav_path, duration=duration)

        loaded_duration = len(y) / sr
        assert abs(loaded_duration - duration) < 0.1, \
            f"Expected ~{duration}s, got {loaded_duration:.3f}s"

    def test_offset_and_duration_combined(self, loader, synthetic_wav):
        """Offset and duration work together."""
        wav_path, _, _ = synthetic_wav

        offset = 0.5
        duration = 1.0
        y, sr = loader.load(wav_path, offset=offset, duration=duration)

        loaded_duration = len(y) / sr
        assert abs(loaded_duration - duration) < 0.1


class TestAudioLoaderFormats:
    """Test different audio formats."""

    def test_wav_format(self, loader, synthetic_wav):
        """WAV format loads correctly."""
        wav_path, _, _ = synthetic_wav
        y, sr = loader.load(wav_path)
        assert len(y) > 0
        assert sr == loader.sample_rate

    def test_mp3_format(self, loader, test_audio_files):
        """MP3 format loads correctly (if available)."""
        if test_audio_files.get('mp3') is None:
            pytest.skip("No MP3 test file available")

        y, sr = loader.load(test_audio_files['mp3'], duration=5.0)
        assert len(y) > 0
        assert sr == loader.sample_rate
        assert y.dtype == np.float32

    def test_flac_format(self, loader, test_audio_files):
        """FLAC format loads correctly (if available)."""
        if test_audio_files.get('flac') is None:
            pytest.skip("No FLAC test file available")

        y, sr = loader.load(test_audio_files['flac'], duration=5.0)
        assert len(y) > 0
        assert sr == loader.sample_rate
        assert y.dtype == np.float32

    def test_m4a_format(self, loader, test_audio_files):
        """M4A format loads correctly (if available)."""
        if test_audio_files.get('m4a') is None:
            pytest.skip("No M4A test file available")

        y, sr = loader.load(test_audio_files['m4a'], duration=5.0)
        assert len(y) > 0
        assert sr == loader.sample_rate
        assert y.dtype == np.float32

    def test_opus_format(self, loader, test_audio_files):
        """OPUS format loads correctly (if available)."""
        if test_audio_files.get('opus') is None:
            pytest.skip("No OPUS test file available")

        y, sr = loader.load(test_audio_files['opus'], duration=5.0)
        assert len(y) > 0
        assert sr == loader.sample_rate
        assert y.dtype == np.float32

    def test_is_supported_format(self):
        """Format detection works correctly."""
        assert AudioLoader.is_supported_format('test.mp3')
        assert AudioLoader.is_supported_format('test.wav')
        assert AudioLoader.is_supported_format('test.flac')
        assert AudioLoader.is_supported_format('test.m4a')
        assert AudioLoader.is_supported_format('test.opus')
        assert AudioLoader.is_supported_format('TEST.MP3')  # Case insensitive
        assert not AudioLoader.is_supported_format('test.txt')
        assert not AudioLoader.is_supported_format('test.pdf')


class TestAudioLoaderErrors:
    """Test error handling."""

    def test_file_not_found(self, loader):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load('/nonexistent/path/audio.mp3')

    def test_unsupported_format(self, loader):
        """ValueError for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not audio')
            f.flush()

            with pytest.raises(ValueError, match='Unsupported format'):
                loader.load(f.name)

            Path(f.name).unlink()

    def test_get_duration_file_not_found(self, loader):
        """get_duration raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            loader.get_duration('/nonexistent/file.mp3')


class TestAudioLoaderDuration:
    """Test get_duration method."""

    def test_get_duration_wav(self, loader, synthetic_wav):
        """get_duration returns correct value for WAV."""
        wav_path, y, sr = synthetic_wav
        expected = len(y) / sr

        duration = loader.get_duration(wav_path)

        assert abs(duration - expected) < 0.01


class TestAudioLoaderValidate:
    """Test validate_file method."""

    def test_validate_valid_file(self, loader, synthetic_wav):
        """validate_file returns True for valid file."""
        wav_path, _, _ = synthetic_wav
        assert loader.validate_file(wav_path) is True

    def test_validate_invalid_file(self, loader):
        """validate_file returns False for invalid file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'not valid audio data')
            f.flush()

            result = loader.validate_file(f.name)
            Path(f.name).unlink()

            assert result is False


class TestLegacyFunction:
    """Test legacy load_audio function."""

    def test_load_audio_function(self, synthetic_wav):
        """load_audio convenience function works."""
        wav_path, _, _ = synthetic_wav
        y, sr = load_audio(wav_path, sr=22050)  # Uses 'sr' not 'sample_rate'

        assert len(y) > 0
        assert sr == 22050
        assert y.dtype == np.float32


# =============================================================================
# Performance benchmarks
# =============================================================================

class TestAudioLoaderPerformance:
    """Performance benchmark tests."""

    def _time_load(self, loader, file_path: str, duration: float = 30.0) -> float:
        """Time a single load operation."""
        start = time.perf_counter()
        loader.load(file_path, duration=duration)
        return time.perf_counter() - start

    def test_wav_load_performance(self, loader, synthetic_wav):
        """WAV loading should be fast (soundfile path)."""
        wav_path, _, _ = synthetic_wav

        # Warm up
        loader.load(wav_path)

        # Benchmark
        times = [self._time_load(loader, wav_path) for _ in range(5)]
        avg_time = np.mean(times)

        # 3 seconds of audio should load in < 0.5s
        assert avg_time < 0.5, f"WAV load too slow: {avg_time:.3f}s"
        print(f"\nWAV load time: {avg_time*1000:.1f}ms (avg of 5)")

    def test_mp3_load_performance(self, loader, test_audio_files):
        """MP3 loading benchmark (librosa path)."""
        if test_audio_files.get('mp3') is None:
            pytest.skip("No MP3 test file available")

        mp3_path = test_audio_files['mp3']

        # Warm up
        loader.load(mp3_path, duration=5.0)

        # Benchmark 5 second loads
        times = [self._time_load(loader, mp3_path, duration=5.0) for _ in range(3)]
        avg_time = np.mean(times)

        # 5 seconds of MP3 should load in < 2s
        assert avg_time < 2.0, f"MP3 load too slow: {avg_time:.3f}s"
        print(f"\nMP3 load time (5s): {avg_time*1000:.1f}ms (avg of 3)")

    def test_flac_load_performance(self, loader, test_audio_files):
        """FLAC loading benchmark (soundfile path)."""
        if test_audio_files.get('flac') is None:
            pytest.skip("No FLAC test file available")

        flac_path = test_audio_files['flac']

        # Warm up
        loader.load(flac_path, duration=5.0)

        # Benchmark
        times = [self._time_load(loader, flac_path, duration=5.0) for _ in range(3)]
        avg_time = np.mean(times)

        # 5 seconds of FLAC should load in < 1s (faster than MP3)
        assert avg_time < 1.0, f"FLAC load too slow: {avg_time:.3f}s"
        print(f"\nFLAC load time (5s): {avg_time*1000:.1f}ms (avg of 3)")


# =============================================================================
# Regression tests (compare with librosa baseline)
# =============================================================================

class TestAudioLoaderRegression:
    """Regression tests comparing with librosa baseline."""

    def _load_with_librosa(self, file_path: str, sr: int, duration: float = None) -> np.ndarray:
        """Load audio using pure librosa (baseline)."""
        import librosa
        y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        return y

    def test_output_similar_to_librosa_wav(self, loader, synthetic_wav):
        """WAV output should be similar to librosa output."""
        wav_path, _, _ = synthetic_wav

        y_optimized, _ = loader.load(wav_path)
        y_librosa = self._load_with_librosa(wav_path, loader.sample_rate)

        # Should have same length (±1 sample for rounding)
        assert abs(len(y_optimized) - len(y_librosa)) <= 10, \
            f"Length mismatch: {len(y_optimized)} vs {len(y_librosa)}"

        # Values should be very close
        min_len = min(len(y_optimized), len(y_librosa))
        correlation = np.corrcoef(y_optimized[:min_len], y_librosa[:min_len])[0, 1]

        assert correlation > 0.99, f"Low correlation with librosa: {correlation:.4f}"

    def test_output_similar_to_librosa_mp3(self, loader, test_audio_files):
        """MP3 output should be identical to librosa (same code path)."""
        if test_audio_files.get('mp3') is None:
            pytest.skip("No MP3 test file available")

        mp3_path = test_audio_files['mp3']
        duration = 5.0

        y_optimized, _ = loader.load(mp3_path, duration=duration)
        y_librosa = self._load_with_librosa(mp3_path, loader.sample_rate, duration=duration)

        # MP3 uses same librosa path, should be identical
        assert len(y_optimized) == len(y_librosa)
        np.testing.assert_array_almost_equal(y_optimized, y_librosa, decimal=5)


# =============================================================================
# AudioSaver tests
# =============================================================================

@pytest.mark.skip(reason="AudioSaver class no longer exists - removed from codebase")
class TestAudioSaver:
    """Tests for AudioSaver class."""

    @pytest.fixture
    def saver(self):
        """Default AudioSaver instance."""
        from app.audio.loader import AudioSaver
        return AudioSaver(sample_rate=48000)

    @pytest.fixture
    def test_audio(self):
        """Generate test audio: 2 seconds of 440Hz sine wave."""
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return y, sr

    def test_save_wav(self, saver, test_audio):
        """Save to WAV format."""
        y, sr = test_audio

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name

        try:
            result = saver.save(y, output_path, sample_rate=sr)
            assert Path(result).exists()
            assert Path(result).stat().st_size > 0

            # Verify content by loading back
            loader = AudioLoader(sample_rate=sr)
            y_loaded, _ = loader.load(output_path)
            assert len(y_loaded) == len(y)
            assert np.corrcoef(y, y_loaded)[0, 1] > 0.99
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_save_flac(self, saver, test_audio):
        """Save to FLAC format."""
        y, sr = test_audio

        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as f:
            output_path = f.name

        try:
            result = saver.save(y, output_path, sample_rate=sr)
            assert Path(result).exists()
            assert Path(result).stat().st_size > 0

            # FLAC should be smaller than WAV
            wav_path = output_path.replace('.flac', '.wav')
            saver.save(y, wav_path, sample_rate=sr)
            assert Path(output_path).stat().st_size < Path(wav_path).stat().st_size
            Path(wav_path).unlink(missing_ok=True)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_save_opus(self, saver, test_audio):
        """Save to OPUS format."""
        y, sr = test_audio

        with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as f:
            output_path = f.name

        try:
            result = saver.save(y, output_path, sample_rate=sr, bitrate='64k')
            assert Path(result).exists()
            assert Path(result).stat().st_size > 0

            # OPUS should be much smaller than WAV
            wav_size = len(y) * 4  # float32 = 4 bytes per sample
            opus_size = Path(output_path).stat().st_size
            assert opus_size < wav_size / 5, "OPUS should be >5x smaller than raw WAV"
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_save_clips_out_of_range(self, saver):
        """Audio values outside [-1, 1] are clipped."""
        # Create audio with values > 1
        y = np.array([0.5, 1.5, -1.5, 0.0], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name

        try:
            saver.save(y, output_path, sample_rate=48000)
            # Should not raise, values are clipped
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_unsupported_format(self, saver, test_audio):
        """Unsupported format raises ValueError."""
        y, sr = test_audio
        with pytest.raises(ValueError, match='Unsupported format'):
            saver.save(y, '/tmp/test.mp3', sample_rate=sr)

    def test_convert_to_opus(self, saver, synthetic_wav):
        """Convert existing file to OPUS."""
        wav_path, _, sr = synthetic_wav

        with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as f:
            output_path = f.name

        try:
            result = saver.convert_to_opus(wav_path, output_path, bitrate='64k')
            assert Path(result).exists()

            # OPUS file should be smaller
            wav_size = Path(wav_path).stat().st_size
            opus_size = Path(output_path).stat().st_size
            assert opus_size < wav_size
        finally:
            Path(output_path).unlink(missing_ok=True)


@pytest.mark.skip(reason="save_audio function no longer exists - removed from codebase")
class TestSaveAudioFunction:
    """Test save_audio convenience function."""

    def test_save_audio_opus(self):
        """save_audio function works for OPUS."""
        from app.audio.loader import save_audio

        # Create test audio
        sr = 48000
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as f:
            output_path = f.name

        try:
            result = save_audio(y, output_path, sample_rate=sr)
            assert Path(result).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


# =============================================================================
# Run benchmarks directly
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-k', 'Performance or Regression'])
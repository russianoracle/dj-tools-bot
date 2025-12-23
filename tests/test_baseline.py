"""
Baseline Tests - Ensure feature extraction produces consistent results.

These tests capture the EXPECTED behavior of the refactored architecture.
If tests fail after refactoring, either:
1. The refactor broke something (fix the code)
2. The new behavior is correct (update the baseline)

Run with: pytest tests/test_baseline.py -v
"""

import pytest
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE_FILE = FIXTURES_DIR / "baseline_results.json"


class TestSTFTCacheBaseline:
    """Test that STFTCache produces consistent results."""

    @pytest.fixture
    def synthetic_audio(self):
        """Create reproducible synthetic audio."""
        np.random.seed(42)
        sr = 22050
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Mix of frequencies for realistic audio
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +   # A4
            0.3 * np.sin(2 * np.pi * 880 * t) +   # A5
            0.2 * np.sin(2 * np.pi * 220 * t) +   # A3
            0.1 * np.random.randn(len(t))          # noise
        ).astype(np.float32)

        return audio, sr

    def _compute_hash(self, arr: np.ndarray) -> str:
        """Compute deterministic hash of array."""
        # Round to avoid floating point differences
        rounded = np.round(arr, decimals=6)
        return hashlib.md5(rounded.tobytes()).hexdigest()[:16]

    def test_stft_cache_deterministic(self, synthetic_audio):
        """STFTCache should produce identical results on same input."""
        from src.core.primitives import compute_stft

        y, sr = synthetic_audio

        # Compute twice
        cache1 = compute_stft(y, sr=sr)
        cache2 = compute_stft(y, sr=sr)

        # Should be identical
        assert np.allclose(cache1.S, cache2.S), "STFT should be deterministic"
        assert np.allclose(cache1.S_db, cache2.S_db), "S_db should be deterministic"

    def test_stft_cache_rms_consistent(self, synthetic_audio):
        """RMS from STFTCache should be consistent."""
        from src.core.primitives import compute_stft

        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)

        # Get RMS twice
        rms1 = cache.get_rms()
        rms2 = cache.get_rms()

        # Should be identical (cached)
        assert np.array_equal(rms1, rms2), "Cached RMS should be identical"

        # Basic sanity checks
        assert len(rms1) > 0, "RMS should not be empty"
        assert np.all(rms1 >= 0), "RMS should be non-negative"
        assert np.mean(rms1) > 0, "Mean RMS should be positive for non-silent audio"

    def test_stft_cache_mfcc_consistent(self, synthetic_audio):
        """MFCC from STFTCache should be consistent."""
        from src.core.primitives import compute_stft

        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)

        mfcc1 = cache.get_mfcc(n_mfcc=13)
        mfcc2 = cache.get_mfcc(n_mfcc=13)

        # Should be identical (cached)
        assert np.array_equal(mfcc1, mfcc2), "Cached MFCC should be identical"

        # Shape check
        assert mfcc1.shape[0] == 13, f"Expected 13 MFCCs, got {mfcc1.shape[0]}"
        assert mfcc1.shape[1] > 0, "MFCC should have time frames"

    def test_stft_cache_spectral_features_consistent(self, synthetic_audio):
        """Spectral features from STFTCache should be consistent."""
        from src.core.primitives import compute_stft

        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)

        # All spectral features
        centroid = cache.get_spectral_centroid()
        rolloff = cache.get_spectral_rolloff()
        flatness = cache.get_spectral_flatness()
        bandwidth = cache.get_spectral_bandwidth()

        # Shape consistency
        n_frames = cache.n_frames
        assert len(centroid) == n_frames, "Centroid frames mismatch"
        assert len(rolloff) == n_frames, "Rolloff frames mismatch"
        assert len(flatness) == n_frames, "Flatness frames mismatch"
        assert len(bandwidth) == n_frames, "Bandwidth frames mismatch"

        # Value sanity
        assert np.all(centroid >= 0), "Centroid should be non-negative"
        assert np.all(rolloff >= 0), "Rolloff should be non-negative"
        assert np.all(flatness >= 0) and np.all(flatness <= 1), "Flatness should be in [0,1]"

    def test_onset_strength_consistent(self, synthetic_audio):
        """Onset strength should be consistent."""
        from src.core.primitives import compute_stft

        y, sr = synthetic_audio
        cache = compute_stft(y, sr=sr)

        onset1 = cache.get_onset_strength()
        onset2 = cache.get_onset_strength()

        assert np.array_equal(onset1, onset2), "Cached onset should be identical"
        assert np.all(onset1 >= 0), "Onset strength should be non-negative"


class TestFeatureExtractionBaseline:
    """Test that FeatureExtractionTask produces expected results."""

    @pytest.fixture
    def audio_context(self):
        """Create AudioContext with synthetic audio."""
        from src.core.tasks.base import AudioContext
        from src.core.primitives import compute_stft

        np.random.seed(42)
        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # More complex audio with beats
        beat_freq = 2.0  # 120 BPM
        audio = (
            0.5 * np.sin(2 * np.pi * 200 * t) *  # bass
            (0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)) +  # pulsing
            0.3 * np.sin(2 * np.pi * 1000 * t) +  # mid
            0.1 * np.random.randn(len(t))
        ).astype(np.float32)

        stft_cache = compute_stft(audio, sr=sr)

        return AudioContext(
            y=audio,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=duration,
        )

    def test_feature_extraction_79_features(self, audio_context):
        """FeatureExtractionTask should produce 79 features."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask, FEATURE_NAMES

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        assert result.success, f"Feature extraction failed: {result.error}"
        assert len(result.features) == 79, f"Expected 79 features, got {len(result.features)}"

        # All feature names should be present
        for name in FEATURE_NAMES:
            assert name in result.features, f"Missing feature: {name}"

    def test_feature_vector_shape(self, audio_context):
        """Feature vector should have correct shape."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        vector = result.to_vector()
        assert vector.shape == (79,), f"Expected shape (79,), got {vector.shape}"
        assert vector.dtype == np.float64 or vector.dtype == np.float32

    def test_feature_values_finite(self, audio_context):
        """All features should be finite values."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        for name, value in result.features.items():
            assert np.isfinite(value), f"Feature {name} is not finite: {value}"


class TestTasksUseSTFTCache:
    """Verify that tasks use STFTCache methods (not direct primitives)."""

    @pytest.fixture
    def audio_context(self):
        """Create AudioContext."""
        from src.core.tasks.base import AudioContext
        from src.core.primitives import compute_stft

        np.random.seed(42)
        sr = 22050
        duration = 5.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
        stft_cache = compute_stft(audio, sr=sr)

        return AudioContext(
            y=audio,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=duration,
        )

    def test_drop_detection_uses_cache(self, audio_context):
        """DropDetectionTask should use STFTCache.get_rms()."""
        from src.core.tasks.drop_detection import DropDetectionTask

        # Record initial cache state
        initial_cache_size = len(audio_context.stft_cache._feature_cache)

        task = DropDetectionTask()
        result = task.execute(audio_context)

        assert result.success, f"Task failed: {result.error}"

        # Cache should have been populated
        assert len(audio_context.stft_cache._feature_cache) >= initial_cache_size
        assert 'rms' in audio_context.stft_cache._feature_cache

    def test_transition_detection_uses_cache(self, audio_context):
        """TransitionDetectionTask should use STFTCache methods."""
        from src.core.tasks.transition_detection import TransitionDetectionTask

        task = TransitionDetectionTask()
        result = task.execute(audio_context)

        assert result.success, f"Task failed: {result.error}"

        # Should have used MFCC and chroma from cache
        # Keys include parameters (e.g., 'mfcc_13_128', 'chroma_12')
        cache_keys = list(audio_context.stft_cache._feature_cache.keys())
        assert any('mfcc' in k for k in cache_keys), f"MFCC should be in cache: {cache_keys}"
        assert any('chroma' in k for k in cache_keys), f"Chroma should be in cache: {cache_keys}"


class TestBlockedImports:
    """Verify that blocked imports raise ImportError."""

    def test_compute_rms_blocked(self):
        """compute_rms should raise ImportError."""
        with pytest.raises(ImportError, match="BLOCKED"):
            from src.core.primitives import compute_rms

    def test_compute_centroid_blocked(self):
        """compute_centroid should raise ImportError."""
        with pytest.raises(ImportError, match="BLOCKED"):
            from src.core.primitives import compute_centroid

    def test_compute_onset_strength_blocked(self):
        """compute_onset_strength should raise ImportError."""
        with pytest.raises(ImportError, match="BLOCKED"):
            from src.core.primitives import compute_onset_strength

    def test_compute_mfcc_blocked(self):
        """compute_mfcc should raise ImportError."""
        with pytest.raises(ImportError, match="BLOCKED"):
            from src.core.primitives import compute_mfcc

    def test_allowed_imports_work(self):
        """Allowed imports should work without error."""
        from src.core.primitives import (
            compute_stft,
            STFTCache,
            smooth_gaussian,
            detect_peaks,
            compute_frequency_bands,
            compute_brightness,
        )

        assert compute_stft is not None
        assert STFTCache is not None
        assert smooth_gaussian is not None


class TestLayerBoundaries:
    """Verify architectural layer boundaries."""

    def test_primitives_no_librosa_in_most_files(self):
        """Primitives (except stft.py, harmonic.py) should not import librosa."""
        import ast
        from pathlib import Path

        primitives_dir = Path("src/core/primitives")
        exceptions = {"stft.py", "harmonic.py", "__init__.py"}

        for py_file in primitives_dir.glob("*.py"):
            if py_file.name in exceptions:
                continue

            content = py_file.read_text()
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert alias.name != "librosa", \
                                f"{py_file.name} should not import librosa"
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("librosa"):
                            pytest.fail(f"{py_file.name} should not import from librosa")
            except SyntaxError:
                pass  # Skip files with syntax errors

    def test_tasks_dont_import_blocked_primitives(self):
        """Tasks should not import blocked primitives directly."""
        import ast
        from pathlib import Path

        tasks_dir = Path("src/core/tasks")
        blocked = {
            "compute_rms", "compute_centroid", "compute_onset_strength",
            "compute_mfcc", "compute_chroma", "compute_tonnetz"
        }

        for py_file in tasks_dir.glob("*.py"):
            if py_file.name in {"__init__.py", "base.py"}:
                continue

            content = py_file.read_text()

            # Check for direct imports from primitives
            for blocked_name in blocked:
                # Pattern: from ..primitives import blocked_name
                if f"from ..primitives import" in content:
                    # This is OK if it's commented or in a NOTE
                    lines = content.split('\n')
                    for line in lines:
                        if f"from ..primitives import" in line and blocked_name in line:
                            if not line.strip().startswith('#') and 'NOTE' not in line:
                                # Check if it's actually imported (not in comment)
                                import re
                                pattern = rf'from \.\.primitives import.*\b{blocked_name}\b'
                                if re.search(pattern, line) and not line.strip().startswith('#'):
                                    pytest.fail(
                                        f"{py_file.name} imports blocked primitive {blocked_name}"
                                    )


# Run baseline capture if executed directly
if __name__ == "__main__":
    print("Running baseline tests...")
    pytest.main([__file__, "-v"])

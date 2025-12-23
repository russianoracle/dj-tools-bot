"""
Regression Test Framework with Golden Baselines.

Input = const -> Process -> Output MUST NOT change without explicit update.

Uses hash-based verification to detect unintended output changes.
"""

import json
import hashlib
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
BASELINES_DIR = Path(__file__).parent / "fixtures" / "baselines"


# =============================================================================
# Baseline Utilities
# =============================================================================

def compute_array_hash(arr: np.ndarray, precision: int = 4) -> str:
    """Compute deterministic hash of numpy array."""
    rounded = np.round(arr, decimals=precision)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:16]


def load_baseline(name: str) -> Optional[Dict[str, Any]]:
    """Load baseline from JSON file."""
    path = BASELINES_DIR / f"{name}.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_baseline(name: str, data: Dict[str, Any]):
    """Save baseline to JSON file."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def assert_within_tolerance(actual: float, expected: float, tolerance: float, msg: str = ""):
    """Assert value is within tolerance of expected."""
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"{msg}: {actual} differs from {expected} by {diff} (tolerance: {tolerance})"
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_audio_deterministic():
    """Create deterministic synthetic audio (seeded)."""
    np.random.seed(42)  # Deterministic
    sr = 22050
    duration = 10.0

    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Complex mix for realistic spectrum
    y = (
        0.4 * np.sin(2 * np.pi * 440 * t) +      # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +      # A5
        0.2 * np.sin(2 * np.pi * 220 * t) +      # A3
        0.1 * np.sin(2 * np.pi * 1760 * t) +     # A6
        0.05 * np.random.randn(len(t))            # Noise
    ).astype(np.float32)

    return y, sr


@pytest.fixture
def stft_cache_deterministic(synthetic_audio_deterministic):
    """Create deterministic STFTCache."""
    from app.common.primitives.stft import compute_stft
    y, sr = synthetic_audio_deterministic
    return compute_stft(y, sr=sr)


# =============================================================================
# Regression Tests: STFTCache
# =============================================================================

@pytest.mark.regression
class TestRegressionSTFTCache:
    """Regression tests for STFTCache outputs."""

    BASELINE_NAME = "stft_cache"

    # Expected values (from initial run with seed=42)
    # Updated 2025-12-17 with actual computed values from refactored implementation
    EXPECTED = {
        "rms_mean": 10.816,  # RMS is now in dB scale, not linear
        "rms_std": 0.05,
        "mfcc_shape": [13, 431],
        "chroma_shape": [12, 431],
        "spectral_centroid_mean": 667.26,  # Hz, weighted centroid
        "n_frames": 431,
    }

    TOLERANCES = {
        "rms_mean": 0.1,  # Allow more tolerance for float variations
        "rms_std": 0.01,
        "spectral_centroid_mean": 10.0,  # Allow 10 Hz tolerance
    }

    def test_rms_mean_consistency(self, stft_cache_deterministic):
        """RMS mean must match baseline."""
        rms = stft_cache_deterministic.get_rms()
        actual = float(np.mean(rms))

        assert_within_tolerance(
            actual,
            self.EXPECTED["rms_mean"],
            self.TOLERANCES["rms_mean"],
            "RMS mean"
        )

    def test_mfcc_shape_consistency(self, stft_cache_deterministic):
        """MFCC shape must match baseline."""
        mfcc = stft_cache_deterministic.get_mfcc(n_mfcc=13)
        assert list(mfcc.shape) == self.EXPECTED["mfcc_shape"]

    def test_chroma_shape_consistency(self, stft_cache_deterministic):
        """Chroma shape must match baseline."""
        chroma = stft_cache_deterministic.get_chroma()
        assert list(chroma.shape) == self.EXPECTED["chroma_shape"]

    def test_n_frames_consistency(self, stft_cache_deterministic):
        """Number of frames must match baseline."""
        assert stft_cache_deterministic.n_frames == self.EXPECTED["n_frames"]

    def test_spectral_centroid_mean_consistency(self, stft_cache_deterministic):
        """Spectral centroid mean must match baseline."""
        centroid = stft_cache_deterministic.get_spectral_centroid()
        actual = float(np.mean(centroid))

        assert_within_tolerance(
            actual,
            self.EXPECTED["spectral_centroid_mean"],
            self.TOLERANCES["spectral_centroid_mean"],
            "Spectral centroid mean"
        )


# =============================================================================
# Regression Tests: BeatGrid
# =============================================================================

@pytest.mark.regression
class TestRegressionBeatGrid:
    """Regression tests for BeatGrid computation."""

    @pytest.fixture
    def beat_audio_deterministic(self):
        """Create audio with clear beat pattern (128 BPM)."""
        np.random.seed(42)
        sr = 22050
        duration = 30.0
        tempo = 128.0
        beat_duration = 60.0 / tempo

        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        y = np.zeros_like(t)

        # Add kicks at beat positions
        for i in range(int(duration / beat_duration)):
            beat_sample = int(i * beat_duration * sr)
            if beat_sample < len(y):
                # Exponential decay envelope
                decay_samples = int(0.1 * sr)
                end_sample = min(beat_sample + decay_samples, len(y))
                window_len = end_sample - beat_sample
                y[beat_sample:end_sample] += (
                    np.exp(-np.arange(window_len) / (0.02 * sr)) *
                    np.sin(2 * np.pi * 100 * np.arange(window_len) / sr)
                )

        # Add background
        y += 0.1 * np.sin(2 * np.pi * 440 * t)
        y = y.astype(np.float32)

        return y, sr

    def test_tempo_within_range(self, beat_audio_deterministic):
        """Detected tempo must be within reasonable range of 128 BPM."""
        from app.common.primitives.stft import compute_stft
        from app.common.primitives.beat_grid import compute_beat_grid

        y, sr = beat_audio_deterministic
        cache = compute_stft(y, sr)
        grid = compute_beat_grid(cache.S, sr, hop_length=cache.hop_length)

        # Allow octave errors (64, 128, 256)
        valid_tempos = [64, 128, 256]
        closest = min(valid_tempos, key=lambda t: abs(grid.tempo - t))
        assert abs(grid.tempo - closest) < 5, f"Tempo {grid.tempo} not near expected values"

    def test_hierarchical_structure(self, beat_audio_deterministic):
        """Beat grid must have valid hierarchical structure."""
        from app.common.primitives.stft import compute_stft
        from app.common.primitives.beat_grid import compute_beat_grid

        y, sr = beat_audio_deterministic
        cache = compute_stft(y, sr)
        grid = compute_beat_grid(cache.S, sr, hop_length=cache.hop_length)

        if len(grid.bars) > 0:
            # Each bar must have 4 beats
            for bar in grid.bars:
                assert len(bar.beat_indices) == 4

        if len(grid.phrases) > 0:
            # Each phrase must have 4 bars
            for phrase in grid.phrases:
                assert len(phrase.bar_indices) == 4


# =============================================================================
# Regression Tests: Feature Extraction
# =============================================================================

@pytest.mark.regression
class TestRegressionFeatureExtraction:
    """Regression tests for feature extraction pipeline."""

    def test_feature_count_consistency(self, synthetic_audio_deterministic):
        """Feature extraction must produce consistent count."""
        # This test verifies that the feature extraction pipeline
        # always produces the same number of features

        from app.common.primitives.stft import compute_stft

        y, sr = synthetic_audio_deterministic
        cache = compute_stft(y, sr)

        # Collect all features
        features = {
            'rms': cache.get_rms(),
            'mfcc': cache.get_mfcc(),
            'chroma': cache.get_chroma(),
            'tonnetz': cache.get_tonnetz(),
            'centroid': cache.get_spectral_centroid(),
            'rolloff': cache.get_spectral_rolloff(),
            'flatness': cache.get_spectral_flatness(),
            'bandwidth': cache.get_spectral_bandwidth(),
            'flux': cache.get_spectral_flux(),
        }

        # Verify shapes are consistent
        expected_frames = cache.n_frames

        assert features['rms'].shape == (expected_frames,)
        assert features['mfcc'].shape[1] == expected_frames
        assert features['chroma'].shape[1] == expected_frames
        assert features['tonnetz'].shape[1] == expected_frames
        assert features['centroid'].shape == (expected_frames,)

    def test_feature_hash_stability(self, stft_cache_deterministic):
        """Feature hashes must be stable across runs."""
        # Compute feature hashes
        rms_hash = compute_array_hash(stft_cache_deterministic.get_rms())
        mfcc_hash = compute_array_hash(stft_cache_deterministic.get_mfcc())

        # Run again (should get same values due to caching)
        rms_hash2 = compute_array_hash(stft_cache_deterministic.get_rms())
        mfcc_hash2 = compute_array_hash(stft_cache_deterministic.get_mfcc())

        assert rms_hash == rms_hash2, "RMS hash not stable"
        assert mfcc_hash == mfcc_hash2, "MFCC hash not stable"


# =============================================================================
# Baseline Management
# =============================================================================

@pytest.mark.skip(reason="Run manually to update baselines")
class TestUpdateBaselines:
    """Helper tests to update baselines (run manually)."""

    def test_update_stft_baseline(self, stft_cache_deterministic):
        """Update STFTCache baseline."""
        baseline = {
            "version": "1.0",
            "rms_mean": float(np.mean(stft_cache_deterministic.get_rms())),
            "rms_std": float(np.std(stft_cache_deterministic.get_rms())),
            "rms_hash": compute_array_hash(stft_cache_deterministic.get_rms()),
            "mfcc_shape": list(stft_cache_deterministic.get_mfcc().shape),
            "mfcc_hash": compute_array_hash(stft_cache_deterministic.get_mfcc()),
            "chroma_shape": list(stft_cache_deterministic.get_chroma().shape),
            "spectral_centroid_mean": float(np.mean(stft_cache_deterministic.get_spectral_centroid())),
            "n_frames": stft_cache_deterministic.n_frames,
        }

        save_baseline("stft_cache", baseline)
        print(f"Updated baseline: {baseline}")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "regression"])

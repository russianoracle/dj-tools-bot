"""
Baseline Tests with Real Audio Files.

Uses Pioneer DJ demo tracks and Josh Baker set for real-world baseline testing.
These tests verify that the refactored architecture produces consistent results.

Run with: pytest tests/test_baseline_real_tracks.py -v
"""

import pytest
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Test audio files
PIONEER_DEMO_1 = Path("/Users/artemgusarov/Music/PioneerDJ/Demo Tracks/Demo Track 1.mp3")
PIONEER_DEMO_2 = Path("/Users/artemgusarov/Music/PioneerDJ/Demo Tracks/Demo Track 2.mp3")
JOSH_BAKER_SET = Path("/Users/artemgusarov/Music/ai_app/mood-classifier/data/dj_sets/josh-baker/boiler-room---josh-baker-boiler-room-london.mp3")

# Baseline storage
BASELINE_FILE = Path(__file__).parent / "fixtures" / "real_track_baseline.json"


def compute_feature_hash(features: Dict[str, float], precision: int = 4) -> str:
    """Compute deterministic hash of feature dict."""
    # Round values to avoid floating point noise
    rounded = {k: round(v, precision) for k, v in sorted(features.items())}
    return hashlib.md5(json.dumps(rounded).encode()).hexdigest()[:16]


class TestRealTrackFeatureExtraction:
    """Test feature extraction on real audio files."""

    @pytest.fixture(scope="class")
    def audio_loader(self):
        """Get AudioLoader instance."""
        from app.core.adapters.loader import AudioLoader
        return AudioLoader()

    @pytest.fixture(scope="class")
    def demo_track_1_context(self, audio_loader):
        """Load Pioneer Demo Track 1 and create AudioContext."""
        if not PIONEER_DEMO_1.exists():
            pytest.skip(f"Demo track not found: {PIONEER_DEMO_1}")

        from app.core.tasks.base import AudioContext
        from app.common.primitives import compute_stft

        y, sr = audio_loader.load(str(PIONEER_DEMO_1))
        stft_cache = compute_stft(y, sr=sr)

        return AudioContext(
            y=y,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=len(y) / sr,
        )

    @pytest.fixture(scope="class")
    def demo_track_2_context(self, audio_loader):
        """Load Pioneer Demo Track 2 and create AudioContext."""
        if not PIONEER_DEMO_2.exists():
            pytest.skip(f"Demo track not found: {PIONEER_DEMO_2}")

        from app.core.tasks.base import AudioContext
        from app.common.primitives import compute_stft

        y, sr = audio_loader.load(str(PIONEER_DEMO_2))
        stft_cache = compute_stft(y, sr=sr)

        return AudioContext(
            y=y,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=len(y) / sr,
        )

    def test_feature_extraction_79_features_demo1(self, demo_track_1_context):
        """Demo Track 1 should produce exactly 79 features."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask, FEATURE_NAMES

        task = FeatureExtractionTask()
        result = task.execute(demo_track_1_context)

        assert result.success, f"Feature extraction failed: {result.error}"
        assert len(result.features) == 79, f"Expected 79 features, got {len(result.features)}"

        # All features must be present
        missing = set(FEATURE_NAMES) - set(result.features.keys())
        assert not missing, f"Missing features: {missing}"

    def test_feature_extraction_79_features_demo2(self, demo_track_2_context):
        """Demo Track 2 should produce exactly 79 features."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask, FEATURE_NAMES

        task = FeatureExtractionTask()
        result = task.execute(demo_track_2_context)

        assert result.success, f"Feature extraction failed: {result.error}"
        assert len(result.features) == 79, f"Expected 79 features, got {len(result.features)}"

    def test_all_features_finite(self, demo_track_1_context):
        """All extracted features must be finite values."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(demo_track_1_context)

        for name, value in result.features.items():
            assert np.isfinite(value), f"Feature {name} is not finite: {value}"

    def test_feature_extraction_deterministic(self, demo_track_1_context):
        """Feature extraction should be deterministic."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()

        # Extract twice
        result1 = task.execute(demo_track_1_context)
        result2 = task.execute(demo_track_1_context)

        assert result1.success and result2.success

        # Compare all features
        for name in result1.features:
            v1, v2 = result1.features[name], result2.features[name]
            assert np.isclose(v1, v2, rtol=1e-5), \
                f"Feature {name} not deterministic: {v1} vs {v2}"

    def test_stft_cache_populated(self, demo_track_1_context):
        """STFTCache should be populated after feature extraction."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        # Clear cache
        demo_track_1_context.stft_cache._feature_cache.clear()

        task = FeatureExtractionTask()
        result = task.execute(demo_track_1_context)

        assert result.success

        # Check cache populated
        cache = demo_track_1_context.stft_cache._feature_cache
        cache_keys = list(cache.keys())

        # Keys include parameters, so check for prefix matches
        assert 'rms' in cache_keys, f"RMS should be in cache: {cache_keys}"
        assert any('mfcc' in k for k in cache_keys), f"MFCC should be in cache: {cache_keys}"
        assert any('chroma' in k for k in cache_keys), f"Chroma should be in cache: {cache_keys}"
        assert any('tonnetz' in k for k in cache_keys), f"Tonnetz should be in cache: {cache_keys}"

    def test_different_tracks_different_features(self, demo_track_1_context, demo_track_2_context):
        """Different tracks should produce different feature values."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()

        result1 = task.execute(demo_track_1_context)
        result2 = task.execute(demo_track_2_context)

        assert result1.success and result2.success

        # Features should be different (not identical)
        hash1 = compute_feature_hash(result1.features)
        hash2 = compute_feature_hash(result2.features)

        assert hash1 != hash2, "Different tracks should have different feature hashes"

    def test_tempo_in_reasonable_range(self, demo_track_1_context):
        """Tempo should be in reasonable DJ range."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(demo_track_1_context)

        tempo = result.features['tempo']
        assert 60 <= tempo <= 200, f"Tempo {tempo} BPM outside reasonable range"


class TestRealTrackSTFTCache:
    """Test STFTCache methods on real audio."""

    @pytest.fixture(scope="class")
    def stft_cache(self):
        """Create STFTCache from Demo Track 1."""
        if not PIONEER_DEMO_1.exists():
            pytest.skip(f"Demo track not found: {PIONEER_DEMO_1}")

        from app.core.adapters.loader import AudioLoader
        from app.common.primitives import compute_stft

        loader = AudioLoader()
        y, sr = loader.load(str(PIONEER_DEMO_1))
        return compute_stft(y, sr=sr)

    def test_rms_consistent(self, stft_cache):
        """RMS should be cached and consistent."""
        rms1 = stft_cache.get_rms()
        rms2 = stft_cache.get_rms()

        assert np.array_equal(rms1, rms2), "RMS should be identical from cache"
        assert len(rms1) > 0, "RMS should not be empty"
        assert np.all(rms1 >= 0), "RMS should be non-negative"

    def test_mfcc_consistent(self, stft_cache):
        """MFCC should be cached and consistent."""
        mfcc1 = stft_cache.get_mfcc(n_mfcc=13)
        mfcc2 = stft_cache.get_mfcc(n_mfcc=13)

        assert np.array_equal(mfcc1, mfcc2), "MFCC should be identical from cache"
        assert mfcc1.shape[0] == 13, f"Expected 13 MFCCs, got {mfcc1.shape[0]}"

    def test_chroma_consistent(self, stft_cache):
        """Chroma should be cached and consistent."""
        chroma1 = stft_cache.get_chroma()
        chroma2 = stft_cache.get_chroma()

        assert np.array_equal(chroma1, chroma2), "Chroma should be identical from cache"
        assert chroma1.shape[0] == 12, f"Expected 12 chroma bins, got {chroma1.shape[0]}"

    def test_spectral_features_consistent(self, stft_cache):
        """All spectral features should be cached."""
        features = {
            'centroid': stft_cache.get_spectral_centroid(),
            'rolloff': stft_cache.get_spectral_rolloff(),
            'flatness': stft_cache.get_spectral_flatness(),
            'flux': stft_cache.get_spectral_flux(),
            'bandwidth': stft_cache.get_spectral_bandwidth(),
        }

        # Get again - should be from cache
        for name, arr in features.items():
            cached = getattr(stft_cache, f'get_spectral_{name}')()
            assert np.array_equal(arr, cached), f"{name} should be identical from cache"

    def test_onset_strength_consistent(self, stft_cache):
        """Onset strength should be cached."""
        onset1 = stft_cache.get_onset_strength()
        onset2 = stft_cache.get_onset_strength()

        assert np.array_equal(onset1, onset2), "Onset should be identical from cache"
        assert np.all(onset1 >= 0), "Onset strength should be non-negative"

    def test_tempo_beats_consistent(self, stft_cache):
        """Tempo and beats should be cached."""
        tempo1, _ = stft_cache.get_tempo()
        tempo2, _ = stft_cache.get_tempo()

        assert tempo1 == tempo2, "Tempo should be identical from cache"

        beats1, times1 = stft_cache.get_beats()
        beats2, times2 = stft_cache.get_beats()

        assert np.array_equal(beats1, beats2), "Beats should be identical from cache"


class TestRealTrackTasks:
    """Test various tasks on real audio."""

    @pytest.fixture(scope="class")
    def audio_context(self):
        """Create AudioContext from Demo Track 1."""
        if not PIONEER_DEMO_1.exists():
            pytest.skip(f"Demo track not found: {PIONEER_DEMO_1}")

        from app.core.adapters.loader import AudioLoader
        from app.core.tasks.base import AudioContext
        from app.common.primitives import compute_stft

        loader = AudioLoader()
        y, sr = loader.load(str(PIONEER_DEMO_1))
        stft_cache = compute_stft(y, sr=sr)

        return AudioContext(
            y=y,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=len(y) / sr,
        )

    def test_drop_detection_uses_cache(self, audio_context):
        """DropDetectionTask should use STFTCache.get_rms()."""
        from app.modules.analysis.tasks.drop_detection import DropDetectionTask

        # Clear cache
        audio_context.stft_cache._feature_cache.clear()

        task = DropDetectionTask()
        result = task.execute(audio_context)

        assert result.success, f"Task failed: {result.error}"
        assert 'rms' in audio_context.stft_cache._feature_cache, \
            "RMS should be in cache after DropDetectionTask"

    def test_transition_detection_uses_cache(self, audio_context):
        """TransitionDetectionTask should use STFTCache methods."""
        from app.modules.analysis.tasks.transition_detection import TransitionDetectionTask

        # Clear cache
        audio_context.stft_cache._feature_cache.clear()

        task = TransitionDetectionTask()
        result = task.execute(audio_context)

        assert result.success, f"Task failed: {result.error}"

        # Should have used MFCC and chroma
        cache_keys = list(audio_context.stft_cache._feature_cache.keys())
        assert any('mfcc' in k for k in cache_keys), f"MFCC should be in cache: {cache_keys}"


class TestBaselineCapture:
    """Capture and verify baseline results."""

    @pytest.fixture(scope="class")
    def audio_context_demo1(self):
        """Create AudioContext from Demo Track 1."""
        if not PIONEER_DEMO_1.exists():
            pytest.skip(f"Demo track not found: {PIONEER_DEMO_1}")

        from app.core.adapters.loader import AudioLoader
        from app.core.tasks.base import AudioContext
        from app.common.primitives import compute_stft

        loader = AudioLoader()
        y, sr = loader.load(str(PIONEER_DEMO_1))
        stft_cache = compute_stft(y, sr=sr)

        return AudioContext(
            y=y,
            sr=sr,
            stft_cache=stft_cache,
            duration_sec=len(y) / sr,
        )

    @pytest.fixture(scope="class")
    def baseline_features(self, audio_context_demo1):
        """Extract features for baseline capture."""
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context_demo1)

        return result.features

    @pytest.fixture(scope="class")
    def baseline_drops(self, audio_context_demo1):
        """Extract drop detection results for baseline."""
        from app.modules.analysis.tasks.drop_detection import DropDetectionTask

        task = DropDetectionTask()
        result = task.execute(audio_context_demo1)

        return {
            'drop_count': len(result.drops) if result.success else 0,
            'drops': [
                {'time': round(d.time, 3), 'confidence': round(d.confidence, 3)}
                for d in (result.drops if result.success else [])
            ]
        }

    @pytest.fixture(scope="class")
    def baseline_transitions(self, audio_context_demo1):
        """Extract transition detection results for baseline."""
        from app.modules.analysis.tasks.transition_detection import TransitionDetectionTask

        task = TransitionDetectionTask()
        result = task.execute(audio_context_demo1)

        return {
            'mixin_count': len(result.mixins) if result.success else 0,
            'mixout_count': len(result.mixouts) if result.success else 0,
            'mixins': [
                {'time': round(m.time, 3), 'confidence': round(m.confidence, 3)}
                for m in (result.mixins if result.success else [])
            ],
            'mixouts': [
                {'time': round(m.time, 3), 'confidence': round(m.confidence, 3)}
                for m in (result.mixouts if result.success else [])
            ]
        }

    @pytest.fixture(scope="class")
    def baseline_segmentation(self, audio_context_demo1):
        """Extract segmentation results for baseline."""
        from app.modules.analysis.tasks.segmentation import SegmentationTask

        task = SegmentationTask()
        result = task.execute(audio_context_demo1)

        # SegmentBoundary has time_sec attribute
        boundaries = result.boundaries if result.success else []
        return {
            'segment_count': len(boundaries),
            'boundaries': [
                {'time': round(b.time_sec, 3), 'confidence': round(b.confidence, 3)}
                for b in boundaries
            ],
        }

    def test_capture_baseline(self, baseline_features, baseline_drops, baseline_transitions, baseline_segmentation):
        """Capture baseline if it doesn't exist or is incomplete."""
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)

        need_capture = False
        if not BASELINE_FILE.exists():
            need_capture = True
        else:
            # Check if baseline has all required keys
            existing = json.loads(BASELINE_FILE.read_text())
            if 'demo_track_1' not in existing:
                need_capture = True
            elif 'drops' not in existing.get('demo_track_1', {}):
                need_capture = True

        if need_capture:
            # Capture new baseline
            baseline = {
                'demo_track_1': {
                    'hash': compute_feature_hash(baseline_features),
                    'features': {k: round(v, 6) for k, v in baseline_features.items()},
                    'drops': baseline_drops,
                    'transitions': baseline_transitions,
                    'segmentation': baseline_segmentation,
                    'captured_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            }
            BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
            pytest.skip("Baseline captured - run again to verify")

    def test_verify_baseline(self, baseline_features):
        """Verify extracted features match baseline."""
        if not BASELINE_FILE.exists():
            pytest.skip("No baseline captured yet")

        baseline = json.loads(BASELINE_FILE.read_text())

        if 'demo_track_1' not in baseline:
            pytest.skip("demo_track_1 not in baseline")

        expected = baseline['demo_track_1']['features']
        expected_hash = baseline['demo_track_1']['hash']

        # Compare hashes
        actual_hash = compute_feature_hash(baseline_features)

        if actual_hash != expected_hash:
            # Find differences
            diffs = []
            for name in expected:
                exp_val = expected[name]
                act_val = baseline_features.get(name, 0.0)
                if not np.isclose(exp_val, act_val, rtol=1e-3):
                    diffs.append(f"{name}: expected {exp_val:.6f}, got {act_val:.6f}")

            if diffs:
                pytest.fail(
                    f"Baseline mismatch!\n"
                    f"Expected hash: {expected_hash}\n"
                    f"Actual hash: {actual_hash}\n"
                    f"Differences:\n" + "\n".join(diffs[:10])
                )

    def test_verify_drops_baseline(self, baseline_drops):
        """Verify drop detection matches baseline."""
        if not BASELINE_FILE.exists():
            pytest.skip("No baseline captured yet")

        baseline = json.loads(BASELINE_FILE.read_text())

        if 'demo_track_1' not in baseline or 'drops' not in baseline['demo_track_1']:
            pytest.skip("drops not in baseline")

        expected = baseline['demo_track_1']['drops']
        assert baseline_drops['drop_count'] == expected['drop_count'], \
            f"Drop count mismatch: {baseline_drops['drop_count']} vs {expected['drop_count']}"

    def test_verify_transitions_baseline(self, baseline_transitions):
        """Verify transition detection matches baseline."""
        if not BASELINE_FILE.exists():
            pytest.skip("No baseline captured yet")

        baseline = json.loads(BASELINE_FILE.read_text())

        if 'demo_track_1' not in baseline or 'transitions' not in baseline['demo_track_1']:
            pytest.skip("transitions not in baseline")

        expected = baseline['demo_track_1']['transitions']
        assert baseline_transitions['mixin_count'] == expected['mixin_count'], \
            f"Mixin count mismatch: {baseline_transitions['mixin_count']} vs {expected['mixin_count']}"

    def test_verify_segmentation_baseline(self, baseline_segmentation):
        """Verify segmentation matches baseline."""
        if not BASELINE_FILE.exists():
            pytest.skip("No baseline captured yet")

        baseline = json.loads(BASELINE_FILE.read_text())

        if 'demo_track_1' not in baseline or 'segmentation' not in baseline['demo_track_1']:
            pytest.skip("segmentation not in baseline")

        expected = baseline['demo_track_1']['segmentation']
        assert baseline_segmentation['segment_count'] == expected['segment_count'], \
            f"Segment count mismatch: {baseline_segmentation['segment_count']} vs {expected['segment_count']}"


# CLI for baseline capture
if __name__ == "__main__":
    print("Running real track baseline tests...")
    pytest.main([__file__, "-v"])

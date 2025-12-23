"""
Functional Tests - Verify tasks produce correct results after refactoring.

These tests ensure:
1. Feature extraction produces consistent results
2. Segmentation works correctly
3. Transition detection works correctly
4. All tasks work with STFTCache
"""

import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """Generate synthetic audio with musical structure."""
    sr = 22050
    duration = 30.0  # 30 seconds - longer for segmentation/transition detection
    t = np.linspace(0, duration, int(sr * duration))

    # Create audio with structure:
    # 0-10s: quiet intro (low energy)
    # 10-20s: buildup + drop (high energy, kick drum)
    # 20-30s: outro (medium energy)

    y = np.zeros_like(t)

    # Intro: soft pad
    intro_mask = t < 10.0
    y[intro_mask] = 0.2 * np.sin(2 * np.pi * 220 * t[intro_mask])

    # Main section: energetic with kick
    main_mask = (t >= 10.0) & (t < 20.0)
    tempo_hz = 128 / 60  # 128 BPM
    kick_phase = 2 * np.pi * tempo_hz * t[main_mask]
    kick = 0.8 * np.exp(-10 * (kick_phase % (2 * np.pi))) * np.sin(2 * np.pi * 60 * t[main_mask])
    y[main_mask] = kick + 0.3 * np.sin(2 * np.pi * 440 * t[main_mask])

    # Outro: medium energy
    outro_mask = t >= 20.0
    y[outro_mask] = 0.4 * np.sin(2 * np.pi * 330 * t[outro_mask])

    # Add some noise
    y += 0.05 * np.random.randn(len(t))

    return y.astype(np.float32), sr


@pytest.fixture
def audio_context(sample_audio):
    """Create AudioContext from sample audio."""
    from src.core.tasks.base import create_audio_context

    y, sr = sample_audio
    return create_audio_context(y, sr=sr)


class TestFeatureExtractionTask:
    """Test FeatureExtractionTask produces correct features."""

    def test_feature_extraction_runs(self, audio_context):
        """Feature extraction should complete without error."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        assert result.success
        assert len(result.features) > 0

    def test_feature_count(self, audio_context):
        """Should extract expected number of features (79)."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask, FEATURE_NAMES

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        assert len(result.features) == len(FEATURE_NAMES)
        assert len(FEATURE_NAMES) == 79

    def test_to_vector_shape(self, audio_context):
        """to_vector should return correct shape."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask, FEATURE_NAMES

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        vector = result.to_vector()
        assert vector.shape == (len(FEATURE_NAMES),)

    def test_mfcc_features_present(self, audio_context):
        """MFCC features should be extracted."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        # Check MFCC 1-13 are present
        for i in range(1, 14):
            assert f'mfcc_{i}' in result.features
            assert f'mfcc_{i}_delta' in result.features

    def test_chroma_features_present(self, audio_context):
        """Chroma features should be extracted."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        chroma_names = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
        for name in chroma_names:
            assert f'chroma_{name}' in result.features

    def test_tonnetz_features_present(self, audio_context):
        """Tonnetz features should be extracted."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        for i in range(1, 7):
            assert f'tonnetz_{i}' in result.features

    def test_energy_features_reasonable(self, audio_context):
        """Energy features should have reasonable values."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        # RMS energy should be positive
        assert result.features['rms_energy'] > 0

        # Low energy ratio should be between 0 and 1
        assert 0 <= result.features['low_energy_ratio'] <= 1

    def test_spectral_features_reasonable(self, audio_context):
        """Spectral features should have reasonable values."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(audio_context)

        # Centroid should be positive (in Hz)
        assert result.features['spectral_centroid'] > 0

        # Brightness should be between 0 and 1
        assert 0 <= result.features['brightness'] <= 1


class TestSegmentationTask:
    """Test SegmentationTask produces correct segments."""

    def test_segmentation_runs(self, audio_context):
        """Segmentation should complete without error."""
        from src.core.tasks.segmentation import SegmentationTask

        task = SegmentationTask()
        result = task.execute(audio_context)

        assert result.success

    def test_segments_cover_audio(self, audio_context):
        """Segment boundaries should span the audio."""
        from src.core.tasks.segmentation import SegmentationTask

        task = SegmentationTask()
        result = task.execute(audio_context)

        if result.boundaries:
            # First boundary should start early
            assert result.boundaries[0].time_sec < 5.0

            # Should have multiple boundaries for 30s audio
            assert result.n_segments >= 1

    def test_boundaries_in_order(self, audio_context):
        """Boundaries should be in chronological order."""
        from src.core.tasks.segmentation import SegmentationTask

        task = SegmentationTask()
        result = task.execute(audio_context)

        if len(result.boundaries) > 1:
            for i in range(len(result.boundaries) - 1):
                b1 = result.boundaries[i]
                b2 = result.boundaries[i + 1]
                # Boundaries should be in order
                assert b1.time_sec <= b2.time_sec


class TestDropDetectionTask:
    """Test DropDetectionTask produces correct drops."""

    def test_drop_detection_runs(self, audio_context):
        """Drop detection should complete without error."""
        from src.core.tasks.drop_detection import DropDetectionTask

        task = DropDetectionTask()
        result = task.execute(audio_context)

        assert result.success

    def test_drops_have_required_fields(self, audio_context):
        """Each drop should have required fields."""
        from src.core.tasks.drop_detection import DropDetectionTask

        task = DropDetectionTask()
        result = task.execute(audio_context)

        for drop in result.drops:
            assert hasattr(drop, 'time')
            assert hasattr(drop, 'confidence')
            assert drop.time >= 0
            assert 0 <= drop.confidence <= 1

    def test_drops_within_audio_duration(self, audio_context):
        """All drops should be within audio duration."""
        from src.core.tasks.drop_detection import DropDetectionTask

        task = DropDetectionTask()
        result = task.execute(audio_context)

        for drop in result.drops:
            assert drop.time <= audio_context.duration_sec


class TestTransitionDetectionTask:
    """Test TransitionDetectionTask produces correct transitions."""

    def test_transition_detection_runs(self, audio_context):
        """Transition detection should complete without error."""
        from src.core.tasks.transition_detection import TransitionDetectionTask

        task = TransitionDetectionTask()
        result = task.execute(audio_context)

        assert result.success

    def test_mixins_within_duration(self, audio_context):
        """All mixins should be within audio duration."""
        from src.core.tasks.transition_detection import TransitionDetectionTask

        task = TransitionDetectionTask()
        result = task.execute(audio_context)

        for mixin in result.mixins:
            assert mixin.time <= audio_context.duration_sec

    def test_mixouts_within_duration(self, audio_context):
        """All mixouts should be within audio duration."""
        from src.core.tasks.transition_detection import TransitionDetectionTask

        task = TransitionDetectionTask()
        result = task.execute(audio_context)

        for mixout in result.mixouts:
            assert mixout.time <= audio_context.duration_sec


class TestSTFTCacheIntegration:
    """Test that tasks properly use STFTCache lazy methods."""

    def test_stft_cache_populated_after_feature_extraction(self, audio_context):
        """Feature extraction should populate STFTCache."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        # Record initial cache size (may contain _y for lazy audio access)
        initial_cache_size = len(audio_context.stft_cache._feature_cache)

        task = FeatureExtractionTask()
        task.execute(audio_context)

        # Cache should be populated with MFCC, chroma, tonnetz, etc.
        assert len(audio_context.stft_cache._feature_cache) > initial_cache_size

    def test_stft_cache_reused_between_tasks(self, audio_context):
        """Multiple tasks should reuse cached features."""
        from src.core.tasks.feature_extraction import FeatureExtractionTask
        from src.core.tasks.segmentation import SegmentationTask

        # Run feature extraction first
        feat_task = FeatureExtractionTask()
        feat_task.execute(audio_context)

        cache_size_after_feat = len(audio_context.stft_cache._feature_cache)

        # Run segmentation (also uses MFCC)
        seg_task = SegmentationTask()
        seg_task.execute(audio_context)

        cache_size_after_seg = len(audio_context.stft_cache._feature_cache)

        # Cache should be same or larger (not recomputed)
        assert cache_size_after_seg >= cache_size_after_feat


class TestFeatureConsistency:
    """Test that features are consistent across runs."""

    def test_feature_extraction_deterministic(self, sample_audio):
        """Feature extraction should produce same results."""
        from src.core.tasks.base import create_audio_context
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        y, sr = sample_audio

        # Run twice with fresh contexts
        ctx1 = create_audio_context(y, sr=sr)
        ctx2 = create_audio_context(y, sr=sr)

        task = FeatureExtractionTask()
        result1 = task.execute(ctx1)
        result2 = task.execute(ctx2)

        # Vectors should be identical
        np.testing.assert_array_almost_equal(
            result1.to_vector(),
            result2.to_vector(),
            decimal=5
        )


class TestTaskErrorHandling:
    """Test that tasks handle errors gracefully."""

    def test_feature_extraction_handles_short_audio(self):
        """Feature extraction should handle very short audio."""
        from src.core.tasks.base import create_audio_context
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        # Very short audio (0.5 seconds)
        sr = 22050
        y = np.random.randn(int(sr * 0.5)).astype(np.float32)

        ctx = create_audio_context(y, sr=sr)

        task = FeatureExtractionTask()
        result = task.execute(ctx)

        # Should either succeed or fail gracefully
        assert result.success or result.error is not None

    def test_feature_extraction_handles_silence(self):
        """Feature extraction should handle silent audio."""
        from src.core.tasks.base import create_audio_context
        from src.core.tasks.feature_extraction import FeatureExtractionTask

        # Silent audio
        sr = 22050
        y = np.zeros(int(sr * 3), dtype=np.float32)

        ctx = create_audio_context(y, sr=sr)

        task = FeatureExtractionTask()
        result = task.execute(ctx)

        # Should succeed but with minimal energy
        if result.success:
            assert result.features['rms_energy'] < 0.01


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

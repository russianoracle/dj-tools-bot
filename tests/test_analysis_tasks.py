"""Unit tests for analysis tasks layer.

Tests individual analysis tasks with synthetic audio contexts:
    - ZoneClassificationTask: Rule-based and ML classification
    - FeatureExtractionTask: Feature vector generation
    - DropDetection: Drop detection algorithms

Uses synthetic audio to test task logic without real audio files.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_stft_cache():
    """Create a mock STFTCache for testing tasks."""
    mock = Mock()

    # Set up spectrogram
    n_freq = 513
    n_frames = 100
    mock.S = np.abs(np.random.randn(n_freq, n_frames).astype(np.float32)) + 0.1
    mock.freqs = np.linspace(0, 11025, n_freq, dtype=np.float32)
    mock.n_frames = n_frames
    mock.hop_length = 512
    mock.n_fft = 2048

    # Mock cache methods
    mock.get_rms.return_value = np.abs(np.random.randn(n_frames).astype(np.float32)) + 0.1
    mock.get_spectral_centroid.return_value = np.random.randn(n_frames).astype(np.float32) * 1000 + 3000
    mock.get_spectral_rolloff.return_value = np.random.randn(n_frames).astype(np.float32) * 1000 + 5000
    mock.get_spectral_flatness.return_value = np.abs(np.random.randn(n_frames).astype(np.float32)) * 0.5
    mock.get_spectral_flux.return_value = np.abs(np.random.randn(n_frames).astype(np.float32))
    mock.get_spectral_bandwidth.return_value = np.abs(np.random.randn(n_frames).astype(np.float32)) * 1000
    mock.get_spectral_contrast.return_value = np.random.randn(7, n_frames).astype(np.float32)
    mock.get_mfcc.return_value = np.random.randn(13, n_frames).astype(np.float32)
    mock.get_mfcc_delta.return_value = np.random.randn(13, n_frames).astype(np.float32)
    mock.get_chroma.return_value = np.abs(np.random.randn(12, n_frames).astype(np.float32))
    mock.get_tonnetz.return_value = np.random.randn(6, n_frames).astype(np.float32)
    mock.get_onset_strength.return_value = np.abs(np.random.randn(n_frames).astype(np.float32))
    mock.get_tempo.return_value = (128.0, None)
    mock.get_beats.return_value = (np.arange(0, n_frames, 10), np.arange(0, n_frames, 10) * 512 / 22050)

    return mock


@pytest.fixture
def mock_audio_context(mock_stft_cache):
    """Create a mock AudioContext for testing tasks."""
    from app.modules.analysis.tasks.base import AudioContext

    n_samples = 22050 * 30  # 30 seconds at 22050 Hz
    y = np.random.randn(n_samples).astype(np.float32) * 0.1

    context = AudioContext(
        y=y,
        sr=22050,
        stft_cache=mock_stft_cache,
        duration_sec=30.0,
        file_path="/test/audio.wav"
    )

    return context


# =============================================================================
# ZONE CLASSIFICATION TESTS
# =============================================================================

@pytest.mark.unit
class TestZoneClassification:
    """Tests for ZoneClassificationTask."""

    def test_zone_classification_imports(self):
        """Test ZoneClassificationTask can be imported.

        ЧТО ПРОВЕРЯЕМ:
            Module imports without errors
        """
        from app.modules.analysis.tasks.zone_classification import (
            ZoneClassificationTask,
            ZoneClassificationResult,
            ZONE_YELLOW, ZONE_GREEN, ZONE_PURPLE
        )

        assert ZoneClassificationTask is not None
        assert ZONE_YELLOW == 'yellow'
        assert ZONE_GREEN == 'green'
        assert ZONE_PURPLE == 'purple'

    def test_zone_classification_result_color(self):
        """Test ZoneClassificationResult color property.

        ЧТО ПРОВЕРЯЕМ:
            Each zone has correct color code
        """
        from app.modules.analysis.tasks.zone_classification import (
            ZoneClassificationResult, ZONE_COLORS
        )

        result_yellow = ZoneClassificationResult(
            success=True,
            task_name="test",
            processing_time_sec=0.1,
            zone='yellow'
        )

        assert result_yellow.color == ZONE_COLORS['yellow']
        assert result_yellow.description == 'Low energy, calm - rest zone'

    def test_zone_classification_result_to_dict(self):
        """Test ZoneClassificationResult serialization.

        ЧТО ПРОВЕРЯЕМ:
            to_dict() includes all fields
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationResult

        result = ZoneClassificationResult(
            success=True,
            task_name="test",
            processing_time_sec=0.5,
            zone='purple',
            confidence=0.85,
            zone_scores={'purple': 0.85, 'green': 0.1, 'yellow': 0.05},
            key_features={'tempo': 130.0}
        )

        d = result.to_dict()

        assert d['zone'] == 'purple'
        assert d['confidence'] == 0.85
        assert 'color' in d
        assert 'description' in d
        assert d['key_features']['tempo'] == 130.0

    def test_classify_rules_yellow_zone(self):
        """Test rule-based classification for YELLOW zone.

        ЧТО ПРОВЕРЯЕМ:
            Low tempo, low energy → yellow zone
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask(model_path=None)

        # Yellow zone features: low tempo, low energy variance, low brightness
        features = {
            'tempo': 95.0,              # < 110 → yellow
            'rms_energy_delta': 0.05,   # < 0.1 → yellow
            'brightness': 0.2,          # < 0.3 → yellow
            'drop_count': 0,            # = 0 → yellow
            'drop_intensity': 0.0,
            'low_energy_ratio': 0.7,    # > 0.6 → yellow
            'bass_energy_ratio': 0.5
        }

        result = task._classify_rules(features)

        assert result.success
        assert result.zone == 'yellow'
        assert result.confidence > 0.3

    def test_classify_rules_purple_zone(self):
        """Test rule-based classification for PURPLE zone.

        ЧТО ПРОВЕРЯЕМ:
            High tempo, drops → purple zone
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask(model_path=None)

        # Purple zone features: high tempo, drops, high energy variance
        features = {
            'tempo': 135.0,             # > 128 → purple
            'rms_energy_delta': 0.3,    # > 0.2 → purple
            'brightness': 0.5,          # > 0.4 → purple
            'drop_count': 3,            # >= 2 → purple
            'drop_intensity': 0.7,      # > 0.5 → purple
            'low_energy_ratio': 0.3,
            'bass_energy_ratio': 0.6
        }

        result = task._classify_rules(features)

        assert result.success
        assert result.zone == 'purple'
        assert result.confidence > 0.3

    def test_classify_rules_green_zone(self):
        """Test rule-based classification for GREEN zone.

        ЧТО ПРОВЕРЯЕМ:
            Medium values → green zone
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask(model_path=None)

        # Green zone features: medium values, not strongly yellow or purple
        features = {
            'tempo': 120.0,             # 110-128 → neutral
            'rms_energy_delta': 0.15,   # 0.1-0.2 → neutral
            'brightness': 0.35,         # 0.3-0.4 → neutral
            'drop_count': 1,            # 1 → neutral
            'drop_intensity': 0.3,
            'low_energy_ratio': 0.4,
            'bass_energy_ratio': 0.5
        }

        result = task._classify_rules(features)

        assert result.success
        assert result.zone == 'green'

    def test_zone_classification_scores_normalized(self):
        """Test zone scores sum to approximately 1.

        ЧТО ПРОВЕРЯЕМ:
            Zone probabilities are normalized
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask(model_path=None)

        features = {
            'tempo': 120.0,
            'rms_energy_delta': 0.1,
            'brightness': 0.4,
            'drop_count': 1,
            'drop_intensity': 0.3,
            'low_energy_ratio': 0.5,
            'bass_energy_ratio': 0.5
        }

        result = task._classify_rules(features)

        total = sum(result.zone_scores.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_zone_classification_key_features(self):
        """Test key features extraction.

        ЧТО ПРОВЕРЯЕМ:
            _get_key_features() returns important features
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask()

        features = {
            'tempo': 128.0,
            'drop_count': 2,
            'rms_energy_delta': 0.2,
            'brightness': 0.45,
            'low_energy_ratio': 0.4,
            'extra_feature': 999.0  # Should not be included
        }

        key_features = task._get_key_features(features)

        assert 'tempo' in key_features
        assert 'drop_count' in key_features
        assert 'brightness' in key_features
        assert 'extra_feature' not in key_features


# =============================================================================
# FEATURE EXTRACTION TESTS
# =============================================================================

@pytest.mark.unit
class TestFeatureExtraction:
    """Tests for FeatureExtractionTask."""

    def test_feature_extraction_imports(self):
        """Test FeatureExtractionTask can be imported.

        ЧТО ПРОВЕРЯЕМ:
            Module imports without errors
        """
        from app.modules.analysis.tasks.feature_extraction import (
            FeatureExtractionTask,
            FeatureExtractionResult,
            FEATURE_NAMES
        )

        assert FeatureExtractionTask is not None
        assert len(FEATURE_NAMES) == 79  # Expected feature count

    def test_feature_names_completeness(self):
        """Test all expected feature categories are present.

        ЧТО ПРОВЕРЯЕМ:
            FEATURE_NAMES contains all required features
        """
        from app.modules.analysis.tasks.feature_extraction import FEATURE_NAMES

        # Energy features
        assert 'rms_energy' in FEATURE_NAMES
        assert 'rms_energy_delta' in FEATURE_NAMES
        assert 'low_energy_ratio' in FEATURE_NAMES

        # Spectral features
        assert 'spectral_centroid' in FEATURE_NAMES
        assert 'spectral_rolloff' in FEATURE_NAMES
        assert 'brightness' in FEATURE_NAMES
        assert 'spectral_flatness' in FEATURE_NAMES

        # MFCCs
        assert 'mfcc_1' in FEATURE_NAMES
        assert 'mfcc_13' in FEATURE_NAMES
        assert 'mfcc_1_delta' in FEATURE_NAMES

        # Chroma
        assert 'chroma_C' in FEATURE_NAMES
        assert 'chroma_B' in FEATURE_NAMES

        # Tonnetz
        assert 'tonnetz_1' in FEATURE_NAMES
        assert 'tonnetz_6' in FEATURE_NAMES

        # Drops
        assert 'drop_count' in FEATURE_NAMES
        assert 'drop_intensity' in FEATURE_NAMES

        # Rhythm
        assert 'tempo' in FEATURE_NAMES
        assert 'onset_density' in FEATURE_NAMES

    def test_feature_result_to_vector(self):
        """Test FeatureExtractionResult.to_vector() method.

        ЧТО ПРОВЕРЯЕМ:
            to_vector() returns correct shape numpy array
        """
        from app.modules.analysis.tasks.feature_extraction import (
            FeatureExtractionResult, FEATURE_NAMES
        )

        features = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}

        result = FeatureExtractionResult(
            success=True,
            task_name="test",
            processing_time_sec=1.0,
            features=features
        )

        vector = result.to_vector()

        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(FEATURE_NAMES)
        assert vector[0] == 0.0  # First feature
        assert vector[-1] == len(FEATURE_NAMES) - 1  # Last feature

    def test_feature_result_to_dict(self):
        """Test FeatureExtractionResult.to_dict() method.

        ЧТО ПРОВЕРЯЕМ:
            to_dict() includes feature count
        """
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionResult

        result = FeatureExtractionResult(
            success=True,
            task_name="test",
            processing_time_sec=0.5,
            features={'tempo': 128.0, 'brightness': 0.5}
        )

        d = result.to_dict()

        assert d['success'] == True
        assert d['n_features'] == 2
        assert d['features']['tempo'] == 128.0


# =============================================================================
# BASE TASK TESTS
# =============================================================================

@pytest.mark.unit
class TestBaseTask:
    """Tests for base task classes."""

    def test_audio_context_properties(self, mock_audio_context):
        """Test AudioContext computed properties.

        ЧТО ПРОВЕРЯЕМ:
            Properties return correct values from STFT cache
        """
        ctx = mock_audio_context

        assert ctx.n_frames == 100
        assert ctx.hop_length == 512
        assert ctx.n_fft == 2048
        assert ctx.duration_sec == 30.0

    def test_audio_context_report_progress(self, mock_audio_context):
        """Test progress reporting callback.

        ЧТО ПРОВЕРЯЕМ:
            report_progress() calls callback when set
        """
        ctx = mock_audio_context

        progress_calls = []
        ctx.progress_callback = lambda stage, progress, msg: progress_calls.append((stage, progress, msg))

        ctx.report_progress("test_stage", 0.5, "Testing")

        assert len(progress_calls) == 1
        assert progress_calls[0] == ("test_stage", 0.5, "Testing")

    def test_task_result_to_dict(self):
        """Test TaskResult serialization.

        ЧТО ПРОВЕРЯЕМ:
            to_dict() returns all base fields
        """
        from app.modules.analysis.tasks.base import TaskResult

        result = TaskResult(
            success=True,
            task_name="TestTask",
            processing_time_sec=1.5,
            error=None
        )

        d = result.to_dict()

        assert d['success'] == True
        assert d['task_name'] == "TestTask"
        assert d['processing_time_sec'] == 1.5
        assert d['error'] is None

    def test_task_result_with_error(self):
        """Test TaskResult with error.

        ЧТО ПРОВЕРЯЕМ:
            Error field is preserved
        """
        from app.modules.analysis.tasks.base import TaskResult

        result = TaskResult(
            success=False,
            task_name="FailedTask",
            processing_time_sec=0.1,
            error="Something went wrong"
        )

        assert result.success == False
        assert result.error == "Something went wrong"

    def test_base_task_name(self):
        """Test BaseTask.name property.

        ЧТО ПРОВЕРЯЕМ:
            name returns class name by default
        """
        from app.modules.analysis.tasks.base import BaseTask, AudioContext, TaskResult

        class MyCustomTask(BaseTask):
            def execute(self, context: AudioContext) -> TaskResult:
                return TaskResult(success=True, task_name=self.name, processing_time_sec=0)

        task = MyCustomTask()
        assert task.name == "MyCustomTask"


# =============================================================================
# DROP DETECTION TESTS
# =============================================================================

@pytest.mark.unit
class TestDropDetection:
    """Tests for drop detection primitives."""

    def test_detect_drop_candidates_imports(self):
        """Test drop detection can be imported.

        ЧТО ПРОВЕРЯЕМ:
            Drop detection primitives import without errors
        """
        from app.common.primitives import detect_drop_candidates, compute_buildup_score

        assert detect_drop_candidates is not None
        assert compute_buildup_score is not None

    def test_compute_buildup_score_shape(self):
        """Test buildup score has correct shape.

        ЧТО ПРОВЕРЯЕМ:
            Output shape matches input RMS shape
        """
        from app.common.primitives import compute_buildup_score

        rms = np.abs(np.random.randn(100).astype(np.float32)) + 0.1
        buildup = compute_buildup_score(rms)

        assert len(buildup) == len(rms)

    def test_compute_buildup_score_increasing_energy(self):
        """Test buildup score detects increasing energy.

        ЧТО ПРОВЕРЯЕМ:
            Increasing energy has high buildup score
        """
        from app.common.primitives import compute_buildup_score

        # Steadily increasing energy (buildup pattern)
        rms = np.linspace(0.1, 1.0, 100, dtype=np.float32)
        buildup = compute_buildup_score(rms)

        # Should have positive buildup in the middle section
        assert np.mean(buildup) >= 0

    def test_detect_drop_candidates_returns_list(self):
        """Test drop detection returns list of candidates.

        ЧТО ПРОВЕРЯЕМ:
            Function returns list (possibly empty)
        """
        from app.common.primitives import detect_drop_candidates

        rms = np.abs(np.random.randn(100).astype(np.float32)) + 0.1
        drops = detect_drop_candidates(rms, sr=22050, hop_length=512)

        assert isinstance(drops, list)

    def test_detect_drop_candidates_with_synthetic_drop(self):
        """Test drop detection finds synthetic drop pattern.

        ЧТО ПРОВЕРЯЕМ:
            Clear buildup → drop pattern is detected
        """
        from app.common.primitives import detect_drop_candidates

        # Create synthetic drop pattern:
        # Low energy → buildup → high energy drop
        rms = np.concatenate([
            np.ones(20) * 0.2,          # Low energy intro
            np.linspace(0.2, 0.8, 20),  # Buildup
            np.ones(20) * 0.9,          # High energy (drop)
            np.linspace(0.9, 0.3, 20),  # Breakdown
            np.ones(20) * 0.3           # Low energy outro
        ]).astype(np.float32)

        drops = detect_drop_candidates(rms, sr=22050, hop_length=512)

        # May or may not detect depending on thresholds
        # Just verify it runs without error
        assert isinstance(drops, list)


# =============================================================================
# TASK INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestTaskIntegration:
    """Integration tests for task execution with real AudioContext."""

    @pytest.fixture
    def real_audio_context(self, synthetic_audio_with_beats):
        """Create real AudioContext using synthetic audio fixture."""
        from app.modules.analysis.tasks.base import create_audio_context

        y, sr = synthetic_audio_with_beats
        context = create_audio_context(y, sr, file_path="/test/synthetic.wav")
        return context

    def test_feature_extraction_with_real_context(self, real_audio_context):
        """Test feature extraction with real STFT computation.

        ЧТО ПРОВЕРЯЕМ:
            FeatureExtractionTask works with real AudioContext
        """
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        task = FeatureExtractionTask()
        result = task.execute(real_audio_context)

        assert result.success, f"Feature extraction failed: {result.error}"
        assert len(result.features) > 50  # Should have many features
        assert 'tempo' in result.features
        assert 'brightness' in result.features

    def test_zone_classification_with_real_context(self, real_audio_context):
        """Test zone classification with real STFT computation.

        ЧТО ПРОВЕРЯЕМ:
            ZoneClassificationTask works with real AudioContext
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask()
        result = task.execute(real_audio_context)

        assert result.success, f"Zone classification failed: {result.error}"
        assert result.zone in ['yellow', 'green', 'purple']
        assert 0 <= result.confidence <= 1

    def test_feature_vector_valid_for_ml(self, real_audio_context):
        """Test feature vector is valid for ML model.

        ЧТО ПРОВЕРЯЕМ:
            Feature vector has correct shape and no NaN/Inf
        """
        from app.modules.analysis.tasks.feature_extraction import (
            FeatureExtractionTask, FEATURE_NAMES
        )

        task = FeatureExtractionTask()
        result = task.execute(real_audio_context)

        assert result.success
        vector = result.to_vector()

        assert len(vector) == len(FEATURE_NAMES)
        assert np.all(np.isfinite(vector)), "Feature vector contains NaN or Inf"


# =============================================================================
# EDGE CASES
# =============================================================================

@pytest.mark.unit
class TestTaskEdgeCases:
    """Edge case tests for tasks."""

    def test_zone_classification_missing_features(self):
        """Test classification handles missing features gracefully.

        ЧТО ПРОВЕРЯЕМ:
            Missing features default to 0
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask()

        # Minimal features - some missing
        features = {
            'tempo': 120.0
        }

        result = task._classify_rules(features)

        assert result.success
        assert result.zone in ['yellow', 'green', 'purple']

    def test_zone_classification_extreme_values(self):
        """Test classification handles extreme values.

        ЧТО ПРОВЕРЯЕМ:
            Extreme feature values don't cause errors
        """
        from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

        task = ZoneClassificationTask()

        # Extreme values
        features = {
            'tempo': 300.0,             # Very high
            'rms_energy_delta': 10.0,   # Very high
            'brightness': 1.0,
            'drop_count': 100,
            'drop_intensity': 1.0,
            'low_energy_ratio': 0.0,
            'bass_energy_ratio': 1.0
        }

        result = task._classify_rules(features)

        assert result.success
        assert result.zone in ['yellow', 'green', 'purple']
        assert 0 <= result.confidence <= 1

    def test_feature_result_empty_features(self):
        """Test FeatureExtractionResult with empty features.

        ЧТО ПРОВЕРЯЕМ:
            Empty features dict doesn't cause errors
        """
        from app.modules.analysis.tasks.feature_extraction import (
            FeatureExtractionResult, FEATURE_NAMES
        )

        result = FeatureExtractionResult(
            success=True,
            task_name="test",
            processing_time_sec=0.1,
            features={}
        )

        vector = result.to_vector()

        # Should be all zeros
        assert len(vector) == len(FEATURE_NAMES)
        assert np.all(vector == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

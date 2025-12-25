"""
Integration Tests - Verify end-to-end functionality after refactoring.

These tests ensure:
1. Pipelines work correctly (LoadAudio → STFT → Tasks)
2. CLI commands work
3. Scripts can import and run
4. Full analysis pipeline produces results
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    sr = 22050
    duration = 30.0  # 30 seconds for segmentation/transition tasks
    t = np.linspace(0, duration, int(sr * duration))

    # Create audio with some structure
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    y += 0.3 * np.sin(2 * np.pi * 880 * t)
    y += 0.1 * np.random.randn(len(t))
    y = y.astype(np.float32)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, y, sr)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Test that pipelines work end-to-end."""

    def test_load_audio_stage(self, temp_audio_file):
        """LoadAudioStage should load audio files."""
        from app.modules.analysis.pipelines.base import LoadAudioStage, PipelineContext

        stage = LoadAudioStage()
        ctx = PipelineContext(input_path=temp_audio_file)

        stage.process(ctx)

        # LoadAudioStage stores in results, not audio_context
        assert ctx.results.get('_audio') is not None
        assert ctx.results.get('_sr') > 0
        assert len(ctx.results['_audio']) > 0

    def test_compute_stft_stage(self, temp_audio_file):
        """ComputeSTFTStage should compute STFT."""
        from app.modules.analysis.pipelines.base import LoadAudioStage, ComputeSTFTStage, PipelineContext

        ctx = PipelineContext(input_path=temp_audio_file)

        # Load audio first
        LoadAudioStage().process(ctx)

        # Then compute STFT
        ComputeSTFTStage().process(ctx)

        assert ctx.audio_context.stft_cache is not None
        assert ctx.audio_context.stft_cache.S is not None
        assert ctx.audio_context.stft_cache.n_frames > 0

    def test_full_pipeline_stages(self, temp_audio_file):
        """Full pipeline should run all stages."""
        from app.modules.analysis.pipelines.base import LoadAudioStage, ComputeSTFTStage, PipelineContext
        from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask

        ctx = PipelineContext(input_path=temp_audio_file)

        # Run stages
        LoadAudioStage().process(ctx)
        ComputeSTFTStage().process(ctx)

        # Use audio_context from pipeline
        audio_ctx = ctx.audio_context

        # Run feature extraction task
        task = FeatureExtractionTask()
        result = task.execute(audio_ctx)

        assert result.success
        assert len(result.features) == 79


class TestTrackAnalysisPipeline:
    """Test TrackAnalysisPipeline."""

    def test_track_analysis_pipeline_import(self):
        """TrackAnalysisPipeline should be importable."""
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline
        assert TrackAnalysisPipeline is not None

    def test_track_analysis_runs(self, temp_audio_file):
        """TrackAnalysisPipeline should run without error."""
        from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline

        pipeline = TrackAnalysisPipeline()
        result = pipeline.run_from_path(temp_audio_file)

        assert result is not None


class TestSetAnalysisPipeline:
    """Test SetAnalysisPipeline."""

    def test_set_analysis_pipeline_import(self):
        """SetAnalysisPipeline should be importable."""
        from app.modules.analysis.pipelines.set_analysis import SetAnalysisPipeline
        assert SetAnalysisPipeline is not None


# =============================================================================
# Tasks Integration Tests
# =============================================================================

class TestTasksIntegration:
    """Test that all tasks are importable and work together."""

    def test_all_tasks_importable(self):
        """All tasks should be importable from app.core.tasks."""
        from app.modules.analysis.tasks import (
            FeatureExtractionTask,
            SegmentationTask,
            DropDetectionTask,
            TransitionDetectionTask,
        )

        assert FeatureExtractionTask is not None
        assert SegmentationTask is not None
        assert DropDetectionTask is not None
        assert TransitionDetectionTask is not None

    def test_tasks_chain(self, temp_audio_file):
        """Tasks should work in sequence."""
        from app.modules.analysis.pipelines.base import LoadAudioStage, ComputeSTFTStage, PipelineContext
        from app.modules.analysis.tasks import (
            FeatureExtractionTask,
            SegmentationTask,
            DropDetectionTask,
        )

        # Setup
        ctx = PipelineContext(input_path=temp_audio_file)
        LoadAudioStage().process(ctx)
        ComputeSTFTStage().process(ctx)

        audio_ctx = ctx.audio_context

        # Run tasks in sequence
        feat_result = FeatureExtractionTask().execute(audio_ctx)
        seg_result = SegmentationTask().execute(audio_ctx)
        drop_result = DropDetectionTask().execute(audio_ctx)

        # All should succeed
        assert feat_result.success
        assert seg_result.success
        assert drop_result.success


# =============================================================================
# Primitives Integration Tests
# =============================================================================

class TestPrimitivesIntegration:
    """Test that primitives are importable and work."""

    def test_all_primitives_importable(self):
        """Public primitives should be importable; deprecated ones should raise ImportError."""
        # These should work
        from app.common.primitives import (
            compute_stft,
            STFTCache,
            smooth_gaussian,
            detect_peaks,
            compute_frequency_bands,
        )

        assert compute_stft is not None
        assert STFTCache is not None
        assert smooth_gaussian is not None
        assert detect_peaks is not None
        assert compute_frequency_bands is not None

        # These are blocked and should raise ImportError
        import pytest

        with pytest.raises(ImportError, match="BLOCKED"):
            from app.common.primitives import compute_rms

        with pytest.raises(ImportError, match="BLOCKED"):
            from app.common.primitives import compute_centroid

        with pytest.raises(ImportError, match="BLOCKED"):
            from app.common.primitives import compute_onset_strength

    def test_stft_cache_lazy_methods_work(self, temp_audio_file):
        """STFTCache lazy methods should work correctly."""
        from app.modules.analysis.pipelines.base import LoadAudioStage, ComputeSTFTStage, PipelineContext

        ctx = PipelineContext(input_path=temp_audio_file)
        LoadAudioStage().process(ctx)
        ComputeSTFTStage().process(ctx)

        # Test lazy methods
        stft_cache = ctx.audio_context.stft_cache
        mfcc = stft_cache.get_mfcc()
        chroma = stft_cache.get_chroma()
        tonnetz = stft_cache.get_tonnetz()
        mel = stft_cache.get_mel()

        assert mfcc.shape[0] == 13
        assert chroma.shape[0] == 12
        assert tonnetz.shape[0] == 6
        assert mel.shape[0] == 128


# =============================================================================
# Import Tests (verify no circular imports)
# =============================================================================

class TestImports:
    """Test that imports work without circular dependencies."""

    def test_core_import(self):
        """app.core should be importable."""
        import app.core
        assert app.core is not None

    def test_core_tasks_import(self):
        """Tasks should be importable from app.modules.analysis.tasks."""
        import app.modules.analysis.tasks
        assert app.modules.analysis.tasks is not None

    def test_core_pipelines_import(self):
        """Pipelines should be importable from app.modules.analysis.pipelines."""
        import app.modules.analysis.pipelines
        assert app.modules.analysis.pipelines is not None

    def test_core_primitives_import(self):
        """Primitives should be importable from app.common.primitives."""
        import app.common.primitives
        assert app.common.primitives is not None

    @pytest.mark.skip(reason="Training module not implemented in new architecture")
    def test_training_import(self):
        """Training module not available in refactored structure."""
        pass

    @pytest.mark.skip(reason="Training pipelines not implemented in new architecture")
    def test_training_pipelines_import(self):
        """Training pipelines not available in refactored structure."""
        pass

    @pytest.mark.skip(reason="Trainers not implemented in new architecture")
    def test_training_trainers_import(self):
        """Trainers not available in refactored structure."""
        pass


# =============================================================================
# CLI Integration Tests
# =============================================================================

class TestCLIIntegration:
    """Test CLI entry points."""

    @pytest.mark.skip(reason="Legacy main.py uses old src imports, app/main.py is the new entry point")
    def test_main_module_importable(self):
        """main.py should be importable (skipped - legacy)."""
        pass

    @pytest.mark.skip(reason="Legacy main.py uses old src imports")
    def test_main_help(self):
        """main.py --help should not crash (skipped - legacy)."""
        pass


# =============================================================================
# Script Import Tests
# =============================================================================

class TestScriptImports:
    """Test that key scripts can be imported."""

    def test_analyze_dj_set_importable(self):
        """analyze_dj_set.py should be importable (syntax check)."""
        script_path = Path('scripts/analyze_dj_set.py')
        if script_path.exists():
            import ast
            with open(script_path, 'r') as f:
                # Just check syntax is valid
                ast.parse(f.read())

    def test_analyze_track_importable(self):
        """analyze_track.py should be importable (syntax check)."""
        script_path = Path('scripts/analyze_track.py')
        if script_path.exists():
            import ast
            with open(script_path, 'r') as f:
                ast.parse(f.read())


# =============================================================================
# Training Module Tests
# =============================================================================

class TestTrainingModule:
    """Test training module isolation."""

    @pytest.mark.skip(reason="Training module not implemented in new architecture")
    def test_training_pipeline_importable(self):
        """TrainingPipeline not available in refactored structure."""
        pass

    @pytest.mark.skip(reason="Training module not implemented in new architecture")
    def test_calibration_pipeline_importable(self):
        """CalibrationPipeline not available in refactored structure."""
        pass

    @pytest.mark.skip(reason="Training module not implemented in new architecture")
    def test_trainers_importable(self):
        """Trainers not available in refactored structure."""
        pass


# =============================================================================
# Cache Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Test cache module integration."""

    def test_cache_repository_importable(self):
        """CacheRepository should be importable."""
        from app.core.cache import CacheRepository
        assert CacheRepository is not None

    def test_cache_models_importable(self):
        """Cache models should be importable."""
        from app.core.cache.models import CachedSetAnalysis
        assert CachedSetAnalysis is not None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

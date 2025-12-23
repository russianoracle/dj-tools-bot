"""
Architectural Invariant Tests.

Tests that MUST pass to ensure system integrity.
These verify critical rules that cannot be violated.
"""

import ast
import numpy as np
import pytest
from pathlib import Path
from typing import Set, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
# Use app structure instead of legacy src
APP_ROOT = PROJECT_ROOT / "app"
SRC_ROOT = PROJECT_ROOT / "src"  # Keep for backward compat, but deprecated


# =============================================================================
# Helper Functions
# =============================================================================

def get_imports_from_file(file_path: Path) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Parse file and return all imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        imports = set()
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                    for alias in node.names:
                        from_imports.append((node.module, alias.name))

        return imports, from_imports
    except SyntaxError:
        return set(), []


# =============================================================================
# Invariant 1: STFTCache is Single Source for Librosa Features
# =============================================================================

@pytest.mark.invariant
class TestInvariantSTFTSingleSource:
    """STFTCache must be the only source for librosa features."""

    PRIMITIVES_DIR = APP_ROOT / "common" / "primitives"
    ALLOWED_LIBROSA_FILE = "stft.py"
    # audio_stft_loader.py also has librosa for loading audio
    ALLOWED_LIBROSA_FILES = {"stft.py", "audio_stft_loader.py"}

    def get_primitive_files(self) -> List[Path]:
        """Get all primitive files except stft.py."""
        return [
            f for f in self.PRIMITIVES_DIR.glob("*.py")
            if f.name not in {"__init__.py"}.union(self.ALLOWED_LIBROSA_FILES)
        ]

    @pytest.mark.parametrize("primitive_file", [
        pytest.param(f, id=f.name)
        for f in (APP_ROOT / "common" / "primitives").glob("*.py")
        if f.name not in {"__init__.py", "stft.py", "audio_stft_loader.py"}
        and (APP_ROOT / "common" / "primitives").exists()
    ])
    def test_no_librosa_in_primitives(self, primitive_file: Path):
        """Primitives (except stft.py) must not import librosa."""
        imports, from_imports = get_imports_from_file(primitive_file)

        assert "librosa" not in imports, (
            f"{primitive_file.name} imports librosa. "
            f"All librosa calls must go through STFTCache in stft.py"
        )

        librosa_from = [m for m, n in from_imports if m and m.startswith("librosa")]
        assert not librosa_from, (
            f"{primitive_file.name} has 'from librosa' imports: {librosa_from}"
        )

    def test_stft_py_exists(self):
        """stft.py must exist as centralized librosa location."""
        stft_file = self.PRIMITIVES_DIR / "stft.py"
        assert stft_file.exists(), "stft.py must exist in primitives"

    def test_stft_py_has_stftcache(self):
        """stft.py must define STFTCache class."""
        from app.common.primitives.stft import STFTCache
        assert STFTCache is not None


# =============================================================================
# Invariant 2: Float32 Contiguity (M2 Optimization)
# =============================================================================

@pytest.mark.invariant
class TestInvariantFloat32Contiguous:
    """All arrays must be float32 and C-contiguous for M2 optimization."""

    @pytest.fixture
    def stft_cache(self):
        """Create STFTCache for testing."""
        from src.core.primitives.stft import compute_stft

        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr).astype(np.float32)
        return compute_stft(y, sr=sr)

    STFT_METHODS = [
        'get_mel', 'get_mfcc', 'get_chroma', 'get_tonnetz',
        'get_rms', 'get_spectral_centroid', 'get_spectral_rolloff',
        'get_spectral_flatness', 'get_spectral_bandwidth',
        'get_spectral_flux', 'get_onset_strength',
    ]

    @pytest.mark.parametrize("method_name", STFT_METHODS)
    def test_stft_method_returns_float32(self, stft_cache, method_name):
        """STFTCache methods must return float32."""
        method = getattr(stft_cache, method_name)
        result = method()

        if isinstance(result, np.ndarray):
            assert result.dtype == np.float32, (
                f"STFTCache.{method_name}() returns {result.dtype}, expected float32"
            )

    @pytest.mark.parametrize("method_name", STFT_METHODS)
    def test_stft_method_returns_contiguous(self, stft_cache, method_name):
        """STFTCache methods must return C-contiguous arrays."""
        method = getattr(stft_cache, method_name)
        result = method()

        if isinstance(result, np.ndarray):
            assert result.flags['C_CONTIGUOUS'], (
                f"STFTCache.{method_name}() not C-contiguous"
            )

    def test_beat_grid_boundaries_float32(self):
        """BeatGrid boundaries must be float32."""
        from src.core.primitives.beat_grid import BeatGridResult, BeatInfo, BarInfo, PhraseInfo

        grid = BeatGridResult(
            beats=[BeatInfo(0.0, 0, 1, 1, 1.0)],
            bars=[BarInfo(0, 0.0, 1.0, [0], 0, 1)],
            phrases=[PhraseInfo(0, 0.0, 7.5, [0], 7.5)],
            tempo=128.0,
            tempo_confidence=0.9,
            beat_duration_sec=0.469,
            bar_duration_sec=1.875,
            phrase_duration_sec=7.5,
        )

        assert grid.get_phrase_boundaries().dtype == np.float32
        assert grid.get_bar_boundaries().dtype == np.float32
        assert grid.get_beat_times().dtype == np.float32


# =============================================================================
# Invariant 3: Three-Layer Architecture
# =============================================================================

@pytest.mark.invariant
class TestInvariantLayerDependencies:
    """Primitives -> Tasks -> Pipelines (no reverse imports)."""

    PRIMITIVES_DIR = SRC_ROOT / "core" / "primitives"
    TASKS_DIR = SRC_ROOT / "core" / "tasks"

    def test_primitives_no_task_imports(self):
        """Primitives cannot import from tasks."""
        for pf in self.PRIMITIVES_DIR.glob("*.py"):
            if pf.name == "__init__.py":
                continue

            imports, from_imports = get_imports_from_file(pf)

            task_imports = [m for m, n in from_imports if m and "tasks" in m]
            assert not task_imports, (
                f"{pf.name} imports from tasks: {task_imports}. "
                f"Primitives cannot depend on tasks layer."
            )

    def test_primitives_no_pipeline_imports(self):
        """Primitives cannot import from pipelines."""
        for pf in self.PRIMITIVES_DIR.glob("*.py"):
            if pf.name == "__init__.py":
                continue

            imports, from_imports = get_imports_from_file(pf)

            pipeline_imports = [m for m, n in from_imports if m and "pipelines" in m]
            assert not pipeline_imports, (
                f"{pf.name} imports from pipelines: {pipeline_imports}. "
                f"Primitives cannot depend on pipelines layer."
            )


# =============================================================================
# Invariant 4: CacheRepository is Only Cache API
# =============================================================================

@pytest.mark.invariant
class TestInvariantCacheAPI:
    """CacheRepository must be the only public cache API."""

    def test_cache_repository_singleton(self):
        """CacheRepository.get_instance() returns same object."""
        import tempfile
        from src.core.cache import CacheRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            CacheRepository._instance = None

            repo1 = CacheRepository.get_instance(tmpdir)
            repo2 = CacheRepository.get_instance(tmpdir)

            assert repo1 is repo2

    def test_cache_manager_not_in_public_exports(self):
        """CacheManager should not be in __all__ of cache module."""
        from src.core import cache

        if hasattr(cache, '__all__'):
            assert 'CacheManager' not in cache.__all__, (
                "CacheManager should not be publicly exported"
            )


# =============================================================================
# Invariant 5: Hierarchical BeatGrid Structure
# =============================================================================

@pytest.mark.invariant
class TestInvariantBeatGridHierarchy:
    """Beat grid must maintain proper hierarchical structure."""

    @pytest.fixture
    def synthetic_spectrogram(self):
        """Create synthetic spectrogram for testing."""
        from src.core.primitives.stft import compute_stft

        sr = 22050
        duration = 30.0  # 30 seconds for multiple phrases
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Simulate kick drum at 128 BPM
        beat_period = 60.0 / 128.0
        kicks = np.zeros_like(t)
        for i in range(int(duration / beat_period)):
            beat_time = i * beat_period
            beat_sample = int(beat_time * sr)
            if beat_sample < len(kicks):
                window = min(int(0.05 * sr), len(kicks) - beat_sample)
                kicks[beat_sample:beat_sample + window] = np.exp(-np.arange(window) / (0.01 * sr))

        y = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.7 * kicks
        y = y.astype(np.float32)

        cache = compute_stft(y, sr=sr)
        return cache.S, sr, 512

    def test_4_beats_per_bar(self, synthetic_spectrogram):
        """Each bar must have 4 beats."""
        from src.core.primitives.beat_grid import compute_beat_grid

        S, sr, hop = synthetic_spectrogram
        grid = compute_beat_grid(S, sr, hop)

        if len(grid.bars) > 0:
            for bar in grid.bars:
                assert len(bar.beat_indices) == 4, (
                    f"Bar {bar.index} has {len(bar.beat_indices)} beats, expected 4"
                )

    def test_4_bars_per_phrase(self, synthetic_spectrogram):
        """Each phrase must have 4 bars."""
        from src.core.primitives.beat_grid import compute_beat_grid

        S, sr, hop = synthetic_spectrogram
        grid = compute_beat_grid(S, sr, hop)

        if len(grid.phrases) > 0:
            for phrase in grid.phrases:
                assert len(phrase.bar_indices) == 4, (
                    f"Phrase {phrase.index} has {len(phrase.bar_indices)} bars, expected 4"
                )

    def test_bar_position_1_to_4(self, synthetic_spectrogram):
        """Beat bar_position must be 1-4."""
        from src.core.primitives.beat_grid import compute_beat_grid

        S, sr, hop = synthetic_spectrogram
        grid = compute_beat_grid(S, sr, hop)

        for beat in grid.beats:
            assert 1 <= beat.bar_position <= 4, (
                f"Beat has invalid bar_position: {beat.bar_position}"
            )

    def test_phrase_position_1_to_16(self, synthetic_spectrogram):
        """Beat phrase_position must be 1-16."""
        from src.core.primitives.beat_grid import compute_beat_grid

        S, sr, hop = synthetic_spectrogram
        grid = compute_beat_grid(S, sr, hop)

        for beat in grid.beats:
            assert 1 <= beat.phrase_position <= 16, (
                f"Beat has invalid phrase_position: {beat.phrase_position}"
            )


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "invariant"])

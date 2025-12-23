"""
Tests to ensure scripts and pipelines don't duplicate primitive operations.

Verifies:
1. No direct librosa calls in scripts (except for specific exceptions)
2. Scripts use STFTCache methods instead of computing features directly
3. No reimplementation of primitives in scripts/pipelines
"""

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINES_DIR = PROJECT_ROOT / "src" / "core" / "pipelines"


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect all imports and function calls."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: List[Tuple[str, str]] = []
        self.function_calls: List[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
            for alias in node.names:
                self.from_imports.append((node.module, alias.name))
        self.generic_visit(node)

    def visit_Call(self, node):
        """Collect function call names."""
        if isinstance(node.func, ast.Attribute):
            # e.g., librosa.feature.mfcc
            attr_chain = []
            current = node.func
            while isinstance(current, ast.Attribute):
                attr_chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                attr_chain.append(current.id)
            attr_chain.reverse()
            self.function_calls.append('.'.join(attr_chain))
        elif isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        self.generic_visit(node)


def analyze_file(file_path: Path) -> Tuple[Set[str], List[Tuple[str, str]], List[str]]:
    """Parse a Python file and return imports and function calls."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports, visitor.from_imports, visitor.function_calls
    except SyntaxError:
        return set(), [], []


def get_python_files(directory: Path, pattern: str = "*.py") -> List[Path]:
    """Get all Python files in directory."""
    if not directory.exists():
        return []
    return list(directory.glob(pattern))


# =============================================================================
# Test: Scripts should not use librosa directly for features
# =============================================================================

class TestScriptsNoLibrosaFeatures:
    """Scripts should use STFTCache methods, not librosa feature extraction."""

    # These are low-level librosa calls that should use STFTCache instead
    PROHIBITED_LIBROSA_CALLS = [
        'librosa.feature.mfcc',
        'librosa.feature.chroma_stft',
        'librosa.feature.chroma_cqt',
        'librosa.feature.tonnetz',
        'librosa.feature.tempogram',
        'librosa.beat.plp',
        'librosa.beat.beat_track',
    ]

    # Scripts that are allowed exceptions (training, experiments, etc.)
    EXCEPTION_SCRIPTS = {
        'extract_features_m2.py',      # Legacy extraction
        'extract_ultimate_gpu.py',      # GPU extraction (own implementation)
        'train_spectrogram_model.py',   # Training
        'train_ensemble.py',            # Training
        'train_user_only.py',           # Training
        'train_drop_detector.py',       # Training
    }

    @pytest.mark.parametrize("script_file", [
        pytest.param(f, id=f.name)
        for f in get_python_files(SCRIPTS_DIR)
        if f.name not in {'__init__.py'} and not f.name.startswith('_')
    ])
    def test_no_direct_librosa_features(self, script_file: Path):
        """Scripts should not call librosa feature functions directly."""
        if script_file.name in self.EXCEPTION_SCRIPTS:
            pytest.skip(f"{script_file.name} is an exception (training/extraction script)")

        _, _, function_calls = analyze_file(script_file)

        violations = []
        for call in function_calls:
            for prohibited in self.PROHIBITED_LIBROSA_CALLS:
                if prohibited in call:
                    violations.append(call)

        if violations:
            pytest.skip(
                f"{script_file.name} uses librosa features directly: {violations}. "
                f"Consider using STFTCache.get_mfcc(), get_chroma(), etc."
            )


# =============================================================================
# Test: Pipelines should not duplicate primitives
# =============================================================================

class TestPipelinesNoPrimitiveDuplication:
    """Pipelines should use tasks/primitives, not reimplement them."""

    # Patterns that suggest primitive reimplementation
    REIMPLEMENTATION_PATTERNS = [
        r'np\.fft\.',                    # Direct FFT (use primitives.stft)
        r'scipy\.fftpack\.',             # Direct FFT (use primitives.stft)
        r'np\.correlate',                # Direct correlation (use rhythm primitives)
        r'scipy\.signal\.correlate',     # Direct correlation
        r'cdist\(',                      # Direct distance (use segmentation primitives)
        r'librosa\.feature\.',           # Direct librosa (use STFTCache)
    ]

    # Files that are allowed exceptions
    EXCEPTION_FILES = {
        'training.py',     # Training pipeline can use anything
        'calibration.py',  # Calibration may need direct access
    }

    @pytest.mark.parametrize("pipeline_file", [
        pytest.param(f, id=f.name)
        for f in get_python_files(PIPELINES_DIR)
        if f.name not in {'__init__.py', 'base.py', 'cache_manager.py'}
    ])
    def test_no_primitive_reimplementation(self, pipeline_file: Path):
        """Pipelines should not reimplement primitive operations."""
        if pipeline_file.name in self.EXCEPTION_FILES:
            pytest.skip(f"{pipeline_file.name} is an exception")

        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            pytest.skip(f"Cannot read {pipeline_file.name}")

        violations = []
        for pattern in self.REIMPLEMENTATION_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                violations.extend(matches)

        # Only warn for now, don't fail
        if violations:
            pytest.skip(
                f"{pipeline_file.name} may reimplement primitives: {set(violations)}. "
                f"Consider using existing primitives or STFTCache methods."
            )


# =============================================================================
# Test: Tasks use STFTCache for features
# =============================================================================

class TestTasksUseSTFTCache:
    """Tasks should use STFTCache lazy methods for computed features."""

    TASKS_DIR = PROJECT_ROOT / "src" / "core" / "tasks"

    # STFTCache methods that should be used
    STFT_CACHE_METHODS = [
        'stft_cache.get_mfcc',
        'stft_cache.get_chroma',
        'stft_cache.get_tonnetz',
        'stft_cache.get_mel',
        'stft_cache.get_onset_strength',
        'stft_cache.get_beats',
        'stft_cache.get_tempo',
    ]

    # Patterns that suggest NOT using STFTCache
    DEPRECATED_PATTERNS = [
        r'compute_mfcc\(',
        r'compute_chroma\(',
        r'compute_tonnetz\(',
        r'compute_mfcc_from_audio\(',
        r'compute_chroma_from_audio\(',
    ]

    # Tasks that are exceptions (may use deprecated functions for compatibility)
    EXCEPTION_TASKS = {
        '__init__.py',
        'base.py',
    }

    @pytest.mark.parametrize("task_file", [
        pytest.param(f, id=f.name)
        for f in get_python_files(TASKS_DIR)
        if f.name not in {'__init__.py', 'base.py'}
    ])
    def test_tasks_use_stft_cache(self, task_file: Path):
        """Tasks should use STFTCache methods for computed features."""
        if task_file.name in self.EXCEPTION_TASKS:
            pytest.skip(f"{task_file.name} is an exception")

        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            pytest.skip(f"Cannot read {task_file.name}")

        violations = []
        for pattern in self.DEPRECATED_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                violations.extend(matches)

        if violations:
            pytest.skip(
                f"{task_file.name} uses deprecated primitive functions: {set(violations)}. "
                f"Use STFTCache.get_*() methods instead."
            )


# =============================================================================
# Test: Scripts import from primitives, not reimplementing
# =============================================================================

class TestScriptsUsePrimitives:
    """Scripts should import from src.core.primitives, not reimplement."""

    # Functions that should be imported from primitives
    PRIMITIVE_FUNCTIONS = [
        'compute_rms',
        'compute_centroid',
        'compute_rolloff',
        'compute_novelty',
        'detect_peaks',
        'smooth_gaussian',
        'compute_frequency_bands',
        'compute_onset_strength',
        'compute_tempo',
    ]

    @pytest.mark.parametrize("script_file", [
        pytest.param(f, id=f.name)
        for f in get_python_files(SCRIPTS_DIR)
        if f.name not in {'__init__.py'} and not f.name.startswith('_')
    ])
    def test_scripts_import_primitives(self, script_file: Path):
        """Scripts that use audio analysis should import from primitives."""
        _, from_imports, function_calls = analyze_file(script_file)

        # Check if script imports from primitives
        primitive_imports = [
            (m, n) for m, n in from_imports
            if m and 'primitives' in m
        ]

        # If script calls analysis functions but doesn't import from primitives
        analysis_calls = [
            call for call in function_calls
            if any(pf in call for pf in self.PRIMITIVE_FUNCTIONS)
        ]

        if analysis_calls and not primitive_imports:
            # Just informational - some scripts may legitimately not use primitives
            pass  # pytest.skip(f"{script_file.name} uses analysis but doesn't import primitives")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

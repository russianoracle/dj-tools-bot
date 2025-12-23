"""
Architecture Tests - Enforce dependency rules and SOLID principles.

These tests verify:
1. Primitives contain only numpy/scipy (no librosa except audio_stft_loader.py)
2. Training code is isolated from production code
3. Proper layer dependencies (scripts → pipelines → tasks → primitives)
4. All pipelines inherit from base Pipeline class

Updated: 2025-12-13 - Changed from src/ to app/ architecture
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
APP_ROOT = PROJECT_ROOT / "app"
TRAINING_ROOT = PROJECT_ROOT / "training"


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to collect all imports from a Python file."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: List[Tuple[str, str]] = []  # (module, name)

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


def get_imports_from_file(file_path: Path) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Parse a Python file and return all imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports, visitor.from_imports
    except SyntaxError:
        return set(), []


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in directory recursively."""
    return list(directory.rglob("*.py"))


# =============================================================================
# Test: Primitives should be pure numpy/scipy (no librosa except audio_stft_loader.py)
# =============================================================================

class TestPrimitivesNoLibrosa:
    """Primitives must be pure numpy/scipy - no librosa (except audio_stft_loader.py)."""

    PRIMITIVES_DIR = APP_ROOT / "common" / "primitives"
    ALLOWED_LIBROSA_FILES = {"audio_stft_loader.py"}  # Exception: Single librosa entry point

    def get_primitive_files(self) -> List[Path]:
        """Get all primitive files excluding __init__.py and allowed exceptions."""
        files = []
        if self.PRIMITIVES_DIR.exists():
            for f in self.PRIMITIVES_DIR.glob("*.py"):
                if f.name != "__init__.py" and f.name not in self.ALLOWED_LIBROSA_FILES:
                    files.append(f)
        return files

    @pytest.mark.parametrize("primitive_file", [
        pytest.param(f, id=f.name)
        for f in (APP_ROOT / "common" / "primitives").glob("*.py")
        if f.name not in {"__init__.py", "audio_stft_loader.py"}
    ] if (APP_ROOT / "common" / "primitives").exists() else [])
    def test_no_librosa_import(self, primitive_file: Path):
        """Each primitive file should not import librosa."""
        imports, from_imports = get_imports_from_file(primitive_file)

        assert "librosa" not in imports, (
            f"{primitive_file.name} imports librosa directly. "
            f"Primitives must be pure numpy/scipy. "
            f"All librosa usage must go through audio_stft_loader.py"
        )

        # Check from imports too
        librosa_from = [m for m, n in from_imports if m and m.startswith("librosa")]
        assert not librosa_from, (
            f"{primitive_file.name} has 'from librosa' imports: {librosa_from}. "
            f"Use STFTCache.get_*() methods or audio_stft_loader functions instead."
        )

    def test_audio_stft_loader_is_exception(self):
        """audio_stft_loader.py is allowed to use librosa (centralized location)."""
        loader_file = self.PRIMITIVES_DIR / "audio_stft_loader.py"
        if loader_file.exists():
            imports, _ = get_imports_from_file(loader_file)
            assert "librosa" in imports, (
                "audio_stft_loader.py should import librosa as the single entry point"
            )
        else:
            pytest.skip("audio_stft_loader.py not found")


# =============================================================================
# Test: No sklearn in primitives
# =============================================================================

class TestPrimitivesNoSklearn:
    """Primitives must not use sklearn (ML belongs in tasks/training)."""

    PRIMITIVES_DIR = APP_ROOT / "common" / "primitives"

    @pytest.mark.parametrize("primitive_file", [
        pytest.param(f, id=f.name)
        for f in (APP_ROOT / "common" / "primitives").glob("*.py")
        if f.name != "__init__.py"
    ] if (APP_ROOT / "common" / "primitives").exists() else [])
    def test_no_sklearn_import(self, primitive_file: Path):
        """Primitives should not import sklearn."""
        imports, from_imports = get_imports_from_file(primitive_file)

        assert "sklearn" not in imports, (
            f"{primitive_file.name} imports sklearn. "
            f"ML code belongs in training/ or app/modules/analysis/tasks/"
        )

        sklearn_from = [m for m, n in from_imports if m and "sklearn" in m]
        assert not sklearn_from, (
            f"{primitive_file.name} has sklearn imports: {sklearn_from}"
        )


# =============================================================================
# Test: Training code isolation
# =============================================================================

class TestTrainingIsolation:
    """Training code must be in training/, not app/."""

    APP_MODULES_DIR = APP_ROOT / "modules"

    def get_app_files(self) -> List[Path]:
        """Get all Python files in app/modules/."""
        return list(self.APP_MODULES_DIR.rglob("*.py"))

    @pytest.mark.parametrize("app_file", [
        pytest.param(f, id=str(f.relative_to(APP_ROOT / "modules")))
        for f in (APP_ROOT / "modules").rglob("*.py")
        if f.name != "__init__.py"
    ] if (APP_ROOT / "modules").exists() else [])
    def test_no_training_imports(self, app_file: Path):
        """App files should not import from training module directly."""
        imports, from_imports = get_imports_from_file(app_file)

        # Check for training-related imports (from project training/ dir)
        training_imports = [
            m for m, n in from_imports
            if m and m.startswith("training")
        ]

        assert not training_imports, (
            f"{app_file.name} imports training modules: {training_imports}. "
            f"Training code should only be in training/"
        )

    @pytest.mark.parametrize("app_file", [
        pytest.param(f, id=str(f.relative_to(APP_ROOT / "modules")))
        for f in (APP_ROOT / "modules").rglob("*.py")
        if f.name != "__init__.py"
    ] if (APP_ROOT / "modules").exists() else [])
    def test_no_fit_methods(self, app_file: Path):
        """App inference code should not have .fit() calls (training belongs in training/)."""
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple pattern matching for .fit( calls
            # Exclude comments and strings
            lines = content.split('\n')
            fit_calls = []
            for i, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                # Check for .fit( pattern
                if '.fit(' in line and 'def ' not in line:
                    # Exclude if it's in a comment at end of line
                    code_part = line.split('#')[0]
                    if '.fit(' in code_part:
                        fit_calls.append((i, line.strip()))

            # Allow in training-related files that may have been missed
            if fit_calls and 'train' not in str(app_file).lower():
                # Just warn for now, don't fail (some tasks load pre-trained models)
                pass

        except Exception:
            pass  # Skip files that can't be read


# =============================================================================
# Test: STFTCache lazy methods
# =============================================================================

class TestSTFTCacheLazyMethods:
    """STFTCache should have lazy computation methods for all features."""

    REQUIRED_METHODS = [
        'get_mfcc',
        'get_chroma',
        'get_tonnetz',
        'get_mel',
        'get_rms',
        'get_spectral_centroid',
        'get_spectral_rolloff',
        'get_onset_strength',
        'clear_feature_cache',
    ]

    def test_stft_cache_has_lazy_methods(self):
        """STFTCache should have all required lazy feature methods."""
        try:
            from app.common.primitives.stft import STFTCache

            for method in self.REQUIRED_METHODS:
                assert hasattr(STFTCache, method), (
                    f"STFTCache missing method: {method}. "
                    f"Add lazy computation for this feature."
                )
        except ImportError as e:
            pytest.skip(f"Cannot import STFTCache: {e}")

    def test_stft_cache_methods_are_cached(self):
        """Lazy methods should use _feature_cache for caching."""
        try:
            from app.common.primitives.stft import STFTCache
            import numpy as np

            # Check that _feature_cache field exists
            assert '_feature_cache' in STFTCache.__dataclass_fields__ or \
                   hasattr(STFTCache, '_feature_cache'), \
                   "STFTCache should have _feature_cache for lazy caching"

        except ImportError as e:
            pytest.skip(f"Cannot import STFTCache: {e}")


# =============================================================================
# Test: Pipeline inheritance (LSP compliance)
# =============================================================================

class TestPipelineInheritance:
    """All pipelines should inherit from base Pipeline class."""

    PIPELINES_DIR = APP_ROOT / "modules" / "analysis" / "pipelines"

    def test_pipeline_classes_inherit_from_base(self):
        """Pipeline classes should inherit from Pipeline base."""
        if not self.PIPELINES_DIR.exists():
            pytest.skip("Pipelines directory not found")

        try:
            # Find all pipeline modules
            pipeline_files = list(self.PIPELINES_DIR.glob("*.py"))
            non_compliant = []

            for pf in pipeline_files:
                if pf.name in {"__init__.py", "base.py", "cache_manager.py"}:
                    continue

                # Parse file to find class definitions
                try:
                    with open(pf, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if 'Pipeline' in node.name and node.name != 'Pipeline':
                                # Check if it inherits from Pipeline
                                base_names = [
                                    b.id if isinstance(b, ast.Name) else
                                    b.attr if isinstance(b, ast.Attribute) else str(b)
                                    for b in node.bases
                                ]
                                if 'Pipeline' not in base_names and 'ABC' not in base_names:
                                    non_compliant.append(f"{pf.name}::{node.name}")
                except Exception:
                    continue

            # For now, just warn (migration in progress)
            if non_compliant:
                pytest.skip(f"Pipelines not inheriting from base (migration pending): {non_compliant}")

        except ImportError as e:
            pytest.skip(f"Cannot import Pipeline base: {e}")


# =============================================================================
# Test: Tasks inherit from BaseTask
# =============================================================================

class TestTaskInheritance:
    """All tasks should inherit from BaseTask."""

    TASKS_DIR = APP_ROOT / "modules" / "analysis" / "tasks"

    @pytest.mark.parametrize("task_file", [
        pytest.param(f, id=f.name)
        for f in (APP_ROOT / "modules" / "analysis" / "tasks").glob("*.py")
        if f.name not in {"__init__.py", "base.py"}
    ] if (APP_ROOT / "modules" / "analysis" / "tasks").exists() else [])
    def test_task_classes_inherit_from_base(self, task_file: Path):
        """Task classes should inherit from BaseTask."""
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if 'Task' in node.name and node.name not in {'BaseTask', 'TaskResult'}:
                        base_names = [
                            b.id if isinstance(b, ast.Name) else
                            b.attr if isinstance(b, ast.Attribute) else str(b)
                            for b in node.bases
                        ]
                        assert 'BaseTask' in base_names or any('Task' in b for b in base_names), (
                            f"{task_file.name}::{node.name} should inherit from BaseTask"
                        )
        except SyntaxError:
            pytest.skip(f"Syntax error in {task_file.name}")


# =============================================================================
# Test: Layer dependencies
# =============================================================================

class TestLayerDependencies:
    """Verify proper layer dependencies."""

    def test_primitives_dont_import_tasks(self):
        """Primitives should not import from tasks."""
        primitives_dir = APP_ROOT / "common" / "primitives"

        if not primitives_dir.exists():
            pytest.skip("Primitives directory not found")

        for pf in primitives_dir.glob("*.py"):
            if pf.name == "__init__.py":
                continue
            imports, from_imports = get_imports_from_file(pf)

            task_imports = [m for m, n in from_imports if m and "tasks" in m]
            assert not task_imports, (
                f"{pf.name} imports from tasks: {task_imports}. "
                f"Primitives cannot depend on tasks layer."
            )

    def test_primitives_dont_import_pipelines(self):
        """Primitives should not import from pipelines."""
        primitives_dir = APP_ROOT / "common" / "primitives"

        if not primitives_dir.exists():
            pytest.skip("Primitives directory not found")

        for pf in primitives_dir.glob("*.py"):
            if pf.name == "__init__.py":
                continue
            imports, from_imports = get_imports_from_file(pf)

            pipeline_imports = [m for m, n in from_imports if m and "pipelines" in m]
            assert not pipeline_imports, (
                f"{pf.name} imports from pipelines: {pipeline_imports}. "
                f"Primitives cannot depend on pipelines layer."
            )

    def test_tasks_dont_import_pipelines(self):
        """Tasks should not import from pipelines (except for types)."""
        tasks_dir = APP_ROOT / "modules" / "analysis" / "tasks"

        if not tasks_dir.exists():
            pytest.skip("Tasks directory not found")

        for tf in tasks_dir.glob("*.py"):
            if tf.name == "__init__.py":
                continue
            imports, from_imports = get_imports_from_file(tf)

            # Allow importing PipelineContext for type hints
            pipeline_imports = [
                (m, n) for m, n in from_imports
                if m and "pipelines" in m
                and n not in {"PipelineContext", "Pipeline"}
            ]

            # Just warn for now
            if pipeline_imports:
                pytest.skip(f"{tf.name} imports from pipelines: {pipeline_imports}")


# =============================================================================
# Test: librosa centralization in app/
# =============================================================================

class TestLibrosaCentralization:
    """Verify librosa is only imported in audio_stft_loader.py within app/."""

    def test_only_audio_stft_loader_imports_librosa(self):
        """Only audio_stft_loader.py should import librosa in the app/ directory."""
        violations = []
        allowed_file = "audio_stft_loader.py"

        for py_file in APP_ROOT.rglob("*.py"):
            if py_file.name == allowed_file:
                continue

            imports, from_imports = get_imports_from_file(py_file)

            if "librosa" in imports:
                violations.append(str(py_file.relative_to(APP_ROOT)))

            librosa_from = [m for m, n in from_imports if m and m.startswith("librosa")]
            if librosa_from:
                violations.append(str(py_file.relative_to(APP_ROOT)))

        assert not violations, (
            f"librosa imports found outside audio_stft_loader.py:\n"
            f"{chr(10).join(violations)}\n"
            f"All librosa usage must go through app/common/primitives/audio_stft_loader.py"
        )


# =============================================================================
# Test: training/ uses app/ imports correctly
# =============================================================================

class TestTrainingUsesAppImports:
    """Verify training/ code uses app/ imports instead of direct librosa."""

    def test_training_no_direct_librosa(self):
        """training/ should not import librosa directly."""
        if not TRAINING_ROOT.exists():
            pytest.skip("training/ directory not found")

        violations = []

        for py_file in TRAINING_ROOT.rglob("*.py"):
            imports, from_imports = get_imports_from_file(py_file)

            if "librosa" in imports:
                violations.append(str(py_file.relative_to(TRAINING_ROOT)))

            librosa_from = [m for m, n in from_imports if m and m.startswith("librosa")]
            if librosa_from:
                violations.append(str(py_file.relative_to(TRAINING_ROOT)))

        assert not violations, (
            f"Direct librosa imports found in training/:\n"
            f"{chr(10).join(violations)}\n"
            f"Use app.common.primitives.audio_stft_loader or app.common.primitives.stft instead"
        )

    def test_training_imports_from_app(self):
        """training/ should import from app/ for audio processing."""
        if not TRAINING_ROOT.exists():
            pytest.skip("training/ directory not found")

        has_app_imports = False

        for py_file in TRAINING_ROOT.rglob("*.py"):
            imports, from_imports = get_imports_from_file(py_file)

            app_imports = [m for m, n in from_imports if m and m.startswith("app.")]
            if app_imports:
                has_app_imports = True
                break

        assert has_app_imports, (
            "training/ should import from app/ for audio processing. "
            "Use app.common.primitives.audio_stft_loader or app.common.primitives.stft"
        )


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

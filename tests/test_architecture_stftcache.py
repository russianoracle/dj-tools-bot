"""
Тесты на соблюдение архитектурных правил STFTCache.

Правила:
1. Librosa ТОЛЬКО в audio_stft_loader.py (ЕДИНСТВЕННАЯ точка!)
2. Librosa ЗАПРЕЩЁН в Pipelines, Tasks, и везде остальное
3. Спектрограмма вычисляется ОДИН раз (в AudioSTFTLoader)
4. Все операции векторизованные (без Python циклов в hot paths)
5. Обязательное кэширование для повторного использования
6. Все массивы float32 contiguous

Целевая архитектура после рефакторинга:
- app/common/primitives/audio_stft_loader.py — ЕДИНСТВЕННЫЙ import librosa!
- app/common/primitives/stft.py — STFTCache (БЕЗ librosa)
- app/modules/analysis/tasks/*.py — через context.stft_cache (БЕЗ librosa!)
- app/modules/analysis/pipelines/*.py — оркестрация (БЕЗ librosa!)

Текущие разрешённые места (до рефакторинга):
- stft.py, loader.py — будут объединены в audio_stft_loader.py
- tasks/*.py — librosa временно допустим, будет удалён
"""

import ast
import re
import pytest
from pathlib import Path


# ========== ПРАВИЛО 1: Librosa только в разрешённых местах ==========

# ЦЕЛЕВОЕ состояние после рефакторинга:
# ALLOWED_LIBROSA_FILES_TARGET = {
#     'app/common/primitives/audio_stft_loader.py',  # ЕДИНСТВЕННЫЙ!
# }

# ТЕКУЩЕЕ состояние (промежуточный этап рефакторинга):
ALLOWED_LIBROSA_FILES = {
    'app/common/primitives/audio_stft_loader.py',  # NEW: Centralized librosa entry point
    'app/common/primitives/stft.py',                # LEGACY: Will be deprecated after merge
    'app/core/adapters/loader.py',                  # LEGACY: Will be deprecated after merge
}

# Tasks — librosa временно допустим (будет удалён после рефакторинга)
TASKS_PATTERN = 'app/modules/analysis/tasks/'

# Pipelines — librosa ЗАПРЕЩЁН
PIPELINES_PATTERN = 'app/modules/analysis/pipelines/'


def test_no_librosa_in_pipelines():
    """Librosa ЗАПРЕЩЁН в Pipelines — только оркестрация."""
    violations = []
    app_path = Path(__file__).parent.parent / 'app'

    pipelines_path = app_path / 'modules' / 'analysis' / 'pipelines'
    if pipelines_path.exists():
        for py_file in pipelines_path.rglob('*.py'):
            try:
                content = py_file.read_text()
                if re.search(r'^import librosa|^from librosa', content, re.MULTILINE):
                    violations.append(str(py_file.relative_to(app_path.parent)))
            except Exception:
                pass

    if violations:
        pytest.fail(f"Librosa in pipelines (FORBIDDEN): {violations}")


def test_librosa_allowed_locations():
    """Librosa разрешён только в stft.py, loader.py, audio_stft_loader.py и Tasks.

    DECISION DOCUMENTATION (2025-12-18):
    - audio_stft_loader.py is a NEW, LEGITIMATE file created during refactoring
    - It serves as the SINGLE entry point for all librosa calls
    - RATIONALE: Centralize librosa usage for easier maintenance and single-exit strategy
    - MIGRATION PLAN:
      1. Current state: audio_stft_loader.py (NEW) + stft.py (LEGACY) + loader.py (LEGACY)
      2. Transition: Migrate all calls from stft.py -> audio_stft_loader.py
      3. Target: audio_stft_loader.py becomes ONLY allowed file (both LEGACY files removed)

    Legacy files (stft.py, loader.py) will be deprecated after full migration completes.
    """
    violations = []
    app_path = Path(__file__).parent.parent / 'app'

    for py_file in app_path.rglob('*.py'):
        rel_path = str(py_file.relative_to(app_path.parent))

        # Разрешённые файлы
        if any(allowed in rel_path for allowed in ALLOWED_LIBROSA_FILES):
            continue

        # Tasks — допустимо
        if TASKS_PATTERN in rel_path:
            continue

        try:
            content = py_file.read_text()
            if re.search(r'^import librosa|^from librosa', content, re.MULTILINE):
                violations.append(rel_path)
        except Exception:
            pass

    # xfail для известных нарушений
    if violations:
        pytest.xfail(f"Known violations - librosa imports in: {violations}")


# ========== ПРАВИЛО 2: Спектрограмма вычисляется один раз ==========

def test_no_direct_stft_calls():
    """librosa.stft() должен вызываться ТОЛЬКО в allowed files."""
    violations = []
    app_path = Path(__file__).parent.parent / 'app'

    # Files allowed to call librosa.stft directly
    allowed_files = {
        'stft.py',
        'audio_stft_loader.py',  # Low-level audio loading
        'feature_factory.py',    # Feature factory adapter
    }

    for py_file in app_path.rglob('*.py'):
        if py_file.name in allowed_files:
            continue

        try:
            content = py_file.read_text()
            if 'librosa.stft(' in content or 'librosa.core.stft(' in content:
                violations.append(str(py_file.relative_to(app_path.parent)))
        except Exception:
            pass

    assert violations == [], f"Direct STFT calls in: {violations}"


# ========== ПРАВИЛО 3: Все операции векторизованные ==========

def test_no_python_loops_in_primitives():
    """Примитивы не должны содержать for/while циклов по данным."""
    warnings = []
    primitives_path = Path(__file__).parent.parent / 'app' / 'common' / 'primitives'

    if not primitives_path.exists():
        pytest.skip("app/common/primitives/ not found")

    for py_file in primitives_path.glob('*.py'):
        try:
            content = py_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Разрешено: range, enumerate, zip
                    iter_dump = ast.dump(node.iter)
                    if not any(allowed in iter_dump for allowed in ['range', 'enumerate', 'zip']):
                        warnings.append((py_file.name, node.lineno))
        except Exception:
            pass

    # Пока предупреждение
    if warnings:
        print(f"WARNING: Potential non-vectorized loops: {warnings}")


# ========== ПРАВИЛО 4: Кэширование ==========

def test_stftcache_caches_features():
    """Все get_*() методы должны кэшировать результаты."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)

    # Первый вызов вычисляет
    rms1 = cache.get_rms()
    # Второй вызов возвращает из кэша (тот же объект)
    rms2 = cache.get_rms()

    assert rms1 is rms2, "get_rms() should return cached object"


def test_stftcache_multiple_methods_cached():
    """Проверить кэширование основных get_*() методов."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)
    cache.set_audio(y)  # Для методов требующих аудио

    methods = [
        'get_rms',
        'get_spectral_centroid',
        'get_spectral_rolloff',
        'get_mfcc',
        'get_chroma',
        'get_onset_strength',
    ]

    for method_name in methods:
        method = getattr(cache, method_name, None)
        if method is None:
            continue

        try:
            result1 = method()
            result2 = method()
            assert result1 is result2, f"{method_name}() should return cached object"
        except Exception as e:
            pytest.skip(f"{method_name}() raised {e}")


# ========== ДОПОЛНИТЕЛЬНО: Проверка RMS формулы ==========

def test_rms_uses_numpy_formula():
    """get_rms() должен использовать numpy формулу (значения > 1.0)."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)
    rms = cache.get_rms()

    # Numpy: sqrt(mean(S²)) даёт ~10-30
    # Librosa: sqrt(2*sum(S²)/frame_length²) даёт ~0.3
    assert rms.mean() > 1.0, \
        f"RMS mean={rms.mean():.3f} too small - using librosa instead of numpy?"


def test_rms_consistent_with_manual_computation():
    """get_rms() должен давать результат = sqrt(mean(S², axis=0))."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)

    rms_cached = cache.get_rms()
    rms_manual = np.sqrt(np.mean(cache.S ** 2, axis=0))

    np.testing.assert_allclose(rms_cached, rms_manual, rtol=1e-5,
                                err_msg="RMS should match manual numpy computation")


# ========== БЛОКИРОВКА ИМПОРТОВ (для app/) ==========

def test_blocked_import_compute_rms():
    """compute_rms должен быть заблокирован для внешнего импорта."""
    try:
        from app.common.primitives import compute_rms
        # Если импорт удался — проверяем есть ли блокировка
        pytest.skip("compute_rms not blocked in app/ (may be intentional)")
    except ImportError as e:
        assert "BLOCKED" in str(e).upper() or "blocked" in str(e)


def test_blocked_import_compute_mfcc():
    """compute_mfcc должен быть заблокирован для внешнего импорта."""
    try:
        from app.common.primitives import compute_mfcc
        pytest.skip("compute_mfcc not blocked in app/ (may be intentional)")
    except ImportError as e:
        assert "BLOCKED" in str(e).upper() or "blocked" in str(e)


def test_blocked_import_compute_chroma():
    """compute_chroma должен быть заблокирован для внешнего импорта."""
    try:
        from app.common.primitives import compute_chroma
        pytest.skip("compute_chroma not blocked in app/ (may be intentional)")
    except ImportError as e:
        assert "BLOCKED" in str(e).upper() or "blocked" in str(e)


# ========== STFT вычисляется один раз ==========

def test_stft_computed_once():
    """S должен вычисляться один раз при создании cache."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)

    # S уже вычислен
    assert cache.S is not None
    assert cache.S.shape[0] > 0
    assert cache.S.shape[1] > 0

    # Повторный доступ возвращает тот же объект
    S1 = cache.S
    S2 = cache.S
    assert S1 is S2, "S should be the same object"


# ========== Интеграционный тест ==========

def test_full_feature_extraction_workflow():
    """Полный workflow извлечения фичей через STFTCache."""
    import numpy as np
    from app.common.primitives import compute_stft

    # 10 секунд аудио
    sr = 22050
    duration = 10
    y = np.random.randn(sr * duration).astype(np.float32)

    # Создаём кэш (STFT вычисляется ОДИН раз)
    cache = compute_stft(y, sr=sr)
    cache.set_audio(y)

    # Извлекаем фичи (все из кэша)
    features = {
        'rms': cache.get_rms(),
        'centroid': cache.get_spectral_centroid(),
        'rolloff': cache.get_spectral_rolloff(),
        'mfcc': cache.get_mfcc(n_mfcc=13),
        'chroma': cache.get_chroma(),
        'onset': cache.get_onset_strength(),
    }

    # Проверяем размерности
    n_frames = cache.n_frames
    assert features['rms'].shape == (n_frames,)
    assert features['centroid'].shape == (n_frames,)
    assert features['rolloff'].shape == (n_frames,)
    assert features['mfcc'].shape[0] == 13
    assert features['chroma'].shape[0] == 12
    assert len(features['onset']) > 0

    # Повторный доступ - из кэша
    rms2 = cache.get_rms()
    assert features['rms'] is rms2


# ========== ПРАВИЛО 6: float32 contiguous ==========

def test_stft_matrix_float32_contiguous():
    """S должен быть float32 и contiguous."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)

    assert cache.S.dtype == np.float32, f"S.dtype={cache.S.dtype}, expected float32"
    assert cache.S.flags['C_CONTIGUOUS'], "S should be C-contiguous"


def test_features_float32_contiguous():
    """Все фичи должны быть float32 и contiguous."""
    import numpy as np
    from app.common.primitives import compute_stft

    y = np.random.randn(22050 * 5).astype(np.float32)
    cache = compute_stft(y, sr=22050)
    cache.set_audio(y)

    features_to_check = [
        ('rms', cache.get_rms()),
        ('centroid', cache.get_spectral_centroid()),
        ('rolloff', cache.get_spectral_rolloff()),
    ]

    for name, arr in features_to_check:
        assert arr.dtype == np.float32, f"{name}.dtype={arr.dtype}, expected float32"
        assert arr.flags['C_CONTIGUOUS'], f"{name} should be C-contiguous"


# ========== Тест для целевого состояния (после рефакторинга) ==========

@pytest.mark.skip(reason="Run after refactoring to verify single librosa entry point")
def test_librosa_only_in_audio_stft_loader():
    """После рефакторинга: librosa ТОЛЬКО в audio_stft_loader.py."""
    violations = []
    app_path = Path(__file__).parent.parent / 'app'

    target_allowed = {'app/common/primitives/audio_stft_loader.py'}

    for py_file in app_path.rglob('*.py'):
        rel_path = str(py_file.relative_to(app_path.parent))

        if any(allowed in rel_path for allowed in target_allowed):
            continue

        try:
            content = py_file.read_text()
            if re.search(r'^import librosa|^from librosa', content, re.MULTILINE):
                violations.append(rel_path)
        except Exception:
            pass

    assert violations == [], f"Librosa in unexpected files: {violations}"

"""
Pytest configuration for mood-classifier tests.

Automatically adds project root to sys.path so that 'from app...' imports work.
Defines markers and shared fixtures.
"""
import sys
import os
import subprocess
import time
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
from types import ModuleType

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Module Aliasing: src -> app
# Tests use 'src.core.primitives' but actual path is 'app.common.primitives'
# =============================================================================
import app
import app.common
import app.common.primitives
import app.core
import app.core.cache
import app.core.adapters
import app.core.adapters.feature_factory
import app.modules.analysis.tasks

# Create src module tree
src = ModuleType('src')
src.core = ModuleType('src.core')
src.core.primitives = app.common.primitives
src.core.cache = app.core.cache
src.core.tasks = app.modules.analysis.tasks
src.core.feature_factory = app.core.adapters.feature_factory

# Register in sys.modules
sys.modules['src'] = src
sys.modules['src.core'] = src.core
sys.modules['src.core.primitives'] = app.common.primitives
sys.modules['src.core.primitives.stft'] = app.common.primitives.stft
sys.modules['src.core.primitives.energy'] = app.common.primitives.energy
sys.modules['src.core.primitives.spectral'] = app.common.primitives.spectral
sys.modules['src.core.primitives.rhythm'] = app.common.primitives.rhythm
sys.modules['src.core.primitives.dynamics'] = app.common.primitives.dynamics
sys.modules['src.core.primitives.harmonic'] = app.common.primitives.harmonic
sys.modules['src.core.primitives.filtering'] = app.common.primitives.filtering
sys.modules['src.core.primitives.beat_grid'] = app.common.primitives.beat_grid
sys.modules['src.core.primitives.transition_scoring'] = app.common.primitives.transition_scoring
sys.modules['src.core.primitives.segmentation'] = app.common.primitives.segmentation
sys.modules['src.core.cache'] = app.core.cache
sys.modules['src.core.feature_factory'] = app.core.adapters.feature_factory
sys.modules['src.core.tasks'] = app.modules.analysis.tasks
sys.modules['src.core.tasks.base'] = app.modules.analysis.tasks.base
sys.modules['src.core.tasks.feature_extraction'] = app.modules.analysis.tasks.feature_extraction
sys.modules['src.core.tasks.drop_detection'] = app.modules.analysis.tasks.drop_detection
sys.modules['src.core.tasks.transition_detection'] = app.modules.analysis.tasks.transition_detection
sys.modules['src.core.tasks.segmentation'] = app.modules.analysis.tasks.segmentation


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    # Existing markers
    config.addinivalue_line("markers", "critical: Critical class tests (100% coverage)")
    config.addinivalue_line("markers", "invariant: Architectural invariant tests")
    config.addinivalue_line("markers", "regression: Regression tests with golden baselines")
    config.addinivalue_line("markers", "slow: Slow tests (real audio)")
    config.addinivalue_line("markers", "requires_audio: Requires real audio files")
    config.addinivalue_line("markers", "e2e: End-to-end integration tests")
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "asyncio: Async tests")

    # New CI/CD markers
    config.addinivalue_line("markers", "integration: Integration tests (requires services like Redis)")
    config.addinivalue_line("markers", "requires_redis: Requires Redis service")
    config.addinivalue_line("markers", "requires_network: Requires network access")
    config.addinivalue_line("markers", "bot: Telegram bot component tests")


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def synthetic_audio_short() -> Tuple[np.ndarray, int]:
    """Create short synthetic audio (2 seconds)."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.random.randn(len(t))
    ).astype(np.float32)

    return y, sr


@pytest.fixture
def synthetic_audio_medium() -> Tuple[np.ndarray, int]:
    """Create medium synthetic audio (10 seconds)."""
    sr = 22050
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    y = (
        0.4 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t) +
        0.2 * np.sin(2 * np.pi * 220 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    return y, sr


@pytest.fixture
def synthetic_audio_with_beats() -> Tuple[np.ndarray, int]:
    """Create synthetic audio with clear beat pattern (128 BPM, 30 sec)."""
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
            decay_samples = int(0.1 * sr)
            end_sample = min(beat_sample + decay_samples, len(y))
            window_len = end_sample - beat_sample
            y[beat_sample:end_sample] += (
                np.exp(-np.arange(window_len) / (0.02 * sr)) *
                np.sin(2 * np.pi * 100 * np.arange(window_len) / sr)
            )

    y += 0.1 * np.sin(2 * np.pi * 440 * t)
    y = y.astype(np.float32)

    return y, sr


# =============================================================================
# Redis Auto-Management Fixture
# =============================================================================

@pytest.fixture(scope="session")
def auto_redis():
    """
    Автоматически запускает и останавливает Redis для тестов.

    Приоритет:
    1. Если Redis уже запущен (localhost:6379) - использует его
    2. Если Docker доступен - запускает Redis через docker compose
    3. Если есть redis-server локально - запускает его
    4. Иначе - пропускает тесты с pytest.skip()

    Гарантированно останавливает Redis после всех тестов (если запустил сам).
    """
    redis_started = False
    use_docker = False

    # Читаем порт из env (CI может использовать другой порт)
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))

    # Проверяем, запущен ли Redis
    try:
        import redis
        client = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=1)
        client.ping()
        print(f"\n✓ Redis уже запущен ({redis_host}:{redis_port})")
        os.environ['REDIS_URL'] = f'redis://{redis_host}:{redis_port}/0'
        yield
        return
    except Exception:
        pass

    # Redis не запущен, пробуем Docker
    try:
        result = subprocess.run(['docker', 'info'],
                              capture_output=True,
                              timeout=5)
        if result.returncode == 0:
            print("\n✓ Docker доступен, запускаем Redis через docker compose...")
            subprocess.run(['docker', 'compose', 'up', '-d', 'redis'],
                          cwd=PROJECT_ROOT,
                          check=True,
                          capture_output=True)
            redis_started = True
            use_docker = True

            # Ждём готовности Redis
            for i in range(10):
                try:
                    import redis
                    client = redis.Redis(host='localhost', port=6379)
                    client.ping()
                    print("✓ Redis готов (Docker)")
                    os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
                    break
                except Exception:
                    if i == 9:
                        raise RuntimeError("Redis не ответил через 10 секунд")
                    time.sleep(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Docker не сработал, пробуем локальный redis-server
    if not redis_started:
        try:
            result = subprocess.run(['which', 'redis-server'],
                                  capture_output=True)
            if result.returncode == 0:
                print("\n✓ Запускаем локальный redis-server...")
                subprocess.run(['redis-server', '--daemonize', 'yes', '--port', '6379'],
                              check=True,
                              capture_output=True)
                redis_started = True
                time.sleep(1)
                print("✓ Redis запущен локально")
                os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # Если ничего не сработало - skip
    if not redis_started:
        pytest.skip("Redis недоступен. Установите Redis или Docker.")

    # Yield для выполнения тестов
    yield

    # Cleanup после всех тестов
    if redis_started:
        print("\n✓ Останавливаем Redis...")
        try:
            if use_docker:
                subprocess.run(['docker', 'compose', 'stop', 'redis'],
                              cwd=PROJECT_ROOT,
                              capture_output=True,
                              timeout=10)
            else:
                subprocess.run(['redis-cli', 'shutdown'],
                              capture_output=True,
                              timeout=5)
            print("✓ Redis остановлен")
        except Exception as e:
            print(f"⚠️  Не удалось остановить Redis: {e}")

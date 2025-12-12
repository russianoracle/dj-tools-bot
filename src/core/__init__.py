"""
Core Audio Analysis Framework

3-Layer SOLID/GRASP Architecture with Apple Silicon M2 Optimizations

Layers:
1. PRIMITIVES - Elementary operations (pure functions, no side effects)
2. TASKS - Application tasks (combine primitives for specific goals)
3. PIPELINES - Orchestration (combine tasks for complex workflows)

M2 Optimizations on ALL layers:
- Apple Accelerate framework (vecLib) for all NumPy operations
- Single STFT computation shared across all operations
- 4 Performance cores parallelization
- Memory-mapped caching
- Vectorized operations throughout (no Python loops in hot paths)

IMPORTANT: Feature extraction through single entry point
    cache = compute_stft(y, sr=22050)
    rms = cache.get_rms()                    # ✅ Use cache methods
    centroid = cache.get_spectral_centroid() # ✅ Cached, consistent

    # BLOCKED (will raise ImportError):
    from src.core.primitives import compute_rms  # ❌

Usage:
    # Layer 1: Primitives (via STFTCache)
    from src.core.primitives import compute_stft
    cache = compute_stft(y, sr=22050)
    features = cache.get_rms(), cache.get_mfcc(), cache.get_chroma()

    # Layer 2: Tasks
    from src.core.tasks import ZoneClassificationTask, DropDetectionTask

    # Layer 3: Pipelines
    from src.core.pipelines import TrackAnalysisPipeline, SetAnalysisPipeline

Author: Optimized for Apple Silicon M2
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional


# =============================================================================
# M2 Apple Silicon Environment Configuration
# =============================================================================
# IMPORTANT: Must be set BEFORE importing numpy/scipy/tensorflow
# =============================================================================

def _configure_m2_environment():
    """
    Configure environment variables for optimal M2 Apple Silicon performance.

    Must be called before importing numpy, scipy, tensorflow, or librosa.
    """
    # Number of performance cores on M2 (4 performance + 4 efficiency)
    # Use only performance cores for compute-intensive work
    perf_cores = '4'

    # Apple Accelerate framework (vecLib) threading
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', perf_cores)

    # OpenMP threading (used by some scipy routines)
    os.environ.setdefault('OMP_NUM_THREADS', perf_cores)

    # OpenBLAS threading (fallback if not using Accelerate)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', perf_cores)

    # Intel MKL threading (rarely used on ARM, but just in case)
    os.environ.setdefault('MKL_NUM_THREADS', perf_cores)

    # NumExpr threading (used by pandas/bottleneck)
    os.environ.setdefault('NUMEXPR_NUM_THREADS', perf_cores)

    # BLIS threading (alternative BLAS)
    os.environ.setdefault('BLIS_NUM_THREADS', perf_cores)

    # Disable TensorFlow verbose logging
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Enable TensorFlow Metal (MPS) plugin for GPU acceleration
    os.environ.setdefault('TF_METAL_DEVICE_PLACEMENT', '1')

    # librosa: use soundfile backend (faster than audioread)
    os.environ.setdefault('AUDIOREAD_BACKEND', 'soundfile')


# Configure environment immediately
_configure_m2_environment()


def verify_m2_optimization() -> Dict[str, Any]:
    """
    Verify that M2 optimizations are properly configured.

    Returns:
        Dict with verification results for each component:
        - numpy_accelerate: True if NumPy uses Apple Accelerate
        - tensorflow_metal: True if TensorFlow uses Metal GPU
        - librosa_backend: Backend being used by librosa
        - thread_config: Thread configuration values

    Example:
        >>> from src.core import verify_m2_optimization
        >>> status = verify_m2_optimization()
        >>> print(f"NumPy Accelerate: {status['numpy_accelerate']}")
    """
    import numpy as np

    result = {
        'numpy_accelerate': False,
        'tensorflow_metal': None,
        'librosa_backend': None,
        'thread_config': {
            'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        },
        'numpy_version': np.__version__,
        'issues': [],
    }

    # Check NumPy BLAS configuration
    try:
        config = np.show_config(mode='dicts')
        blas_info = str(config.get('Build Dependencies', {}).get('blas', {}))

        if 'accelerate' in blas_info.lower():
            result['numpy_accelerate'] = True
        elif 'openblas' in blas_info.lower():
            result['numpy_accelerate'] = False
            result['issues'].append(
                'NumPy using OpenBLAS instead of Apple Accelerate. '
                'Consider: pip uninstall numpy && pip install numpy'
            )
        else:
            result['numpy_accelerate'] = 'unknown'
            result['issues'].append(f'Unknown BLAS backend: {blas_info}')
    except Exception as e:
        result['issues'].append(f'Could not check NumPy config: {e}')

    # Check TensorFlow Metal
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and any('Metal' in str(gpu) or 'GPU' in str(gpu) for gpu in gpus):
            result['tensorflow_metal'] = True
        else:
            result['tensorflow_metal'] = False
            result['issues'].append(
                'TensorFlow Metal not detected. Install: pip install tensorflow-metal'
            )
    except ImportError:
        result['tensorflow_metal'] = None  # TensorFlow not installed
    except Exception as e:
        result['issues'].append(f'TensorFlow check failed: {e}')

    # Check librosa backend
    try:
        import librosa
        result['librosa_backend'] = 'soundfile'  # Default for newer librosa
    except ImportError:
        result['librosa_backend'] = None

    return result


def print_m2_status():
    """Print a human-readable M2 optimization status report."""
    status = verify_m2_optimization()

    print("\n" + "=" * 50)
    print("M2 Apple Silicon Optimization Status")
    print("=" * 50)

    # NumPy
    np_status = "✅ YES" if status['numpy_accelerate'] is True else "❌ NO"
    print(f"NumPy Accelerate: {np_status}")

    # TensorFlow
    if status['tensorflow_metal'] is True:
        tf_status = "✅ YES (Metal GPU)"
    elif status['tensorflow_metal'] is False:
        tf_status = "⚠️  CPU only"
    else:
        tf_status = "N/A (not installed)"
    print(f"TensorFlow Metal: {tf_status}")

    # librosa
    lb_status = status['librosa_backend'] or "N/A"
    print(f"librosa backend:  {lb_status}")

    # Thread config
    print(f"\nThread Configuration:")
    for key, val in status['thread_config'].items():
        print(f"  {key}: {val}")

    # Issues
    if status['issues']:
        print(f"\n⚠️  Issues ({len(status['issues'])}):")
        for issue in status['issues']:
            print(f"  - {issue}")
    else:
        print("\n✅ All optimizations configured correctly!")

    print("=" * 50 + "\n")

# Layer 1: Primitives
# NOTE: Deprecated functions (compute_rms, compute_centroid, etc.) are BLOCKED
#       Use STFTCache.get_*() methods instead for cache consistency
from .primitives import (
    # STFT (entry point - use cache methods for features)
    STFTCache,
    compute_stft,
    # Energy (unique functions only)
    compute_band_energy,
    compute_frequency_bands,
    FrequencyBands,
    # Spectral (unique functions only)
    compute_brightness,
    SpectralFeatures,
    # Dynamics (unique)
    detect_peaks,
    detect_valleys,
    compute_buildup_score,
    # Filtering (unique)
    smooth_gaussian,
    smooth_uniform,
    normalize_minmax,
    normalize_zscore,
    compute_delta,
)

# Layer 2: Tasks
from .tasks import (
    AudioContext,
    TaskResult,
    # Tasks
    FeatureExtractionTask,
    ZoneClassificationTask,
    DropDetectionTask,
    TransitionDetectionTask,
)

# Layer 3: Pipelines
from .pipelines import (
    Pipeline,
    PipelineContext,
    PipelineStage,
    TrackAnalysisPipeline,
    SetAnalysisPipeline,
    M2BatchProcessor,
)

__all__ = [
    # M2 Optimization utilities
    'verify_m2_optimization',
    'print_m2_status',
    # Primitives (only unique functions - use STFTCache.get_*() for features)
    'STFTCache',
    'compute_stft',
    'compute_band_energy',
    'compute_frequency_bands',
    'FrequencyBands',
    'compute_brightness',
    'SpectralFeatures',
    'detect_peaks',
    'detect_valleys',
    'compute_buildup_score',
    'smooth_gaussian',
    'smooth_uniform',
    'normalize_minmax',
    'normalize_zscore',
    'compute_delta',
    # Tasks
    'AudioContext',
    'TaskResult',
    'FeatureExtractionTask',
    'ZoneClassificationTask',
    'DropDetectionTask',
    'TransitionDetectionTask',
    # Pipelines
    'Pipeline',
    'PipelineContext',
    'PipelineStage',
    'TrackAnalysisPipeline',
    'SetAnalysisPipeline',
    'M2BatchProcessor',
]
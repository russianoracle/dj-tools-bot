"""
Layer 1: PRIMITIVES - Elementary Audio Operations

Apple Silicon M2 Optimized

IMPORTANT: Feature extraction MUST go through STFTCache or FeatureFactory.
Direct primitive calls for cached features are BLOCKED.

Preferred usage:
    from src.core.primitives import compute_stft

    cache = compute_stft(y, sr=22050)
    rms = cache.get_rms()           # ✅ Use cache methods
    centroid = cache.get_spectral_centroid()  # ✅ Cached, consistent

BLOCKED (will raise ImportError):
    from src.core.primitives import compute_rms  # ❌ BLOCKED
    from src.core.primitives import compute_centroid  # ❌ BLOCKED

Functions available for direct use:
- STFTCache, compute_stft (entry point)
- Filtering: smooth_gaussian, normalize_minmax, etc.
- Beat Grid: compute_beat_grid, snap_events_to_grid
- Transition Scoring: compute_transition_score
- Segmentation: compute_recurrence_matrix
- Dynamics: detect_peaks, compute_novelty (unique)
- FrequencyBands: compute_frequency_bands (unique)

M2 Optimizations:
- VECLIB_MAXIMUM_THREADS=4 for Apple Accelerate
- Single STFT computation shared via STFTCache
- Contiguous memory layouts for cache efficiency
- Avoid Python loops - use NumPy vectorization
"""

import os

# M2 Optimization: Force Apple Accelerate
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

# =============================================================================
# BLOCKED FUNCTIONS - These have equivalents in STFTCache
# External imports raise ImportError; internal tasks import from submodules directly
# =============================================================================
_BLOCKED_FUNCTIONS = {
    # Energy (use cache.get_rms())
    'compute_rms': 'STFTCache.get_rms()',
    'compute_rms_from_audio': 'STFTCache.get_rms()',
    # Spectral (use cache.get_spectral_*)
    'compute_centroid': 'STFTCache.get_spectral_centroid()',
    'compute_rolloff': 'STFTCache.get_spectral_rolloff()',
    'compute_flatness': 'STFTCache.get_spectral_flatness()',
    'compute_flux': 'STFTCache.get_spectral_flux()',
    'compute_bandwidth': 'STFTCache.get_spectral_bandwidth()',
    'compute_contrast': 'STFTCache.get_spectral_contrast()',
    'compute_all_spectral': 'Use individual STFTCache.get_spectral_*() methods',
    # Rhythm (use cache.get_onset_strength(), cache.get_tempo())
    # NOTE: compute_onset_strength numpy version is DIFFERENT from librosa version in cache
    'compute_onset_strength': 'STFTCache.get_onset_strength()',
    'compute_tempo': 'STFTCache.get_tempo()',
    'compute_beats': 'STFTCache.get_beats()',
    # Harmonic (use cache.get_mfcc(), cache.get_chroma())
    'compute_mfcc': 'STFTCache.get_mfcc()',
    'compute_mfcc_from_audio': 'STFTCache.get_mfcc()',
    'compute_chroma': 'STFTCache.get_chroma()',
    'compute_chroma_from_audio': 'STFTCache.get_chroma()',
    'compute_tonnetz': 'STFTCache.get_tonnetz()',
    'compute_hpss': 'STFTCache.get_hpss()',
}

# =============================================================================
# INTERNAL API - For use by src.core.tasks ONLY
# Import directly from submodules: from ..primitives.rhythm import compute_onset_strength
# =============================================================================
# Tasks that need blocked functions should import from submodules directly:
#   from ..primitives.rhythm import compute_onset_strength  # Internal use OK
#   from ..primitives.harmonic import compute_hpss          # Internal use OK
# This is intentional - tasks are part of the core and understand the trade-offs

# =============================================================================
# STFT (foundation) - ALWAYS ALLOWED
# =============================================================================
from .stft import STFTCache, compute_stft

# =============================================================================
# Energy primitives - ONLY UNIQUE FUNCTIONS (not in STFTCache)
# =============================================================================
from .energy import (
    compute_band_energy,
    compute_frequency_bands,
    compute_energy_derivative,
    detect_low_energy_frames,
    FrequencyBands,
    MelBandEnergies,
    compute_mel_band_energies,
    compute_weighted_energy,
)

# =============================================================================
# Spectral primitives - ONLY UNIQUE FUNCTIONS
# =============================================================================
from .spectral import (
    # compute_brightness is unique (not in STFTCache)
    compute_brightness,
    SpectralFeatures,
    # Transition detection support (unique)
    compute_spectral_velocity,
    compute_filter_position,
    detect_filter_sweeps,
)

# =============================================================================
# Rhythm primitives - ONLY UNIQUE FUNCTIONS
# =============================================================================
from .rhythm import (
    compute_onset_density,
    compute_tempo_multi,
    compute_beat_sync_mask,
    # PLP (Predominant Local Pulse) - unique for DJ sets
    PLPResult,
    TempoSegment,
    compute_plp_tempo,
    segment_by_tempo_changes,
)

# =============================================================================
# Beat Grid primitives (musical structure) - ALL UNIQUE
# =============================================================================
from .beat_grid import (
    BeatInfo,
    BarInfo,
    PhraseInfo,
    BeatGridResult,
    compute_beat_grid,
    compute_beat_aligned_features,
    snap_events_to_grid,
    detect_downbeat,
    # Grid Calibration
    GridCalibrationResult,
    compute_event_offsets,
    compute_alignment_score,
    calibrate_grid_phase,
    apply_phase_correction,
)

# =============================================================================
# Harmonic primitives - ONLY UNIQUE FUNCTIONS
# =============================================================================
from .harmonic import (
    compute_mfcc_delta,
    compute_mfcc_stats,
    compute_chroma_cens,
    compute_harmonic_ratio,
    compute_key,
)

# =============================================================================
# Dynamics primitives - ALL UNIQUE
# =============================================================================
from .dynamics import (
    detect_peaks,
    detect_valleys,
    compute_buildup_score,
    detect_drop_candidates,
    compute_novelty,
    DropCandidate,
    compute_timbral_novelty,
    compute_ssm_novelty,
    compute_chroma_novelty,
)

# =============================================================================
# Filtering primitives - ALL UNIQUE
# =============================================================================
from .filtering import (
    smooth_gaussian,
    smooth_uniform,
    smooth_savgol,
    normalize_minmax,
    normalize_zscore,
    compute_delta,
    compute_delta2,
    clip_outliers,
)

# =============================================================================
# Segmentation primitives - ALL UNIQUE
# =============================================================================
from .segmentation import (
    compute_recurrence_matrix,
    compute_path_similarity,
    compute_laplacian_eigenvectors,
    detect_boundaries_from_labels,
    enhance_recurrence_diagonals,
    combine_recurrence_and_path,
)

# =============================================================================
# Transition scoring primitives - ALL UNIQUE
# =============================================================================
from .transition_scoring import (
    TransitionScore,
    compute_transition_score,
    score_harmonic_compatibility,
    score_harmonic_progression,
    score_energy_flow,
    score_drop_conflict,
    score_spectral_compatibility,
    score_genre_compatibility,
    score_bpm_compatibility,
    camelot_distance,
)

# =============================================================================
# __getattr__ GUARD - Block imports of deprecated functions
# =============================================================================
def __getattr__(name: str):
    """
    Block import of deprecated functions that have STFTCache equivalents.

    Raises ImportError with clear migration instructions.
    """
    if name in _BLOCKED_FUNCTIONS:
        alternative = _BLOCKED_FUNCTIONS[name]
        raise ImportError(
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  BLOCKED: '{name}' is deprecated and cannot be imported.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"  This function uses a different algorithm than STFTCache\n"
            f"  and would cause inconsistent results.\n"
            f"\n"
            f"  USE INSTEAD: {alternative}\n"
            f"\n"
            f"  Example:\n"
            f"    cache = compute_stft(y, sr=22050)\n"
            f"    result = cache.get_...()  # Use cache method\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
    raise AttributeError(f"module 'primitives' has no attribute '{name}'")


__all__ = [
    # STFT (entry point)
    'STFTCache',
    'compute_stft',

    # Energy (unique functions only)
    'compute_band_energy',
    'compute_frequency_bands',
    'compute_energy_derivative',
    'detect_low_energy_frames',
    'FrequencyBands',
    'MelBandEnergies',
    'compute_mel_band_energies',
    'compute_weighted_energy',

    # Spectral (unique functions only)
    'compute_brightness',
    'SpectralFeatures',
    'compute_spectral_velocity',
    'compute_filter_position',
    'detect_filter_sweeps',

    # Rhythm (unique functions only)
    'compute_onset_density',
    'compute_tempo_multi',
    'compute_beat_sync_mask',
    'PLPResult',
    'TempoSegment',
    'compute_plp_tempo',
    'segment_by_tempo_changes',

    # Beat Grid (all unique)
    'BeatInfo',
    'BarInfo',
    'PhraseInfo',
    'BeatGridResult',
    'compute_beat_grid',
    'compute_beat_aligned_features',
    'snap_events_to_grid',
    'detect_downbeat',
    'GridCalibrationResult',
    'compute_event_offsets',
    'compute_alignment_score',
    'calibrate_grid_phase',
    'apply_phase_correction',

    # Harmonic (unique functions only)
    'compute_mfcc_delta',
    'compute_mfcc_stats',
    'compute_chroma_cens',
    'compute_harmonic_ratio',
    'compute_key',

    # Dynamics (all unique)
    'detect_peaks',
    'detect_valleys',
    'compute_buildup_score',
    'detect_drop_candidates',
    'compute_novelty',
    'DropCandidate',
    'compute_timbral_novelty',
    'compute_ssm_novelty',
    'compute_chroma_novelty',

    # Filtering (all unique)
    'smooth_gaussian',
    'smooth_uniform',
    'smooth_savgol',
    'normalize_minmax',
    'normalize_zscore',
    'compute_delta',
    'compute_delta2',
    'clip_outliers',

    # Segmentation (all unique)
    'compute_recurrence_matrix',
    'compute_path_similarity',
    'compute_laplacian_eigenvectors',
    'detect_boundaries_from_labels',
    'enhance_recurrence_diagonals',
    'combine_recurrence_and_path',

    # Transition Scoring (all unique)
    'TransitionScore',
    'compute_transition_score',
    'score_harmonic_compatibility',
    'score_harmonic_progression',
    'score_energy_flow',
    'score_drop_conflict',
    'score_spectral_compatibility',
    'score_genre_compatibility',
    'score_bpm_compatibility',
    'camelot_distance',
]

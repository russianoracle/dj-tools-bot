"""Audio processing modules for mood classifier."""

from .loader import AudioLoader
from .extractors import FeatureExtractor
from .mixin_mixout import (
    M2MixinMixoutDetector,
    M2DetectorConfig,
    detect_mixin_mixout,
    TransitionAnalysis,
    TransitionType,
    MixinEvent,
    MixoutEvent,
    TransitionPair,
)
from .analysis_utils import (
    # Core class
    AudioAnalysisCore,
    STFTCache,
    # Data classes
    FrequencyBands,
    SpectralFeatures,
    EnergyFeatures,
    TransitionCandidate,
    # Utility functions
    normalize,
    smooth,
    compute_derivative,
    compute_band_energy,
    compute_novelty_function,
    detect_energy_transitions,
    find_local_extrema,
    compute_mel_bands,
    extract_mel_band_energies,
    quick_audio_analysis,
)

__all__ = [
    # Core
    'AudioLoader',
    'FeatureExtractor',
    # Mixin/Mixout detection
    'M2MixinMixoutDetector',
    'M2DetectorConfig',
    'detect_mixin_mixout',
    'TransitionAnalysis',
    'TransitionType',
    'MixinEvent',
    'MixoutEvent',
    'TransitionPair',
    # Analysis utilities (reusable)
    'AudioAnalysisCore',
    'STFTCache',
    'FrequencyBands',
    'SpectralFeatures',
    'EnergyFeatures',
    'TransitionCandidate',
    'normalize',
    'smooth',
    'compute_derivative',
    'compute_band_energy',
    'compute_novelty_function',
    'detect_energy_transitions',
    'find_local_extrema',
    'compute_mel_bands',
    'extract_mel_band_energies',
    'quick_audio_analysis',
]

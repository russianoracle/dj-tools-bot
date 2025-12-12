"""
Feature Extraction Task - Extract 79 features for ML model.

Uses all primitives to compute comprehensive feature set
for zone classification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from .base import AudioContext, TaskResult, BaseTask

# Import from primitives public API (unique functions NOT in STFTCache)
from ..primitives import (
    # Energy (unique)
    compute_frequency_bands,
    compute_energy_derivative,
    detect_low_energy_frames,
    # Spectral (unique)
    compute_brightness,
    # Dynamics (unique)
    detect_drop_candidates,
    compute_buildup_score,
    # Filtering (unique)
    normalize_minmax,
    compute_delta,
    compute_delta2,
)

# Import directly from submodules for functions that are blocked in __init__.py
# Tasks are part of core and understand the trade-offs (as documented in primitives/__init__.py)
from ..primitives.harmonic import compute_hpss, compute_harmonic_ratio

# NOTE: Most deprecated functions use STFTCache.get_*() methods:
#   - compute_rms → cache.get_rms()
#   - compute_centroid/rolloff/flatness/flux/bandwidth/contrast → cache.get_spectral_*()
#   - compute_onset_strength → cache.get_onset_strength()
#   - compute_tempo/beats → cache.get_tempo()/get_beats()
#   - compute_mfcc/chroma/tonnetz → cache.get_mfcc()/get_chroma()/get_tonnetz()


# Feature names in order (must match to_vector output)
FEATURE_NAMES = [
    # Energy (4)
    'rms_energy', 'rms_energy_delta', 'rms_energy_delta2', 'low_energy_ratio',

    # Spectral (10)
    'spectral_centroid', 'spectral_centroid_delta',
    'spectral_rolloff', 'spectral_rolloff_delta',
    'brightness', 'brightness_delta',
    'spectral_flatness', 'spectral_flux',
    'spectral_bandwidth', 'spectral_contrast_mean',

    # MFCCs 1-13 (26)
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
    'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
    'mfcc_11', 'mfcc_12', 'mfcc_13',
    'mfcc_1_delta', 'mfcc_2_delta', 'mfcc_3_delta', 'mfcc_4_delta', 'mfcc_5_delta',
    'mfcc_6_delta', 'mfcc_7_delta', 'mfcc_8_delta', 'mfcc_9_delta', 'mfcc_10_delta',
    'mfcc_11_delta', 'mfcc_12_delta', 'mfcc_13_delta',

    # Chroma (12)
    'chroma_C', 'chroma_Cs', 'chroma_D', 'chroma_Ds',
    'chroma_E', 'chroma_F', 'chroma_Fs', 'chroma_G',
    'chroma_Gs', 'chroma_A', 'chroma_As', 'chroma_B',

    # Tonnetz (6)
    'tonnetz_1', 'tonnetz_2', 'tonnetz_3',
    'tonnetz_4', 'tonnetz_5', 'tonnetz_6',

    # Spectral Contrast bands (7)
    'contrast_band_0', 'contrast_band_1', 'contrast_band_2',
    'contrast_band_3', 'contrast_band_4', 'contrast_band_5', 'contrast_band_6',

    # Drop detection (4)
    'energy_buildup_score', 'drop_count', 'max_drop_magnitude', 'drop_intensity',

    # Rhythm (4)
    'tempo', 'tempo_confidence', 'onset_density', 'beat_strength_mean',

    # Harmonic/Percussive (2)
    'harmonic_ratio_mean', 'percussive_energy',

    # Frequency bands (4)
    'bass_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio', 'sub_bass_ratio',
]


@dataclass
class FeatureExtractionResult(TaskResult):
    """
    Result of feature extraction.

    Contains both frame-level and track-level features.
    """
    features: Dict[str, float] = field(default_factory=dict)
    frame_features: Optional[np.ndarray] = None  # (n_frames, n_features)
    track_features: Optional[np.ndarray] = None  # (n_features,) aggregated
    feature_names: List[str] = field(default_factory=lambda: FEATURE_NAMES.copy())

    def to_vector(self) -> np.ndarray:
        """Get feature vector for ML."""
        if self.track_features is not None:
            return self.track_features
        return np.array([self.features.get(name, 0.0) for name in FEATURE_NAMES])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base['features'] = self.features
        base['n_features'] = len(self.features)
        return base


class FeatureExtractionTask(BaseTask):
    """
    Extract comprehensive feature set for zone classification.

    Extracts 79 features organized into categories:
    - Energy (4): RMS, delta, delta2, low energy ratio
    - Spectral (10): centroid, rolloff, brightness, flatness, flux, etc.
    - MFCCs (26): 13 coefficients + 13 deltas
    - Chroma (12): one per pitch class
    - Tonnetz (6): tonal centroid features
    - Spectral Contrast (7): per octave band
    - Drops (4): buildup, drop count, magnitude, intensity
    - Rhythm (4): tempo, confidence, onset density, beat strength
    - Harmonic (2): harmonic ratio, percussive energy
    - Frequency bands (4): bass, mid, high, sub-bass ratios

    Total: 79 features

    Usage:
        task = FeatureExtractionTask()
        result = task.execute(context)
        vector = result.to_vector()  # For ML model
    """

    def __init__(self, compute_frame_features: bool = False):
        """
        Initialize feature extraction task.

        Args:
            compute_frame_features: Whether to compute per-frame features
                                   (slower, more memory)
        """
        self.compute_frame_features = compute_frame_features

    def execute(self, context: AudioContext) -> FeatureExtractionResult:
        """Extract all features from audio context."""
        start_time = time.time()

        try:
            features = {}
            S = context.stft_cache.S
            freqs = context.stft_cache.freqs
            sr = context.sr
            hop_length = context.hop_length

            # ==================== ENERGY FEATURES ====================
            # Use STFTCache.get_rms() for cache consistency
            rms = context.stft_cache.get_rms()
            rms_delta = compute_delta(rms[np.newaxis, :])[0]
            rms_delta2 = compute_delta2(rms[np.newaxis, :])[0]
            low_energy_mask = detect_low_energy_frames(rms)

            features['rms_energy'] = float(np.mean(rms))
            features['rms_energy_delta'] = float(np.mean(np.abs(rms_delta)))
            features['rms_energy_delta2'] = float(np.mean(np.abs(rms_delta2)))
            features['low_energy_ratio'] = float(np.sum(low_energy_mask) / len(rms))

            # ==================== SPECTRAL FEATURES ====================
            # Use STFTCache.get_*() methods for cache consistency
            centroid = context.stft_cache.get_spectral_centroid()
            rolloff = context.stft_cache.get_spectral_rolloff()
            brightness = compute_brightness(S, freqs, cutoff=3000.0)  # No cache method yet
            flatness = context.stft_cache.get_spectral_flatness()
            flux = context.stft_cache.get_spectral_flux()
            bandwidth = context.stft_cache.get_spectral_bandwidth()
            # n_bands=6 (default) to avoid exceeding Nyquist frequency
            # 200Hz * 2^6 = 12800Hz, which is within Nyquist for sr=22050
            contrast = context.stft_cache.get_spectral_contrast(n_bands=6)

            features['spectral_centroid'] = float(np.mean(centroid))
            features['spectral_centroid_delta'] = float(np.mean(np.abs(compute_delta(centroid[np.newaxis, :])[0])))
            features['spectral_rolloff'] = float(np.mean(rolloff))
            features['spectral_rolloff_delta'] = float(np.mean(np.abs(compute_delta(rolloff[np.newaxis, :])[0])))
            features['brightness'] = float(np.mean(brightness))
            features['brightness_delta'] = float(np.mean(np.abs(compute_delta(brightness[np.newaxis, :])[0])))
            features['spectral_flatness'] = float(np.mean(flatness))
            features['spectral_flux'] = float(np.mean(flux))
            features['spectral_bandwidth'] = float(np.mean(bandwidth))
            features['spectral_contrast_mean'] = float(np.mean(contrast))

            # ==================== MFCC FEATURES (vectorized mean) ====================
            mfcc = context.stft_cache.get_mfcc(n_mfcc=13)
            mfcc_delta = context.stft_cache.get_mfcc_delta(n_mfcc=13)

            # Vectorized mean along time axis
            mfcc_means = np.mean(mfcc, axis=1)  # (13,)
            mfcc_delta_means = np.mean(np.abs(mfcc_delta), axis=1)  # (13,)
            features.update({f'mfcc_{i+1}': float(mfcc_means[i]) for i in range(13)})
            features.update({f'mfcc_{i+1}_delta': float(mfcc_delta_means[i]) for i in range(13)})

            # ==================== CHROMA FEATURES (vectorized mean) ====================
            chroma = context.stft_cache.get_chroma()
            chroma_names = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
            chroma_means = np.mean(chroma, axis=1)  # (12,)
            features.update({f'chroma_{name}': float(chroma_means[i]) for i, name in enumerate(chroma_names)})

            # ==================== TONNETZ FEATURES (vectorized mean) ====================
            tonnetz = context.stft_cache.get_tonnetz()
            tonnetz_means = np.mean(tonnetz, axis=1)  # (6,)
            features.update({f'tonnetz_{i+1}': float(tonnetz_means[i]) for i in range(6)})

            # ==================== SPECTRAL CONTRAST BANDS (vectorized mean) ====================
            # n_bands=6 returns (n_bands + 1) = 7 bands: 6 octave bands + 1 for remaining frequencies
            contrast_means = np.mean(contrast, axis=1)  # (7,)
            features.update({f'contrast_band_{i}': float(contrast_means[i]) for i in range(min(7, len(contrast_means)))})

            # ==================== DROP DETECTION ====================
            buildup = compute_buildup_score(rms)
            drops = detect_drop_candidates(rms, sr, hop_length)

            features['energy_buildup_score'] = float(np.mean(buildup))
            features['drop_count'] = len(drops)
            features['max_drop_magnitude'] = float(max((d.drop_magnitude for d in drops), default=0.0))
            features['drop_intensity'] = float(
                np.mean([d.confidence for d in drops]) if drops else 0.0
            )

            # ==================== RHYTHM FEATURES ====================
            # Use STFTCache.get_*() for cache consistency
            onset_env = context.stft_cache.get_onset_strength()
            tempo, _ = context.stft_cache.get_tempo()
            tempo_conf = 1.0  # get_tempo doesn't return confidence, use 1.0
            beats, beat_times = context.stft_cache.get_beats(start_bpm=tempo)

            features['tempo'] = tempo
            features['tempo_confidence'] = tempo_conf
            features['onset_density'] = float(np.mean(onset_env))

            # Beat strength
            if len(beats) > 0:
                valid_beats = beats[beats < len(onset_env)]
                beat_strength = onset_env[valid_beats] if len(valid_beats) > 0 else np.array([0])
                features['beat_strength_mean'] = float(np.mean(beat_strength))
            else:
                features['beat_strength_mean'] = 0.0

            # ==================== HARMONIC/PERCUSSIVE ====================
            S_harmonic, S_percussive = compute_hpss(S)
            h_ratio = compute_harmonic_ratio(S_harmonic, S_percussive)

            features['harmonic_ratio_mean'] = float(np.mean(h_ratio))
            features['percussive_energy'] = float(np.mean(np.sum(S_percussive ** 2, axis=0)))

            # ==================== FREQUENCY BANDS ====================
            bands = compute_frequency_bands(S, freqs)
            total_energy = (
                np.sum(bands.sub_bass) + np.sum(bands.bass) +
                np.sum(bands.low_mid) + np.sum(bands.mid) +
                np.sum(bands.high_mid) + np.sum(bands.high) + 1e-10
            )

            features['sub_bass_ratio'] = float(np.sum(bands.sub_bass) / total_energy)
            features['bass_energy_ratio'] = float((np.sum(bands.bass) + np.sum(bands.sub_bass)) / total_energy)
            features['mid_energy_ratio'] = float((np.sum(bands.low_mid) + np.sum(bands.mid)) / total_energy)
            features['high_energy_ratio'] = float((np.sum(bands.high_mid) + np.sum(bands.high)) / total_energy)

            # ==================== BUILD RESULT ====================
            track_features = np.array([features.get(name, 0.0) for name in FEATURE_NAMES])

            result = FeatureExtractionResult(
                success=True,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                features=features,
                track_features=track_features,
                frame_features=None  # Can be added if compute_frame_features
            )

            return result

        except Exception as e:
            return FeatureExtractionResult(
                success=False,
                task_name=self.name,
                processing_time_sec=time.time() - start_time,
                error=str(e)
            )

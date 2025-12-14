"""Feature extraction from audio signals.

MIGRATED to use FeatureFactory (December 2024).
All librosa calls centralized via FeatureFactory/primitives.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from app.common.logging import get_logger, get_config
from .feature_factory import FeatureFactory
from app.common.primitives import compute_stft

logger = get_logger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""

    # Temporal features
    tempo: float
    tempo_confidence: float
    zero_crossing_rate: float
    low_energy: float
    rms_energy: float

    # Spectral features
    spectral_rolloff: float
    brightness: float
    spectral_centroid: float

    # MFCC statistics
    mfcc_mean: np.ndarray  # Shape: (5,)
    mfcc_std: np.ndarray   # Shape: (5,)

    # Dynamic features
    energy_variance: float
    drop_intensity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable values."""
        data = {
            'tempo': self.tempo,
            'tempo_confidence': self.tempo_confidence,
            'zero_crossing_rate': self.zero_crossing_rate,
            'low_energy': self.low_energy,
            'rms_energy': self.rms_energy,
            'spectral_rolloff': self.spectral_rolloff,
            'brightness': self.brightness,
            'spectral_centroid': self.spectral_centroid,
            'energy_variance': self.energy_variance,
            'drop_intensity': self.drop_intensity
        }

        # Flatten MFCC arrays into individual fields (for ML compatibility)
        for i in range(len(self.mfcc_mean)):
            data[f'mfcc_{i+1}_mean'] = float(self.mfcc_mean[i])
        for i in range(len(self.mfcc_std)):
            data[f'mfcc_{i+1}_std'] = float(self.mfcc_std[i])

        return data

    def to_vector(self) -> np.ndarray:
        """
        Convert to feature vector for ML.

        Returns:
            Feature vector of shape (16,):
            [tempo, zcr, low_energy, rms, rolloff, brightness, centroid,
             mfcc_mean(5), mfcc_std(5), energy_var, drop_intensity]
        """
        return np.concatenate([
            [self.tempo],
            [self.zero_crossing_rate],
            [self.low_energy],
            [self.rms_energy],
            [self.spectral_rolloff],
            [self.brightness],
            [self.spectral_centroid],
            self.mfcc_mean,
            self.mfcc_std,
            [self.energy_variance],
            [self.drop_intensity]
        ])


class FeatureExtractor:
    """
    Extracts audio features for classification.

    Uses FeatureFactory for centralized, cached feature extraction.
    """

    def __init__(self, config: Any = None, bpm_correction_model: Any = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration object (uses global config if None)
            bpm_correction_model: Optional trained BPM correction model
        """
        if config is None:
            config = get_config()

        self.config = config
        self.bpm_correction_model = bpm_correction_model

        # Feature extraction parameters
        self.n_mfcc = config.get('features.mfcc.n_mfcc', 5)
        self.n_fft = config.get('features.spectral.n_fft', 2048)
        self.hop_length = config.get('features.spectral.hop_length', 512)
        self.brightness_threshold = config.get('features.spectral.brightness_threshold', 3000)

        # Load BPM correction model if path provided
        if isinstance(bpm_correction_model, str):
            self._load_bpm_correction_model(bpm_correction_model)

    def extract(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """
        Extract all features from audio signal.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            AudioFeatures object with all extracted features
        """
        logger.info("Extracting audio features...")

        # Create FeatureFactory - single STFT, all features cached
        factory = FeatureFactory.from_audio(
            y, sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc,
            brightness_cutoff=self.brightness_threshold
        )

        # Get RMS from factory for multiple uses
        rms_frames = factory.rms()

        # Extract features using factory and helper methods
        tempo, tempo_conf = self._extract_tempo_advanced(y, sr, factory)
        zcr = self._extract_zero_crossing_rate(y, factory)
        low_energy = self._extract_low_energy_cached(rms_frames)
        rms = self._extract_rms_energy_cached(rms_frames)

        # Use factory for spectral features
        spectral_rolloff = float(np.mean(factory.spectral_rolloff()))
        brightness = float(np.mean(factory.spectral_brightness()))
        spectral_centroid = float(np.mean(factory.spectral_centroid()))

        # MFCC from factory
        mfcc = factory.mfcc(n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        energy_var = self._extract_energy_variance_cached(rms_frames)
        drop_intensity = self._detect_drop_intensity_cached(rms_frames, sr)

        features = AudioFeatures(
            tempo=tempo,
            tempo_confidence=tempo_conf,
            zero_crossing_rate=zcr,
            low_energy=low_energy,
            rms_energy=rms,
            spectral_rolloff=spectral_rolloff,
            brightness=brightness,
            spectral_centroid=spectral_centroid,
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            energy_variance=energy_var,
            drop_intensity=drop_intensity
        )

        logger.info("Feature extraction complete")
        return features

    def _extract_tempo_advanced(
        self,
        y: np.ndarray,
        sr: int,
        factory: FeatureFactory
    ) -> Tuple[float, float]:
        """
        Extract tempo (BPM) using multiple methods and select the best result.

        Uses FeatureFactory for onset strength where possible.

        Returns the result with highest confidence.
        """
        results = []

        # Get onset strength from factory (pure numpy implementation)
        onset_env = factory.onset_strength_pure()

        # Method 1: Use factory tempo (pure numpy)
        try:
            tempo_val, confidence = factory.tempo()
            if 60 <= tempo_val <= 200:
                results.append(('factory', tempo_val, confidence))
                logger.debug(f"Factory method: {tempo_val:.1f} BPM (conf: {confidence:.2f})")
        except Exception as e:
            logger.debug(f"Factory tempo failed: {e}")

        # Method 2: Beat tracking with percussive separation
        # Uses centralized audio_stft_loader for librosa calls
        try:
            from app.common.primitives.audio_stft_loader import (
                get_harmonic_percussive, get_beat_frames, frames_to_time
            )
            y_harmonic, y_percussive = get_harmonic_percussive(y)
            tempo_beat, beats = get_beat_frames(y_percussive, sr, self.hop_length)

            # Calculate confidence from beat consistency
            if len(beats) > 2:
                beat_times = frames_to_time(beats, sr, self.hop_length)
                beat_intervals = np.diff(beat_times)

                if len(beat_intervals) > 0:
                    mean_interval = np.mean(beat_intervals)
                    std_interval = np.std(beat_intervals)
                    consistency = 1.0 - min(std_interval / mean_interval, 1.0)
                    confidence = max(consistency, 0.3)
                    results.append(('beat_track', float(tempo_beat), confidence))
                    logger.debug(f"Beat track method: {tempo_beat:.1f} BPM (conf: {confidence:.2f})")
        except ImportError:
            logger.debug("audio_stft_loader not available for HPSS")
        except Exception as e:
            logger.debug(f"Beat tracking failed: {e}")

        # Method 3: Autocorrelation of onset strength (pure numpy from factory)
        try:
            from scipy.signal import correlate
            # Compute autocorrelation
            ac_full = correlate(onset_env, onset_env, mode='full')
            ac = ac_full[len(onset_env)-1:]  # positive lags only

            # Limit search to reasonable BPM range (60-200)
            frame_rate = sr / self.hop_length
            min_lag = int(60 * frame_rate / 200)  # 200 BPM max
            max_lag = int(60 * frame_rate / 60)   # 60 BPM min

            if max_lag < len(ac):
                ac_range = ac[min_lag:max_lag]
                if len(ac_range) > 0:
                    peak_idx = np.argmax(ac_range) + min_lag
                    tempo_ac = 60 * frame_rate / peak_idx

                    # Confidence from peak strength
                    confidence = min(ac_range[np.argmax(ac_range)] / np.max(ac), 1.0)
                    results.append(('autocorr', tempo_ac, confidence))
                    logger.debug(f"Autocorr method: {tempo_ac:.1f} BPM (conf: {confidence:.2f})")
        except Exception as e:
            logger.debug(f"Autocorrelation tempo failed: {e}")

        # Select best result using scoring algorithm
        return self._select_best_tempo(results)

    def _select_best_tempo(self, results: list) -> Tuple[float, float]:
        """Select best tempo from multiple candidates."""
        if not results:
            logger.warning("All tempo extraction methods failed, using default")
            return 120.0, 0.0

        # Collect all tempo candidates (including octaves)
        all_candidates = []
        for method, tempo, conf in results:
            all_candidates.append((tempo, conf, method, 'original'))
            all_candidates.append((tempo * 2, conf * 0.8, method, '2x'))
            all_candidates.append((tempo / 2, conf * 0.8, method, '0.5x'))

        # Score each candidate
        scored_candidates = []
        for tempo, conf, method, octave in all_candidates:
            if tempo < 50 or tempo > 210:
                continue

            score = conf

            # Prefer typical electronic music BPM ranges
            if 115 <= tempo <= 145:
                score *= 1.3
            elif 90 <= tempo <= 115 or 145 <= tempo <= 165:
                score *= 1.1
            elif tempo < 80 or tempo > 175:
                score *= 0.7

            # Check agreement with other candidates
            agreements = sum(1 for other_tempo, _, _, _ in all_candidates
                           if other_tempo != tempo and abs(tempo - other_tempo) < 3)
            if agreements > 0:
                score *= (1 + agreements * 0.1)

            scored_candidates.append((tempo, score, conf, method, octave))

        if not scored_candidates:
            logger.warning("No valid tempo candidates, using default")
            return 120.0, 0.0

        # Select best scored candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_tempo, best_score, original_conf, best_method, octave_used = scored_candidates[0]

        # Calculate honest confidence
        top_candidates = scored_candidates[:3]
        if len(top_candidates) >= 2:
            top_tempos = [c[0] for c in top_candidates]
            std_dev = np.std(top_tempos)

            if std_dev < 3:
                agreement_factor = 1.0
            elif std_dev < 5:
                agreement_factor = 0.85
            else:
                agreement_factor = 0.7
        else:
            agreement_factor = 0.8

        octave_penalty = 0.85 if octave_used != 'original' else 1.0
        final_confidence = min(original_conf * agreement_factor * octave_penalty, 0.95)

        logger.debug(f"Best tempo: {best_tempo:.1f} BPM via {best_method} ({octave_used}) (conf: {final_confidence:.2f})")
        return float(best_tempo), float(final_confidence)

    def _extract_zero_crossing_rate(self, y: np.ndarray, factory: Optional[FeatureFactory] = None) -> float:
        """
        Extract mean zero crossing rate.

        Note: ZCR is computed directly from audio, not spectrogram.
        """
        # Pure numpy ZCR computation
        signs = np.sign(y)
        signs[signs == 0] = 1  # treat zeros as positive
        zero_crossings = np.abs(np.diff(signs)) / 2

        # Frame-based mean
        frame_length = self.n_fft
        hop_length = self.hop_length
        n_frames = 1 + (len(y) - frame_length) // hop_length

        if n_frames <= 0:
            return 0.0

        # Compute ZCR per frame
        zcr_frames = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(zero_crossings):
                frame_zcr = np.sum(zero_crossings[start:end]) / frame_length
                zcr_frames.append(frame_zcr)

        mean_zcr = float(np.mean(zcr_frames)) if zcr_frames else 0.0
        logger.debug(f"Zero crossing rate: {mean_zcr:.4f}")
        return mean_zcr

    def _extract_low_energy_cached(self, rms_frames: np.ndarray) -> float:
        """Extract percentage of low-energy frames using pre-computed RMS."""
        mean_energy = np.mean(rms_frames)
        low_energy_frames = np.sum(rms_frames < mean_energy)
        total_frames = len(rms_frames)
        low_energy_pct = float(low_energy_frames / total_frames)
        logger.debug(f"Low energy percentage: {low_energy_pct:.2%}")
        return low_energy_pct

    def _extract_rms_energy_cached(self, rms_frames: np.ndarray) -> float:
        """Extract mean RMS energy using pre-computed RMS."""
        mean_rms = float(np.mean(rms_frames))
        logger.debug(f"RMS energy: {mean_rms:.4f}")
        return mean_rms

    def _extract_energy_variance_cached(self, rms_frames: np.ndarray) -> float:
        """Extract energy variance using pre-computed RMS."""
        mean_rms = np.mean(rms_frames)
        if mean_rms > 0:
            cv = np.std(rms_frames) / mean_rms
            energy_var = float(cv * cv)
        else:
            energy_var = 0.0
        logger.debug(f"Energy variance: {energy_var:.4f}")
        return energy_var

    def _detect_drop_intensity_cached(self, rms_frames: np.ndarray, sr: int) -> float:
        """Detect drop intensity using pre-computed RMS."""
        # Smooth energy curve
        rms_smooth = uniform_filter1d(rms_frames, size=11)

        # Compute energy derivative (changes)
        energy_diff = np.diff(rms_smooth)

        # Find sudden increases (potential drops)
        threshold = np.percentile(np.abs(energy_diff), 90)
        drops = energy_diff > threshold

        # Calculate drop intensity
        if len(energy_diff) > 0:
            drop_count = np.sum(drops)
            max_drop = np.max(energy_diff) if len(energy_diff) > 0 else 0
            mean_energy = np.mean(rms_frames)

            if mean_energy > 0:
                intensity = min(1.0, (drop_count / len(energy_diff)) * (max_drop / mean_energy) * 10.0)
            else:
                intensity = 0.0
        else:
            intensity = 0.0

        logger.debug(f"Drop intensity: {intensity:.2f}")
        return float(intensity)

    def _load_bpm_correction_model(self, model_path: str):
        """Load BPM correction model from file."""
        try:
            from ..training.models import XGBoostBPMModel, NeuralBPMModel, EnsembleBPMModel
            from pathlib import Path

            model_path = Path(model_path)

            # Detect model type from filename
            if 'ensemble' in model_path.stem:
                self.bpm_correction_model = EnsembleBPMModel()
            elif 'neural' in model_path.stem or 'nn' in model_path.stem:
                self.bpm_correction_model = NeuralBPMModel()
            else:
                self.bpm_correction_model = XGBoostBPMModel()

            self.bpm_correction_model.load(str(model_path))
            logger.info(f"Loaded BPM correction model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load BPM correction model: {e}")
            self.bpm_correction_model = None

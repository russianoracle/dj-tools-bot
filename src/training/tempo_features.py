"""Extended tempo feature extraction for BPM correction model training."""

import librosa
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class TempoFeatures:
    """Container for tempo-specific features used in BPM correction."""

    # Basic librosa tempo detection
    detected_bpm: float
    tempo_confidence: float

    # Onset features
    onset_strength_mean: float
    onset_strength_std: float
    onset_strength_max: float
    onset_peaks_count: int
    onset_peak_spacing_mean: float
    onset_peak_spacing_std: float
    onset_autocorr_strength: float

    # Beat tracking features
    beat_count: int
    beat_consistency: float  # How regular are the beats
    beat_intervals_mean: float
    beat_intervals_std: float

    # Tempogram features
    tempogram_mean: float
    tempogram_std: float
    tempogram_peaks_count: int
    tempogram_dominant_peak: float

    # Autocorrelation features
    autocorr_peak_1: float  # Primary peak (likely correct tempo)
    autocorr_peak_2: float  # Secondary peak (might be octave)
    autocorr_peak_3: float  # Third peak
    autocorr_strength_1: float
    autocorr_strength_2: float
    autocorr_ratio_12: float  # Ratio peak1/peak2 (octave detector)

    # Spectral features
    spectral_flux_mean: float
    spectral_flux_std: float

    # Harmonic/Percussive separation
    harmonic_ratio: float
    percussive_strength: float

    # Octave candidates
    bpm_half: float
    bpm_double: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'detected_bpm': self.detected_bpm,
            'tempo_confidence': self.tempo_confidence,
            'onset_strength_mean': self.onset_strength_mean,
            'onset_strength_std': self.onset_strength_std,
            'onset_strength_max': self.onset_strength_max,
            'onset_peaks_count': float(self.onset_peaks_count),
            'onset_peak_spacing_mean': self.onset_peak_spacing_mean,
            'onset_peak_spacing_std': self.onset_peak_spacing_std,
            'onset_autocorr_strength': self.onset_autocorr_strength,
            'beat_count': float(self.beat_count),
            'beat_consistency': self.beat_consistency,
            'beat_intervals_mean': self.beat_intervals_mean,
            'beat_intervals_std': self.beat_intervals_std,
            'tempogram_mean': self.tempogram_mean,
            'tempogram_std': self.tempogram_std,
            'tempogram_peaks_count': float(self.tempogram_peaks_count),
            'tempogram_dominant_peak': self.tempogram_dominant_peak,
            'autocorr_peak_1': self.autocorr_peak_1,
            'autocorr_peak_2': self.autocorr_peak_2,
            'autocorr_peak_3': self.autocorr_peak_3,
            'autocorr_strength_1': self.autocorr_strength_1,
            'autocorr_strength_2': self.autocorr_strength_2,
            'autocorr_ratio_12': self.autocorr_ratio_12,
            'spectral_flux_mean': self.spectral_flux_mean,
            'spectral_flux_std': self.spectral_flux_std,
            'harmonic_ratio': self.harmonic_ratio,
            'percussive_strength': self.percussive_strength,
            'bpm_half': self.bpm_half,
            'bpm_double': self.bpm_double,
        }


class TempoFeatureExtractor:
    """Extracts extended features for BPM correction model training."""

    def __init__(self, sr: int = 22050):
        """
        Initialize tempo feature extractor.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr
        self.hop_length = 512

    def extract(self, y: np.ndarray, sr: int = None) -> TempoFeatures:
        """
        Extract all tempo-related features from audio signal.

        Args:
            y: Audio time series
            sr: Sample rate (uses self.sr if None)

        Returns:
            TempoFeatures object with all extracted features
        """
        if sr is None:
            sr = self.sr

        logger.debug("Extracting tempo features for training...")

        # Basic tempo detection
        detected_bpm, tempo_conf = self._extract_basic_tempo(y, sr)

        # Onset features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_features = self._extract_onset_features(onset_env, sr)

        # Beat tracking features
        beat_features = self._extract_beat_features(y, sr)

        # Tempogram features
        tempogram_features = self._extract_tempogram_features(onset_env, sr)

        # Autocorrelation features
        autocorr_features = self._extract_autocorr_features(onset_env, sr)

        # Spectral features
        spectral_features = self._extract_spectral_features(y, sr)

        # Harmonic/Percussive features
        hp_features = self._extract_harmonic_percussive_features(y)

        # Combine all features
        features = TempoFeatures(
            detected_bpm=detected_bpm,
            tempo_confidence=tempo_conf,
            **onset_features,
            **beat_features,
            **tempogram_features,
            **autocorr_features,
            **spectral_features,
            **hp_features,
            bpm_half=detected_bpm / 2,
            bpm_double=detected_bpm * 2
        )

        logger.debug(f"Extracted {len(features.to_dict())} tempo features")
        return features

    def _extract_basic_tempo(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Extract basic tempo using librosa (simplified version of current implementation)."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)

            if len(tempo) > 0:
                hist, bins = np.histogram(tempo, bins=50, range=(60, 200))
                dominant_idx = np.argmax(hist)
                tempo_val = (bins[dominant_idx] + bins[dominant_idx + 1]) / 2
                confidence = hist[dominant_idx] / len(tempo)
                return float(tempo_val), float(confidence)
        except Exception as e:
            logger.warning(f"Basic tempo extraction failed: {e}")

        return 120.0, 0.0

    def _extract_onset_features(self, onset_env: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract onset envelope features."""
        features = {}

        # Basic statistics
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_std'] = float(np.std(onset_env))
        features['onset_strength_max'] = float(np.max(onset_env))

        # Peak analysis
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(onset_env, height=np.mean(onset_env))
        features['onset_peaks_count'] = len(peaks)

        if len(peaks) > 1:
            peak_spacing = np.diff(peaks)
            features['onset_peak_spacing_mean'] = float(np.mean(peak_spacing))
            features['onset_peak_spacing_std'] = float(np.std(peak_spacing))
        else:
            features['onset_peak_spacing_mean'] = 0.0
            features['onset_peak_spacing_std'] = 0.0

        # Autocorrelation of onset strength
        onset_autocorr = librosa.autocorrelate(onset_env)
        if len(onset_autocorr) > 1:
            features['onset_autocorr_strength'] = float(np.max(onset_autocorr[1:]) / onset_autocorr[0])
        else:
            features['onset_autocorr_strength'] = 0.0

        return features

    def _extract_beat_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract beat tracking features."""
        features = {}

        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['beat_count'] = len(beats)

            if len(beats) > 2:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)

                features['beat_intervals_mean'] = float(np.mean(beat_intervals))
                features['beat_intervals_std'] = float(np.std(beat_intervals))

                # Beat consistency (inverse of coefficient of variation)
                if features['beat_intervals_mean'] > 0:
                    cv = features['beat_intervals_std'] / features['beat_intervals_mean']
                    features['beat_consistency'] = float(1.0 - min(cv, 1.0))
                else:
                    features['beat_consistency'] = 0.0
            else:
                features['beat_intervals_mean'] = 0.0
                features['beat_intervals_std'] = 0.0
                features['beat_consistency'] = 0.0

        except Exception as e:
            logger.warning(f"Beat feature extraction failed: {e}")
            features['beat_count'] = 0
            features['beat_intervals_mean'] = 0.0
            features['beat_intervals_std'] = 0.0
            features['beat_consistency'] = 0.0

        return features

    def _extract_tempogram_features(self, onset_env: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract tempogram features."""
        features = {}

        try:
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)

            features['tempogram_mean'] = float(np.mean(tempogram))
            features['tempogram_std'] = float(np.std(tempogram))

            # Find peaks in averaged tempogram
            avg_tempogram = np.mean(tempogram, axis=1)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(avg_tempogram, height=np.mean(avg_tempogram))

            features['tempogram_peaks_count'] = len(peaks)

            if len(peaks) > 0:
                dominant_peak_idx = peaks[np.argmax(properties['peak_heights'])]
                # Convert to BPM (tempogram is in tempo bins)
                tempo_bins = librosa.tempo_frequencies(len(avg_tempogram), hop_length=self.hop_length, sr=sr)
                features['tempogram_dominant_peak'] = float(tempo_bins[dominant_peak_idx])
            else:
                features['tempogram_dominant_peak'] = 0.0

        except Exception as e:
            logger.warning(f"Tempogram feature extraction failed: {e}")
            features['tempogram_mean'] = 0.0
            features['tempogram_std'] = 0.0
            features['tempogram_peaks_count'] = 0
            features['tempogram_dominant_peak'] = 0.0

        return features

    def _extract_autocorr_features(self, onset_env: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract autocorrelation features for octave error detection."""
        features = {}

        try:
            # Compute autocorrelation
            ac = librosa.autocorrelate(onset_env)

            # Frame rate
            frame_rate = sr / self.hop_length

            # Search range for reasonable BPM (60-200)
            min_lag = int(60 * frame_rate / 200)  # 200 BPM max
            max_lag = int(60 * frame_rate / 60)   # 60 BPM min

            if max_lag < len(ac):
                ac_range = ac[min_lag:max_lag]

                # Find top 3 peaks
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(ac_range, height=np.mean(ac_range))

                if len(peaks) > 0:
                    # Sort by height
                    sorted_indices = np.argsort(properties['peak_heights'])[::-1]
                    top_peaks = peaks[sorted_indices[:3]]
                    top_heights = properties['peak_heights'][sorted_indices[:3]]

                    # Convert to BPM
                    for i in range(min(3, len(top_peaks))):
                        lag = top_peaks[i] + min_lag
                        bpm = 60 * frame_rate / lag
                        features[f'autocorr_peak_{i+1}'] = float(bpm)
                        features[f'autocorr_strength_{i+1}'] = float(top_heights[i] / np.max(ac))

                    # Fill remaining with zeros
                    for i in range(len(top_peaks), 3):
                        features[f'autocorr_peak_{i+1}'] = 0.0
                        features[f'autocorr_strength_{i+1}'] = 0.0

                    # Ratio of first two peaks (octave detector)
                    if len(top_peaks) >= 2 and features['autocorr_peak_2'] > 0:
                        features['autocorr_ratio_12'] = float(features['autocorr_peak_1'] / features['autocorr_peak_2'])
                    else:
                        features['autocorr_ratio_12'] = 1.0
                else:
                    # No peaks found
                    for i in range(1, 4):
                        features[f'autocorr_peak_{i}'] = 0.0
                        features[f'autocorr_strength_{i}'] = 0.0
                    features['autocorr_ratio_12'] = 1.0
            else:
                # Signal too short
                for i in range(1, 4):
                    features[f'autocorr_peak_{i}'] = 0.0
                    features[f'autocorr_strength_{i}'] = 0.0
                features['autocorr_ratio_12'] = 1.0

        except Exception as e:
            logger.warning(f"Autocorr feature extraction failed: {e}")
            for i in range(1, 4):
                features[f'autocorr_peak_{i}'] = 0.0
                features[f'autocorr_strength_{i}'] = 0.0
            features['autocorr_ratio_12'] = 1.0

        return features

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral flux features."""
        features = {}

        try:
            # Spectral flux
            spec = np.abs(librosa.stft(y, hop_length=self.hop_length))
            flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec, ref=np.max), sr=sr)

            features['spectral_flux_mean'] = float(np.mean(flux))
            features['spectral_flux_std'] = float(np.std(flux))
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            features['spectral_flux_mean'] = 0.0
            features['spectral_flux_std'] = 0.0

        return features

    def _extract_harmonic_percussive_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract harmonic/percussive separation features."""
        features = {}

        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Ratio of harmonic to total energy
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2)

            if total_energy > 0:
                features['harmonic_ratio'] = float(harmonic_energy / total_energy)
            else:
                features['harmonic_ratio'] = 0.0

            # Percussive strength
            features['percussive_strength'] = float(np.mean(np.abs(y_percussive)))

        except Exception as e:
            logger.warning(f"H/P separation feature extraction failed: {e}")
            features['harmonic_ratio'] = 0.0
            features['percussive_strength'] = 0.0

        return features

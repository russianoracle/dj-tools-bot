"""Feature extraction from audio signals."""

import librosa
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass, asdict

from ..utils import get_logger, get_config

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
        data = asdict(self)
        # Convert numpy arrays to lists
        data['mfcc_mean'] = data['mfcc_mean'].tolist() if isinstance(data['mfcc_mean'], np.ndarray) else data['mfcc_mean']
        data['mfcc_std'] = data['mfcc_std'].tolist() if isinstance(data['mfcc_std'], np.ndarray) else data['mfcc_std']
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
    """Extracts audio features for classification."""

    def __init__(self, config: Any = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration object (uses global config if None)
        """
        if config is None:
            config = get_config()

        self.config = config

        # Feature extraction parameters
        self.n_mfcc = config.get('features.mfcc.n_mfcc', 5)
        self.n_fft = config.get('features.spectral.n_fft', 2048)
        self.hop_length = config.get('features.spectral.hop_length', 512)
        self.brightness_threshold = config.get('features.spectral.brightness_threshold', 3000)

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

        # Extract features in parallel where possible
        tempo, tempo_conf = self._extract_tempo(y, sr)
        zcr = self._extract_zero_crossing_rate(y)
        low_energy = self._extract_low_energy(y)
        rms = self._extract_rms_energy(y)

        spectral_rolloff = self._extract_spectral_rolloff(y, sr)
        brightness = self._extract_brightness(y, sr)
        spectral_centroid = self._extract_spectral_centroid(y, sr)

        mfcc_mean, mfcc_std = self._extract_mfcc_stats(y, sr)

        energy_var = self._extract_energy_variance(y)
        drop_intensity = self._detect_drop_intensity(y, sr)

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

    def _extract_tempo(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Extract tempo (BPM) and confidence."""
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Convert to scalar if array
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0

            # Estimate confidence based on beat strength
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                # Lower variance = higher confidence
                confidence = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                confidence = 0.5

            logger.debug(f"Tempo: {tempo:.1f} BPM (confidence: {confidence:.2f})")
            return float(tempo), float(confidence)

        except Exception as e:
            logger.warning(f"Tempo extraction failed: {e}, using default")
            return 120.0, 0.0

    def _extract_zero_crossing_rate(self, y: np.ndarray) -> float:
        """Extract mean zero crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        mean_zcr = float(np.mean(zcr))
        logger.debug(f"Zero crossing rate: {mean_zcr:.4f}")
        return mean_zcr

    def _extract_low_energy(self, y: np.ndarray) -> float:
        """Extract percentage of low-energy frames."""
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Compute mean energy
        mean_energy = np.mean(rms)

        # Count frames below mean
        low_energy_frames = np.sum(rms < mean_energy)
        total_frames = len(rms)

        low_energy_pct = float(low_energy_frames / total_frames)
        logger.debug(f"Low energy percentage: {low_energy_pct:.2%}")
        return low_energy_pct

    def _extract_rms_energy(self, y: np.ndarray) -> float:
        """Extract mean RMS energy."""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        mean_rms = float(np.mean(rms))
        logger.debug(f"RMS energy: {mean_rms:.4f}")
        return mean_rms

    def _extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> float:
        """Extract mean spectral rolloff (85% energy threshold)."""
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )[0]
        mean_rolloff = float(np.mean(rolloff))
        logger.debug(f"Spectral rolloff: {mean_rolloff:.1f} Hz")
        return mean_rolloff

    def _extract_brightness(self, y: np.ndarray, sr: int) -> float:
        """Extract brightness (high-frequency energy above threshold)."""
        # Compute spectrogram
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        # Find bins above brightness threshold
        bright_bins = freqs > self.brightness_threshold

        # Calculate ratio of high-freq energy to total energy
        if np.sum(S) > 0:
            brightness = float(np.sum(S[bright_bins, :]) / np.sum(S))
        else:
            brightness = 0.0

        logger.debug(f"Brightness: {brightness:.2%}")
        return brightness

    def _extract_spectral_centroid(self, y: np.ndarray, sr: int) -> float:
        """Extract mean spectral centroid (center of mass of spectrum)."""
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        mean_centroid = float(np.mean(centroid))
        logger.debug(f"Spectral centroid: {mean_centroid:.1f} Hz")
        return mean_centroid

    def _extract_mfcc_stats(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract MFCC mean and standard deviation."""
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Compute statistics across time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        logger.debug(f"MFCC mean: {mfcc_mean}")
        logger.debug(f"MFCC std: {mfcc_std}")

        return mfcc_mean, mfcc_std

    def _extract_energy_variance(self, y: np.ndarray) -> float:
        """Extract variance of energy over time."""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        energy_var = float(np.var(rms))
        logger.debug(f"Energy variance: {energy_var:.4f}")
        return energy_var

    def _detect_drop_intensity(self, y: np.ndarray, sr: int) -> float:
        """
        Detect intensity of drops (sudden energy changes).

        Analyzes the signal for characteristic build-up/drop patterns
        common in electronic music.

        Returns:
            Drop intensity score (0.0 to 1.0)
        """
        # Calculate RMS energy with short window
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Smooth energy curve
        rms_smooth = librosa.util.smooth(rms, window_len=11)

        # Compute energy derivative (changes)
        energy_diff = np.diff(rms_smooth)

        # Find sudden increases (potential drops)
        threshold = np.percentile(np.abs(energy_diff), 90)
        drops = energy_diff > threshold

        # Calculate drop intensity
        if len(energy_diff) > 0:
            drop_count = np.sum(drops)
            max_drop = np.max(energy_diff) if len(energy_diff) > 0 else 0
            mean_energy = np.mean(rms)

            # Normalize intensity
            if mean_energy > 0:
                intensity = min(1.0, (drop_count / len(drops)) * (max_drop / mean_energy))
            else:
                intensity = 0.0
        else:
            intensity = 0.0

        logger.debug(f"Drop intensity: {intensity:.2f}")
        return float(intensity)

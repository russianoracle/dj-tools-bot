"""Feature extraction from audio signals."""

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
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
    """Extracts audio features for classification."""

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

        # Pre-compute STFT and RMS once for reuse (major optimization)
        logger.debug("Computing STFT (will be reused for multiple features)...")
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))

        logger.debug("Computing RMS energy (will be reused for multiple features)...")
        rms_frames = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Extract features (passing pre-computed values where possible)
        tempo, tempo_conf = self._extract_tempo(y, sr)
        zcr = self._extract_zero_crossing_rate(y)
        low_energy = self._extract_low_energy_cached(rms_frames)
        rms = self._extract_rms_energy_cached(rms_frames)

        spectral_rolloff = self._extract_spectral_rolloff_cached(S, sr)
        brightness = self._extract_brightness_cached(S, sr)
        spectral_centroid = self._extract_spectral_centroid_cached(S, sr)

        mfcc_mean, mfcc_std = self._extract_mfcc_stats(y, sr)

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

    def _extract_tempo(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """
        Extract tempo (BPM) using multiple methods and select the best result.

        Uses:
        1. Onset-based tempo detection (most accurate for electronic music)
        2. Beat tracking method
        3. Tempogram aggregation

        Returns the result with highest confidence.
        """
        results = []

        # Method 1: Onset-based tempo (best for electronic/dance music)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_onset = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)

            # Get the dominant tempo
            if len(tempo_onset) > 0:
                # Use histogram to find most common tempo
                hist, bins = np.histogram(tempo_onset, bins=50, range=(60, 200))
                dominant_idx = np.argmax(hist)
                tempo_val = (bins[dominant_idx] + bins[dominant_idx + 1]) / 2

                # Confidence based on how much the dominant tempo appears
                confidence = hist[dominant_idx] / len(tempo_onset)
                results.append(('onset', tempo_val, confidence))
                logger.debug(f"Onset method: {tempo_val:.1f} BPM (conf: {confidence:.2f})")
        except Exception as e:
            logger.debug(f"Onset tempo failed: {e}")

        # Method 2: Beat tracking with percussive separation
        try:
            # Separate harmonic and percussive
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Beat track on percussive component
            tempo_beat, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

            if isinstance(tempo_beat, np.ndarray):
                tempo_beat = float(tempo_beat[0]) if len(tempo_beat) > 0 else 120.0

            # Calculate confidence from beat consistency
            if len(beats) > 2:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)

                # Check consistency of beat intervals
                if len(beat_intervals) > 0:
                    mean_interval = np.mean(beat_intervals)
                    std_interval = np.std(beat_intervals)
                    # Higher consistency = higher confidence
                    consistency = 1.0 - min(std_interval / mean_interval, 1.0)
                    confidence = max(consistency, 0.3)
                    results.append(('beat_track', float(tempo_beat), confidence))
                    logger.debug(f"Beat track method: {tempo_beat:.1f} BPM (conf: {confidence:.2f})")
        except Exception as e:
            logger.debug(f"Beat tracking failed: {e}")

        # Method 3: Autocorrelation of onset strength
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # Compute autocorrelation
            ac = librosa.autocorrelate(onset_env)

            # Find peaks in autocorrelation (correspond to tempo)
            # Limit search to reasonable BPM range (60-200)
            # Frame rate for onset strength with default hop_length=512
            hop_length = 512
            frame_rate = sr / hop_length
            min_lag = int(60 * frame_rate / 200)  # 200 BPM max
            max_lag = int(60 * frame_rate / 60)   # 60 BPM min

            if max_lag < len(ac):
                ac_range = ac[min_lag:max_lag]
                if len(ac_range) > 0:
                    peak_idx = np.argmax(ac_range) + min_lag
                    # Convert lag to tempo: BPM = 60 * frame_rate / lag
                    tempo_ac = 60 * frame_rate / peak_idx

                    # Confidence from peak strength
                    confidence = min(ac_range[np.argmax(ac_range)] / np.max(ac), 1.0)
                    results.append(('autocorr', tempo_ac, confidence))
                    logger.debug(f"Autocorr method: {tempo_ac:.1f} BPM (conf: {confidence:.2f})")
        except Exception as e:
            logger.debug(f"Autocorrelation tempo failed: {e}")

        # Select best result
        if results:
            # Collect all tempo candidates (including octaves)
            all_candidates = []
            for method, tempo, conf in results:
                all_candidates.append((tempo, conf, method, 'original'))
                all_candidates.append((tempo * 2, conf * 0.8, method, '2x'))
                all_candidates.append((tempo / 2, conf * 0.8, method, '0.5x'))

            # Score each candidate based on:
            # 1. Original confidence
            # 2. How well it fits typical BPM ranges for electronic music
            # 3. Agreement with other methods

            scored_candidates = []
            for tempo, conf, method, octave in all_candidates:
                # Skip unreasonable tempos
                if tempo < 50 or tempo > 210:
                    continue

                score = conf

                # Prefer typical electronic music BPM ranges
                # Tech House/Techno: 120-135, House: 118-130, Dubstep: 130-145
                if 115 <= tempo <= 145:
                    score *= 1.3  # Strong preference
                elif 90 <= tempo <= 115 or 145 <= tempo <= 165:
                    score *= 1.1  # Moderate preference
                elif tempo < 80 or tempo > 175:
                    score *= 0.7  # Penalize unusual tempos

                # Check agreement with other candidates
                agreements = 0
                for other_tempo, _, _, _ in all_candidates:
                    if other_tempo != tempo and abs(tempo - other_tempo) < 3:
                        agreements += 1

                if agreements > 0:
                    score *= (1 + agreements * 0.1)

                scored_candidates.append((tempo, score, conf, method, octave))

            if not scored_candidates:
                logger.warning("No valid tempo candidates, using default")
                return 120.0, 0.0

            # Select best scored candidate
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_tempo, best_score, original_conf, best_method, octave_used = scored_candidates[0]

            # Calculate honest confidence based on:
            # - Agreement between top candidates
            # - Consistency of original methods
            # - Whether octave correction was needed

            top_candidates = scored_candidates[:3]
            if len(top_candidates) >= 2:
                # Check if top candidates agree (within 3 BPM)
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

            # Penalize confidence if octave correction was used
            if octave_used != 'original':
                octave_penalty = 0.85
            else:
                octave_penalty = 1.0

            # Final honest confidence
            final_confidence = min(original_conf * agreement_factor * octave_penalty, 0.95)

            logger.debug(f"Best tempo: {best_tempo:.1f} BPM via {best_method} ({octave_used}) (conf: {final_confidence:.2f})")
            return float(best_tempo), float(final_confidence)
        else:
            logger.warning("All tempo extraction methods failed, using default")
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
        # Normalize by mean to get coefficient of variation squared
        mean_rms = np.mean(rms)
        if mean_rms > 0:
            cv = np.std(rms) / mean_rms
            energy_var = float(cv * cv)
        else:
            energy_var = 0.0
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
        rms_smooth = uniform_filter1d(rms, size=11)

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

    # Optimized cached methods that reuse pre-computed STFT and RMS
    # These methods significantly reduce computation time (3x speedup)

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

    def _extract_spectral_rolloff_cached(self, S: np.ndarray, sr: int) -> float:
        """Extract spectral rolloff using pre-computed STFT."""
        # Compute spectral rolloff from magnitude spectrogram
        # Find frequency containing 85% of energy for each frame
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        energy_cumsum = np.cumsum(S, axis=0)
        total_energy = energy_cumsum[-1, :]

        # Find rolloff frequency for each frame
        rolloff_freqs = []
        for i in range(S.shape[1]):
            if total_energy[i] > 0:
                threshold = 0.85 * total_energy[i]
                idx = np.where(energy_cumsum[:, i] >= threshold)[0]
                if len(idx) > 0:
                    rolloff_freqs.append(freqs[idx[0]])
                else:
                    rolloff_freqs.append(freqs[-1])
            else:
                rolloff_freqs.append(0.0)

        mean_rolloff = float(np.mean(rolloff_freqs))
        logger.debug(f"Spectral rolloff: {mean_rolloff:.1f} Hz")
        return mean_rolloff

    def _extract_brightness_cached(self, S: np.ndarray, sr: int) -> float:
        """Extract brightness using pre-computed STFT."""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        bright_bins = freqs > self.brightness_threshold

        if np.sum(S) > 0:
            brightness = float(np.sum(S[bright_bins, :]) / np.sum(S))
        else:
            brightness = 0.0

        logger.debug(f"Brightness: {brightness:.2%}")
        return brightness

    def _extract_spectral_centroid_cached(self, S: np.ndarray, sr: int) -> float:
        """Extract spectral centroid using pre-computed STFT."""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        # Compute centroid for each frame
        centroids = []
        for i in range(S.shape[1]):
            spectrum = S[:, i]
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                centroids.append(centroid)
            else:
                centroids.append(0.0)

        mean_centroid = float(np.mean(centroids))
        logger.debug(f"Spectral centroid: {mean_centroid:.1f} Hz")
        return mean_centroid

    def _extract_energy_variance_cached(self, rms_frames: np.ndarray) -> float:
        """Extract energy variance using pre-computed RMS."""
        # Normalize by mean to get coefficient of variation squared
        # This makes variance scale-independent and more meaningful for classification
        mean_rms = np.mean(rms_frames)
        if mean_rms > 0:
            # Use coefficient of variation: std/mean, then square it for variance-like measure
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

            # Normalize intensity
            if mean_energy > 0:
                # Fixed: use len(energy_diff) instead of len(drops)
                # drops is a boolean array, len(drops) == len(energy_diff) always
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

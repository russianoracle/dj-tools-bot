"""
Unified Feature Extraction for DEAM-Compatible Training

Extracts 21 common features from both audio files (librosa) and DEAM CSVs (openSMILE).
This ensures feature compatibility between User and DEAM datasets.
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# Frame-level features (per 0.5s frame, DEAM-compatible)
# Maximum features that can be computed per-frame (38 features)
FRAME_FEATURES = [
    'frameTime',
    # Energy (4)
    'rms_energy',
    'rms_energy_delta',
    'rms_energy_delta2',  # Second derivative (acceleration)
    'low_energy_flag',    # 1 if below mean, 0 otherwise
    # Zero Crossing (2)
    'zero_crossing_rate',
    'zcr_delta',
    # Spectral basic (6)
    'spectral_centroid',
    'spectral_centroid_delta',
    'spectral_rolloff',
    'spectral_rolloff_delta',
    'brightness',
    'brightness_delta',
    # Spectral advanced (4)
    'spectral_flux',
    'spectral_flatness',
    'spectral_contrast',  # Mean across bands per frame
    'spectral_bandwidth',
    # MFCCs (10)
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
    'mfcc_1_delta', 'mfcc_2_delta', 'mfcc_3_delta', 'mfcc_4_delta', 'mfcc_5_delta',
    # Onset/Rhythm (3)
    'onset_strength',
    'onset_strength_delta',
    'beat_sync',  # 1 if near beat, 0 otherwise
    # Harmonic (3)
    'chroma_energy',  # Sum of chroma per frame
    'harmonic_ratio',
    'percussive_ratio',
    # Pitch (2)
    'pitch',
    'pitch_confidence',
    # === Drop detection (4) - critical for zone classification ===
    'energy_buildup_score',  # Rising energy slope in window
    'drop_candidate',        # Binary: potential drop moment
    'energy_valley',         # Binary: local energy minimum
    'energy_peak',           # Binary: local energy maximum
]  # Total: 38 frame features

# 47 common features for unified training (expanded from ZoneFeatures)
COMMON_FEATURES = [
    # === Basic temporal (4) ===
    'tempo',
    'zero_crossing_rate',
    'low_energy',
    'rms_energy',

    # === Spectral (7) ===
    'spectral_rolloff',
    'brightness',
    'spectral_centroid',
    'spectral_flux',
    'spectral_flatness',
    'spectral_contrast_mean',  # Mean across all 7 bands
    'spectral_contrast_std',   # Std across all 7 bands

    # === MFCCs (10) ===
    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
    'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std',

    # === Advanced temporal (4) ===
    'onset_strength_mean',
    'onset_strength_std',
    'beat_strength',
    'tempo_stability',

    # === Harmonic-Percussive (2) ===
    'harmonic_ratio',
    'percussive_ratio',

    # === Energy dynamics (3) ===
    'energy_variance',
    'drop_strength',
    'peak_energy_ratio',

    # === Build-up detection (3) - critical for GREEN zone ===
    'energy_slope',           # -1 falling, +1 rising
    'energy_buildup_ratio',   # Ratio of rising segments (0-1)
    'onset_acceleration',     # Acceleration of onset strength

    # === Drop detection (5) - critical for PURPLE zone ===
    'drop_frequency',         # Drops per minute
    'drop_contrast_mean',
    'drop_contrast_max',
    'drop_count',
    'drop_intensity',

    # === Euphoria indicators (2) - for PURPLE zone ===
    'rhythmic_regularity',    # Rhythm stability (0-1)
    'harmonic_complexity',    # Harmonic richness (entropy)

    # === Climax structure (3) - for PURPLE zone ===
    'has_climax',             # Binary: clear climax present
    'climax_position',        # Position in track (0-1)
    'dynamic_range',          # Dynamic range in dB

    # === Rhythm (1) ===
    'onset_rate',

    # === Emotion (2) ===
    'arousal',
    'valence',

    # === Pitch (1) ===
    'pitch_mean',
]


@dataclass
class UnifiedFeatures:
    """47 unified features compatible with both DEAM and User datasets."""

    # Basic temporal (4)
    tempo: float
    zero_crossing_rate: float
    low_energy: float
    rms_energy: float

    # Spectral (7)
    spectral_rolloff: float
    brightness: float
    spectral_centroid: float
    spectral_flux: float
    spectral_flatness: float
    spectral_contrast_mean: float
    spectral_contrast_std: float

    # MFCCs (10)
    mfcc_1_mean: float
    mfcc_2_mean: float
    mfcc_3_mean: float
    mfcc_4_mean: float
    mfcc_5_mean: float
    mfcc_1_std: float
    mfcc_2_std: float
    mfcc_3_std: float
    mfcc_4_std: float
    mfcc_5_std: float

    # Advanced temporal (4)
    onset_strength_mean: float
    onset_strength_std: float
    beat_strength: float
    tempo_stability: float

    # Harmonic-Percussive (2)
    harmonic_ratio: float
    percussive_ratio: float

    # Energy dynamics (3)
    energy_variance: float
    drop_strength: float
    peak_energy_ratio: float

    # Build-up detection (3)
    energy_slope: float
    energy_buildup_ratio: float
    onset_acceleration: float

    # Drop detection (5)
    drop_frequency: float
    drop_contrast_mean: float
    drop_contrast_max: float
    drop_count: float
    drop_intensity: float

    # Euphoria indicators (2)
    rhythmic_regularity: float
    harmonic_complexity: float

    # Climax structure (3)
    has_climax: float
    climax_position: float
    dynamic_range: float

    # Rhythm (1)
    onset_rate: float

    # Emotion (2)
    arousal: float
    valence: float

    # Pitch (1)
    pitch_mean: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {f: getattr(self, f) for f in COMMON_FEATURES}

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector."""
        return np.array([getattr(self, f) for f in COMMON_FEATURES])


class UnifiedFeatureExtractor:
    """
    Extracts 21 common features from audio files or DEAM CSV files.

    Supports two sources:
    - 'audio': Extract from audio files using librosa
    - 'deam': Extract from DEAM openSMILE CSV files
    """

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize extractor.

        Args:
            sample_rate: Sample rate for audio loading (default 22050)
        """
        self.sample_rate = sample_rate

    def extract_from_audio(self, audio_path: str) -> Optional[Dict[str, float]]:
        """
        Extract 47 unified features from audio file using librosa.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with 47 features or None if extraction failed
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            features = {}

            # === Basic temporal (4) ===

            # 1. Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)

            # 2. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))

            # 3-4. RMS Energy and Low Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy'] = float(np.mean(rms))
            features['low_energy'] = float(np.sum(rms < np.mean(rms)) / len(rms))

            # === Spectral (7) ===

            # 5. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.9)[0]
            features['spectral_rolloff'] = float(np.mean(rolloff))

            # 6. Brightness (high frequency energy ratio)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_energy = np.sum(S[freqs > 3000, :], axis=0)
            total_energy = np.sum(S, axis=0) + 1e-10
            features['brightness'] = float(np.mean(high_freq_energy / total_energy))

            # 7. Spectral Centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(centroid))

            # 8. Spectral Flux (rate of spectral change)
            spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            features['spectral_flux'] = float(np.mean(spectral_flux))

            # 9. Spectral Flatness (tonal vs noise)
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            features['spectral_flatness'] = float(np.mean(flatness))

            # 10-11. Spectral Contrast (safe band count for various sample rates)
            n_bands = 6 if sr <= 24000 else 7
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=200.0)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            features['spectral_contrast_std'] = float(np.std(contrast))

            # === MFCCs (10) ===
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)  # 0-5, we use 1-5
            for i in range(1, 6):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

            # === Advanced temporal (4) ===

            # Onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_env))
            features['onset_strength_std'] = float(np.std(onset_env))

            # Beat strength
            if len(beats) > 0:
                beat_frames = librosa.frames_to_samples(beats)
                beat_energies = [rms[min(b // 512, len(rms)-1)] for b in beat_frames]
                features['beat_strength'] = float(np.mean(beat_energies))
            else:
                features['beat_strength'] = 0.0

            # Tempo stability (tempogram variance)
            tempogram = librosa.feature.tempogram(y=y, sr=sr)
            features['tempo_stability'] = float(1.0 / (np.std(np.argmax(tempogram, axis=0)) + 1))

            # === Harmonic-Percussive (2) ===
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_hp = harmonic_energy + percussive_energy + 1e-10
            features['harmonic_ratio'] = float(harmonic_energy / total_hp)
            features['percussive_ratio'] = float(percussive_energy / total_hp)

            # === Energy dynamics (3) ===
            features['energy_variance'] = float(np.std(rms))

            # Drop Strength (90th percentile of energy derivative)
            rms_delta = np.diff(rms)
            if len(rms_delta) > 0:
                features['drop_strength'] = float(np.percentile(np.abs(rms_delta), 90))
            else:
                features['drop_strength'] = 0.0

            # Peak energy ratio
            if len(rms) > 0 and np.mean(rms) > 0:
                features['peak_energy_ratio'] = float(np.max(rms) / np.mean(rms))
            else:
                features['peak_energy_ratio'] = 1.0

            # === Build-up detection (3) ===

            # Energy slope (linear regression of energy over time)
            if len(rms) > 1:
                time_normalized = np.arange(len(rms)) / len(rms)
                slope, _ = np.polyfit(time_normalized, rms, 1)
                features['energy_slope'] = float(np.clip(slope / (np.std(rms) + 1e-6), -1, 1))
            else:
                features['energy_slope'] = 0.0

            # Energy buildup ratio (ratio of rising segments)
            energy_diff = np.diff(rms)
            if len(energy_diff) > 0:
                rising_segments = np.sum(energy_diff > 0)
                features['energy_buildup_ratio'] = float(rising_segments / len(energy_diff))
            else:
                features['energy_buildup_ratio'] = 0.5

            # Onset acceleration (second derivative of onset strength)
            if len(onset_env) > 2:
                onset_velocity = np.diff(onset_env)
                onset_accel = np.diff(onset_velocity)
                features['onset_acceleration'] = float(np.mean(np.abs(onset_accel)))
            else:
                features['onset_acceleration'] = 0.0

            # === Drop detection (5) ===

            # Drop frequency (sharp energy drops per minute)
            if len(rms_delta) > 0:
                drop_threshold = -np.percentile(np.abs(rms_delta), 75)
                drops = np.sum(rms_delta < drop_threshold)
                duration = len(y) / sr / 60.0
                features['drop_frequency'] = float(drops / max(duration, 0.1))
            else:
                features['drop_frequency'] = 0.0

            # Full drop features
            drop_features = self._extract_drop_features(rms)
            features['drop_contrast_mean'] = drop_features['drop_contrast_mean']
            features['drop_contrast_max'] = drop_features.get('drop_contrast_max', 0.0)
            features['drop_count'] = drop_features['drop_count']
            features['drop_intensity'] = drop_features.get('drop_intensity', 0.0)

            # === Euphoria indicators (2) ===

            # Rhythmic regularity (from beat intervals)
            if len(beats) > 2:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                if np.mean(beat_intervals) > 0:
                    cv = np.std(beat_intervals) / np.mean(beat_intervals)
                    features['rhythmic_regularity'] = float(1.0 / (1.0 + cv))
                else:
                    features['rhythmic_regularity'] = 0.0
            else:
                features['rhythmic_regularity'] = 0.0

            # Harmonic complexity (spectral entropy)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
            entropy_per_frame = -np.sum(chroma_norm * np.log(chroma_norm + 1e-10), axis=0)
            features['harmonic_complexity'] = float(np.mean(entropy_per_frame))

            # === Climax structure (3) ===

            if len(rms) > 0:
                max_energy = np.max(rms)
                mean_energy = np.mean(rms)
                std_energy = np.std(rms)

                # Has climax (peak > 2 std above mean)
                features['has_climax'] = float(1.0 if (max_energy - mean_energy) > 2 * std_energy else 0.0)

                # Climax position (where in track is peak energy)
                max_idx = np.argmax(rms)
                features['climax_position'] = float(max_idx / len(rms))

                # Dynamic range (in dB)
                if np.max(rms) > 0 and np.min(rms) > 0:
                    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
                    features['dynamic_range'] = float(np.max(rms_db) - np.min(rms_db))
                else:
                    features['dynamic_range'] = 0.0
            else:
                features['has_climax'] = 0.0
                features['climax_position'] = 0.5
                features['dynamic_range'] = 0.0

            # === Rhythm (1) ===

            # Onset Rate (onsets per second)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            duration = len(y) / sr
            features['onset_rate'] = float(len(onsets) / duration) if duration > 0 else 0.0

            # === Emotion (2) ===

            # Arousal estimation
            arousal = self._estimate_arousal(
                tempo=features['tempo'],
                rms_energy=features['rms_energy'],
                energy_variance=features['energy_variance'],
                drop_count=features['drop_count']
            )
            features['arousal'] = arousal

            # Valence estimation
            valence = self._estimate_valence(
                brightness=features['brightness'],
                spectral_centroid=features['spectral_centroid'],
                tempo=features['tempo']
            )
            features['valence'] = valence

            # === Pitch (1) ===

            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                f0_voiced = f0[~np.isnan(f0)]
                features['pitch_mean'] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            except Exception:
                features['pitch_mean'] = 0.0

            return features

        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None

    def extract_from_deam_csv(
        self,
        csv_path: str,
        audio_path: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """
        Extract 27 unified features from DEAM openSMILE CSV file.

        Args:
            csv_path: Path to DEAM feature CSV file
            audio_path: Optional path to audio for tempo extraction

        Returns:
            Dictionary with 27 features or None if extraction failed
        """
        try:
            # Load DEAM CSV (semicolon separated)
            df = pd.read_csv(csv_path, sep=';')

            features = {}

            # 1. Tempo - need to extract from audio or use default
            if audio_path and Path(audio_path).exists():
                try:
                    y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=60)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    features['tempo'] = float(tempo)
                except Exception as e:
                    logger.warning(f"Could not extract tempo from {audio_path}: {e}")
                    features['tempo'] = 120.0  # Default
            else:
                features['tempo'] = 120.0  # Default

            # 2. Zero Crossing Rate
            if 'pcm_zcr_sma_amean' in df.columns:
                features['zero_crossing_rate'] = float(df['pcm_zcr_sma_amean'].mean())
            else:
                features['zero_crossing_rate'] = 0.0

            # 3-4. RMS Energy and Low Energy
            if 'pcm_RMSenergy_sma_amean' in df.columns:
                rms_values = df['pcm_RMSenergy_sma_amean'].values
                features['rms_energy'] = float(np.mean(rms_values))
                features['low_energy'] = float(np.sum(rms_values < np.mean(rms_values)) / len(rms_values))
            else:
                features['rms_energy'] = 0.0
                features['low_energy'] = 0.5

            # 5. Spectral Rolloff (90%)
            if 'pcm_fftMag_spectralRollOff90.0_sma_amean' in df.columns:
                features['spectral_rolloff'] = float(df['pcm_fftMag_spectralRollOff90.0_sma_amean'].mean())
            else:
                features['spectral_rolloff'] = 0.0

            # 6. Brightness (high bands / total bands from audSpec_Rfilt)
            high_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(20, 26)]
            all_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(0, 26)]

            if all(col in df.columns for col in high_bands + all_bands):
                high_energy = df[high_bands].sum(axis=1)
                total_energy = df[all_bands].sum(axis=1) + 1e-10
                features['brightness'] = float((high_energy / total_energy).mean())
            else:
                features['brightness'] = 0.0

            # 7. Spectral Centroid
            if 'pcm_fftMag_spectralCentroid_sma_amean' in df.columns:
                features['spectral_centroid'] = float(df['pcm_fftMag_spectralCentroid_sma_amean'].mean())
            else:
                features['spectral_centroid'] = 0.0

            # 8. Spectral Flux
            if 'pcm_fftMag_spectralFlux_sma_amean' in df.columns:
                features['spectral_flux'] = float(df['pcm_fftMag_spectralFlux_sma_amean'].mean())
            else:
                features['spectral_flux'] = 0.0

            # 9. Spectral Flatness (using harmonicity inverse as proxy)
            if 'pcm_fftMag_spectralHarmonicity_sma_amean' in df.columns:
                # Flatness is inverse of harmonicity (more harmonic = less flat)
                harmonicity = df['pcm_fftMag_spectralHarmonicity_sma_amean'].mean()
                features['spectral_flatness'] = float(1.0 / (1.0 + abs(harmonicity)))
            else:
                features['spectral_flatness'] = 0.5

            # 10-19. MFCCs (1-5 mean and std)
            for i in range(1, 6):
                mean_col = f'pcm_fftMag_mfcc_sma[{i}]_amean'
                std_col = f'pcm_fftMag_mfcc_sma[{i}]_stddev'

                if mean_col in df.columns:
                    features[f'mfcc_{i}_mean'] = float(df[mean_col].mean())
                else:
                    features[f'mfcc_{i}_mean'] = 0.0

                if std_col in df.columns:
                    features[f'mfcc_{i}_std'] = float(df[std_col].mean())
                else:
                    features[f'mfcc_{i}_std'] = 0.0

            # 18. Energy Variance
            if 'pcm_RMSenergy_sma_stddev' in df.columns:
                features['energy_variance'] = float(df['pcm_RMSenergy_sma_stddev'].mean())
            elif 'pcm_RMSenergy_sma_amean' in df.columns:
                features['energy_variance'] = float(df['pcm_RMSenergy_sma_amean'].std())
            else:
                features['energy_variance'] = 0.0

            # 19. Drop Strength (from energy delta)
            if 'pcm_RMSenergy_sma_de_amean' in df.columns:
                energy_delta = df['pcm_RMSenergy_sma_de_amean'].values
                features['drop_strength'] = float(np.percentile(np.abs(energy_delta), 90))
            else:
                features['drop_strength'] = 0.0

            # 20-23. Drop detection from RMS frames
            if 'pcm_RMSenergy_sma_amean' in df.columns:
                rms_frames = df['pcm_RMSenergy_sma_amean'].values
                drop_features = self._extract_drop_features(rms_frames, frame_size=0.5)
                features['drop_contrast_mean'] = drop_features['drop_contrast_mean']
                features['drop_count'] = drop_features['drop_count']
            else:
                features['drop_contrast_mean'] = 0.0
                features['drop_count'] = 0.0

            # 24. Onset Rate (estimate from energy derivative)
            if 'pcm_RMSenergy_sma_de_amean' in df.columns:
                # Count significant energy changes as onset proxy
                energy_delta = np.abs(df['pcm_RMSenergy_sma_de_amean'].values)
                threshold = np.percentile(energy_delta, 75)
                onsets = np.sum(energy_delta > threshold)
                duration = len(df) * 0.5  # 0.5s per frame
                features['onset_rate'] = float(onsets / duration) if duration > 0 else 0.0
            else:
                features['onset_rate'] = 0.0

            # 25. Pitch Mean (F0)
            if 'F0final_sma_amean' in df.columns:
                f0_values = df['F0final_sma_amean'].values
                f0_voiced = f0_values[f0_values > 0]
                features['pitch_mean'] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            else:
                features['pitch_mean'] = 0.0

            # 26-27. Arousal and Valence - will be set from annotations externally
            # Use estimates as fallback
            features['arousal'] = self._estimate_arousal(
                tempo=features['tempo'],
                rms_energy=features['rms_energy'],
                energy_variance=features['energy_variance'],
                drop_count=features['drop_count']
            )
            features['valence'] = self._estimate_valence(
                brightness=features['brightness'],
                spectral_centroid=features['spectral_centroid'],
                tempo=features['tempo']
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting features from DEAM CSV {csv_path}: {e}")
            return None

    def extract_frames_from_audio(
        self,
        audio_path: str,
        frame_size: float = 0.5
    ) -> Optional[pd.DataFrame]:
        """
        Extract frame-level features from audio (DEAM-compatible 0.5s frames).

        Args:
            audio_path: Path to audio file
            frame_size: Frame size in seconds (default 0.5s = DEAM format)

        Returns:
            DataFrame with 38 features per frame, or None if failed
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(y) / sr

            # Calculate frame parameters
            hop_length = int(frame_size * sr)
            n_frames = int(np.ceil(len(y) / hop_length))

            # Pre-compute all features
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

            # Spectral features
            S = np.abs(librosa.stft(y, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.9)[0]
            flatness = librosa.feature.spectral_flatness(S=S)[0]
            bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

            # Brightness (high freq ratio)
            high_freq_mask = freqs > 3000
            high_energy = np.sum(S[high_freq_mask, :], axis=0)
            total_energy = np.sum(S, axis=0) + 1e-10
            brightness = high_energy / total_energy

            # Spectral flux
            flux = np.zeros(S.shape[1])
            flux[1:] = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

            # Spectral contrast (mean across bands per frame)
            n_bands = 6 if sr <= 24000 else 7
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=n_bands, fmin=200.0)
            contrast_mean = np.mean(contrast, axis=0)

            # MFCCs
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=sr, n_mfcc=6)

            # Onset strength
            onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr)

            # Chroma
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            chroma_energy = np.sum(chroma, axis=0)

            # Beat tracking for beat_sync
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            beat_frames = set(beats)

            # HPSS for harmonic/percussive ratio per frame
            H, P = librosa.decompose.hpss(S)
            h_energy = np.sum(H ** 2, axis=0)
            p_energy = np.sum(P ** 2, axis=0)
            total_hp = h_energy + p_energy + 1e-10
            harmonic_ratio = h_energy / total_hp
            percussive_ratio = p_energy / total_hp

            # Pitch (simplified - use spectral centroid as proxy, pyin is too slow)
            # For real pitch, would need pyin but it's very slow
            pitch_proxy = centroid / 100  # Scaled spectral centroid as pitch proxy

            # Ensure all arrays have same length
            min_len = min(len(rms), len(zcr), len(centroid), len(onset_env), n_frames)

            # Compute deltas
            def safe_delta(arr, min_len):
                arr = arr[:min_len]
                delta = np.zeros(min_len)
                delta[1:] = np.diff(arr)
                return delta

            def safe_delta2(arr, min_len):
                arr = arr[:min_len]
                delta = np.zeros(min_len)
                if len(arr) > 2:
                    delta[2:] = np.diff(arr, n=2)
                return delta

            rms_delta = safe_delta(rms, min_len)
            rms_delta2 = safe_delta2(rms, min_len)
            zcr_delta = safe_delta(zcr, min_len)
            centroid_delta = safe_delta(centroid, min_len)
            rolloff_delta = safe_delta(rolloff, min_len)
            brightness_delta = safe_delta(brightness, min_len)
            onset_delta = safe_delta(onset_env, min_len)

            # MFCC deltas
            mfcc_deltas = [safe_delta(mfccs[i], min_len) for i in range(1, 6)]

            # Drop detection features (per-frame)
            mean_rms = np.mean(rms[:min_len])
            low_energy_flag = (rms[:min_len] < mean_rms).astype(float)

            # Energy buildup score (slope in 4-frame window)
            buildup_score = np.zeros(min_len)
            window = 4
            for i in range(window, min_len):
                segment = rms[i-window:i]
                if len(segment) > 1:
                    slope = (segment[-1] - segment[0]) / (window * frame_size)
                    buildup_score[i] = max(0, slope / (mean_rms + 1e-6))

            # Drop candidates (sharp energy increase after valley)
            drop_candidate = np.zeros(min_len)
            energy_valley = np.zeros(min_len)
            energy_peak = np.zeros(min_len)

            # Find local minima/maxima
            from scipy.signal import argrelextrema
            min_dist = max(1, int(2.0 / frame_size))  # 2 second minimum distance

            local_mins = argrelextrema(rms[:min_len], np.less, order=min_dist)[0]
            local_maxs = argrelextrema(rms[:min_len], np.greater, order=min_dist)[0]

            energy_valley[local_mins] = 1.0
            energy_peak[local_maxs] = 1.0

            # Mark drop candidates (frames after valley with strong energy increase)
            for valley_idx in local_mins:
                for offset in range(1, min(8, min_len - valley_idx)):
                    idx = valley_idx + offset
                    if idx < min_len and rms[idx] > rms[valley_idx] * 1.5:
                        drop_candidate[idx] = 1.0
                        break

            # Build DataFrame
            records = []
            for i in range(min_len):
                frame = {
                    'frameTime': i * frame_size,
                    'rms_energy': float(rms[i]),
                    'rms_energy_delta': float(rms_delta[i]),
                    'rms_energy_delta2': float(rms_delta2[i]),
                    'low_energy_flag': float(low_energy_flag[i]),
                    'zero_crossing_rate': float(zcr[i]) if i < len(zcr) else 0.0,
                    'zcr_delta': float(zcr_delta[i]),
                    'spectral_centroid': float(centroid[i]) if i < len(centroid) else 0.0,
                    'spectral_centroid_delta': float(centroid_delta[i]),
                    'spectral_rolloff': float(rolloff[i]) if i < len(rolloff) else 0.0,
                    'spectral_rolloff_delta': float(rolloff_delta[i]),
                    'brightness': float(brightness[i]) if i < len(brightness) else 0.0,
                    'brightness_delta': float(brightness_delta[i]),
                    'spectral_flux': float(flux[i]) if i < len(flux) else 0.0,
                    'spectral_flatness': float(flatness[i]) if i < len(flatness) else 0.0,
                    'spectral_contrast': float(contrast_mean[i]) if i < len(contrast_mean) else 0.0,
                    'spectral_bandwidth': float(bandwidth[i]) if i < len(bandwidth) else 0.0,
                    'onset_strength': float(onset_env[i]) if i < len(onset_env) else 0.0,
                    'onset_strength_delta': float(onset_delta[i]),
                    'beat_sync': 1.0 if i in beat_frames else 0.0,
                    'chroma_energy': float(chroma_energy[i]) if i < len(chroma_energy) else 0.0,
                    'harmonic_ratio': float(harmonic_ratio[i]) if i < len(harmonic_ratio) else 0.5,
                    'percussive_ratio': float(percussive_ratio[i]) if i < len(percussive_ratio) else 0.5,
                    'pitch': float(pitch_proxy[i]) if i < len(pitch_proxy) else 0.0,
                    'pitch_confidence': 0.5,  # Placeholder without pyin
                    'energy_buildup_score': float(buildup_score[i]),
                    'drop_candidate': float(drop_candidate[i]),
                    'energy_valley': float(energy_valley[i]),
                    'energy_peak': float(energy_peak[i]),
                }

                # MFCCs
                for j in range(1, 6):
                    frame[f'mfcc_{j}'] = float(mfccs[j][i]) if i < mfccs.shape[1] else 0.0
                    frame[f'mfcc_{j}_delta'] = float(mfcc_deltas[j-1][i])

                records.append(frame)

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error extracting frames from {audio_path}: {e}")
            return None

    def extract_frames_from_deam_csv(
        self,
        csv_path: str
    ) -> Optional[pd.DataFrame]:
        """
        Extract frame-level features from DEAM CSV (already frame-based).

        Args:
            csv_path: Path to DEAM feature CSV

        Returns:
            DataFrame with 38 features per frame, or None if failed
        """
        try:
            df = pd.read_csv(csv_path, sep=';')

            records = []
            n_frames = len(df)

            # Pre-compute mean RMS for low_energy_flag
            rms_col = 'pcm_RMSenergy_sma_amean'
            mean_rms = df[rms_col].mean() if rms_col in df.columns else 0.1

            # Compute deltas
            def get_delta(col_name):
                if col_name in df.columns:
                    vals = df[col_name].values
                    delta = np.zeros(len(vals))
                    delta[1:] = np.diff(vals)
                    return delta
                return np.zeros(n_frames)

            def get_delta2(col_name):
                if col_name in df.columns:
                    vals = df[col_name].values
                    delta2 = np.zeros(len(vals))
                    if len(vals) > 2:
                        delta2[2:] = np.diff(vals, n=2)
                    return delta2
                return np.zeros(n_frames)

            rms_delta = get_delta(rms_col)
            rms_delta2 = get_delta2(rms_col)
            zcr_delta = get_delta('pcm_zcr_sma_amean')
            centroid_delta = get_delta('pcm_fftMag_spectralCentroid_sma_amean')
            rolloff_delta = get_delta('pcm_fftMag_spectralRollOff90.0_sma_amean')

            # Brightness from audSpec bands
            high_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(20, 26)]
            all_bands = [f'audSpec_Rfilt_sma[{i}]_amean' for i in range(0, 26)]

            if all(col in df.columns for col in high_bands + all_bands):
                high_energy = df[high_bands].sum(axis=1).values
                total_energy = df[all_bands].sum(axis=1).values + 1e-10
                brightness = high_energy / total_energy
            else:
                brightness = np.zeros(n_frames)

            brightness_delta = np.zeros(n_frames)
            brightness_delta[1:] = np.diff(brightness)

            # Drop detection
            if rms_col in df.columns:
                rms_values = df[rms_col].values
                low_energy_flag = (rms_values < mean_rms).astype(float)

                # Buildup score
                buildup_score = np.zeros(n_frames)
                window = 4
                for i in range(window, n_frames):
                    segment = rms_values[i-window:i]
                    slope = (segment[-1] - segment[0]) / (window * 0.5)
                    buildup_score[i] = max(0, slope / (mean_rms + 1e-6))

                # Local minima/maxima
                from scipy.signal import argrelextrema
                min_dist = max(1, 4)  # 2 seconds at 0.5s frames

                energy_valley = np.zeros(n_frames)
                energy_peak = np.zeros(n_frames)
                drop_candidate = np.zeros(n_frames)

                if n_frames > min_dist * 2:
                    local_mins = argrelextrema(rms_values, np.less, order=min_dist)[0]
                    local_maxs = argrelextrema(rms_values, np.greater, order=min_dist)[0]

                    energy_valley[local_mins] = 1.0
                    energy_peak[local_maxs] = 1.0

                    for valley_idx in local_mins:
                        for offset in range(1, min(8, n_frames - valley_idx)):
                            idx = valley_idx + offset
                            if idx < n_frames and rms_values[idx] > rms_values[valley_idx] * 1.5:
                                drop_candidate[idx] = 1.0
                                break
            else:
                low_energy_flag = np.zeros(n_frames)
                buildup_score = np.zeros(n_frames)
                energy_valley = np.zeros(n_frames)
                energy_peak = np.zeros(n_frames)
                drop_candidate = np.zeros(n_frames)

            # Onset strength delta
            onset_delta = get_delta('pcm_RMSenergy_sma_de_amean')

            for i in range(n_frames):
                row = df.iloc[i]

                frame = {
                    'frameTime': row['frameTime'] if 'frameTime' in df.columns else i * 0.5,
                    'rms_energy': float(row.get(rms_col, 0)),
                    'rms_energy_delta': float(rms_delta[i]),
                    'rms_energy_delta2': float(rms_delta2[i]),
                    'low_energy_flag': float(low_energy_flag[i]),
                    'zero_crossing_rate': float(row.get('pcm_zcr_sma_amean', 0)),
                    'zcr_delta': float(zcr_delta[i]),
                    'spectral_centroid': float(row.get('pcm_fftMag_spectralCentroid_sma_amean', 0)),
                    'spectral_centroid_delta': float(centroid_delta[i]),
                    'spectral_rolloff': float(row.get('pcm_fftMag_spectralRollOff90.0_sma_amean', 0)),
                    'spectral_rolloff_delta': float(rolloff_delta[i]),
                    'brightness': float(brightness[i]),
                    'brightness_delta': float(brightness_delta[i]),
                    'spectral_flux': float(row.get('pcm_fftMag_spectralFlux_sma_amean', 0)),
                    'spectral_flatness': float(1.0 / (1.0 + abs(row.get('pcm_fftMag_spectralHarmonicity_sma_amean', 0)))),
                    'spectral_contrast': 0.0,  # Not in DEAM
                    'spectral_bandwidth': 0.0,  # Not in DEAM
                    'onset_strength': float(row.get('pcm_RMSenergy_sma_de_amean', 0)),
                    'onset_strength_delta': float(onset_delta[i]),
                    'beat_sync': 0.0,  # Not in DEAM
                    'chroma_energy': 0.0,  # Not in DEAM
                    'harmonic_ratio': 0.5,  # Not in DEAM
                    'percussive_ratio': 0.5,  # Not in DEAM
                    'pitch': float(row.get('F0final_sma_amean', 0)),
                    'pitch_confidence': float(row.get('voicingFinalUnclipped_sma_amean', 0)),
                    'energy_buildup_score': float(buildup_score[i]),
                    'drop_candidate': float(drop_candidate[i]),
                    'energy_valley': float(energy_valley[i]),
                    'energy_peak': float(energy_peak[i]),
                }

                # MFCCs
                for j in range(1, 6):
                    mean_col = f'pcm_fftMag_mfcc_sma[{j}]_amean'
                    std_col = f'pcm_fftMag_mfcc_sma[{j}]_stddev'
                    frame[f'mfcc_{j}'] = float(row.get(mean_col, 0))
                    # Use stddev as proxy for delta (not exact but available)
                    frame[f'mfcc_{j}_delta'] = float(row.get(std_col, 0)) * 0.1

                records.append(frame)

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error extracting frames from DEAM CSV {csv_path}: {e}")
            return None

    def extract_batch_frames_from_audio(
        self,
        audio_paths: List[str],
        zones: Optional[List[str]] = None,
        frame_size: float = 0.5,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract frame features from multiple audio files.

        Args:
            audio_paths: List of audio file paths
            zones: Optional list of zone labels (same length as audio_paths)
            frame_size: Frame size in seconds
            show_progress: Show progress bar

        Returns:
            DataFrame with all frames from all tracks
        """
        all_frames = []

        iterator = enumerate(audio_paths)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Extracting frames")
            except ImportError:
                iterator = list(iterator)

        for idx, audio_path in iterator:
            frames_df = self.extract_frames_from_audio(audio_path, frame_size)
            if frames_df is not None:
                frames_df['track_id'] = idx
                frames_df['path'] = audio_path
                frames_df['source'] = 'user'
                if zones and idx < len(zones):
                    frames_df['zone'] = zones[idx]
                all_frames.append(frames_df)

        if all_frames:
            return pd.concat(all_frames, ignore_index=True)
        return pd.DataFrame()

    def extract_batch_frames_from_deam(
        self,
        csv_dir: str,
        zones: Optional[Dict[int, str]] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract frame features from DEAM CSV directory.

        Args:
            csv_dir: Directory with DEAM CSV files
            zones: Optional dict mapping track_id to zone
            show_progress: Show progress bar

        Returns:
            DataFrame with all frames from all tracks
        """
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("*.csv"))

        all_frames = []

        iterator = csv_files
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(csv_files, desc="Extracting DEAM frames")
            except ImportError:
                pass

        for csv_path in iterator:
            track_id = csv_path.stem
            track_id_int = int(track_id) if track_id.isdigit() else 0

            frames_df = self.extract_frames_from_deam_csv(str(csv_path))
            if frames_df is not None:
                frames_df['track_id'] = track_id_int
                frames_df['path'] = str(csv_path)
                frames_df['source'] = 'deam'
                if zones and track_id_int in zones:
                    frames_df['zone'] = zones[track_id_int]
                all_frames.append(frames_df)

        if all_frames:
            return pd.concat(all_frames, ignore_index=True)
        return pd.DataFrame()

    def _estimate_arousal(
        self,
        tempo: float,
        rms_energy: float,
        energy_variance: float,
        drop_count: float
    ) -> float:
        """
        Estimate arousal (energy level) from audio features.

        Returns value on 1-9 scale (DEAM compatible).

        Arousal correlates with:
        - Higher tempo → higher arousal
        - Higher RMS energy → higher arousal
        - More drops → higher arousal (EDM tracks)
        - Higher energy variance → higher arousal
        """
        # Normalize features to 0-1 range
        tempo_norm = np.clip((tempo - 60) / (160 - 60), 0, 1)  # 60-160 BPM
        rms_norm = np.clip(rms_energy / 0.3, 0, 1)  # 0-0.3 typical range
        variance_norm = np.clip(energy_variance / 0.1, 0, 1)  # 0-0.1 typical
        drops_norm = np.clip(drop_count / 10, 0, 1)  # 0-10 drops

        # Weighted combination
        arousal_raw = (
            0.35 * tempo_norm +
            0.30 * rms_norm +
            0.20 * variance_norm +
            0.15 * drops_norm
        )

        # Scale to 1-9
        arousal = 1 + arousal_raw * 8
        return float(np.clip(arousal, 1, 9))

    def _estimate_valence(
        self,
        brightness: float,
        spectral_centroid: float,
        tempo: float
    ) -> float:
        """
        Estimate valence (mood positivity) from audio features.

        Returns value on 1-9 scale (DEAM compatible).

        Valence correlates with:
        - Higher brightness → more positive
        - Higher spectral centroid → more positive
        - Medium-high tempo → more positive
        """
        # Normalize features
        brightness_norm = np.clip(brightness / 0.3, 0, 1)  # 0-0.3 typical
        centroid_norm = np.clip((spectral_centroid - 1000) / (4000 - 1000), 0, 1)  # 1k-4k Hz

        # Tempo affects valence (very slow or very fast = less positive)
        tempo_centered = abs(tempo - 120) / 60  # Distance from 120 BPM
        tempo_valence = 1 - np.clip(tempo_centered, 0, 1)

        # Weighted combination
        valence_raw = (
            0.40 * brightness_norm +
            0.35 * centroid_norm +
            0.25 * tempo_valence
        )

        # Scale to 1-9
        valence = 1 + valence_raw * 8
        return float(np.clip(valence, 1, 9))

    def _extract_drop_features(
        self,
        rms: np.ndarray,
        frame_size: float = 0.023  # librosa default hop ~23ms
    ) -> Dict[str, float]:
        """
        Extract drop detection features from RMS energy frames.

        Finds breakdown→drop patterns (valley followed by peak).

        Args:
            rms: RMS energy values per frame
            frame_size: Frame size in seconds

        Returns:
            Dictionary with drop_contrast_mean, drop_contrast_max, drop_count, drop_intensity
        """
        features = {
            'drop_contrast_mean': 0.0,
            'drop_contrast_max': 0.0,
            'drop_count': 0.0,
            'drop_intensity': 0.0,
        }

        if len(rms) < 20:
            return features

        mean_energy = np.mean(rms)
        std_energy = np.std(rms)

        if mean_energy < 1e-10:
            return features

        # Minimum distance between peaks/valleys (~4 seconds)
        min_distance = max(1, int(4.0 / frame_size))

        # Find valleys (potential breakdowns)
        valley_threshold = mean_energy - 0.3 * std_energy
        try:
            valleys, _ = find_peaks(-rms, height=-valley_threshold, distance=min_distance)
        except Exception:
            valleys = np.array([])

        # Find peaks (potential drops)
        peak_threshold = mean_energy + 0.3 * std_energy
        try:
            peaks, _ = find_peaks(rms, height=peak_threshold, distance=min_distance)
        except Exception:
            peaks = np.array([])

        if len(valleys) == 0 or len(peaks) == 0:
            return features

        # Match valleys to following peaks (breakdown→drop pattern)
        drop_contrasts = []
        drop_peak_energies = []
        max_distance = int(8.0 / frame_size)  # Max ~8 seconds between valley and peak

        for valley_idx in valleys:
            following_peaks = peaks[peaks > valley_idx]
            if len(following_peaks) == 0:
                continue

            next_peak = following_peaks[0]
            distance = next_peak - valley_idx

            if distance > max_distance:
                continue

            valley_energy = rms[valley_idx]
            peak_energy = rms[next_peak]
            contrast = (peak_energy - valley_energy) / (mean_energy + 1e-10)

            # Only count significant drops (>50% of mean energy contrast)
            if contrast > 0.5:
                drop_contrasts.append(contrast)
                drop_peak_energies.append(peak_energy)

        if len(drop_contrasts) > 0:
            features['drop_contrast_mean'] = float(np.mean(drop_contrasts))
            features['drop_contrast_max'] = float(np.max(drop_contrasts))
            features['drop_count'] = float(len(drop_contrasts))
            features['drop_intensity'] = float(np.mean(drop_peak_energies) / mean_energy)

        return features

    def extract_batch_from_audio(
        self,
        audio_paths: List[str],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of audio file paths
            show_progress: Show progress bar

        Returns:
            DataFrame with features for all tracks
        """
        records = []

        iterator = audio_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Extracting from audio")
            except ImportError:
                pass

        for audio_path in iterator:
            features = self.extract_from_audio(audio_path)
            if features:
                features['path'] = audio_path
                features['source'] = 'audio'
                records.append(features)

        return pd.DataFrame(records)

    def extract_batch_from_deam(
        self,
        csv_dir: str,
        audio_dir: Optional[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from DEAM CSV directory.

        Args:
            csv_dir: Directory with DEAM feature CSV files
            audio_dir: Optional directory with corresponding audio files
            show_progress: Show progress bar

        Returns:
            DataFrame with features for all tracks
        """
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob("*.csv"))

        records = []

        iterator = csv_files
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(csv_files, desc="Extracting from DEAM")
            except ImportError:
                pass

        for csv_path in iterator:
            track_id = csv_path.stem

            # Find corresponding audio file
            audio_path = None
            if audio_dir:
                audio_dir_path = Path(audio_dir)
                for ext in ['.mp3', '.wav', '.flac', '.m4a']:
                    candidate = audio_dir_path / f"{track_id}{ext}"
                    if candidate.exists():
                        audio_path = str(candidate)
                        break

            features = self.extract_from_deam_csv(str(csv_path), audio_path)
            if features:
                features['track_id'] = int(track_id) if track_id.isdigit() else track_id
                features['path'] = str(csv_path)
                features['source'] = 'deam'
                if audio_path:
                    features['audio_path'] = audio_path
                records.append(features)

        return pd.DataFrame(records)


def merge_datasets(
    user_df: pd.DataFrame,
    deam_df: pd.DataFrame,
    user_weight: float = 1.0,
    deam_weight: float = 1.0
) -> pd.DataFrame:
    """
    Merge User and DEAM datasets with optional weighting.

    Args:
        user_df: User dataset DataFrame
        deam_df: DEAM dataset DataFrame
        user_weight: Sample weight for User tracks
        deam_weight: Sample weight for DEAM tracks

    Returns:
        Merged DataFrame with sample_weight column
    """
    # Ensure both have source column
    if 'source' not in user_df.columns:
        user_df = user_df.copy()
        user_df['source'] = 'user'

    if 'source' not in deam_df.columns:
        deam_df = deam_df.copy()
        deam_df['source'] = 'deam'

    # Add weights
    user_df = user_df.copy()
    user_df['sample_weight'] = user_weight

    deam_df = deam_df.copy()
    deam_df['sample_weight'] = deam_weight

    # Find common columns (features only)
    common_cols = set(COMMON_FEATURES) & set(user_df.columns) & set(deam_df.columns)
    meta_cols = ['path', 'source', 'sample_weight', 'zone', 'track_id', 'audio_path']

    # Select columns that exist
    select_cols = list(common_cols) + [c for c in meta_cols if c in user_df.columns or c in deam_df.columns]

    user_subset = user_df[[c for c in select_cols if c in user_df.columns]]
    deam_subset = deam_df[[c for c in select_cols if c in deam_df.columns]]

    # Merge
    merged = pd.concat([user_subset, deam_subset], ignore_index=True)

    logger.info(f"Merged dataset: {len(user_subset)} user + {len(deam_subset)} deam = {len(merged)} total")

    return merged


if __name__ == "__main__":
    # Quick test
    import sys
    logging.basicConfig(level=logging.INFO)

    extractor = UnifiedFeatureExtractor()

    # Test with a DEAM file
    deam_csv = Path("dataset/features/2.csv")
    if deam_csv.exists():
        print(f"\nTesting DEAM extraction from {deam_csv}...")
        features = extractor.extract_from_deam_csv(str(deam_csv))
        if features:
            print(f"Extracted {len(features)} features:")
            for k, v in sorted(features.items()):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n21 Common Features:")
    for i, f in enumerate(COMMON_FEATURES, 1):
        print(f"  {i}. {f}")

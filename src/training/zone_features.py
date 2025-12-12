"""Zone classification feature extraction combining librosa and torchaudio."""

import numpy as np
import librosa
import torch
import torchaudio
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ZoneFeatures:
    """Audio features for energy zone classification."""

    # Librosa features (16 features from existing classifier)
    tempo: float
    zero_crossing_rate: float
    low_energy: float
    rms_energy: float
    spectral_rolloff: float
    brightness: float
    spectral_centroid: float
    mfcc_1_mean: float
    mfcc_1_std: float
    mfcc_2_mean: float
    mfcc_2_std: float
    mfcc_3_mean: float
    mfcc_3_std: float
    mfcc_4_mean: float
    mfcc_4_std: float
    mfcc_5_mean: float
    mfcc_5_std: float
    energy_variance: float
    drop_strength: float

    # Music emotion features (arousal-valence model)
    arousal: float = 0.0  # Energy level: -1 (low) to +1 (high)
    valence: float = 0.0  # Mood: -1 (negative) to +1 (positive)

    # Advanced torchaudio features (embeddings from pre-trained models)
    wav2vec2_embedding: Optional[np.ndarray] = None  # 768-dim by default
    hubert_embedding: Optional[np.ndarray] = None  # 768-dim (optional)

    # Additional temporal features
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    beat_strength: float = 0.0
    tempo_stability: float = 0.0

    # Spectral contrast features
    spectral_contrast_mean: np.ndarray = None  # 7 bands
    spectral_contrast_std: np.ndarray = None  # 7 bands

    # Harmonic-percussive features
    harmonic_ratio: float = 0.0
    percussive_ratio: float = 0.0

    # Build-up detection features (CRITICAL for GREEN zone)
    energy_slope: float = 0.0  # Linear regression of energy (-1 falling, +1 rising)
    energy_buildup_ratio: float = 0.0  # Ratio of rising energy segments (0-1)
    onset_acceleration: float = 0.0  # Acceleration of onset strength

    # Drive enhancement features (for PURPLE zone)
    drop_frequency: float = 0.0  # Drops per minute
    peak_energy_ratio: float = 0.0  # Peak energy / mean energy

    # Improved drop detection features (breakdown→drop contrast)
    drop_contrast_mean: float = 0.0  # Average contrast between breakdown and drop
    drop_contrast_max: float = 0.0  # Maximum single drop contrast
    drop_count: float = 0.0  # Number of true breakdown→drop patterns
    drop_intensity: float = 0.0  # Average drop peak energy / mean track energy

    # Euphoria indicators (for PURPLE zone)
    rhythmic_regularity: float = 0.0  # Rhythm stability (0-1, 1=very regular)
    harmonic_complexity: float = 0.0  # Harmonic richness (entropy)

    # Climax structure detection (for PURPLE zone)
    has_climax: float = 0.0  # Binary indicator (0 or 1) for clear climax
    climax_position: float = 0.5  # Position in track (0=start, 1=end)
    dynamic_range: float = 0.0  # Dynamic range in dB

    def to_vector(self, include_embeddings: bool = True) -> np.ndarray:
        """
        Convert features to flat vector.

        Args:
            include_embeddings: Whether to include deep learning embeddings

        Returns:
            Feature vector
        """
        # Basic librosa features (19)
        basic_features = [
            self.tempo,
            self.zero_crossing_rate,
            self.low_energy,
            self.rms_energy,
            self.spectral_rolloff,
            self.brightness,
            self.spectral_centroid,
            self.mfcc_1_mean,
            self.mfcc_1_std,
            self.mfcc_2_mean,
            self.mfcc_2_std,
            self.mfcc_3_mean,
            self.mfcc_3_std,
            self.mfcc_4_mean,
            self.mfcc_4_std,
            self.mfcc_5_mean,
            self.mfcc_5_std,
            self.energy_variance,
            self.drop_strength,
        ]

        # Music emotion features (2) - arousal-valence model
        emotion_features = [
            self.arousal,
            self.valence,
        ]

        # Additional temporal features (4)
        temporal_features = [
            self.onset_strength_mean,
            self.onset_strength_std,
            self.beat_strength,
            self.tempo_stability,
        ]

        # Spectral contrast (14)
        if self.spectral_contrast_mean is not None:
            spectral_features = np.concatenate([
                self.spectral_contrast_mean,
                self.spectral_contrast_std
            ])
        else:
            spectral_features = np.zeros(14)

        # Harmonic-percussive (2)
        hp_features = [
            self.harmonic_ratio,
            self.percussive_ratio,
        ]

        # Build-up detection (3)
        buildup_features = [
            self.energy_slope,
            self.energy_buildup_ratio,
            self.onset_acceleration,
        ]

        # Drive enhancement (2)
        drive_features = [
            self.drop_frequency,
            self.peak_energy_ratio,
        ]

        # Improved drop detection (4)
        drop_features = [
            self.drop_contrast_mean,
            self.drop_contrast_max,
            self.drop_count,
            self.drop_intensity,
        ]

        # Euphoria indicators (2)
        euphoria_features = [
            self.rhythmic_regularity,
            self.harmonic_complexity,
        ]

        # Climax structure (3)
        climax_features = [
            self.has_climax,
            self.climax_position,
            self.dynamic_range,
        ]

        # Combine all traditional features (19 + 2 + 4 + 14 + 2 + 3 + 2 + 4 + 2 + 3 = 55)
        traditional = np.concatenate([
            basic_features,
            emotion_features,
            temporal_features,
            spectral_features,
            hp_features,
            buildup_features,
            drive_features,
            drop_features,
            euphoria_features,
            climax_features
        ])

        if not include_embeddings:
            return traditional

        # Add embeddings
        features_list = [traditional]

        if self.wav2vec2_embedding is not None:
            features_list.append(self.wav2vec2_embedding)

        if self.hubert_embedding is not None:
            features_list.append(self.hubert_embedding)

        return np.concatenate(features_list)


class ZoneFeatureExtractor:
    """Extract comprehensive features for zone classification."""

    def __init__(self, sample_rate: int = 22050, use_gpu: bool = True, use_embeddings: bool = False, use_music_emotion: bool = False):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate
            use_gpu: Use GPU for torchaudio models if available
            use_embeddings: Use deep learning embeddings (wav2vec2, HuBERT) - slow but accurate
            use_music_emotion: Use pretrained music emotion model for arousal/valence - VERY slow but music-specific
        """
        self.sample_rate = sample_rate
        self.use_embeddings = use_embeddings
        self.use_music_emotion = use_music_emotion
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Load pre-trained models for embeddings (only if requested)
        if use_embeddings:
            self._load_embedding_models()
        else:
            logger.info("Skipping deep learning embeddings (use_embeddings=False)")
            self.wav2vec2_model = None
            self.hubert_model = None

        # Load music emotion model (only if requested)
        if use_music_emotion:
            self._load_music_emotion_model()
        else:
            logger.info("Skipping music emotion model (use_music_emotion=False)")
            self.music_emotion_extractor = None

    def _load_embedding_models(self):
        """Load pre-trained torchaudio models."""
        try:
            # Wav2Vec2 for audio representations
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.wav2vec2_model = bundle.get_model().to(self.device)
            self.wav2vec2_model.eval()
            self.wav2vec2_sample_rate = bundle.sample_rate
            logger.info(f"Loaded Wav2Vec2 model on {self.device}")

        except Exception as e:
            logger.warning(f"Could not load Wav2Vec2: {e}. Using fallback.")
            self.wav2vec2_model = None

        try:
            # HuBERT for additional representations (optional)
            bundle = torchaudio.pipelines.HUBERT_BASE
            self.hubert_model = bundle.get_model().to(self.device)
            self.hubert_model.eval()
            self.hubert_sample_rate = bundle.sample_rate
            logger.info(f"Loaded HuBERT model on {self.device}")

        except Exception as e:
            logger.warning(f"Could not load HuBERT: {e}. Skipping.")
            self.hubert_model = None

    def _load_music_emotion_model(self):
        """Load pretrained music emotion model for arousal/valence."""
        try:
            from .music_emotion_model import MusicEmotionFeatureExtractor
            self.music_emotion_extractor = MusicEmotionFeatureExtractor(
                device=str(self.device),
                use_gpu=(self.device.type == 'cuda')
            )
            logger.info(f"Loaded MusicEmotionRegressor on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load music emotion model: {e}. Using zero values.")
            self.music_emotion_extractor = None

    def extract(self, audio_path: str) -> ZoneFeatures:
        """
        Extract all features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            ZoneFeatures object
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Extract librosa features
        librosa_features = self._extract_librosa_features(y, sr)

        # Extract torchaudio embeddings (optional)
        if self.use_embeddings:
            embeddings = self._extract_embeddings(audio_path)
        else:
            # Use zero embeddings if disabled (fast)
            embeddings = {
                'wav2vec2_embedding': np.zeros(768),
                'hubert_embedding': None
            }

        # Extract music emotion features (arousal/valence) - optional
        if self.use_music_emotion and self.music_emotion_extractor is not None:
            try:
                arousal, valence = self.music_emotion_extractor.extract_arousal_valence(audio_path)
                librosa_features['arousal'] = arousal
                librosa_features['valence'] = valence
            except Exception as e:
                logger.warning(f"Music emotion extraction failed for {audio_path}: {e}")
                librosa_features['arousal'] = 0.0
                librosa_features['valence'] = 0.0
        else:
            # Use neutral values if disabled
            librosa_features['arousal'] = 0.0
            librosa_features['valence'] = 0.0

        # Combine all features
        return ZoneFeatures(
            **librosa_features,
            **embeddings
        )

    def _extract_librosa_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract traditional librosa-based features."""
        features = {}

        # 1. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)

        # 2. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))

        # 3. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy'] = float(np.mean(rms))

        # 4. Low energy (percentage of frames below mean)
        features['low_energy'] = float(np.sum(rms < np.mean(rms)) / len(rms))

        # 5. Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))

        # 6. Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(centroid))

        # 7. Brightness (high-frequency energy)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_energy = np.sum(S[freqs > 3000, :], axis=0)
        total_energy = np.sum(S, axis=0)
        brightness = high_freq_energy / (total_energy + 1e-6)
        features['brightness'] = float(np.mean(brightness))

        # 8-17. MFCCs (first 5 coefficients, mean and std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        for i in range(5):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))

        # 18. Energy variance
        features['energy_variance'] = float(np.std(rms))

        # 19. Drop detection (sharp energy changes)
        energy_derivative = np.diff(rms)
        features['drop_strength'] = float(np.percentile(np.abs(energy_derivative), 90))

        # Additional temporal features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_std'] = float(np.std(onset_env))

        # Beat strength
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) > 0:
            beat_frames = librosa.frames_to_samples(beats)
            beat_energies = [rms[min(b // 512, len(rms)-1)] for b in beat_frames]
            features['beat_strength'] = float(np.mean(beat_energies))
        else:
            features['beat_strength'] = 0.0

        # Tempo stability (tempogram variance)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features['tempo_stability'] = float(1.0 / (np.std(np.argmax(tempogram, axis=0)) + 1))

        # Spectral contrast (safe band count for various sample rates)
        # At 22050 Hz, max 6 bands to avoid exceeding Nyquist frequency
        # At 44100 Hz, can use 7 bands safely
        n_bands = 6 if sr <= 24000 else 7
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=200.0)
        features['spectral_contrast_mean'] = np.mean(contrast, axis=1)
        features['spectral_contrast_std'] = np.std(contrast, axis=1)

        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total = harmonic_energy + percussive_energy
        features['harmonic_ratio'] = float(harmonic_energy / (total + 1e-6))
        features['percussive_ratio'] = float(percussive_energy / (total + 1e-6))

        # Build-up detection features
        # 1. Energy slope (linear regression of energy over time)
        time_frames = np.arange(len(rms))
        if len(rms) > 1:
            # Normalize time to [0, 1]
            time_normalized = time_frames / len(rms)
            # Linear regression: slope
            slope, _ = np.polyfit(time_normalized, rms, 1)
            # Normalize slope to [-1, 1] range
            features['energy_slope'] = float(np.clip(slope / (np.std(rms) + 1e-6), -1, 1))
        else:
            features['energy_slope'] = 0.0

        # 2. Energy buildup ratio (ratio of rising segments)
        energy_diff = np.diff(rms)
        if len(energy_diff) > 0:
            rising_segments = np.sum(energy_diff > 0)
            features['energy_buildup_ratio'] = float(rising_segments / len(energy_diff))
        else:
            features['energy_buildup_ratio'] = 0.5

        # 3. Onset acceleration (second derivative of onset strength)
        if len(onset_env) > 2:
            onset_velocity = np.diff(onset_env)
            onset_accel = np.diff(onset_velocity)
            # Use mean absolute acceleration
            features['onset_acceleration'] = float(np.mean(np.abs(onset_accel)))
        else:
            features['onset_acceleration'] = 0.0

        # Drive enhancement features
        # 1. Drop frequency (sharp energy drops per minute)
        # A drop is a significant negative energy change
        if len(energy_derivative) > 0:
            drop_threshold = -np.percentile(np.abs(energy_derivative), 75)  # Top 25% drops
            drops = np.sum(energy_derivative < drop_threshold)
            # Convert to drops per minute
            track_duration_minutes = len(y) / sr / 60.0
            features['drop_frequency'] = float(drops / max(track_duration_minutes, 0.1))
        else:
            features['drop_frequency'] = 0.0

        # 2. Peak energy ratio (max / mean energy)
        if len(rms) > 0 and np.mean(rms) > 0:
            features['peak_energy_ratio'] = float(np.max(rms) / np.mean(rms))
        else:
            features['peak_energy_ratio'] = 1.0

        # Improved drop detection: find actual breakdown→drop patterns
        drop_features = self._extract_drop_features(rms, sr)
        features.update(drop_features)

        # Euphoria indicators
        # 1. Rhythmic regularity (from beat intervals)
        if len(beats) > 2:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times)
            # Coefficient of variation (lower = more regular)
            if np.mean(beat_intervals) > 0:
                cv = np.std(beat_intervals) / np.mean(beat_intervals)
                # Invert to get regularity (0=irregular, 1=very regular)
                features['rhythmic_regularity'] = float(1.0 / (1.0 + cv))
            else:
                features['rhythmic_regularity'] = 0.0
        else:
            features['rhythmic_regularity'] = 0.0

        # 2. Harmonic complexity (spectral entropy)
        # Use chromagram to capture harmonic content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Calculate entropy across chromatic bins (average over time)
        chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
        entropy_per_frame = -np.sum(chroma_norm * np.log(chroma_norm + 1e-10), axis=0)
        features['harmonic_complexity'] = float(np.mean(entropy_per_frame))

        # Climax structure detection
        # 1. Has climax (detect if there's a clear peak in energy)
        # Find global maximum and check if it's significantly above mean
        if len(rms) > 0:
            max_energy = np.max(rms)
            mean_energy = np.mean(rms)
            std_energy = np.std(rms)
            # Climax exists if peak is > 2 standard deviations above mean
            features['has_climax'] = float(1.0 if (max_energy - mean_energy) > 2 * std_energy else 0.0)

            # 2. Climax position (where in the track is the peak energy)
            max_idx = np.argmax(rms)
            features['climax_position'] = float(max_idx / len(rms))
        else:
            features['has_climax'] = 0.0
            features['climax_position'] = 0.5

        # 3. Dynamic range (in dB)
        if len(rms) > 0 and np.max(rms) > 0 and np.min(rms) > 0:
            # Convert RMS to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            features['dynamic_range'] = float(np.max(rms_db) - np.min(rms_db))
        else:
            features['dynamic_range'] = 0.0

        return features

    def _extract_drop_features(self, rms: np.ndarray, sr: int) -> dict:
        """
        Extract improved drop detection features.

        Finds actual breakdown→drop patterns by detecting:
        1. Energy valleys (breakdowns)
        2. Following energy peaks (drops)
        3. Contrast between them

        Args:
            rms: RMS energy array
            sr: Sample rate

        Returns:
            Dictionary with drop features
        """
        from scipy.signal import find_peaks
        from scipy.ndimage import median_filter

        features = {
            'drop_contrast_mean': 0.0,
            'drop_contrast_max': 0.0,
            'drop_count': 0.0,
            'drop_intensity': 0.0,
        }

        if len(rms) < 100:  # Need enough data for pattern detection
            return features

        # Smooth RMS to reduce noise (use ~0.5s window)
        hop_length = 512  # librosa default
        window_frames = int(0.5 * sr / hop_length)
        window_frames = max(3, window_frames if window_frames % 2 == 1 else window_frames + 1)
        rms_smooth = median_filter(rms, size=window_frames)

        mean_energy = np.mean(rms_smooth)
        std_energy = np.std(rms_smooth)

        if mean_energy < 1e-6:
            return features

        # Find valleys (breakdowns) - local minima below mean
        valley_threshold = mean_energy - 0.3 * std_energy
        valleys, valley_props = find_peaks(-rms_smooth, height=-valley_threshold, distance=50)

        # Find peaks (drops) - local maxima above mean
        peak_threshold = mean_energy + 0.3 * std_energy
        peaks, peak_props = find_peaks(rms_smooth, height=peak_threshold, distance=50)

        if len(valleys) == 0 or len(peaks) == 0:
            return features

        # Match valleys to following peaks (breakdown→drop pattern)
        drop_contrasts = []
        drop_peak_energies = []

        for valley_idx in valleys:
            # Find nearest peak AFTER this valley (within ~8 seconds / ~350 frames)
            following_peaks = peaks[peaks > valley_idx]
            if len(following_peaks) == 0:
                continue

            # Get first peak after valley within reasonable distance
            next_peak = following_peaks[0]
            distance = next_peak - valley_idx

            # Drop should happen within ~8 seconds after breakdown
            max_distance = int(8.0 * sr / hop_length)
            if distance > max_distance:
                continue

            # Calculate contrast
            valley_energy = rms_smooth[valley_idx]
            peak_energy = rms_smooth[next_peak]
            contrast = (peak_energy - valley_energy) / (mean_energy + 1e-6)

            # Only count significant drops (contrast > 0.5 = 50% of mean energy)
            if contrast > 0.5:
                drop_contrasts.append(contrast)
                drop_peak_energies.append(peak_energy)

        # Calculate features
        if len(drop_contrasts) > 0:
            features['drop_contrast_mean'] = float(np.mean(drop_contrasts))
            features['drop_contrast_max'] = float(np.max(drop_contrasts))
            features['drop_count'] = float(len(drop_contrasts))
            features['drop_intensity'] = float(np.mean(drop_peak_energies) / mean_energy)

        return features

    def _extract_embeddings(self, audio_path: str) -> dict:
        """Extract deep learning embeddings using torchaudio models."""
        embeddings = {}

        # Load audio with torchaudio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Extract Wav2Vec2 embedding
            if self.wav2vec2_model is not None:
                embeddings['wav2vec2_embedding'] = self._get_wav2vec2_embedding(
                    waveform, sample_rate
                )
            else:
                embeddings['wav2vec2_embedding'] = np.zeros(768)

            # Extract HuBERT embedding (optional)
            if self.hubert_model is not None:
                embeddings['hubert_embedding'] = self._get_hubert_embedding(
                    waveform, sample_rate
                )
            else:
                embeddings['hubert_embedding'] = None

        except Exception as e:
            logger.warning(f"Could not extract embeddings from {audio_path}: {e}")
            embeddings['wav2vec2_embedding'] = np.zeros(768)
            embeddings['hubert_embedding'] = None

        return embeddings

    def _get_wav2vec2_embedding(self, waveform: torch.Tensor,
                                sample_rate: int) -> np.ndarray:
        """Extract Wav2Vec2 embedding from audio."""
        try:
            # Resample if needed
            if sample_rate != self.wav2vec2_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.wav2vec2_sample_rate
                )
                waveform = resampler(waveform)

            # Move to device
            waveform = waveform.to(self.device)

            # Extract features
            with torch.no_grad():
                features, _ = self.wav2vec2_model.extract_features(waveform)

                # Use last layer, average over time
                embedding = features[-1].mean(dim=1).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.warning(f"Wav2Vec2 extraction failed: {e}")
            return np.zeros(768)

    def _get_hubert_embedding(self, waveform: torch.Tensor,
                             sample_rate: int) -> np.ndarray:
        """Extract HuBERT embedding from audio."""
        try:
            # Resample if needed
            if sample_rate != self.hubert_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.hubert_sample_rate
                )
                waveform = resampler(waveform)

            # Move to device
            waveform = waveform.to(self.device)

            # Extract features
            with torch.no_grad():
                features, _ = self.hubert_model.extract_features(waveform)

                # Use last layer, average over time
                embedding = features[-1].mean(dim=1).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.warning(f"HuBERT extraction failed: {e}")
            return np.zeros(768)

    def extract_frames(self, audio_path: str, frame_size: float = 0.5) -> 'pd.DataFrame':
        """
        Extract frame-level features (DEAM-compatible format).

        Fast extraction that outputs features per frame (default 0.5s).
        Compatible with DEAM dataset format for unified training.

        Args:
            audio_path: Path to audio file
            frame_size: Frame size in seconds (default 0.5s like DEAM)

        Returns:
            DataFrame with columns per frame:
            - frameTime: time in seconds
            - rms_energy, rms_energy_delta
            - zcr
            - spectral_centroid, spectral_rolloff
            - mfcc_1..5, mfcc_1_delta..5_delta
        """
        import pandas as pd

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Calculate hop_length for desired frame_size
        # frame_size = hop_length / sr => hop_length = frame_size * sr
        hop_length = int(frame_size * sr)
        n_fft = 2048  # Standard FFT size

        # Extract frame-level features
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]

        # MFCCs (first 5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, n_fft=n_fft, hop_length=hop_length)

        # Calculate deltas
        rms_delta = np.concatenate([[0], np.diff(rms)])
        mfcc_deltas = np.vstack([
            np.concatenate([[0], np.diff(mfccs[i])]) for i in range(5)
        ])

        # Frame times
        n_frames = len(rms)
        frame_times = np.arange(n_frames) * frame_size

        # Build DataFrame
        data = {
            'frameTime': frame_times,
            'rms_energy': rms,
            'rms_energy_delta': rms_delta,
            'zcr': zcr,
            'spectral_centroid': centroid,
            'spectral_rolloff': rolloff,
        }

        # Add MFCCs
        for i in range(5):
            data[f'mfcc_{i+1}'] = mfccs[i]
            data[f'mfcc_{i+1}_delta'] = mfcc_deltas[i]

        df = pd.DataFrame(data)
        return df

    def extract_frames_fast(self, audio_path: str, frame_size: float = 0.5) -> dict:
        """
        Extract frame-level features and aggregate to track-level (fastest method).

        Similar to DEAM processing: extract per-frame, then aggregate.

        Args:
            audio_path: Path to audio file
            frame_size: Frame size in seconds

        Returns:
            Dictionary with aggregated features (ready for classifier)
        """
        frames_df = self.extract_frames(audio_path, frame_size)

        # Aggregate to track level
        features = {}

        # RMS features
        rms = frames_df['rms_energy'].values
        features['rms_energy'] = float(np.mean(rms))
        features['energy_variance'] = float(np.std(rms))
        features['low_energy'] = float(np.sum(rms < np.mean(rms)) / len(rms))
        features['peak_energy_ratio'] = float(np.max(rms) / (np.mean(rms) + 1e-6))

        # Drop detection from delta
        rms_delta = frames_df['rms_energy_delta'].values
        features['drop_strength'] = float(np.percentile(np.abs(rms_delta), 90))

        # ZCR
        features['zero_crossing_rate'] = float(frames_df['zcr'].mean())

        # Spectral features
        features['spectral_centroid'] = float(frames_df['spectral_centroid'].mean())
        features['spectral_rolloff'] = float(frames_df['spectral_rolloff'].mean())

        # Calculate brightness proxy (rolloff / centroid ratio)
        features['brightness'] = float(
            frames_df['spectral_rolloff'].mean() / (frames_df['spectral_centroid'].mean() + 1e-6)
        )

        # MFCCs
        for i in range(1, 6):
            features[f'mfcc_{i}_mean'] = float(frames_df[f'mfcc_{i}'].mean())
            features[f'mfcc_{i}_std'] = float(frames_df[f'mfcc_{i}'].std())

        # Drop detection from frames
        drop_features = self._extract_drop_features_from_frames(rms, frame_size)
        features.update(drop_features)

        return features

    def _extract_drop_features_from_frames(self, rms: np.ndarray, frame_size: float) -> dict:
        """
        Extract drop detection features from frame-level RMS.

        Args:
            rms: Frame-level RMS values
            frame_size: Frame size in seconds

        Returns:
            Drop detection features
        """
        from scipy.signal import find_peaks

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

        if mean_energy < 1e-6:
            return features

        # Find valleys and peaks
        # Distance in frames (~4 seconds minimum between drops)
        min_distance = int(4.0 / frame_size)

        valley_threshold = mean_energy - 0.3 * std_energy
        valleys, _ = find_peaks(-rms, height=-valley_threshold, distance=min_distance)

        peak_threshold = mean_energy + 0.3 * std_energy
        peaks, _ = find_peaks(rms, height=peak_threshold, distance=min_distance)

        if len(valleys) == 0 or len(peaks) == 0:
            return features

        # Match valleys to following peaks
        drop_contrasts = []
        drop_peak_energies = []

        # Max frames between valley and peak (~8 seconds)
        max_distance = int(8.0 / frame_size)

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
            contrast = (peak_energy - valley_energy) / (mean_energy + 1e-6)

            if contrast > 0.5:
                drop_contrasts.append(contrast)
                drop_peak_energies.append(peak_energy)

        if len(drop_contrasts) > 0:
            features['drop_contrast_mean'] = float(np.mean(drop_contrasts))
            features['drop_contrast_max'] = float(np.max(drop_contrasts))
            features['drop_count'] = float(len(drop_contrasts))
            features['drop_intensity'] = float(np.mean(drop_peak_energies) / mean_energy)

        return features

    def extract_batch(self, audio_paths: list, show_progress: bool = True) -> list:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of audio file paths
            show_progress: Show progress bar

        Returns:
            List of ZoneFeatures
        """
        features_list = []

        iterator = audio_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Extracting features")
            except ImportError:
                pass

        for audio_path in iterator:
            try:
                features = self.extract(audio_path)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {audio_path}: {e}")
                features_list.append(None)

        return features_list

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

    # Advanced torchaudio features (embeddings from pre-trained models)
    wav2vec2_embedding: np.ndarray  # 768-dim by default
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

        # Combine all traditional features
        traditional = np.concatenate([
            basic_features,
            temporal_features,
            spectral_features,
            hp_features
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

    def __init__(self, sample_rate: int = 22050, use_gpu: bool = True, use_embeddings: bool = False):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate
            use_gpu: Use GPU for torchaudio models if available
            use_embeddings: Use deep learning embeddings (wav2vec2, HuBERT) - slow but accurate
        """
        self.sample_rate = sample_rate
        self.use_embeddings = use_embeddings
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Load pre-trained models for embeddings (only if requested)
        if use_embeddings:
            self._load_embedding_models()
        else:
            logger.info("Skipping deep learning embeddings (use_embeddings=False)")
            self.wav2vec2_model = None
            self.hubert_model = None

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

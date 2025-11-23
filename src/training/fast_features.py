"""Fast feature extraction for zone classification (optimized version)."""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class FastZoneFeatures:
    """Lightweight features for quick zone classification."""

    # Core features (10 total - fast to compute)
    tempo: float
    tempo_confidence: float

    # Energy features
    rms_energy: float
    energy_variance: float

    # Spectral features
    spectral_centroid: float
    spectral_rolloff: float
    brightness: float

    # Temporal features
    zero_crossing_rate: float
    drop_intensity: float
    low_energy: float

    def to_vector(self, include_embeddings: bool = False) -> np.ndarray:
        """
        Convert to feature vector.

        Args:
            include_embeddings: Ignored for FastZoneFeatures (no embeddings in fast mode)

        Returns:
            Feature vector
        """
        return np.array([
            self.tempo,
            self.tempo_confidence,
            self.rms_energy,
            self.energy_variance,
            self.spectral_centroid,
            self.spectral_rolloff,
            self.brightness,
            self.zero_crossing_rate,
            self.drop_intensity,
            self.low_energy
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'tempo': self.tempo,
            'tempo_confidence': self.tempo_confidence,
            'rms_energy': self.rms_energy,
            'energy_variance': self.energy_variance,
            'spectral_centroid': self.spectral_centroid,
            'spectral_rolloff': self.spectral_rolloff,
            'brightness': self.brightness,
            'zero_crossing_rate': self.zero_crossing_rate,
            'drop_intensity': self.drop_intensity,
            'low_energy': self.low_energy
        }


class FastZoneFeatureExtractor:
    """Fast feature extractor using only essential features."""

    def __init__(self, sample_rate: int = 22050):
        """Initialize fast extractor."""
        self.sample_rate = sample_rate

    def extract(self, audio_path: str) -> FastZoneFeatures:
        """
        Extract essential features quickly.

        Args:
            audio_path: Path to audio file

        Returns:
            FastZoneFeatures object
        """
        from pathlib import Path
        logger.info(f"🎵 Loading audio file: {Path(audio_path).name}")

        # Load audio (already fast)
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        logger.info(f"✓ Audio loaded: duration={len(y)/sr:.1f}s, sr={sr}Hz")

        logger.info(f"⚙️  Computing spectral features...")
        # Pre-compute common values
        S = np.abs(librosa.stft(y))
        rms = librosa.feature.rms(y=y)[0]
        logger.info(f"✓ STFT and RMS computed")

        # 1-2. Tempo (using fast method)
        logger.info(f"⚙️  Detecting tempo...")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # Convert tempo to float if it's an array
        tempo_val = float(tempo) if isinstance(tempo, np.ndarray) else float(tempo)
        logger.info(f"✓ Tempo detected: {tempo_val:.1f} BPM ({len(beats)} beats)")

        # Calculate confidence from beat consistency
        if len(beats) > 2:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            intervals = np.diff(beat_times)
            tempo_confidence = 1.0 - min(np.std(intervals) / np.mean(intervals), 1.0) if len(intervals) > 0 else 0.5
        else:
            tempo_confidence = 0.5

        # 3. RMS energy
        rms_energy = float(np.mean(rms))

        # 4. Energy variance
        energy_variance = float(np.var(rms))

        # 5. Spectral centroid (using pre-computed STFT)
        freqs = librosa.fft_frequencies(sr=sr)
        spectral_centroid = float(np.mean([
            np.sum(freqs * S[:, i]) / (np.sum(S[:, i]) + 1e-10)
            for i in range(S.shape[1])
        ]))

        # 6. Spectral rolloff (using pre-computed STFT)
        rolloff_threshold = 0.85
        cumsum_energy = np.cumsum(S, axis=0)
        total_energy = cumsum_energy[-1, :]
        rolloff_freqs = []
        for i in range(S.shape[1]):
            threshold = rolloff_threshold * total_energy[i]
            idx = np.where(cumsum_energy[:, i] >= threshold)[0]
            if len(idx) > 0:
                rolloff_freqs.append(freqs[idx[0]])
        spectral_rolloff = float(np.mean(rolloff_freqs)) if rolloff_freqs else 0.0

        # 7. Brightness (high-frequency energy ratio)
        brightness_threshold = 3000  # Hz
        bright_bins = freqs > brightness_threshold
        brightness = float(np.sum(S[bright_bins, :]) / (np.sum(S) + 1e-10))

        # 8. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zero_crossing_rate = float(np.mean(zcr))

        # 9. Drop intensity (simplified)
        energy_diff = np.abs(np.diff(rms))
        drop_intensity = float(np.percentile(energy_diff, 90))

        # 10. Low energy percentage
        mean_energy = np.mean(rms)
        low_energy = float(np.sum(rms < mean_energy) / len(rms))

        return FastZoneFeatures(
            tempo=float(tempo) if isinstance(tempo, np.ndarray) else float(tempo),
            tempo_confidence=tempo_confidence,
            rms_energy=rms_energy,
            energy_variance=energy_variance,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            brightness=brightness,
            zero_crossing_rate=zero_crossing_rate,
            drop_intensity=drop_intensity,
            low_energy=low_energy
        )

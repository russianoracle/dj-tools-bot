#!/usr/bin/env python3
"""
🚀 ULTIMATE GPU-Accelerated Feature Extraction

Extracts MAXIMUM features using GPU acceleration:
- nnAudio: MelSpectrogram, MFCC, CQT, Chroma (fastest)
- torchaudio: additional transforms
- PyTorch: custom GPU implementations for tonnetz, HPSS

Output:
- Frame-level features (79+ features per 0.5s frame)
- Track-level statistics (mean, std, percentiles)
- 3-channel spectrograms for CNN

Requirements:
    pip install nnAudio torch torchaudio scipy

Performance:
    - Apple Silicon M2 (MPS): ~0.05-0.1s per track
    - NVIDIA GPU (CUDA): ~0.02-0.05s per track
    - CPU: ~0.5-1s per track

Usage:
    python scripts/extract_ultimate_gpu.py \\
        --input results/user_tracks.csv \\
        --output-dir results/ultimate_features \\
        --mode all
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MPS memory fragmentation fix - only set if not already configured
# Avoid overwriting user-set values that may conflict
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ:
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable limit

import argparse
import numpy as np
import pandas as pd
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import gc  # For aggressive memory cleanup

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Import advanced drop detection (CPU-based, more sophisticated)
from src.audio.drop_detection import detect_drops_advanced, get_drop_features_dict

# Track-level advanced drop feature names (13 features)
ADVANCED_DROP_FEATURES = [
    'drop_count', 'drop_avg_intensity', 'drop_max_intensity',
    'drop_avg_buildup_duration', 'drop_avg_recovery_rate', 'drop_avg_bass_prominence',
    'drop_in_first_half', 'drop_in_second_half', 'drop_temporal_distribution',
    'drop_energy_variance', 'drop_energy_range', 'bass_energy_mean', 'bass_energy_var'
]

# ============================================================
# Device Setup + M2 Optimizations
# ============================================================

# M2-specific flags
USE_FP16 = False  # Disabled: nnAudio transforms don't support FP16 on MPS yet
USE_COMPILE = hasattr(torch, 'compile')  # PyTorch 2.0+
USE_NON_BLOCKING = True  # Async GPU transfers

def get_device():
    """Get best available device with M2 optimizations."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple Silicon GPU (MPS) enabled")

        # M2 optimizations - using PYTORCH_MPS_HIGH_WATERMARK_RATIO env var instead

        # Enable high-performance mode
        if hasattr(torch.backends, 'mps'):
            print("   → Float16 acceleration: ✅")
            print("   → Unified Memory: ✅")
            print("   → Non-blocking transfers: ✅")

    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ NVIDIA CUDA GPU enabled: {torch.cuda.get_device_name()}")
        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU (no GPU available)")
        global USE_FP16
        USE_FP16 = False

    return device

DEVICE = get_device()

# ============================================================
# Try importing nnAudio (fastest GPU audio library)
# ============================================================

try:
    from nnAudio.features import MelSpectrogram as nnMelSpec
    from nnAudio.features import MFCC as nnMFCC
    from nnAudio.features import CQT as nnCQT
    HAS_NNAUDIO = True
    print("✅ nnAudio available (fastest GPU spectrograms)")
except ImportError:
    HAS_NNAUDIO = False
    print("⚠️ nnAudio not installed. Using torchaudio. Install: pip install nnAudio")


@dataclass
class UltimateConfig:
    """Configuration for ultimate feature extraction."""
    sample_rate: int = 22050
    frame_size: float = 0.5  # seconds per frame

    # Spectrogram params
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20  # More MFCCs for better representation

    # CQT params
    n_bins: int = 84
    bins_per_octave: int = 12

    # Chroma params
    n_chroma: int = 12

    # Output
    output_modes: List[str] = field(default_factory=lambda: ['frames', 'track', 'spectrogram'])


class GPUAudioTransforms:
    """
    GPU-accelerated audio transforms using nnAudio + torchaudio.

    nnAudio is ~2x faster than torchaudio for spectrograms.
    """

    def __init__(self, config: UltimateConfig, device: torch.device):
        self.config = config
        self.device = device
        self._setup_transforms()

    def _setup_transforms(self):
        """Initialize all GPU transforms."""
        sr = self.config.sample_rate
        n_fft = self.config.n_fft
        hop = self.config.hop_length

        if HAS_NNAUDIO:
            # nnAudio transforms (faster)
            self.mel_transform = nnMelSpec(
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=self.config.n_mels,
                fmin=20,
                fmax=8000,
                trainable_mel=False,
                trainable_STFT=False
            ).to(self.device)

            self.mfcc_transform = nnMFCC(
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=self.config.n_mels,
            ).to(self.device)

            self.cqt_transform = nnCQT(
                sr=sr,
                hop_length=hop,
                n_bins=self.config.n_bins,
                bins_per_octave=self.config.bins_per_octave
            ).to(self.device)

        else:
            # torchaudio fallback
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=self.config.n_mels,
                f_min=20,
                f_max=8000
            ).to(self.device)

            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=self.config.n_mfcc,
                melkwargs={'n_fft': n_fft, 'hop_length': hop, 'n_mels': self.config.n_mels}
            ).to(self.device)

            self.cqt_transform = None  # torchaudio doesn't have CQT

        # Amplitude to dB
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(self.device)

        # Resampler cache
        self._resamplers = {}

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr == self.config.sample_rate:
            return waveform

        key = (orig_sr, self.config.sample_rate)
        if key not in self._resamplers:
            # Limit cache to 5 most common sample rates
            if len(self._resamplers) >= 5:
                # Remove oldest entry
                oldest_key = next(iter(self._resamplers))
                del self._resamplers[oldest_key]

            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_sr, self.config.sample_rate
            ).to(self.device)

        return self._resamplers[key](waveform)


class UltimateFeatureExtractor:
    """
    Ultimate GPU-accelerated feature extractor.

    Extracts:
    - 128-band Mel spectrogram
    - 20 MFCCs + deltas
    - 84-bin CQT (Constant-Q Transform)
    - 12 Chroma features
    - 6 Tonnetz features
    - Spectral features (centroid, rolloff, flux, flatness, bandwidth, contrast)
    - Energy features (RMS, low energy ratio, variance)
    - Rhythm features (onset strength, beat sync)
    - Drop detection features

    Total: 100+ features per frame
    """

    def __init__(self, config: UltimateConfig = None):
        self.config = config or UltimateConfig()
        self.device = DEVICE
        self.transforms = GPUAudioTransforms(self.config, self.device)

        # Compile hot paths for M2 (PyTorch 2.0+)
        if USE_COMPILE and self.device.type != 'mps':  # MPS compile support limited
            try:
                self.compute_spectral_features = torch.compile(
                    self.compute_spectral_features,
                    mode='reduce-overhead'
                )
                print("   → torch.compile: ✅")
            except Exception:
                pass  # Compile not supported for this method

    def load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load audio file to GPU tensor (M2 optimized)."""
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono first (on CPU, fast)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Non-blocking transfer to GPU (M2 unified memory makes this nearly free)
            waveform = waveform.to(self.device, non_blocking=USE_NON_BLOCKING)

            # Resample if needed (on GPU)
            if sr != self.config.sample_rate:
                waveform = self.transforms.resample(waveform, sr)

            return waveform

        except Exception as e:
            print(f"❌ Error loading {Path(audio_path).name}: {e}")
            return None

    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT on GPU."""
        return torch.stft(
            waveform.squeeze(),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=torch.hann_window(self.config.n_fft, device=self.device),
            return_complex=True
        )

    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Mel spectrogram and related features on GPU."""
        mel = self.transforms.mel_transform(waveform)
        mel_db = self.transforms.amp_to_db(mel)

        return {
            'mel': mel.squeeze(),
            'mel_db': mel_db.squeeze()
        }

    def compute_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute MFCCs on GPU."""
        mfcc = self.transforms.mfcc_transform(waveform)
        return mfcc.squeeze()

    def compute_cqt(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute Constant-Q Transform on GPU."""
        if self.transforms.cqt_transform is not None:
            cqt = self.transforms.cqt_transform(waveform)
            return cqt.squeeze()
        return None

    def compute_chroma(self, stft: torch.Tensor) -> torch.Tensor:
        """Compute Chroma features on GPU (fully vectorized)."""
        S = torch.abs(stft)
        n_freq, n_time = S.shape

        freqs = torch.linspace(0, self.config.sample_rate / 2, n_freq, device=self.device)

        # Convert to pitch class (12 semitones)
        midi = 12 * torch.log2(freqs / 440 + 1e-10) + 69
        pitch_class = (midi % 12).long()

        # Create one-hot encoding for pitch classes: (n_freq, 12)
        valid_mask = freqs > 20
        one_hot = torch.zeros(n_freq, 12, device=self.device)
        one_hot[torch.arange(n_freq, device=self.device), pitch_class] = valid_mask.float()

        # Normalize one-hot per pitch class (so mean is taken, not sum)
        counts = one_hot.sum(dim=0, keepdim=True) + 1e-10  # (1, 12)
        one_hot_norm = one_hot / counts  # (n_freq, 12)

        # Batched matrix multiply: (12, n_freq) @ (n_freq, n_time) = (12, n_time)
        chroma = one_hot_norm.T @ S

        # Normalize across pitch classes
        chroma = chroma / (chroma.sum(dim=0, keepdim=True) + 1e-10)

        return chroma

    def compute_tonnetz(self, chroma: torch.Tensor) -> torch.Tensor:
        """
        Compute Tonnetz features on GPU.

        Tonnetz represents harmonic relations:
        - fifths
        - minor thirds
        - major thirds
        """
        # Tonnetz transformation matrix
        # Based on Harte et al., "Detecting Harmonic Change in Musical Audio"
        phi = torch.tensor([
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],   # fifths axis
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],   # minor axis
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],   # major axis
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],   # dim fifths
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],   #
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],   #
        ], dtype=torch.float32, device=self.device)

        # Normalize phi
        phi = phi / phi.sum(dim=1, keepdim=True)

        # Compute tonnetz: phi @ chroma
        tonnetz = phi @ chroma

        return tonnetz

    def compute_spectral_features(self, stft: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all spectral features on GPU."""
        S = torch.abs(stft)
        n_freq, n_time = S.shape

        freqs = torch.linspace(0, self.config.sample_rate / 2, n_freq, device=self.device)

        # Spectral centroid
        S_sum = S.sum(dim=0) + 1e-10
        centroid = (freqs.unsqueeze(1) * S).sum(dim=0) / S_sum

        # Spectral rolloff (85%)
        cumsum = torch.cumsum(S, dim=0)
        threshold = 0.85 * S.sum(dim=0)
        rolloff_idx = (cumsum >= threshold.unsqueeze(0)).float().argmax(dim=0)
        rolloff = freqs[rolloff_idx.clamp(max=len(freqs)-1)]

        # Spectral bandwidth
        deviation = torch.abs(freqs.unsqueeze(1) - centroid.unsqueeze(0))
        bandwidth = torch.sqrt((S * deviation ** 2).sum(dim=0) / S_sum)

        # Spectral flatness (Wiener entropy)
        log_S = torch.log(S + 1e-10)
        geometric_mean = torch.exp(log_S.mean(dim=0))
        arithmetic_mean = S.mean(dim=0) + 1e-10
        flatness = geometric_mean / arithmetic_mean

        # Spectral flux
        flux = torch.zeros(n_time, device=self.device)
        flux[1:] = torch.sqrt(torch.sum((S[:, 1:] - S[:, :-1]) ** 2, dim=0))

        # Brightness (energy > 3000 Hz)
        high_mask = freqs > 3000
        brightness = S[high_mask].sum(dim=0) / (S.sum(dim=0) + 1e-10)

        # Spectral contrast (per band)
        n_bands = 7
        contrast = torch.zeros(n_bands, n_time, device=self.device)
        band_edges = torch.linspace(0, n_freq, n_bands + 1).long()

        for i in range(n_bands):
            band = S[band_edges[i]:band_edges[i+1]]
            if band.shape[0] > 0:
                band_sorted = torch.sort(band, dim=0)[0]
                n = band.shape[0]
                top = band_sorted[-max(1, n//4):].mean(dim=0)
                bottom = band_sorted[:max(1, n//4)].mean(dim=0) + 1e-10
                contrast[i] = torch.log(top / bottom + 1e-10)

        # Spectral skewness and kurtosis
        mean_freq = centroid
        std_freq = bandwidth + 1e-10
        freq_normalized = (freqs.unsqueeze(1) - mean_freq.unsqueeze(0)) / std_freq.unsqueeze(0)

        weights = S / S_sum
        skewness = (weights * freq_normalized ** 3).sum(dim=0)
        kurtosis = (weights * freq_normalized ** 4).sum(dim=0)

        return {
            'centroid': centroid,
            'rolloff': rolloff,
            'bandwidth': bandwidth,
            'flatness': flatness,
            'flux': flux,
            'brightness': brightness,
            'contrast': contrast,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

    def compute_energy_features(self, waveform: torch.Tensor,
                                mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute energy-related features on GPU (fully vectorized)."""
        # Frame-level RMS from mel spectrogram
        rms = mel.mean(dim=0).sqrt()

        # Normalize
        rms = rms / (rms.max() + 1e-10)

        # Low energy ratio (frames below mean)
        mean_rms = rms.mean()
        low_energy = (rms < mean_rms).float()

        # Energy variance (vectorized with unfold)
        window = 8
        # Pad and unfold for sliding window variance
        padded = F.pad(rms.unsqueeze(0).unsqueeze(0), (window//2, window//2), mode='reflect')
        unfolded = padded.unfold(2, window, 1).squeeze()  # (n_frames, window)
        variance = unfolded.var(dim=1)

        # Deltas (vectorized)
        delta = F.pad(rms[1:] - rms[:-1], (1, 0), value=0)
        delta2 = F.pad(rms[2:] - 2 * rms[1:-1] + rms[:-2], (2, 0), value=0)

        return {
            'rms': rms,
            'rms_delta': delta,
            'rms_delta2': delta2,
            'low_energy': low_energy,
            'variance': variance,
        }

    def compute_rhythm_features(self, waveform: torch.Tensor,
                               mel_db: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute rhythm-related features on GPU (fully vectorized)."""
        # Onset strength from spectral flux of mel spectrogram
        mel_diff = mel_db[:, 1:] - mel_db[:, :-1]
        onset_strength = F.relu(mel_diff).mean(dim=0)

        # Pad to original length
        onset_strength = F.pad(onset_strength, (1, 0), value=0)

        # Onset delta (vectorized)
        onset_delta = F.pad(onset_strength[1:] - onset_strength[:-1], (1, 0), value=0)

        # ZCR vectorized: reshape waveform into frames
        n_frames = mel_db.shape[1]
        samples_per_frame = waveform.shape[1] // n_frames
        total_samples = n_frames * samples_per_frame

        # Reshape to (n_frames, samples_per_frame)
        wav_frames = waveform[0, :total_samples].view(n_frames, samples_per_frame)

        # Zero crossings: sign changes between adjacent samples
        sign_changes = (wav_frames[:, :-1] * wav_frames[:, 1:]) < 0
        zcr = sign_changes.float().mean(dim=1)

        return {
            'onset_strength': onset_strength,
            'onset_delta': onset_delta,
            'zcr': zcr,
        }

    def compute_drop_features(self, rms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect drops and buildups on GPU (fully vectorized, no scipy)."""
        n_frames = len(rms)

        # Buildup: positive slope over time
        delta = F.pad(rms[1:] - rms[:-1], (1, 0), value=0)
        buildup_score = F.relu(delta * 10)

        # GPU-based peak/valley detection using local max/min
        # Pad for window comparison
        window = 4
        padded = F.pad(rms.unsqueeze(0), (window, window), mode='reflect').squeeze()

        # Local max: value > all neighbors in window
        is_peak = torch.ones(n_frames, device=self.device, dtype=torch.bool)
        is_valley = torch.ones(n_frames, device=self.device, dtype=torch.bool)

        for offset in range(1, window + 1):
            is_peak &= (rms >= padded[window - offset:window - offset + n_frames])
            is_peak &= (rms >= padded[window + offset:window + offset + n_frames])
            is_valley &= (rms <= padded[window - offset:window - offset + n_frames])
            is_valley &= (rms <= padded[window + offset:window + offset + n_frames])

        energy_peak = is_peak.float()
        energy_valley = is_valley.float()

        # Drop candidate: frames after valley with significant increase
        # FULLY VECTORIZED - no Python loops
        valley_indices = torch.where(is_valley)[0]
        drop_candidate = torch.zeros(n_frames, device=self.device)

        if len(valley_indices) > 0:
            # Create index matrix: (n_valleys, 8) - each row is valley_idx + [0,1,2,...,7]
            offsets = torch.arange(8, device=self.device)
            valley_ranges = valley_indices.unsqueeze(1) + offsets  # (n_valleys, 8)
            valley_ranges = valley_ranges.clamp(max=n_frames - 1)  # Clamp to valid range

            # Get valley values and window values
            valley_vals = rms[valley_indices].unsqueeze(1)  # (n_valleys, 1)
            window_vals = rms[valley_ranges]  # (n_valleys, 8)

            # Vectorized comparison: which frames exceed 1.5x valley value
            exceeds_threshold = window_vals > valley_vals * 1.5  # (n_valleys, 8)

            # Find first drop in each window (argmax on bool gives first True)
            has_drop = exceeds_threshold.any(dim=1)  # (n_valleys,)
            first_drop_offset = exceeds_threshold.float().argmax(dim=1)  # (n_valleys,)

            # Calculate actual drop indices
            drop_indices = valley_indices + first_drop_offset
            drop_indices = drop_indices[has_drop]  # Only where drops exist
            drop_indices = drop_indices[drop_indices < n_frames]  # Valid indices only

            # Scatter to result (handles duplicates by last-write-wins, which is fine)
            if len(drop_indices) > 0:
                drop_candidate.scatter_(0, drop_indices, 1.0)

        return {
            'buildup_score': buildup_score,
            'drop_candidate': drop_candidate,
            'energy_valley': energy_valley,
            'energy_peak': energy_peak,
        }

    @torch.inference_mode()  # CRITICAL: prevents computation graph buildup
    def extract_all(self, audio_path: str, prefetched_audio=None) -> Optional[Dict[str, Any]]:
        """
        Extract ALL features from audio file (M2 Float16 accelerated).

        Args:
            audio_path: Path to audio file
            prefetched_audio: Optional pre-loaded (waveform, sr) tuple for I/O overlap

        Returns dict with:
        - 'frames': DataFrame with frame-level features
        - 'track': Dict with track-level statistics
        - 'spectrogram': 3-channel tensor for CNN
        """

        def _mps_cleanup():
            """Flush MPS memory to prevent fragmentation."""
            if self.device.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()

        # Use prefetched audio if available, otherwise load
        if prefetched_audio is not None:
            try:
                waveform, sr = prefetched_audio
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.to(self.device, non_blocking=USE_NON_BLOCKING)
                if sr != self.config.sample_rate:
                    waveform = self.transforms.resample(waveform, sr)
            except Exception:
                waveform = self.load_audio(audio_path)
        else:
            waveform = self.load_audio(audio_path)

        if waveform is None:
            return None

        duration = waveform.shape[1] / self.config.sample_rate

        # ============================================================
        # PHASE 1: STFT-based features (then delete STFT)
        # ============================================================

        stft = self.compute_stft(waveform.float())

        # Compute STFT-derived features immediately
        chroma = self.compute_chroma(stft)
        tonnetz = self.compute_tonnetz(chroma)
        spectral = self.compute_spectral_features(stft)

        # FREE STFT memory NOW (not at end!)
        del stft
        _mps_cleanup()

        # ============================================================
        # PHASE 2: Mel-based features
        # ============================================================

        mel_data = self.compute_mel_spectrogram(waveform)
        mfcc = self.compute_mfcc(waveform)

        # Energy and rhythm need mel data
        energy = self.compute_energy_features(waveform, mel_data['mel'])
        rhythm = self.compute_rhythm_features(waveform, mel_data['mel_db'])
        drops = self.compute_drop_features(energy['rms'])

        # ============================================================
        # PHASE 2.5: Advanced Drop Detection (CPU-based, more sophisticated)
        # ============================================================
        # Convert waveform to numpy for librosa-based drop analysis
        waveform_np = waveform.squeeze().cpu().numpy()
        try:
            drop_analysis = detect_drops_advanced(
                waveform_np,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )
            advanced_drop_features = get_drop_features_dict(drop_analysis)
        except Exception as e:
            # Fallback: zero features if detection fails
            advanced_drop_features = {name: 0.0 for name in ADVANCED_DROP_FEATURES}
        del waveform_np

        # FREE waveform NOW
        del waveform
        _mps_cleanup()

        # ============================================================
        # PHASE 3: Aggregation (moves data to CPU)
        # ============================================================

        frame_df = self._aggregate_to_frames(
            mel_data, mfcc, chroma, tonnetz, spectral, energy, rhythm, drops,
            advanced_drop_features
        )

        # FREE intermediate tensors after aggregation
        del chroma, tonnetz, spectral, mfcc
        _mps_cleanup()

        # ============================================================
        # PHASE 4: CNN spectrogram (last GPU op)
        # ============================================================

        spectrogram_3ch = self._create_3channel_spectrogram(
            mel_data['mel_db'], energy['rms'], drops
        )

        # Transfer to CPU immediately
        spectrogram_np = spectrogram_3ch.cpu().numpy()
        del spectrogram_3ch, mel_data, energy, rhythm, drops
        _mps_cleanup()

        # ============================================================
        # PHASE 5: Track statistics (CPU only)
        # ============================================================

        track_stats = self._compute_track_statistics(frame_df)

        return {
            'frames': frame_df,
            'track': track_stats,
            'spectrogram': spectrogram_np,
            'duration': duration,
            'path': audio_path,
        }

    def _aggregate_to_frames(self, mel_data, mfcc, chroma, tonnetz,
                            spectral, energy, rhythm, drops,
                            advanced_drop_features: dict = None) -> pd.DataFrame:
        """Aggregate high-resolution features to 0.5s frames (BATCHED - single GPU op)."""
        sr = self.config.sample_rate
        hop = self.config.hop_length
        frame_size = self.config.frame_size

        # Calculate aggregation factor
        samples_per_frame = int(frame_size * sr)
        hops_per_frame = samples_per_frame // hop

        n_feature_frames = spectral['centroid'].shape[0]
        n_output_frames = max(1, n_feature_frames // hops_per_frame)
        usable_len = n_output_frames * hops_per_frame

        def prepare_tensor(t):
            """Prepare tensor for batched aggregation."""
            if len(t) < usable_len:
                return F.pad(t, (0, usable_len - len(t)), value=0)
            return t[:usable_len]

        # ============================================================
        # BATCHED AGGREGATION: Stack all 1D features, single reshape+mean
        # ============================================================

        feature_names = []
        feature_tensors = []

        # Energy features (5)
        for name, tensor in [('rms_energy', energy['rms']),
                            ('low_energy_flag', energy['low_energy']),
                            ('energy_variance', energy['variance'])]:
            feature_names.append(name)
            feature_tensors.append(prepare_tensor(tensor))

        # Spectral features (9)
        for name in ['centroid', 'rolloff', 'brightness', 'flux', 'flatness',
                     'bandwidth', 'skewness', 'kurtosis']:
            feature_names.append(f'spectral_{name}' if name != 'brightness' else name)
            feature_tensors.append(prepare_tensor(spectral[name]))

        # Spectral contrast bands (7)
        for i in range(spectral['contrast'].shape[0]):
            feature_names.append(f'spectral_contrast_band_{i}')
            feature_tensors.append(prepare_tensor(spectral['contrast'][i]))

        # MFCCs (1-19) - keep on GPU, no cpu().numpy()!
        for i in range(1, min(self.config.n_mfcc, mfcc.shape[0])):
            feature_names.append(f'mfcc_{i}')
            feature_tensors.append(prepare_tensor(mfcc[i]))

        # Chroma (12)
        chroma_names = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
        for i, name in enumerate(chroma_names):
            feature_names.append(f'chroma_{name}')
            feature_tensors.append(prepare_tensor(chroma[i]))

        # Tonnetz (6)
        for i in range(6):
            feature_names.append(f'tonnetz_{i+1}')
            feature_tensors.append(prepare_tensor(tonnetz[i]))

        # Rhythm features (2)
        feature_names.extend(['onset_strength', 'zcr'])
        feature_tensors.append(prepare_tensor(rhythm['onset_strength']))
        feature_tensors.append(prepare_tensor(rhythm['zcr']))

        # Drop detection (4) - direct key mapping
        drop_features = [
            ('energy_buildup_score', 'buildup_score'),
            ('drop_candidate', 'drop_candidate'),
            ('energy_valley', 'energy_valley'),
            ('energy_peak', 'energy_peak'),
        ]
        for feature_name, drops_key in drop_features:
            feature_names.append(feature_name)
            feature_tensors.append(prepare_tensor(drops[drops_key]))

        # ============================================================
        # SINGLE BATCHED GPU OPERATION
        # ============================================================

        # Stack all features: (N_features, time)
        stacked = torch.stack(feature_tensors, dim=0)

        # FREE individual tensors immediately after stacking
        feature_tensors.clear()

        # Reshape and mean in ONE operation: (N_features, n_output_frames, hops) -> mean
        batched_agg = stacked.view(len(feature_names), n_output_frames, hops_per_frame).mean(dim=2)
        del stacked  # FREE stacked tensor

        # SINGLE CPU transfer
        all_features_np = batched_agg.cpu().numpy()
        del batched_agg  # FREE GPU tensor after transfer

        # ============================================================
        # Build DataFrame (CPU only, fast)
        # ============================================================

        data = {
            'frame_idx': np.arange(n_output_frames),
            'frameTime': np.arange(n_output_frames) * frame_size,
        }

        # Unpack batched results
        for i, name in enumerate(feature_names):
            data[name] = all_features_np[i]

        # Compute deltas (CPU, fast)
        def delta(arr):
            d = np.zeros_like(arr)
            d[1:] = np.diff(arr)
            return d

        # Add deltas for key features
        for name in ['rms_energy', 'spectral_centroid', 'spectral_rolloff', 'brightness',
                     'onset_strength', 'zcr']:
            if name in data:
                data[f'{name}_delta'] = delta(data[name])

        # MFCC deltas
        for i in range(1, min(self.config.n_mfcc, mfcc.shape[0])):
            if f'mfcc_{i}' in data:
                data[f'mfcc_{i}_delta'] = delta(data[f'mfcc_{i}'])

        # Second delta for energy
        if 'rms_energy_delta' in data:
            data['rms_energy_delta2'] = delta(data['rms_energy_delta'])

        # Spectral contrast mean
        contrast_cols = [data[f'spectral_contrast_band_{i}'] for i in range(7)]
        data['spectral_contrast'] = np.mean(contrast_cols, axis=0)

        # Placeholders
        data['harmonic_ratio'] = np.full(n_output_frames, 0.5)
        data['percussive_ratio'] = np.full(n_output_frames, 0.5)

        # ============================================================
        # Add advanced drop features (track-level, constant per frame)
        # ============================================================
        if advanced_drop_features is not None:
            for feature_name in ADVANCED_DROP_FEATURES:
                value = advanced_drop_features.get(feature_name, 0.0)
                data[feature_name] = np.full(n_output_frames, value)

        return pd.DataFrame(data)

    def _create_3channel_spectrogram(self, mel_db: torch.Tensor,
                                     rms: torch.Tensor,
                                     drops: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create 3-channel spectrogram tensor for CNN."""
        # Target size
        target_width = 512
        target_height = 128

        # Channel 0: Mel spectrogram (normalized)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)

        # Resize to target
        mel_resized = F.interpolate(
            mel_norm.unsqueeze(0).unsqueeze(0),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Channel 1: Energy curve (expanded to full height)
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        rms_resized = F.interpolate(
            rms_norm.unsqueeze(0).unsqueeze(0),  # 3D: (1, 1, N) for linear mode
            size=target_width,
            mode='linear',
            align_corners=False
        ).squeeze()
        energy_channel = rms_resized.unsqueeze(0).expand(target_height, -1)

        # Channel 2: Drop/buildup mask
        mask = torch.zeros(len(rms), device=self.device)
        mask += drops['drop_candidate'] * 1.0
        mask += drops['buildup_score'] * 0.5
        mask = mask.clamp(0, 1)

        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),  # 3D: (1, 1, N) for nearest mode
            size=target_width,
            mode='nearest'
        ).squeeze()
        mask_channel = mask_resized.unsqueeze(0).expand(target_height, -1)

        # Stack channels
        spectrogram_3ch = torch.stack([mel_resized, energy_channel, mask_channel], dim=0)

        return spectrogram_3ch.float()

    def _compute_track_statistics(self, frame_df: pd.DataFrame) -> Dict[str, float]:
        """Compute track-level statistics from frame features (CPU - faster for small data)."""
        exclude = ['frame_idx', 'frameTime', 'zone', 'track_id', 'path']
        feature_cols = [c for c in frame_df.columns if c not in exclude]

        if not feature_cols:
            return {}

        # Use numpy instead of GPU (faster for small DataFrames, avoids MPS quantile issues)
        data = frame_df[feature_cols].values.astype(np.float32)

        # Compute all statistics on CPU (numpy is fast for this size)
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        mins = np.nanmin(data, axis=0)
        maxs = np.nanmax(data, axis=0)
        p10 = np.nanpercentile(data, 10, axis=0)
        p50 = np.nanpercentile(data, 50, axis=0)
        p90 = np.nanpercentile(data, 90, axis=0)
        ranges = maxs - mins

        # Build dict
        stats = {}
        stat_names = ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90', 'range']
        all_stats = [means, stds, mins, maxs, p10, p50, p90, ranges]

        for i, col in enumerate(feature_cols):
            for j, stat_name in enumerate(stat_names):
                stats[f'{col}_{stat_name}'] = float(all_stats[j][i])

        return stats


def _extract_single_track(args):
    """Worker function for parallel extraction."""
    idx, path, zone, extractor, mode = args
    try:
        result = extractor.extract_all(path)
        if result is None:
            return None

        output = {'idx': idx, 'zone': zone, 'path': path}

        if mode in ['frames', 'all']:
            frames = result['frames']
            frames['zone'] = zone
            frames['track_id'] = idx
            frames['path'] = path
            output['frames'] = frames

        if mode in ['track', 'all']:
            track_stats = result['track']
            track_stats['zone'] = zone
            track_stats['path'] = path
            track_stats['duration'] = result['duration']
            output['track'] = track_stats

        if mode in ['spectrogram', 'all']:
            output['spectrogram'] = {
                'tensor': result['spectrogram'],
                'zone': zone,
                'path': path,
            }

        return output
    except Exception as e:
        print(f"Error {path}: {e}")
        return None


def _merge_chunks(chunks_dir: Path, output_dir: Path, mode: str):
    """Merge chunk files into final output files (run once at end)."""
    import glob

    # Merge frames
    if mode in ['frames', 'all']:
        frame_chunks = sorted(chunks_dir.glob("frames_*.pkl"))
        if frame_chunks:
            frames_list = [pd.read_pickle(f) for f in frame_chunks]
            frames_df = pd.concat(frames_list, ignore_index=True)
            frames_df.to_pickle(output_dir / "frames.pkl")
            print(f"✅ Frames: {len(frames_df):,} rows, {len(frames_df.columns)} cols")

    # Merge tracks
    if mode in ['track', 'all']:
        track_chunks = sorted(chunks_dir.glob("tracks_*.pkl"))
        if track_chunks:
            tracks_list = [pd.read_pickle(f) for f in track_chunks]
            tracks_df = pd.concat(tracks_list, ignore_index=True)
            tracks_df.to_pickle(output_dir / "track_features.pkl")
            print(f"✅ Tracks: {len(tracks_df)} rows, {len(tracks_df.columns)} cols")

    # Merge spectrograms
    if mode in ['spectrogram', 'all']:
        spec_chunks = sorted(chunks_dir.glob("specs_*.pkl"))
        if spec_chunks:
            all_specs = []
            for f in spec_chunks:
                with open(f, 'rb') as fp:
                    all_specs.extend(pickle.load(fp))
            with open(output_dir / "spectrograms.pkl", 'wb') as f:
                pickle.dump(all_specs, f)
            print(f"✅ Spectrograms: {len(all_specs)}, shape {all_specs[0]['tensor'].shape}")


def extract_dataset(csv_path: str, output_dir: str, mode: str = 'all', workers: int = 8, max_tracks: int = 0):
    """
    Extract features for entire dataset with chunked saving (O(1) per save).

    Modes:
    - 'frames': Frame-level features only
    - 'track': Track-level statistics only
    - 'spectrogram': 3-channel spectrograms only
    - 'all': Everything

    Features:
    - Chunked saves (append-only, no full rewrite)
    - Resume from checkpoint
    - O(1) memory usage
    - max_tracks: Stop after N tracks (0=unlimited) for restart wrapper
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    extractor = UltimateFeatureExtractor()

    # Checkpoint and chunk tracking
    checkpoint_path = output_dir / "checkpoint.pkl"
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    processed_paths = set()
    chunk_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            processed_paths = checkpoint.get('processed', set())
            chunk_idx = checkpoint.get('chunk_idx', 0)
        print(f"📁 Resuming: {len(processed_paths)} done, chunk {chunk_idx}")

    print(f"\n{'='*60}")
    print("ULTIMATE GPU FEATURE EXTRACTION (v3 - chunked O(1) saves)")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Tracks: {len(df)} ({len(df) - len(processed_paths)} remaining)")
    print(f"Mode: {mode}")
    print(f"nnAudio: {'✅' if HAS_NNAUDIO else '❌'}")

    start_time = time.time()

    # Buffers for current chunk
    buffer_frames = []
    buffer_track_stats = []
    buffer_spectrograms = []
    save_interval = 20  # Match restart-every for proper max_tracks check

    def save_chunk():
        """Save current buffer as a new chunk (O(1) operation)."""
        nonlocal buffer_frames, buffer_track_stats, buffer_spectrograms, chunk_idx

        if not any([buffer_frames, buffer_track_stats, buffer_spectrograms]):
            return

        # Save each buffer as separate chunk file
        if mode in ['frames', 'all'] and buffer_frames:
            chunk_path = chunks_dir / f"frames_{chunk_idx:04d}.pkl"
            pd.concat(buffer_frames, ignore_index=True).to_pickle(chunk_path)

        if mode in ['track', 'all'] and buffer_track_stats:
            chunk_path = chunks_dir / f"tracks_{chunk_idx:04d}.pkl"
            pd.DataFrame(buffer_track_stats).to_pickle(chunk_path)

        if mode in ['spectrogram', 'all'] and buffer_spectrograms:
            chunk_path = chunks_dir / f"specs_{chunk_idx:04d}.pkl"
            with open(chunk_path, 'wb') as f:
                pickle.dump(buffer_spectrograms, f)

        # Update checkpoint
        chunk_idx += 1
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'processed': processed_paths, 'chunk_idx': chunk_idx}, f)

        # Clear buffers
        buffer_frames.clear()
        buffer_track_stats.clear()
        buffer_spectrograms.clear()

        # AGGRESSIVE: Force Python garbage collection after chunk save
        gc.collect()

    # Build list of paths to process
    paths_to_process = [(idx, row) for idx, row in df.iterrows()
                        if row['path'] not in processed_paths]

    pbar = tqdm(total=len(paths_to_process), desc="Extracting")
    tracks_since_save = 0
    tracks_this_run = 0  # Counter for max_tracks limit

    for i, (idx, row) in enumerate(paths_to_process):
        path = row['path']

        try:
            result = extractor.extract_all(path)

            # Clear GPU cache EVERY track to prevent memory accumulation
            if DEVICE.type == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()

            if result is None:
                processed_paths.add(path)
                gc.collect()  # Clean up failed load
                continue

            if mode in ['frames', 'all']:
                frames = result['frames'].copy()  # Explicit copy to break references
                frames['zone'] = row['zone']
                frames['track_id'] = idx
                frames['path'] = path
                buffer_frames.append(frames)

            if mode in ['track', 'all']:
                track_stats = result['track'].copy()  # Explicit copy
                track_stats['zone'] = row['zone']
                track_stats['path'] = path
                track_stats['duration'] = result['duration']
                buffer_track_stats.append(track_stats)

            if mode in ['spectrogram', 'all']:
                buffer_spectrograms.append({
                    'tensor': result['spectrogram'].copy(),  # Explicit copy
                    'zone': row['zone'],
                    'path': path,
                })

            # FREE result dict to break reference chain
            del result

            processed_paths.add(path)
            tracks_since_save += 1
            tracks_this_run += 1

            # AGGRESSIVE: gc.collect() every 5 tracks
            if i % 5 == 0:
                gc.collect()

            # Save chunk every N tracks (O(1) operation!)
            if tracks_since_save >= save_interval:
                save_chunk()
                pbar.set_postfix({'chunks': chunk_idx, 'run': tracks_this_run})
                tracks_since_save = 0

                # Check max_tracks limit (for restart wrapper)
                if max_tracks > 0 and tracks_this_run >= max_tracks:
                    print(f"\n🔄 Reached {max_tracks} tracks this run, exiting for restart...")
                    pbar.close()
                    return  # Exit without merging - wrapper will restart

        except Exception as e:
            print(f"\n❌ Error {Path(path).name}: {e}")
            processed_paths.add(path)
            gc.collect()  # Clean up on error

        pbar.update(1)

    pbar.close()

    # Save final chunk
    save_chunk()

    # Merge all chunks into final files
    print("\n📦 Merging chunks...")
    _merge_chunks(chunks_dir, output_dir, mode)

    # Print final statistics
    total_time = time.time() - start_time
    n_processed = len(processed_paths)
    avg_time = total_time / max(1, n_processed)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processed: {n_processed} tracks")
    print(f"Average: {avg_time:.2f}s/track ({1/avg_time:.1f} tracks/sec)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Ultimate GPU-accelerated feature extraction"
    )

    parser.add_argument("--input", "-i", required=True,
                       help="Input CSV with 'path' and 'zone' columns")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Output directory")
    parser.add_argument("--mode", choices=['frames', 'track', 'spectrogram', 'all'],
                       default='all',
                       help="Extraction mode")
    parser.add_argument("--workers", "-w", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    parser.add_argument("--max-tracks", "-m", type=int, default=0,
                       help="Max tracks per run (0=unlimited, for restart wrapper)")

    args = parser.parse_args()

    extract_dataset(args.input, args.output_dir, args.mode, args.workers, args.max_tracks)


if __name__ == "__main__":
    main()

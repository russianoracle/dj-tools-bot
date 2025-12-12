"""
STFT Primitives - Foundation for all spectral analysis.

Apple Silicon M2 Optimized:
- Single STFT computation cached in STFTCache dataclass
- Contiguous memory layout for Apple Accelerate
- All downstream operations reuse this cache
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class STFTCache:
    """
    Cached STFT computation results with lazy feature computation.

    This is the foundation for all spectral analysis.
    Compute once, reuse everywhere.

    Core attributes (always computed):
        S: Magnitude spectrogram (n_freq, n_frames)
        S_db: dB-scaled spectrogram
        phase: Phase information
        freqs: Frequency bins in Hz
        times: Time points in seconds
        sr: Sample rate
        hop_length: Hop length used
        n_fft: FFT size used

    Lazy-computed features (computed on first access):
        - MFCC via get_mfcc()
        - Chroma via get_chroma()
        - Mel spectrogram via get_mel()
        - Tonnetz via get_tonnetz()

    This centralizes all librosa feature extraction in one place,
    keeping primitives pure numpy/scipy.
    """
    S: np.ndarray           # Magnitude spectrogram
    S_db: np.ndarray        # dB spectrogram
    phase: np.ndarray       # Phase
    freqs: np.ndarray       # Frequency bins
    times: np.ndarray       # Time frames
    sr: int
    hop_length: int
    n_fft: int

    # Lazy feature cache (private, not part of dataclass comparison)
    _feature_cache: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_frames(self) -> int:
        """Number of time frames."""
        return self.S.shape[1]

    @property
    def n_freq(self) -> int:
        """Number of frequency bins."""
        return self.S.shape[0]

    @property
    def duration_sec(self) -> float:
        """Duration in seconds."""
        return self.times[-1] if len(self.times) > 0 else 0.0

    @property
    def frame_duration(self) -> float:
        """Duration of each frame in seconds."""
        return self.hop_length / self.sr

    # ============== Lazy Feature Computation ==============
    # All librosa feature extraction centralized here

    def get_mel(self, n_mels: int = 128, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
        """
        Get mel spectrogram (lazy computation).

        Args:
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (default: sr/2)

        Returns:
            Mel spectrogram (n_mels, n_frames)
        """
        cache_key = f"mel_{n_mels}_{fmin}_{fmax}"
        if cache_key not in self._feature_cache:
            if fmax is None:
                fmax = self.sr / 2
            mel_basis = librosa.filters.mel(
                sr=self.sr, n_fft=self.n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
            mel_S = np.dot(mel_basis, self.S ** 2)
            self._feature_cache[cache_key] = np.ascontiguousarray(mel_S, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_mfcc(self, n_mfcc: int = 13, n_mels: int = 128) -> np.ndarray:
        """
        Get MFCCs (lazy computation).

        Args:
            n_mfcc: Number of MFCCs to return
            n_mels: Number of mel bands for computation

        Returns:
            MFCCs (n_mfcc, n_frames)
        """
        cache_key = f"mfcc_{n_mfcc}_{n_mels}"
        if cache_key not in self._feature_cache:
            mel_S = self.get_mel(n_mels=n_mels)
            log_mel = librosa.power_to_db(mel_S)
            mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc)
            self._feature_cache[cache_key] = np.ascontiguousarray(mfcc, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_chroma(self, n_chroma: int = 12) -> np.ndarray:
        """
        Get chroma features (lazy computation).

        Args:
            n_chroma: Number of chroma bins (default: 12)

        Returns:
            Chroma features (n_chroma, n_frames)
        """
        cache_key = f"chroma_{n_chroma}"
        if cache_key not in self._feature_cache:
            chroma = librosa.feature.chroma_stft(S=self.S ** 2, sr=self.sr, n_chroma=n_chroma)
            self._feature_cache[cache_key] = np.ascontiguousarray(chroma, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_tonnetz(self) -> np.ndarray:
        """
        Get tonnetz features (lazy computation).

        Returns:
            Tonnetz features (6, n_frames)
        """
        cache_key = "tonnetz"
        if cache_key not in self._feature_cache:
            chroma = self.get_chroma()
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            self._feature_cache[cache_key] = np.ascontiguousarray(tonnetz, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_mfcc_delta(self, n_mfcc: int = 13, width: int = 9, order: int = 1) -> np.ndarray:
        """
        Get MFCC deltas (lazy computation).

        Args:
            n_mfcc: Number of MFCCs
            width: Window width for delta computation
            order: Derivative order (1 or 2)

        Returns:
            Delta MFCCs (n_mfcc, n_frames)
        """
        cache_key = f"mfcc_delta_{n_mfcc}_{width}_{order}"
        if cache_key not in self._feature_cache:
            mfcc = self.get_mfcc(n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc, width=width, order=order)
            self._feature_cache[cache_key] = np.ascontiguousarray(delta, dtype=np.float32)
        return self._feature_cache[cache_key]

    def clear_feature_cache(self):
        """Clear lazy-computed feature cache to free memory."""
        self._feature_cache.clear()

    # ============== Rhythm/Beat Features ==============
    # Centralized librosa beat tracking

    def get_onset_strength(self, aggregate: bool = True) -> np.ndarray:
        """
        Get onset strength envelope (lazy computation).

        Args:
            aggregate: If True, return mean across frequency; else per-band

        Returns:
            Onset strength envelope (n_frames,) or (n_bands, n_frames)
        """
        cache_key = f"onset_strength_{aggregate}"
        if cache_key not in self._feature_cache:
            onset = librosa.onset.onset_strength(
                S=self.S_db,
                sr=self.sr,
                hop_length=self.hop_length,
                aggregate=np.mean if aggregate else None
            )
            self._feature_cache[cache_key] = np.ascontiguousarray(onset, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_tempo(self, start_bpm: float = 120.0) -> tuple:
        """
        Get tempo estimate (lazy computation).

        Args:
            start_bpm: Prior tempo for estimation

        Returns:
            Tuple of (tempo_bpm, tempo_confidence)
        """
        cache_key = f"tempo_{start_bpm}"
        if cache_key not in self._feature_cache:
            onset_env = self.get_onset_strength()
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
                start_bpm=start_bpm,
                trim=True
            )
            # Confidence from beat regularity
            if len(beats) > 1:
                intervals = np.diff(beats)
                confidence = 1.0 - min(1.0, np.std(intervals) / np.mean(intervals))
            else:
                confidence = 0.0
            self._feature_cache[cache_key] = (float(tempo), float(confidence))
        return self._feature_cache[cache_key]

    def get_beats(self, start_bpm: float = 120.0, tightness: float = 100.0) -> tuple:
        """
        Get beat positions (lazy computation).

        Args:
            start_bpm: Starting tempo estimate
            tightness: How tightly to follow tempo

        Returns:
            Tuple of (beat_frames, beat_times)
        """
        cache_key = f"beats_{start_bpm}_{tightness}"
        if cache_key not in self._feature_cache:
            onset_env = self.get_onset_strength()
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
                start_bpm=start_bpm,
                tightness=tightness,
                trim=True
            )
            beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
            self._feature_cache[cache_key] = (
                np.ascontiguousarray(beats, dtype=np.int32),
                np.ascontiguousarray(beat_times, dtype=np.float32)
            )
        return self._feature_cache[cache_key]

    def get_tempogram(self, win_length: int = 384) -> tuple:
        """
        Get tempogram (lazy computation).

        Args:
            win_length: Window length for tempogram

        Returns:
            Tuple of (tempogram, tempo_axis)
        """
        cache_key = f"tempogram_{win_length}"
        if cache_key not in self._feature_cache:
            onset_env = self.get_onset_strength()
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
                win_length=win_length
            )
            tempo_axis = librosa.tempo_frequencies(
                tempogram.shape[0],
                sr=self.sr,
                hop_length=self.hop_length
            )
            self._feature_cache[cache_key] = (
                np.ascontiguousarray(tempogram, dtype=np.float32),
                np.ascontiguousarray(tempo_axis, dtype=np.float32)
            )
        return self._feature_cache[cache_key]

    def get_plp(self, tempo_min: float = 60.0, tempo_max: float = 200.0) -> np.ndarray:
        """
        Get Predominant Local Pulse (lazy computation).

        Args:
            tempo_min: Minimum tempo to consider
            tempo_max: Maximum tempo to consider

        Returns:
            PLP curve (n_frames,)
        """
        cache_key = f"plp_{tempo_min}_{tempo_max}"
        if cache_key not in self._feature_cache:
            onset_env = self.get_onset_strength()
            try:
                plp = librosa.beat.plp(
                    onset_envelope=onset_env,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    tempo_min=tempo_min,
                    tempo_max=tempo_max
                )
            except Exception:
                plp = np.zeros(len(onset_env))
            self._feature_cache[cache_key] = np.ascontiguousarray(plp, dtype=np.float32)
        return self._feature_cache[cache_key]

    # ============== Spectral Features ==============
    # Centralized librosa spectral feature extraction

    def get_rms(self) -> np.ndarray:
        """
        Get RMS energy from spectrogram (lazy computation).

        Returns:
            RMS energy (n_frames,)
        """
        cache_key = "rms"
        if cache_key not in self._feature_cache:
            rms = librosa.feature.rms(S=self.S, hop_length=self.hop_length)[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(rms, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_centroid(self) -> np.ndarray:
        """
        Get spectral centroid (lazy computation).

        Returns:
            Spectral centroid in Hz (n_frames,)
        """
        cache_key = "spectral_centroid"
        if cache_key not in self._feature_cache:
            centroid = librosa.feature.spectral_centroid(
                S=self.S, sr=self.sr, hop_length=self.hop_length
            )[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(centroid, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_rolloff(self, roll_percent: float = 0.85) -> np.ndarray:
        """
        Get spectral rolloff (lazy computation).

        Args:
            roll_percent: Roll-off percentage (default: 0.85)

        Returns:
            Spectral rolloff in Hz (n_frames,)
        """
        cache_key = f"spectral_rolloff_{roll_percent}"
        if cache_key not in self._feature_cache:
            rolloff = librosa.feature.spectral_rolloff(
                S=self.S, sr=self.sr, roll_percent=roll_percent, hop_length=self.hop_length
            )[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(rolloff, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_flatness(self) -> np.ndarray:
        """
        Get spectral flatness (lazy computation).

        Returns:
            Spectral flatness (n_frames,)
        """
        cache_key = "spectral_flatness"
        if cache_key not in self._feature_cache:
            flatness = librosa.feature.spectral_flatness(S=self.S)[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(flatness, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_bandwidth(self, p: int = 2) -> np.ndarray:
        """
        Get spectral bandwidth (lazy computation).

        Args:
            p: Power for bandwidth computation (default: 2)

        Returns:
            Spectral bandwidth in Hz (n_frames,)
        """
        cache_key = f"spectral_bandwidth_{p}"
        if cache_key not in self._feature_cache:
            bandwidth = librosa.feature.spectral_bandwidth(
                S=self.S, sr=self.sr, p=p, hop_length=self.hop_length
            )[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(bandwidth, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_contrast(self, n_bands: int = 6, fmin: float = 200.0) -> np.ndarray:
        """
        Get spectral contrast (lazy computation).

        Args:
            n_bands: Number of frequency bands
            fmin: Minimum frequency

        Returns:
            Spectral contrast (n_bands + 1, n_frames)
        """
        cache_key = f"spectral_contrast_{n_bands}_{fmin}"
        if cache_key not in self._feature_cache:
            contrast = librosa.feature.spectral_contrast(
                S=self.S, sr=self.sr, n_bands=n_bands, fmin=fmin, hop_length=self.hop_length
            )
            self._feature_cache[cache_key] = np.ascontiguousarray(contrast, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_spectral_flux(self) -> np.ndarray:
        """
        Get spectral flux (lazy computation).

        Returns:
            Spectral flux (n_frames,)
        """
        cache_key = "spectral_flux"
        if cache_key not in self._feature_cache:
            # Spectral flux = sum of squared differences between frames
            diff = np.diff(self.S, axis=1, prepend=0)
            flux = np.sqrt(np.sum(diff ** 2, axis=0))
            self._feature_cache[cache_key] = np.ascontiguousarray(flux, dtype=np.float32)
        return self._feature_cache[cache_key]

    # ============== Audio Processing ==============
    # Centralized librosa audio processing

    def get_zcr(self, y: np.ndarray) -> np.ndarray:
        """
        Get zero crossing rate (requires original audio).

        Args:
            y: Original audio time series

        Returns:
            Zero crossing rate (n_frames,)
        """
        cache_key = "zcr"
        if cache_key not in self._feature_cache:
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            self._feature_cache[cache_key] = np.ascontiguousarray(zcr, dtype=np.float32)
        return self._feature_cache[cache_key]

    def set_audio(self, y: np.ndarray):
        """Store original audio for features that require it (HPSS, ZCR)."""
        self._feature_cache['_y'] = np.ascontiguousarray(y, dtype=np.float32)

    def get_hpss(self) -> tuple:
        """
        Get harmonic-percussive source separation (lazy computation).

        Returns:
            Tuple of (harmonic_audio, percussive_audio)
        """
        cache_key = "hpss"
        if cache_key not in self._feature_cache:
            if '_y' not in self._feature_cache:
                raise ValueError("Original audio not set. Call set_audio(y) first.")
            y_h, y_p = librosa.effects.hpss(self._feature_cache['_y'])
            self._feature_cache[cache_key] = (
                np.ascontiguousarray(y_h, dtype=np.float32),
                np.ascontiguousarray(y_p, dtype=np.float32)
            )
        return self._feature_cache[cache_key]

    def get_harmonic(self) -> np.ndarray:
        """
        Get harmonic component (lazy computation).

        Returns:
            Harmonic audio (n_samples,)
        """
        cache_key = "harmonic"
        if cache_key not in self._feature_cache:
            if '_y' not in self._feature_cache:
                raise ValueError("Original audio not set. Call set_audio(y) first.")
            y_h = librosa.effects.harmonic(self._feature_cache['_y'])
            self._feature_cache[cache_key] = np.ascontiguousarray(y_h, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_percussive(self) -> np.ndarray:
        """
        Get percussive component (lazy computation).

        Returns:
            Percussive audio (n_samples,)
        """
        cache_key = "percussive"
        if cache_key not in self._feature_cache:
            if '_y' not in self._feature_cache:
                raise ValueError("Original audio not set. Call set_audio(y) first.")
            y_p = librosa.effects.percussive(self._feature_cache['_y'])
            self._feature_cache[cache_key] = np.ascontiguousarray(y_p, dtype=np.float32)
        return self._feature_cache[cache_key]

    def get_cqt(self, n_bins: int = 84, bins_per_octave: int = 12) -> np.ndarray:
        """
        Get Constant-Q Transform (lazy computation).

        Args:
            n_bins: Number of frequency bins
            bins_per_octave: Bins per octave

        Returns:
            CQT (n_bins, n_frames)
        """
        cache_key = f"cqt_{n_bins}_{bins_per_octave}"
        if cache_key not in self._feature_cache:
            if '_y' not in self._feature_cache:
                raise ValueError("Original audio not set. Call set_audio(y) first.")
            C = np.abs(librosa.cqt(
                self._feature_cache['_y'], sr=self.sr, hop_length=self.hop_length,
                n_bins=n_bins, bins_per_octave=bins_per_octave
            ))
            self._feature_cache[cache_key] = np.ascontiguousarray(C, dtype=np.float32)
        return self._feature_cache[cache_key]

    def frames_to_time(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert frame indices to time in seconds.

        Args:
            frames: Frame indices

        Returns:
            Time values in seconds
        """
        return librosa.frames_to_time(frames, sr=self.sr, hop_length=self.hop_length)


def compute_stft(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    ref: float = 1.0,
    top_db: Optional[float] = 80.0
) -> STFTCache:
    """
    Compute STFT and cache all derived values.

    This is THE function to call at the start of analysis.
    All spectral primitives should use the returned cache.

    Args:
        y: Audio time series (mono)
        sr: Sample rate (default: 22050)
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length in samples (default: 512)
        ref: Reference amplitude for dB conversion
        top_db: Maximum dB range (clips values below)

    Returns:
        STFTCache with all precomputed values

    Example:
        >>> cache = compute_stft(y, sr=22050)
        >>> print(f"Computed {cache.n_frames} frames")
        >>> rms = compute_rms(cache.S)  # Reuse cache
    """
    # Ensure contiguous array for Apple Accelerate
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Compute complex STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Extract magnitude and phase
    S = np.abs(D)
    phase = np.angle(D)

    # Ensure contiguous for downstream operations
    S = np.ascontiguousarray(S, dtype=np.float32)

    # Compute dB spectrogram
    S_db = librosa.amplitude_to_db(S, ref=ref, top_db=top_db)
    S_db = np.ascontiguousarray(S_db, dtype=np.float32)

    # Compute frequency and time axes
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]),
        sr=sr,
        hop_length=hop_length
    )

    cache = STFTCache(
        S=S,
        S_db=S_db,
        phase=phase,
        freqs=freqs,
        times=times,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft
    )
    # Store original audio for HPSS and other audio-domain features
    cache.set_audio(y)
    return cache


def stft_to_mel(
    cache: STFTCache,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Convert STFT cache to mel spectrogram.

    Args:
        cache: Pre-computed STFT cache
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (default: sr/2)

    Returns:
        Mel spectrogram (n_mels, n_frames)
    """
    if fmax is None:
        fmax = cache.sr / 2

    mel_basis = librosa.filters.mel(
        sr=cache.sr,
        n_fft=cache.n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    mel_S = np.dot(mel_basis, cache.S ** 2)
    return np.ascontiguousarray(mel_S, dtype=np.float32)

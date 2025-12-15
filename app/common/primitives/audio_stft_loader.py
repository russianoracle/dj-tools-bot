"""
Audio STFT Loader - ЕДИНСТВЕННЫЙ файл с import librosa во ВСЁМ ПРОЕКТЕ.

ВСЕ вызовы librosa должны проходить через этот файл.
Другие модули НЕ ДОЛЖНЫ импортировать librosa напрямую.

Provides:
- load_audio() - загрузка аудио файла
- load_audio_and_stft() - загрузка + STFT
- get_duration() - длительность файла
- Все остальные librosa-зависимые операции

Apple Silicon M2 Optimized:
- Single librosa import point
- Contiguous float32 arrays
"""

import numpy as np
import librosa  # ЕДИНСТВЕННЫЙ import librosa во ВСЁМ ПРОЕКТЕ!
from typing import Optional, Tuple


def load_audio(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.

    This is the ONLY place where librosa.load() is called in the entire project.

    Args:
        path: Path to audio file
        sr: Target sample rate (None = native)
        mono: Convert to mono
        duration: Duration to load in seconds (None = entire file)
        offset: Start offset in seconds

    Returns:
        Tuple of (y, sr) where:
        - y: Audio signal (float32, contiguous)
        - sr: Sample rate
    """
    y, actual_sr = librosa.load(
        path, sr=sr, mono=mono, duration=duration, offset=offset
    )
    return np.ascontiguousarray(y, dtype=np.float32), actual_sr


def get_duration(path: str) -> float:
    """
    Get audio file duration without loading entire file.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    return librosa.get_duration(path=path)


def load_audio_and_stft(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load audio file and compute STFT.

    Convenience function that combines load_audio + compute_stft_from_audio.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        Tuple of (y, D, freqs, actual_sr) where:
        - y: Audio signal (float32)
        - D: Complex STFT (complex64)
        - freqs: Frequency bins (float32)
        - actual_sr: Actual sample rate used
    """
    # Use centralized load_audio (no duplication!)
    y, actual_sr = load_audio(path, sr=sr, mono=mono)

    # Compute STFT
    D, freqs = compute_stft_from_audio(y, sr=actual_sr, n_fft=n_fft, hop_length=hop_length)

    return y, D, freqs, actual_sr


def compute_stft_from_audio(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT from audio signal using librosa.

    For when audio is already loaded (e.g., from different source).

    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        Tuple of (D, freqs) where:
        - D: Complex STFT (complex64)
        - freqs: Frequency bins (float32)
    """
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Compute STFT with librosa
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D = np.ascontiguousarray(D, dtype=np.complex64)

    # Compute frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = np.ascontiguousarray(freqs, dtype=np.float32)

    return D, freqs


def get_beat_frames(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[float, np.ndarray]:
    """
    Get tempo and beat frames using librosa.

    This is the ONLY place for beat detection with librosa.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Tuple of (tempo, beat_frames)
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    # Handle different librosa versions (tempo may be array)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)
    return tempo, np.ascontiguousarray(beats, dtype=np.int32)


def get_onset_strength(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute onset strength envelope using librosa.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Onset strength envelope (float32)
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return np.ascontiguousarray(onset, dtype=np.float32)


def get_harmonic_percussive(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    HPSS (Harmonic-Percussive Source Separation) using librosa.

    Args:
        y: Audio signal

    Returns:
        Tuple of (y_harmonic, y_percussive)
    """
    y_h, y_p = librosa.effects.hpss(y)
    return (
        np.ascontiguousarray(y_h, dtype=np.float32),
        np.ascontiguousarray(y_p, dtype=np.float32)
    )


def get_cqt(y: np.ndarray, sr: int, hop_length: int = 512,
            n_bins: int = 84, bins_per_octave: int = 12) -> np.ndarray:
    """
    Compute Constant-Q Transform using librosa.

    Args:
        y: Audio signal
        sr: Sample rate
        hop_length: Hop length
        n_bins: Number of frequency bins
        bins_per_octave: Bins per octave

    Returns:
        CQT magnitude (float32)
    """
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length,
                           n_bins=n_bins, bins_per_octave=bins_per_octave))
    return np.ascontiguousarray(C, dtype=np.float32)


def get_tempogram(onset_env: np.ndarray, sr: int, hop_length: int = 512,
                  win_length: int = 384) -> np.ndarray:
    """
    Compute tempogram using librosa.

    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Hop length
        win_length: Window length

    Returns:
        Tempogram (float32)
    """
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr,
        hop_length=hop_length, win_length=win_length
    )
    return np.ascontiguousarray(tempogram, dtype=np.float32)


def frames_to_time(frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Convert frame indices to time using librosa.

    Args:
        frames: Frame indices
        sr: Sample rate
        hop_length: Hop length

    Returns:
        Time in seconds (float32)
    """
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return np.ascontiguousarray(times, dtype=np.float32)


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio signal using librosa.

    Args:
        y: Audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio (float32)
    """
    if orig_sr == target_sr:
        return np.ascontiguousarray(y, dtype=np.float32)
    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return np.ascontiguousarray(y_resampled, dtype=np.float32)

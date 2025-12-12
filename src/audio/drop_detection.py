"""
🎵 Advanced Drop Detection for Energy Zone Classification

Multi-level drop detection algorithm that captures:
1. Energy buildups (tension before drop)
2. Drop events (sudden energy release)
3. Bass prominence
4. Recovery patterns

This is critical for PURPLE zone classification.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DropEvent:
    """Represents a single drop event in the track."""
    frame_idx: int           # Frame index where drop occurs
    time_sec: float          # Time in seconds
    intensity: float         # Drop intensity (0-1)
    buildup_duration: float  # Duration of buildup before drop (seconds)
    recovery_rate: float     # How fast energy recovers after drop
    bass_prominence: float   # Bass energy at drop moment


@dataclass
class DropAnalysis:
    """Complete drop analysis for a track."""
    # Core metrics
    drop_count: int
    drop_events: List[DropEvent]

    # Aggregate features
    avg_drop_intensity: float
    max_drop_intensity: float
    avg_buildup_duration: float
    avg_recovery_rate: float
    avg_bass_prominence: float

    # Temporal distribution
    drops_in_first_half: int
    drops_in_second_half: int
    drop_temporal_distribution: float  # 0=all early, 1=all late

    # Energy dynamics
    energy_variance: float
    energy_range: float
    bass_energy_mean: float
    bass_energy_var: float


def detect_drops_advanced(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    min_drop_distance_sec: float = 2.0,
    buildup_window_sec: float = 4.0,
    recovery_window_sec: float = 2.0
) -> DropAnalysis:
    """
    Multi-level drop detection algorithm.

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        hop_length: Hop length for feature extraction
        min_drop_distance_sec: Minimum time between drops
        buildup_window_sec: Window to look for buildup before drop
        recovery_window_sec: Window to measure recovery after drop

    Returns:
        DropAnalysis with all metrics
    """
    # Convert time to frames
    min_drop_frames = int(min_drop_distance_sec * sr / hop_length)
    buildup_frames = int(buildup_window_sec * sr / hop_length)
    recovery_frames = int(recovery_window_sec * sr / hop_length)

    # ==== Level 1: Compute multi-band energy ====

    # Mel spectrogram for frequency analysis
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Frequency bands
    # Bass: 0-250 Hz (mels 0-10 approximately)
    bass_energy = np.mean(S_db[0:12], axis=0)

    # Low-mid: 250-500 Hz (kick drums)
    kick_energy = np.mean(S_db[12:20], axis=0)

    # Mid: 500-2000 Hz (snare, vocals)
    mid_energy = np.mean(S_db[20:50], axis=0)

    # High-mid: 2000-8000 Hz (hi-hats, cymbals, brightness)
    high_energy = np.mean(S_db[50:90], axis=0)

    # ==== Level 2: Compute overall RMS energy ====
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # ==== Level 3: Combined energy with weights ====
    # Weight bass and kick higher for drop detection
    combined_energy = (
        0.35 * _normalize(bass_energy) +
        0.25 * _normalize(kick_energy) +
        0.20 * _normalize(mid_energy) +
        0.10 * _normalize(high_energy) +
        0.10 * _normalize(rms)
    )

    # ==== Level 4: Compute energy derivative ====
    energy_gradient = np.gradient(combined_energy)
    energy_gradient_smooth = _smooth(energy_gradient, window=5)

    # ==== Level 5: Find drop candidates ====
    # Drop = sharp negative gradient (energy falling)
    # followed by sharp positive gradient (energy release)

    # Find local minima in energy (potential drop points)
    negative_peaks, _ = find_peaks(-combined_energy, distance=min_drop_frames)

    # Find local maxima (potential release points)
    positive_peaks, _ = find_peaks(combined_energy, distance=min_drop_frames)

    # ==== Level 6: Validate drops ====
    drop_events = []
    n_frames = len(combined_energy)

    for valley_idx in negative_peaks:
        # Check for buildup before (energy decreasing)
        buildup_start = max(0, valley_idx - buildup_frames)
        buildup_region = combined_energy[buildup_start:valley_idx]

        if len(buildup_region) < 3:
            continue

        # Buildup should show energy decrease
        buildup_trend = np.mean(np.gradient(buildup_region))

        # Check for release after (energy spike)
        release_end = min(n_frames, valley_idx + recovery_frames)
        release_region = combined_energy[valley_idx:release_end]

        if len(release_region) < 3:
            continue

        # Find peak after valley
        future_peaks = positive_peaks[(positive_peaks > valley_idx) &
                                       (positive_peaks < valley_idx + recovery_frames * 2)]

        if len(future_peaks) == 0:
            continue

        peak_idx = future_peaks[0]
        peak_energy = combined_energy[peak_idx]
        valley_energy = combined_energy[valley_idx]

        # Calculate drop intensity
        energy_drop = peak_energy - valley_energy
        if energy_drop < 0.15:  # Minimum drop threshold
            continue

        # Calculate metrics
        time_sec = valley_idx * hop_length / sr
        intensity = min(1.0, energy_drop / 0.5)  # Normalize to 0-1

        # Buildup duration
        buildup_duration = (valley_idx - buildup_start) * hop_length / sr

        # Recovery rate (how fast energy comes back)
        recovery_time = (peak_idx - valley_idx) * hop_length / sr
        recovery_rate = energy_drop / max(recovery_time, 0.1)

        # Bass prominence at drop
        bass_at_drop = bass_energy[min(peak_idx, len(bass_energy) - 1)]
        bass_prominence = (bass_at_drop - np.min(bass_energy)) / (np.max(bass_energy) - np.min(bass_energy) + 1e-6)

        drop_events.append(DropEvent(
            frame_idx=valley_idx,
            time_sec=time_sec,
            intensity=intensity,
            buildup_duration=buildup_duration,
            recovery_rate=recovery_rate,
            bass_prominence=bass_prominence
        ))

    # ==== Level 7: Compute aggregate features ====
    drop_count = len(drop_events)

    if drop_count > 0:
        avg_drop_intensity = np.mean([d.intensity for d in drop_events])
        max_drop_intensity = np.max([d.intensity for d in drop_events])
        avg_buildup_duration = np.mean([d.buildup_duration for d in drop_events])
        avg_recovery_rate = np.mean([d.recovery_rate for d in drop_events])
        avg_bass_prominence = np.mean([d.bass_prominence for d in drop_events])

        # Temporal distribution
        track_duration_sec = len(y) / sr
        mid_point = track_duration_sec / 2
        drops_in_first_half = sum(1 for d in drop_events if d.time_sec < mid_point)
        drops_in_second_half = drop_count - drops_in_first_half
        drop_temporal_distribution = drops_in_second_half / drop_count
    else:
        avg_drop_intensity = 0.0
        max_drop_intensity = 0.0
        avg_buildup_duration = 0.0
        avg_recovery_rate = 0.0
        avg_bass_prominence = 0.0
        drops_in_first_half = 0
        drops_in_second_half = 0
        drop_temporal_distribution = 0.5

    # Energy dynamics
    energy_variance = float(np.var(combined_energy))
    energy_range = float(np.max(combined_energy) - np.min(combined_energy))
    bass_energy_mean = float(np.mean(bass_energy))
    bass_energy_var = float(np.var(bass_energy))

    return DropAnalysis(
        drop_count=drop_count,
        drop_events=drop_events,
        avg_drop_intensity=avg_drop_intensity,
        max_drop_intensity=max_drop_intensity,
        avg_buildup_duration=avg_buildup_duration,
        avg_recovery_rate=avg_recovery_rate,
        avg_bass_prominence=avg_bass_prominence,
        drops_in_first_half=drops_in_first_half,
        drops_in_second_half=drops_in_second_half,
        drop_temporal_distribution=drop_temporal_distribution,
        energy_variance=energy_variance,
        energy_range=energy_range,
        bass_energy_mean=bass_energy_mean,
        bass_energy_var=bass_energy_var
    )


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-1 range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def get_drop_features_dict(analysis: DropAnalysis) -> dict:
    """Convert DropAnalysis to feature dictionary for ML."""
    return {
        'drop_count': analysis.drop_count,
        'drop_avg_intensity': analysis.avg_drop_intensity,
        'drop_max_intensity': analysis.max_drop_intensity,
        'drop_avg_buildup_duration': analysis.avg_buildup_duration,
        'drop_avg_recovery_rate': analysis.avg_recovery_rate,
        'drop_avg_bass_prominence': analysis.avg_bass_prominence,
        'drop_in_first_half': analysis.drops_in_first_half,
        'drop_in_second_half': analysis.drops_in_second_half,
        'drop_temporal_distribution': analysis.drop_temporal_distribution,
        'drop_energy_variance': analysis.energy_variance,
        'drop_energy_range': analysis.energy_range,
        'bass_energy_mean': analysis.bass_energy_mean,
        'bass_energy_var': analysis.bass_energy_var,
    }


# Quick test
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Analyzing: {audio_path}")

        y, sr = librosa.load(audio_path, sr=22050)
        analysis = detect_drops_advanced(y, sr)

        print(f"\n📊 Drop Analysis Results:")
        print(f"   Drop count: {analysis.drop_count}")
        print(f"   Avg intensity: {analysis.avg_drop_intensity:.3f}")
        print(f"   Max intensity: {analysis.max_drop_intensity:.3f}")
        print(f"   Avg buildup: {analysis.avg_buildup_duration:.2f}s")
        print(f"   Avg recovery rate: {analysis.avg_recovery_rate:.3f}")
        print(f"   Bass prominence: {analysis.avg_bass_prominence:.3f}")
        print(f"   Temporal dist: {analysis.drop_temporal_distribution:.2f} (1=all late)")
        print(f"   Energy variance: {analysis.energy_variance:.4f}")

        if analysis.drop_events:
            print(f"\n🎵 Drop Events:")
            for i, drop in enumerate(analysis.drop_events[:5], 1):
                print(f"   {i}. t={drop.time_sec:.1f}s, intensity={drop.intensity:.2f}, "
                      f"buildup={drop.buildup_duration:.1f}s")
    else:
        print("Usage: python drop_detection.py <audio_file.mp3>")

#!/usr/bin/env python3
"""
Train XGBoost Beat Grid Detector using Universal Training Pipeline.

Architecture (Primitives -> Tasks -> Pipelines):
- Uses TrainingPipeline from src/training/pipelines/training.py
- Uses BeatGridGroundTruthLoader for Rekordbox ground truth
- Uses existing AudioLoader, CacheRepository
- Follows project's Single Responsibility Principle

The model learns to:
1. Correct BPM detection (classify between candidate tempos)
2. Find first downbeat position
3. Improve phrase boundary detection

Usage:
    python scripts/train_beatgrid_detector.py
    python scripts/train_beatgrid_detector.py --sample 100 --no-cache
    python scripts/train_beatgrid_detector.py --task bpm  # BPM correction only
    python scripts/train_beatgrid_detector.py --task downbeat  # Downbeat detection
"""

import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === Import from project architecture ===
from src.training.pipelines import (
    TrainingPipeline,
    BaseGroundTruthLoader,
    XGBoostTrainer,
    TrainingResult,
)
from src.audio.loader import AudioLoader
from src.core.primitives import STFTCache


# =============================================================================
# Beat Grid Ground Truth Loader (follows BaseGroundTruthLoader interface)
# =============================================================================

class BeatGridGroundTruthLoader(BaseGroundTruthLoader):
    """
    Load beat grid ground truth from Rekordbox-extracted JSON.

    Expected JSON format (from extract_all_beatgrid.py):
    [
        {
            "file": "/path/to/audio.mp3",
            "bpm": 128.0,
            "first_downbeat_sec": 0.5,
            "beat_times": [...],
            "downbeat_times": [...],
            "phrase_times": [...],
            ...
        },
        ...
    ]
    """

    def __init__(self, gt_path: str, max_tracks: Optional[int] = None):
        self.gt_path = gt_path
        self.max_tracks = max_tracks

    def load(self) -> Dict[str, Dict]:
        """Load ground truth in TrainingPipeline format."""
        with open(self.gt_path) as f:
            data = json.load(f)

        result = {}
        for i, track in enumerate(data):
            if self.max_tracks and i >= self.max_tracks:
                break

            file_path = track.get('file', '')
            if not file_path or not Path(file_path).exists():
                continue

            # Create safe key from filename
            name = Path(file_path).stem
            safe_name = name.lower().replace(' ', '_').replace('-', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:50]

            # Labels = downbeat times (for downbeat detection)
            # Or BPM as single label (for BPM classification)
            result[safe_name] = {
                'file': file_path,
                'labels': track.get('downbeat_times', []),  # Primary labels
                'metadata': {
                    'bpm': track.get('bpm', 0),
                    'first_downbeat_sec': track.get('first_downbeat_sec', 0),
                    'beat_times': track.get('beat_times', []),
                    'phrase_times': track.get('phrase_times', []),
                    'n_beats': track.get('n_beats', 0),
                    'n_bars': track.get('n_bars', 0),
                },
            }

        logger.info(f"Loaded {len(result)} tracks from {self.gt_path}")
        return result


# =============================================================================
# Feature Extractors (pure functions that Tasks would use)
# =============================================================================

def extract_tempo_features(y: np.ndarray, sr: int, metadata: dict) -> List[dict]:
    """
    Extract features for BPM classification.

    Creates samples at different BPM candidates and extracts features
    to train a model that selects the correct BPM.
    """
    import librosa

    gt_bpm = metadata.get('bpm', 128)
    duration = len(y) / sr

    # Detect tempo candidates
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_candidates = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=None
    )

    # Get top tempo estimate
    tempo_detected, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo_detected, '__len__'):
        tempo_detected = float(tempo_detected[0]) if len(tempo_detected) > 0 else 120.0

    # Generate candidate BPMs (detected, half, double, harmonics)
    candidates = [
        tempo_detected,
        tempo_detected * 0.5,
        tempo_detected * 2.0,
        tempo_detected * 0.75,
        tempo_detected * 1.5,
    ]

    # Spectral features for tempo analysis
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))

    # Rhythm features
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo_strength = np.max(tempogram, axis=0).mean()

    samples = []
    for cand_bpm in candidates:
        if cand_bpm < 60 or cand_bpm > 200:
            continue

        # Is this candidate correct? (within 2 BPM of GT)
        is_correct = abs(cand_bpm - gt_bpm) < 2.0

        # Features for this candidate
        beat_interval = 60.0 / cand_bpm
        expected_beats = duration / beat_interval

        # Tempo ratio features
        ratio_to_detected = cand_bpm / tempo_detected if tempo_detected > 0 else 1.0

        sample = {
            'time': 0,  # Dummy for compatibility
            'candidate_bpm': cand_bpm,
            'detected_bpm': tempo_detected,
            'gt_bpm': gt_bpm,
            'bpm_ratio': ratio_to_detected,
            'bpm_diff_detected': cand_bpm - tempo_detected,
            'is_half': 1 if abs(ratio_to_detected - 0.5) < 0.1 else 0,
            'is_double': 1 if abs(ratio_to_detected - 2.0) < 0.1 else 0,
            'is_base': 1 if abs(ratio_to_detected - 1.0) < 0.1 else 0,
            'expected_beats': expected_beats,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'tempo_strength': tempo_strength,
            'duration': duration,
            'is_correct': 1 if is_correct else 0,
        }
        samples.append(sample)

    return samples


def extract_downbeat_features(y: np.ndarray, sr: int, metadata: dict) -> List[dict]:
    """
    Extract features for downbeat detection.

    Samples at beat positions and classifies which are downbeats (beat 1).
    """
    import librosa

    gt_bpm = metadata.get('bpm', 128)
    gt_downbeats = metadata.get('beat_times', [])[:100]  # Limit for speed

    if len(gt_downbeats) < 4:
        return []

    duration = len(y) / sr
    beat_interval = 60.0 / gt_bpm

    # Detect beats using librosa
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 4:
        return []

    # STFT for spectral features
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    hop_length = 512

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Spectral contrast (bass vs high frequencies)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    bass_band = contrast[0]  # Lowest frequency band

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    samples = []

    for i, beat_time in enumerate(beat_times[:200]):  # Limit samples
        # Time to frame
        frame = int(beat_time * sr / hop_length)
        frame = min(frame, len(rms) - 1)

        # Window features (4 beats context)
        window_start = max(0, frame - 4)
        window_end = min(len(rms), frame + 4)

        # Local energy features
        local_rms = rms[window_start:window_end]
        local_bass = bass_band[window_start:window_end] if frame < len(bass_band) else np.zeros(8)

        # Is this beat a downbeat? (nearest neighbor to GT downbeats)
        distances = np.abs(np.array(gt_downbeats) - beat_time)
        min_dist = np.min(distances) if len(distances) > 0 else float('inf')
        is_downbeat = min_dist < (beat_interval * 0.5)  # Within half beat

        # Beat position in bar (estimated)
        beat_in_bar = (i % 4) + 1

        sample = {
            'time': beat_time,
            'beat_index': i,
            'beat_in_bar_est': beat_in_bar,
            'rms_local_mean': np.mean(local_rms) if len(local_rms) > 0 else 0,
            'rms_local_std': np.std(local_rms) if len(local_rms) > 0 else 0,
            'rms_at_beat': rms[frame] if frame < len(rms) else 0,
            'bass_at_beat': bass_band[frame] if frame < len(bass_band) else 0,
            'bass_local_mean': np.mean(local_bass) if len(local_bass) > 0 else 0,
            'onset_strength': onset_env[frame] if frame < len(onset_env) else 0,
            'is_first_beat': 1 if i == 0 else 0,
            'time_mod_4beats': (beat_time % (beat_interval * 4)) / (beat_interval * 4),
            'is_positive': 1 if is_downbeat else 0,  # Label
        }
        samples.append(sample)

    return samples


# =============================================================================
# Feature columns for each task
# =============================================================================

BPM_FEATURE_COLS = [
    'candidate_bpm', 'detected_bpm', 'bpm_ratio', 'bpm_diff_detected',
    'is_half', 'is_double', 'is_base', 'expected_beats',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'tempo_strength', 'duration',
]

DOWNBEAT_FEATURE_COLS = [
    'beat_index', 'beat_in_bar_est', 'rms_local_mean', 'rms_local_std',
    'rms_at_beat', 'bass_at_beat', 'bass_local_mean', 'onset_strength',
    'is_first_beat', 'time_mod_4beats',
]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Beat Grid Detector')
    parser.add_argument('--gt-path', default='data/beatgrid_training.json',
                        help='Path to beat grid ground truth JSON')
    parser.add_argument('--output', '-o', default='models/beatgrid_detector.pkl',
                        help='Output model path')
    parser.add_argument('--task', choices=['bpm', 'downbeat'], default='downbeat',
                        help='Which model to train')
    parser.add_argument('--sample', '-n', type=int, help='Limit tracks for testing')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Label tolerance in seconds (for downbeat)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')

    args = parser.parse_args()

    print("=" * 60)
    print(f"BEAT GRID TRAINING ({args.task.upper()})")
    print("=" * 60)

    # Select feature extractor and columns based on task
    if args.task == 'bpm':
        feature_extractor = extract_tempo_features
        feature_cols = BPM_FEATURE_COLS
        label_col = 'is_correct'
        output_path = args.output.replace('.pkl', '_bpm.pkl')
    else:
        feature_extractor = extract_downbeat_features
        feature_cols = DOWNBEAT_FEATURE_COLS
        label_col = 'is_positive'
        output_path = args.output.replace('.pkl', '_downbeat.pkl')

    # Build pipeline using project architecture
    pipeline = TrainingPipeline(
        gt_loader=BeatGridGroundTruthLoader(args.gt_path, max_tracks=args.sample),
        feature_extractor=feature_extractor,
        trainer=XGBoostTrainer(feature_cols=feature_cols),
        cache_dir='cache',
        sr=22050,
        label_tolerance=args.tolerance,
    )

    # Run training
    result = pipeline.run(
        output_path=output_path,
        project_root=PROJECT_ROOT,
        use_cache=not args.no_cache,
    )

    print(f"\nFinal Metrics: F1={result.metrics['f1']:.3f}, CV_F1={result.metrics['cv_f1']:.3f}")
    print(f"Model saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

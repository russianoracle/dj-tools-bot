#!/usr/bin/env python3
"""
Train Beat Grid ML Models using Universal Training Pipeline.

ARCHITECTURE COMPLIANCE (Primitives → Tasks → Pipelines):
- Uses create_audio_context() for AudioContext creation
- Uses STFTCache for ALL feature extraction (lazy computation)
- Feature extractors receive AudioContext, NOT raw audio
- Uses CacheRepository for caching (single entry point)
- ALL OPERATIONS FULLY VECTORIZED (numpy broadcasting, no Python loops)

Key Changes from Legacy:
- NO direct librosa calls in extractors
- NO compute_stft() - use create_audio_context() instead
- NO librosa.beat.beat_track() - use stft_cache.get_beats()
- ALL features via stft_cache.get_*() methods

Models trained:
1. Downbeat detector - classifies which beats are downbeats (beat 1 of bar)
2. BPM corrector - selects correct tempo from candidates (octave disambiguation)

Usage:
    python scripts/train_beatgrid_ml.py --task downbeat --sample 500
    python scripts/train_beatgrid_ml.py --task bpm --full
    python scripts/train_beatgrid_ml.py --task all --verbose
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === Import from project architecture ===
from src.training.pipelines import (
    TrainingPipeline,
    BaseGroundTruthLoader,
    XGBoostTrainer,
    TrainingResult,
    FeatureDataset,
    LabelAssigner,
)
from src.audio.loader import AudioLoader
from src.core.cache import CacheRepository
from src.core.tasks.base import AudioContext, create_audio_context

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
            yield item
        print()


# =============================================================================
# Setup Logging
# =============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging with console and file handlers."""
    logger = logging.getLogger('beatgrid_training')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Beat Grid Ground Truth Loader (extends BaseGroundTruthLoader)
# =============================================================================

class BeatGridGroundTruthLoader(BaseGroundTruthLoader):
    """
    Load beat grid ground truth from Rekordbox-extracted JSON.
    Extends BaseGroundTruthLoader following project architecture.
    """

    def __init__(
        self,
        gt_path: str,
        max_tracks: Optional[int] = None,
        require_file: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.gt_path = gt_path
        self.max_tracks = max_tracks
        self.require_file = require_file
        self.logger = logger or logging.getLogger(__name__)

    def load(self) -> Dict[str, Dict]:
        """Load ground truth following BaseGroundTruthLoader interface."""
        self.logger.info(f"Loading GT from {self.gt_path}")

        with open(self.gt_path) as f:
            data = json.load(f)

        result = {}
        skipped = 0

        for i, track in enumerate(data):
            if self.max_tracks and len(result) >= self.max_tracks:
                break

            file_path = track.get('file', '')

            if self.require_file:
                if not file_path or not Path(file_path).exists():
                    skipped += 1
                    continue

            name = Path(file_path).stem
            safe_name = name.lower().replace(' ', '_').replace('-', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:50]

            if safe_name in result:
                safe_name = f"{safe_name}_{i}"

            result[safe_name] = {
                'file': file_path,
                'labels': track.get('downbeat_times', []),
                'metadata': {
                    'bpm': track.get('bpm', 0),
                    'first_downbeat_sec': track.get('first_downbeat_sec', 0),
                    'beat_times': track.get('beat_times', []),
                    'phrase_times': track.get('phrase_times', []),
                    'n_beats': track.get('n_beats', 0),
                    'n_bars': track.get('n_bars', 0),
                    'tempo_mean': track.get('tempo_mean', track.get('bpm', 0)),
                },
            }

        self.logger.info(f"Loaded {len(result)} tracks (skipped {skipped} without files)")
        return result


# =============================================================================
# VECTORIZED Feature Extractors (AudioContext-based)
# =============================================================================

def extract_downbeat_features_vectorized(context: AudioContext, metadata: dict) -> List[dict]:
    """
    Extract features for downbeat detection - FULLY VECTORIZED.

    ARCHITECTURE COMPLIANT:
    - Uses AudioContext (not raw y, sr)
    - Uses STFTCache.get_*() for ALL features (lazy, cached)
    - NO direct librosa calls

    Args:
        context: AudioContext with pre-computed STFTCache
        metadata: Dict with bpm, beat_times (ground truth)

    Returns:
        List of feature dicts (converted from vectorized arrays)
    """
    stft_cache = context.stft_cache
    sr = context.sr
    hop_length = stft_cache.hop_length

    gt_bpm = metadata.get('bpm', 128)
    gt_downbeats = np.ascontiguousarray(metadata.get('beat_times', [])[:100], dtype=np.float32)

    if len(gt_downbeats) < 4:
        return []

    beat_interval = 60.0 / gt_bpm

    # === ALL features from STFTCache (lazy, cached) ===
    _, beat_times = stft_cache.get_beats(start_bpm=gt_bpm)  # librosa beat tracking
    beat_times = np.ascontiguousarray(beat_times, dtype=np.float32)

    if len(beat_times) < 4:
        return []

    # Limit beats
    beat_times = beat_times[:200]
    n_beats = len(beat_times)

    # === VECTORIZED feature computation from STFTCache ===
    rms = stft_cache.get_rms()  # (n_frames,)
    contrast = stft_cache.get_spectral_contrast()  # (7, n_frames)
    bass_band = contrast[0]
    onset_env = stft_cache.get_onset_strength()  # (n_frames,)

    # Convert beat times to frame indices - VECTORIZED
    beat_frames_idx = (beat_times * sr / hop_length).astype(np.int32)
    beat_frames_idx = np.clip(beat_frames_idx, 0, len(rms) - 1)

    # === Extract features at beats - VECTORIZED ===
    rms_at_beat = rms[beat_frames_idx]
    bass_at_beat = bass_band[np.clip(beat_frames_idx, 0, len(bass_band) - 1)]
    onset_at_beat = onset_env[np.clip(beat_frames_idx, 0, len(onset_env) - 1)]

    # Local window features - VECTORIZED with broadcasting
    window_half = 4
    offsets = np.arange(-window_half, window_half)
    window_frames = beat_frames_idx[:, np.newaxis] + offsets[np.newaxis, :]
    window_frames = np.clip(window_frames, 0, len(rms) - 1)

    local_rms = rms[window_frames]
    rms_local_mean = np.mean(local_rms, axis=1)
    rms_local_std = np.std(local_rms, axis=1)

    window_frames_bass = np.clip(window_frames, 0, len(bass_band) - 1)
    local_bass = bass_band[window_frames_bass]
    bass_local_mean = np.mean(local_bass, axis=1)

    # === Label assignment - VECTORIZED nearest neighbor ===
    diffs = np.abs(beat_times[:, np.newaxis] - gt_downbeats[np.newaxis, :])
    min_dist_per_beat = np.min(diffs, axis=1)
    is_downbeat = (min_dist_per_beat < (beat_interval * 0.5)).astype(np.int32)

    # === Build feature arrays - all VECTORIZED ===
    beat_indices = np.arange(n_beats, dtype=np.int32)
    beat_in_bar_est = (beat_indices % 4) + 1
    is_first_beat = (beat_indices == 0).astype(np.int32)
    time_mod_4beats = (beat_times % (beat_interval * 4)) / (beat_interval * 4)

    # Convert to list of dicts (required by TrainingPipeline interface)
    return [
        {
            'time': float(beat_times[i]),
            'beat_index': int(beat_indices[i]),
            'beat_in_bar_est': int(beat_in_bar_est[i]),
            'rms_local_mean': float(rms_local_mean[i]),
            'rms_local_std': float(rms_local_std[i]),
            'rms_at_beat': float(rms_at_beat[i]),
            'bass_at_beat': float(bass_at_beat[i]),
            'bass_local_mean': float(bass_local_mean[i]),
            'onset_strength': float(onset_at_beat[i]),
            'is_first_beat': int(is_first_beat[i]),
            'time_mod_4beats': float(time_mod_4beats[i]),
            'is_positive': int(is_downbeat[i]),
        }
        for i in range(n_beats)
    ]


def extract_bpm_features_vectorized(context: AudioContext, metadata: dict) -> List[dict]:
    """
    Extract features for BPM correction - FULLY VECTORIZED.

    ARCHITECTURE COMPLIANT:
    - Uses AudioContext (not raw y, sr)
    - Uses STFTCache.get_*() for ALL features
    - NO direct librosa calls

    Args:
        context: AudioContext with pre-computed STFTCache
        metadata: Dict with bpm (ground truth)

    Returns:
        List of feature dicts
    """
    stft_cache = context.stft_cache
    sr = context.sr
    duration = context.duration_sec

    gt_bpm = metadata.get('bpm', 128)

    # === ALL features from STFTCache ===
    tempo_detected, _ = stft_cache.get_tempo(start_bpm=gt_bpm)

    # Ensure tempo is a scalar
    if hasattr(tempo_detected, '__len__'):
        tempo_detected = float(tempo_detected[0]) if len(tempo_detected) > 0 else 120.0
    else:
        tempo_detected = float(tempo_detected)

    # Generate candidate BPMs - VECTORIZED
    multipliers = np.array([1.0, 0.5, 2.0, 0.75, 1.5], dtype=np.float32)
    candidates = tempo_detected * multipliers

    # Filter valid range - VECTORIZED
    valid_mask = (candidates >= 60) & (candidates <= 200)
    candidates = candidates[valid_mask]

    if len(candidates) == 0:
        return []

    # Spectral features from STFTCache - VECTORIZED (computed once)
    spectral_centroid = float(np.mean(stft_cache.get_spectral_centroid()))
    spectral_bandwidth = float(np.mean(stft_cache.get_spectral_bandwidth()))
    spectral_rolloff = float(np.mean(stft_cache.get_spectral_rolloff()))

    # Rhythm features from STFTCache
    tempogram, _ = stft_cache.get_tempogram()
    tempo_strength = float(np.max(tempogram, axis=0).mean())

    # === VECTORIZED feature computation for all candidates ===
    n_cand = len(candidates)

    bpm_ratio = candidates / tempo_detected
    bpm_diff = candidates - tempo_detected

    is_half = (np.abs(bpm_ratio - 0.5) < 0.1).astype(np.int32)
    is_double = (np.abs(bpm_ratio - 2.0) < 0.1).astype(np.int32)
    is_base = (np.abs(bpm_ratio - 1.0) < 0.1).astype(np.int32)

    beat_intervals = 60.0 / candidates
    expected_beats = duration / beat_intervals

    is_correct = (np.abs(candidates - gt_bpm) < 2.0).astype(np.int32)

    # Convert to list of dicts
    return [
        {
            'time': 0.0,
            'candidate_bpm': float(candidates[i]),
            'detected_bpm': float(tempo_detected),
            'gt_bpm': float(gt_bpm),
            'bpm_ratio': float(bpm_ratio[i]),
            'bpm_diff_detected': float(bpm_diff[i]),
            'is_half': int(is_half[i]),
            'is_double': int(is_double[i]),
            'is_base': int(is_base[i]),
            'expected_beats': float(expected_beats[i]),
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'tempo_strength': tempo_strength,
            'duration': duration,
            'is_correct': int(is_correct[i]),
            'is_positive': int(is_correct[i]),
        }
        for i in range(n_cand)
    ]


# =============================================================================
# Feature Columns
# =============================================================================

DOWNBEAT_FEATURE_COLS = [
    'beat_index', 'beat_in_bar_est', 'rms_local_mean', 'rms_local_std',
    'rms_at_beat', 'bass_at_beat', 'bass_local_mean', 'onset_strength',
    'is_first_beat', 'time_mod_4beats',
]

BPM_FEATURE_COLS = [
    'candidate_bpm', 'detected_bpm', 'bpm_ratio', 'bpm_diff_detected',
    'is_half', 'is_double', 'is_base', 'expected_beats',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'tempo_strength', 'duration',
]


# =============================================================================
# AudioContext-aware Feature Dataset
# =============================================================================

@dataclass
class AudioContextFeatureDataset:
    """
    Feature extraction using AudioContext (architecture-compliant).

    Replaces FeatureDataset with AudioContext-based extraction:
    - Creates AudioContext via create_audio_context()
    - Passes AudioContext to feature_extractor (not y, sr)
    - All features computed via STFTCache (lazy, cached)
    """
    cache: CacheRepository
    loader: AudioLoader
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512

    def extract(
        self,
        audio_path: str,
        extractor_fn: Callable[[AudioContext, Dict], List[dict]],
        metadata: Dict,
        use_cache: bool = True,
        extractor_name: str = None,
    ) -> pd.DataFrame:
        """
        Extract features via AudioContext.

        Args:
            audio_path: Path to audio file
            extractor_fn: Function(context: AudioContext, metadata) -> List[dict]
            metadata: Metadata dict (bpm, beat_times, etc.)
            use_cache: Whether to use feature cache
            extractor_name: Name to differentiate cache keys

        Returns:
            DataFrame with features
        """
        file_hash = self.cache.compute_file_hash(audio_path)

        # Include extractor name in cache key
        cache_key = f"{file_hash}_{extractor_name}" if extractor_name else file_hash

        # Try cache first
        if use_cache:
            cached = self.cache.get_features(cache_key)
            if cached is not None:
                return pd.DataFrame(cached)

        # Load audio via AudioLoader
        y, sr = self.loader.load(audio_path)

        # Create AudioContext (computes STFT once)
        context = create_audio_context(
            y=y,
            sr=sr,
            file_path=audio_path,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Extract features via AudioContext
        features = extractor_fn(context, metadata)

        if not features:
            return pd.DataFrame()

        # Cache features
        self.cache.save_features(cache_key, features)

        return pd.DataFrame(features)


# =============================================================================
# Training Runner (Architecture-Compliant)
# =============================================================================

def run_training(
    task: str,
    gt_path: str,
    output_dir: str,
    max_tracks: Optional[int] = None,
    use_cache: bool = True,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> TrainingResult:
    """
    Run training for a specific task.

    ARCHITECTURE COMPLIANT:
    - Uses create_audio_context() for AudioContext creation
    - Feature extractors receive AudioContext (not y, sr)
    - All features via STFTCache.get_*() (lazy, cached)
    """
    logger = logger or logging.getLogger(__name__)
    output_path = Path(output_dir) / f'beatgrid_detector_{task}.pkl'

    logger.info("=" * 60)
    logger.info(f"BEAT GRID TRAINING: {task.upper()}")
    logger.info("=" * 60)

    # Select task-specific components
    if task == 'downbeat':
        feature_extractor = extract_downbeat_features_vectorized
        feature_cols = DOWNBEAT_FEATURE_COLS
    elif task == 'bpm':
        feature_extractor = extract_bpm_features_vectorized
        feature_cols = BPM_FEATURE_COLS
    else:
        raise ValueError(f"Unknown task: {task}")

    # Load ground truth
    gt_loader = BeatGridGroundTruthLoader(
        gt_path=gt_path,
        max_tracks=max_tracks,
        logger=logger,
    )
    gt_data = gt_loader.load()

    if not gt_data:
        logger.error("No tracks loaded!")
        return None

    # Setup components (AudioContext-aware)
    cache_repo = CacheRepository.get_instance()
    audio_loader = AudioLoader(sample_rate=22050)
    dataset = AudioContextFeatureDataset(
        cache=cache_repo,
        loader=audio_loader,
        sr=22050,
        n_fft=2048,
        hop_length=512,
    )
    labeler = LabelAssigner(tolerance=0.5, label_col='is_positive')

    # Extract features with progress bar
    logger.info(f"\nExtracting features from {len(gt_data)} tracks...")
    all_features = []

    items = list(gt_data.items())
    for name, entry in tqdm(items, desc="Feature extraction", unit="track"):
        try:
            df = dataset.extract(
                audio_path=entry['file'],
                extractor_fn=feature_extractor,
                metadata=entry['metadata'],
                use_cache=use_cache,
                extractor_name=task,
            )

            if df.empty:
                logger.debug(f"  {name}: No features")
                continue

            df = labeler.assign(df, entry['labels'], time_col='time')
            df['track_name'] = name

            n_pos = df['is_positive'].sum()
            logger.debug(f"  {name}: {len(df)} samples, {n_pos} positive")

            all_features.append(df)

        except Exception as e:
            logger.warning(f"  {name}: Error - {e}")
            continue

    if not all_features:
        logger.error("No features extracted!")
        return None

    # Combine features - VECTORIZED concatenation
    combined_df = pd.concat(all_features, ignore_index=True)
    n_pos = combined_df['is_positive'].sum()
    n_neg = len(combined_df) - n_pos

    logger.info(f"\nDataset: {len(combined_df)} samples")
    logger.info(f"  Positive: {n_pos} ({100*n_pos/len(combined_df):.1f}%)")
    logger.info(f"  Negative: {n_neg} ({100*n_neg/len(combined_df):.1f}%)")
    logger.info(f"  Tracks: {len(all_features)}")

    # Train model
    logger.info("\nTraining XGBoost model...")
    trainer = XGBoostTrainer(
        feature_cols=feature_cols,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
    )

    result = trainer.train(combined_df, label_col='is_positive')
    trainer.save(result, str(output_path))

    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  F1 Score: {result.metrics['f1']:.3f}")
    logger.info(f"  CV F1: {result.metrics['cv_f1']:.3f}")
    logger.info(f"  Precision: {result.metrics['precision']:.3f}")
    logger.info(f"  Recall: {result.metrics['recall']:.3f}")
    logger.info(f"  Model saved: {output_path}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Beat Grid ML Models')
    parser.add_argument('--gt-path', default='data/beatgrid_training.json')
    parser.add_argument('--output-dir', '-o', default='models')
    parser.add_argument('--task', choices=['downbeat', 'bpm', 'all'], default='downbeat')
    parser.add_argument('--sample', '-n', type=int, help='Limit tracks')
    parser.add_argument('--full', action='store_true', help='Use all tracks')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--log-file', default='logs/beatgrid_training.log')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file.replace('.log', f'_{timestamp}.log')
    logger = setup_logging(verbose=args.verbose, log_file=log_file)

    logger.info(f"Beat Grid ML Training (VECTORIZED)")
    logger.info(f"Log: {log_file}")
    logger.info(f"Task: {args.task}")

    max_tracks = None if args.full else (args.sample or 200)
    tasks = ['downbeat', 'bpm'] if args.task == 'all' else [args.task]

    results = {}
    for task in tasks:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# TASK: {task.upper()}")
        logger.info(f"{'#'*60}")

        result = run_training(
            task=task,
            gt_path=args.gt_path,
            output_dir=args.output_dir,
            max_tracks=max_tracks,
            use_cache=not args.no_cache,
            verbose=args.verbose,
            logger=logger,
        )

        if result:
            results[task] = result

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for task, result in results.items():
        logger.info(f"  {task}: F1={result.metrics['f1']:.3f}, CV_F1={result.metrics['cv_f1']:.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
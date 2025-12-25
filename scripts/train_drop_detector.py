#!/usr/bin/env python3
"""
Train XGBoost Drop Detector using Universal Training Pipeline.

Architecture (Primitives -> Tasks -> Pipelines):
- Uses TrainingPipeline from src/training/pipelines/training.py
- Uses DropDetectorML for feature extraction
- Uses existing CacheManager and AudioLoader

Usage:
    python scripts/train_drop_detector.py
    python scripts/train_drop_detector.py --no-cache  # Force re-extraction
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# === Import from Training Pipelines (moved to src/training/) ===
from src.training.pipelines import (
    TrainingPipeline,
    DropGroundTruthLoader,
    XGBoostTrainer,
)
from src.core.tasks.drop_detector_ml import DropDetectorML


# Feature columns for drop detection
DROP_FEATURE_COLS = [
    'rms_change', 'rms_change_short', 'bass_change', 'high_change', 'centroid_change',
    'bass_before', 'bass_after', 'rms_before', 'rms_after',
    'valley_depth', 'drop_contrast', 'buildup_ratio', 'has_buildup_pattern',
]


def create_feature_extractor(detector: DropDetectorML, beats_per_sample: int = 4):
    """
    Create feature extractor function for TrainingPipeline.

    Returns:
        Callable[[np.ndarray, int, Dict], List[dict]]
    """
    def extractor(y: np.ndarray, sr: int, metadata: dict) -> list:
        bpm = metadata.get('bpm', 132)
        music_start = metadata.get('music_start', 0)
        duration = len(y) / sr

        # Sample grid (vectorized)
        interval = (60.0 / bpm) * beats_per_sample
        sample_times = np.arange(music_start + interval, duration - interval, interval)

        # Extract via Task
        return detector._extract_features_at_boundaries(y, sr, sample_times, tempo=bpm)

    return extractor


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost Drop Detector')
    parser.add_argument('--gt-path', default='data/ground_truth_drops.json',
                        help='Path to ground truth JSON')
    parser.add_argument('--output', '-o', default='models/drop_detector_xgb.pkl',
                        help='Output model path')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Label tolerance in seconds')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable feature caching')
    args = parser.parse_args()

    print("="*60 + "\nDROP DETECTOR TRAINING (Universal Pipeline)\n" + "="*60)

    # Create feature extractor using existing Task
    detector = DropDetectorML(auto_load_model=False)
    feature_extractor = create_feature_extractor(detector)

    # Build pipeline using universal components
    pipeline = TrainingPipeline(
        gt_loader=DropGroundTruthLoader(args.gt_path),
        feature_extractor=feature_extractor,
        trainer=XGBoostTrainer(feature_cols=DROP_FEATURE_COLS),
        cache_dir='cache',
        sr=22050,
        label_tolerance=args.tolerance,
    )

    # Run training
    result = pipeline.run(
        output_path=args.output,
        project_root=PROJECT_ROOT,
        use_cache=not args.no_cache,
    )

    print(f"\nFinal Metrics: F1={result.metrics['f1']:.3f}, CV_F1={result.metrics['cv_f1']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

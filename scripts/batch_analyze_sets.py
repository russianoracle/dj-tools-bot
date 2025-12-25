#!/usr/bin/env python3
"""
Batch analyze DJ sets and extract style features.

Runs SetAnalysisPipeline on all sets in a directory and saves results.

Supports parallel processing with --workers N for ~Nx speedup on multicore.
"""

import sys
import json
import time
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

warnings.filterwarnings('ignore')

# Clear cached modules for fresh import
for mod in list(sys.modules.keys()):
    if 'src.core' in mod:
        del sys.modules[mod]

from src.core.pipelines.set_analysis import SetAnalysisPipeline, MixingStyle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def analyze_set_standalone(
    path_str: str,
    skip_genres: bool = False
) -> Dict[str, Any]:
    """
    Analyze a single set (standalone function for multiprocessing).

    Creates its own pipeline instance to work in subprocess.
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Import in subprocess
    from src.core.pipelines.set_analysis import SetAnalysisPipeline, MixingStyle

    path = Path(path_str)
    pipeline = SetAnalysisPipeline(
        mixing_style=MixingStyle.SMOOTH,
        analyze_genres=not skip_genres,
    )

    start = time.time()
    try:
        result = pipeline.analyze(str(path))
        elapsed = time.time() - start

        features = {
            'file_name': path.name,
            'duration_min': result.duration_sec / 60,
            'n_segments': result.n_segments,
            'n_transitions': result.n_transitions,
            'transition_density': result.transition_density,
            'total_drops': result.total_drops,
            'drop_density': result.drop_density,
            'processing_time_sec': elapsed,
        }

        if result.segments:
            durations = [s.duration / 60 for s in result.segments]
            features['segment_duration_mean'] = float(np.mean(durations))
            features['segment_duration_std'] = float(np.std(durations))
            features['segment_duration_min'] = float(np.min(durations))
            features['segment_duration_max'] = float(np.max(durations))

        if result.genre_distribution:
            gd = result.genre_distribution
            features['primary_genre'] = gd.primary_category
            features['primary_genre_ratio'] = gd.primary_category_ratio
            features['genre_diversity'] = gd.genre_diversity
            features['genre_transitions'] = gd.genre_transitions
            features['n_unique_subgenres'] = gd.n_unique_subgenres
            features['category_distribution'] = gd.category_distribution
            features['top_subgenres'] = gd.top_subgenres[:5]
            features['mood_tags'] = dict(sorted(gd.mood_tags.items(), key=lambda x: -x[1])[:5])

        features['segments'] = [
            {
                'start_min': s.start_time / 60,
                'end_min': s.end_time / 60,
                'duration_min': s.duration / 60,
                'genre': s.genre.genre if s.genre else None,
                'dj_category': s.genre.dj_category if s.genre else None,
                'confidence': s.genre.confidence if s.genre else None,
            }
            for s in result.segments
        ]

        return features

    except Exception as e:
        return {
            'file_name': path.name,
            'error': str(e),
            'processing_time_sec': time.time() - start,
        }


def analyze_set(pipeline: SetAnalysisPipeline, path: Path) -> Dict[str, Any]:
    """Analyze a single set and return features (for sequential mode)."""
    start = time.time()
    result = pipeline.analyze(str(path))
    elapsed = time.time() - start

    # Extract key features
    features = {
        'file_name': path.name,
        'duration_min': result.duration_sec / 60,
        'n_segments': result.n_segments,
        'n_transitions': result.n_transitions,
        'transition_density': result.transition_density,
        'total_drops': result.total_drops,
        'drop_density': result.drop_density,
        'processing_time_sec': elapsed,
    }

    # Segment stats
    if result.segments:
        durations = [s.duration / 60 for s in result.segments]
        features['segment_duration_mean'] = float(np.mean(durations))
        features['segment_duration_std'] = float(np.std(durations))
        features['segment_duration_min'] = float(np.min(durations))
        features['segment_duration_max'] = float(np.max(durations))

    # Genre distribution
    if result.genre_distribution:
        gd = result.genre_distribution
        features['primary_genre'] = gd.primary_category
        features['primary_genre_ratio'] = gd.primary_category_ratio
        features['genre_diversity'] = gd.genre_diversity
        features['genre_transitions'] = gd.genre_transitions
        features['n_unique_subgenres'] = gd.n_unique_subgenres
        features['category_distribution'] = gd.category_distribution
        features['top_subgenres'] = gd.top_subgenres[:5]
        features['mood_tags'] = dict(sorted(gd.mood_tags.items(), key=lambda x: -x[1])[:5])

    # Segments detail
    features['segments'] = [
        {
            'start_min': s.start_time / 60,
            'end_min': s.end_time / 60,
            'duration_min': s.duration / 60,
            'genre': s.genre.genre if s.genre else None,
            'dj_category': s.genre.dj_category if s.genre else None,
            'confidence': s.genre.confidence if s.genre else None,
        }
        for s in result.segments
    ]

    return features


def main():
    parser = argparse.ArgumentParser(description='Batch analyze DJ sets')
    parser.add_argument('--input-dir', type=str, default='data/dj_sets/nina-kraviz',
                        help='Directory with audio files')
    parser.add_argument('--output', type=str, default='results/nk_set_analysis.json',
                        help='Output JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of sets to analyze')
    parser.add_argument('--skip-genres', action='store_true',
                        help='Skip genre analysis (faster)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, recommend 2 for M2)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    files = sorted(input_dir.glob('*.m4a'))
    if args.limit:
        files = files[:args.limit]

    logger.info(f"Found {len(files)} sets to analyze")

    # Load existing results (for resuming)
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results.get('sets', []))} existing results")
    else:
        results = {
            'analysis_date': datetime.now().isoformat(),
            'input_dir': str(input_dir),
            'sets': [],
        }

    # Track already analyzed
    analyzed_names = {r['file_name'] for r in results.get('sets', [])}

    # Filter out already analyzed
    files_to_analyze = [f for f in files if f.name not in analyzed_names]
    logger.info(f"Need to analyze: {len(files_to_analyze)} sets")

    if not files_to_analyze:
        logger.info("All files already analyzed!")
        return

    total_start = time.time()

    if args.workers > 1:
        # Parallel mode with ProcessPoolExecutor
        logger.info(f"Running in parallel mode with {args.workers} workers")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(analyze_set_standalone, str(path), args.skip_genres): path
                for path in files_to_analyze
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                completed += 1
                try:
                    features = future.result()
                    results['sets'].append(features)

                    # Save incrementally
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)

                    if 'error' in features:
                        logger.error(f"[{completed}/{len(files_to_analyze)}] {path.name} -> Failed: {features['error']}")
                    else:
                        logger.info(
                            f"[{completed}/{len(files_to_analyze)}] {path.name} -> "
                            f"{features['n_segments']} seg, {features['duration_min']:.1f}min, "
                            f"{features.get('primary_genre', 'N/A')}, {features['processing_time_sec']:.1f}s"
                        )
                except Exception as e:
                    logger.error(f"[{completed}/{len(files_to_analyze)}] {path.name} -> Exception: {e}")
                    results['sets'].append({
                        'file_name': path.name,
                        'error': str(e),
                    })
    else:
        # Sequential mode (original behavior)
        logger.info("Running in sequential mode")

        pipeline = SetAnalysisPipeline(
            mixing_style=MixingStyle.SMOOTH,
            analyze_genres=not args.skip_genres,
        )

        for i, path in enumerate(files_to_analyze):
            logger.info(f"[{i+1}/{len(files_to_analyze)}] Analyzing: {path.name}")

            try:
                features = analyze_set(pipeline, path)
                results['sets'].append(features)

                # Save incrementally
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)

                logger.info(
                    f"  -> {features['n_segments']} segments, "
                    f"{features['duration_min']:.1f}min, "
                    f"{features.get('primary_genre', 'N/A')}, "
                    f"{features['processing_time_sec']:.1f}s"
                )
            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                results['sets'].append({
                    'file_name': path.name,
                    'error': str(e),
                })

    # Compute aggregate stats
    valid_sets = [s for s in results['sets'] if 'error' not in s]
    if valid_sets:
        results['aggregate'] = {
            'n_sets': len(valid_sets),
            'total_duration_hours': sum(s['duration_min'] for s in valid_sets) / 60,
            'avg_segments_per_set': np.mean([s['n_segments'] for s in valid_sets]),
            'avg_segment_duration_min': np.mean([s.get('segment_duration_mean', 0) for s in valid_sets]),
            'total_processing_time_min': (time.time() - total_start) / 60,
        }

        # Genre distribution across all sets
        genre_counts = {}
        for s in valid_sets:
            pg = s.get('primary_genre', 'Unknown')
            genre_counts[pg] = genre_counts.get(pg, 0) + 1
        results['aggregate']['genre_distribution'] = genre_counts

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nDone! Results saved to {output_path}")
    logger.info(f"Analyzed {len(valid_sets)} sets in {(time.time() - total_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
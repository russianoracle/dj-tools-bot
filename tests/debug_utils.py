"""Debugging utilities for DJ Tools Bot development.

Provides helper functions for:
    - Generating synthetic test audio
    - Inspecting analysis results
    - Profiling performance
    - Visualizing features (text-based)
    - Comparing analysis runs

Usage:
    from tests.debug_utils import (
        generate_test_audio,
        inspect_analysis_result,
        profile_pipeline,
        compare_features
    )

    # Generate test audio
    audio = generate_test_audio(duration=30, bpm=128, zone='purple')

    # Inspect results
    inspect_analysis_result(result)

    # Profile performance
    profile_pipeline('track_sample.wav')
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# SYNTHETIC AUDIO GENERATION
# =============================================================================

def generate_test_audio(
    duration: float = 30.0,
    sr: int = 22050,
    bpm: float = 128.0,
    zone: str = 'green',
    add_drops: bool = False,
    add_noise: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Generate synthetic test audio with known characteristics.

    Args:
        duration: Duration in seconds
        sr: Sample rate
        bpm: Tempo in BPM
        zone: Target energy zone ('yellow', 'green', 'purple')
        add_drops: Add synthetic drop patterns
        add_noise: Add background noise

    Returns:
        Tuple of (audio_array, sample_rate)

    Example:
        >>> audio, sr = generate_test_audio(duration=10, bpm=128, zone='purple')
        >>> print(f"Generated {len(audio)/sr:.1f}s of audio")
    """
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Base frequency for kick drum simulation
    kick_freq = 60.0  # Hz

    # Beat timing
    beat_period = 60.0 / bpm
    beat_samples = int(beat_period * sr)

    # Generate base signal based on zone
    if zone == 'yellow':
        # Calm: low frequency, low energy, smooth
        audio = 0.2 * np.sin(2 * np.pi * 220 * t)  # Low pad
        audio += 0.1 * np.sin(2 * np.pi * 110 * t)  # Sub bass
        # Soft beats
        for i in range(0, n_samples, beat_samples):
            if i + 1000 < n_samples:
                envelope = np.exp(-np.linspace(0, 5, 1000))
                audio[i:i+1000] += 0.15 * envelope * np.sin(2 * np.pi * kick_freq * np.linspace(0, 1000/sr, 1000))

    elif zone == 'purple':
        # High energy: strong beats, bright, complex
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # High synth
        audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # Brightness
        audio += 0.15 * np.sin(2 * np.pi * 220 * t)  # Mid
        # Strong kicks
        for i in range(0, n_samples, beat_samples):
            if i + 2000 < n_samples:
                envelope = np.exp(-np.linspace(0, 3, 2000))
                audio[i:i+2000] += 0.4 * envelope * np.sin(2 * np.pi * kick_freq * np.linspace(0, 2000/sr, 2000))

        if add_drops:
            # Add drop pattern: buildup then energy spike
            drop_positions = [int(duration * 0.3 * sr), int(duration * 0.7 * sr)]
            for drop_pos in drop_positions:
                if drop_pos + sr < n_samples:
                    # Buildup (1 second before)
                    buildup_start = max(0, drop_pos - sr)
                    buildup = np.linspace(0.3, 1.0, drop_pos - buildup_start)
                    audio[buildup_start:drop_pos] *= buildup
                    # Drop impact
                    drop_impact = np.exp(-np.linspace(0, 2, sr))
                    audio[drop_pos:drop_pos+sr] += 0.3 * drop_impact

    else:  # green
        # Medium energy: balanced
        audio = 0.25 * np.sin(2 * np.pi * 330 * t)
        audio += 0.15 * np.sin(2 * np.pi * 165 * t)
        # Medium kicks
        for i in range(0, n_samples, beat_samples):
            if i + 1500 < n_samples:
                envelope = np.exp(-np.linspace(0, 4, 1500))
                audio[i:i+1500] += 0.25 * envelope * np.sin(2 * np.pi * kick_freq * np.linspace(0, 1500/sr, 1500))

    if add_noise:
        audio += 0.02 * np.random.randn(n_samples).astype(np.float32)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.9

    return audio.astype(np.float32), sr


def generate_drop_pattern(
    duration: float = 10.0,
    sr: int = 22050,
    drop_time: float = 5.0
) -> Tuple[np.ndarray, int]:
    """
    Generate audio with clear drop pattern for testing drop detection.

    Args:
        duration: Total duration in seconds
        sr: Sample rate
        drop_time: Time when drop occurs

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    audio = np.zeros(n_samples, dtype=np.float32)

    drop_sample = int(drop_time * sr)
    buildup_start = int((drop_time - 2.0) * sr)

    # Intro (low energy)
    audio[:buildup_start] = 0.2 * np.sin(2 * np.pi * 220 * t[:buildup_start])

    # Buildup (increasing energy)
    buildup_len = drop_sample - buildup_start
    if buildup_len > 0:
        buildup_envelope = np.linspace(0.2, 0.8, buildup_len)
        audio[buildup_start:drop_sample] = buildup_envelope * np.sin(2 * np.pi * 440 * t[buildup_start:drop_sample])

    # Drop (high energy)
    post_drop = n_samples - drop_sample
    if post_drop > 0:
        drop_envelope = np.exp(-np.linspace(0, 1, min(post_drop, sr)))
        full_envelope = np.concatenate([drop_envelope, np.ones(max(0, post_drop - sr)) * 0.6])[:post_drop]
        audio[drop_sample:] = full_envelope * (
            0.5 * np.sin(2 * np.pi * 880 * t[drop_sample:]) +
            0.3 * np.sin(2 * np.pi * 60 * t[drop_sample:])
        )

    # Add beats
    bpm = 128.0
    beat_period = int(60.0 / bpm * sr)
    for i in range(0, n_samples, beat_period):
        if i + 1000 < n_samples:
            kick = 0.3 * np.exp(-np.linspace(0, 5, 1000)) * np.sin(2 * np.pi * 60 * np.linspace(0, 1000/sr, 1000))
            audio[i:i+1000] += kick

    return audio.astype(np.float32), sr


# =============================================================================
# RESULT INSPECTION
# =============================================================================

def inspect_analysis_result(result: Any, verbose: bool = True) -> Dict[str, Any]:
    """
    Inspect and summarize an analysis result.

    Args:
        result: Analysis result object (TrackAnalysisResult, etc.)
        verbose: Print detailed output

    Returns:
        Dictionary with inspection summary
    """
    summary = {
        'type': type(result).__name__,
        'success': getattr(result, 'success', None),
        'error': getattr(result, 'error', None),
        'processing_time': getattr(result, 'processing_time_sec', None),
    }

    # Zone classification specific
    if hasattr(result, 'zone'):
        summary['zone'] = result.zone
        summary['zone_confidence'] = getattr(result, 'zone_confidence', None)
        summary['zone_scores'] = getattr(result, 'zone_scores', None)

    # Feature extraction specific
    if hasattr(result, 'features'):
        features = result.features
        summary['n_features'] = len(features)
        summary['feature_sample'] = dict(list(features.items())[:5])

    # Set analysis specific
    if hasattr(result, 'n_segments'):
        summary['n_segments'] = result.n_segments
        summary['n_transitions'] = getattr(result, 'n_transitions', 0)
        summary['total_drops'] = getattr(result, 'total_drops', 0)

    # Duration
    if hasattr(result, 'duration_sec'):
        summary['duration_sec'] = result.duration_sec

    if verbose:
        print("\n" + "=" * 60)
        print(f"ANALYSIS RESULT INSPECTION: {summary['type']}")
        print("=" * 60)

        print(f"\nStatus: {'SUCCESS' if summary['success'] else 'FAILED'}")
        if summary['error']:
            print(f"Error: {summary['error']}")

        if summary.get('processing_time'):
            print(f"Processing Time: {summary['processing_time']:.2f}s")

        if summary.get('zone'):
            print(f"\nZone: {summary['zone'].upper()}")
            print(f"Confidence: {summary['zone_confidence']:.2%}")
            if summary.get('zone_scores'):
                print("Zone Scores:")
                for zone, score in summary['zone_scores'].items():
                    bar = '#' * int(score * 30)
                    print(f"  {zone:8s}: {bar} {score:.2%}")

        if summary.get('n_features'):
            print(f"\nFeatures: {summary['n_features']} total")
            print("Sample features:")
            for name, value in summary.get('feature_sample', {}).items():
                print(f"  {name}: {value:.4f}")

        if summary.get('n_segments') is not None:
            print(f"\nSegments: {summary['n_segments']}")
            print(f"Transitions: {summary['n_transitions']}")
            print(f"Drops: {summary['total_drops']}")

        print("=" * 60 + "\n")

    return summary


def inspect_features(features: Dict[str, float], top_n: int = 20) -> None:
    """
    Print feature values in a readable format.

    Args:
        features: Dictionary of feature names to values
        top_n: Number of top features to show
    """
    print("\n" + "=" * 50)
    print("FEATURE INSPECTION")
    print("=" * 50)

    # Group by category
    categories = {
        'energy': ['rms_', 'low_energy'],
        'spectral': ['spectral_', 'brightness', 'rolloff', 'centroid'],
        'mfcc': ['mfcc_'],
        'chroma': ['chroma_'],
        'rhythm': ['tempo', 'onset', 'beat'],
        'drop': ['drop_', 'buildup'],
        'band': ['bass_', 'mid_', 'high_', 'sub_bass']
    }

    categorized = {cat: {} for cat in categories}
    categorized['other'] = {}

    for name, value in features.items():
        assigned = False
        for cat, prefixes in categories.items():
            if any(name.startswith(p) or p in name for p in prefixes):
                categorized[cat][name] = value
                assigned = True
                break
        if not assigned:
            categorized['other'][name] = value

    for category, feats in categorized.items():
        if feats:
            print(f"\n{category.upper()}:")
            for name, value in sorted(feats.items())[:top_n]:
                print(f"  {name:30s}: {value:>10.4f}")

    print("=" * 50 + "\n")


# =============================================================================
# PERFORMANCE PROFILING
# =============================================================================

@dataclass
class ProfileResult:
    """Result of profiling a pipeline."""
    total_time: float
    stages: Dict[str, float]
    memory_peak_mb: float = 0.0


def profile_pipeline(
    audio_path: str,
    pipeline_type: str = 'track',
    iterations: int = 1
) -> ProfileResult:
    """
    Profile pipeline execution time.

    Args:
        audio_path: Path to audio file
        pipeline_type: 'track' or 'set'
        iterations: Number of iterations to average

    Returns:
        ProfileResult with timing breakdown
    """
    times = []
    stage_times: Dict[str, List[float]] = {}

    for i in range(iterations):
        if pipeline_type == 'track':
            from app.modules.analysis.pipelines.track_analysis import TrackAnalysisPipeline
            pipeline = TrackAnalysisPipeline(sr=22050)
        else:
            from app.modules.analysis.pipelines.set_analysis import SetAnalysisPipeline
            pipeline = SetAnalysisPipeline(verbose=False)

        start = time.time()
        result = pipeline.analyze(audio_path)
        elapsed = time.time() - start
        times.append(elapsed)

        # Collect stage times if available
        if hasattr(result, 'stage_times'):
            for stage, t in result.stage_times.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(t)

    avg_time = np.mean(times)
    avg_stages = {stage: np.mean(ts) for stage, ts in stage_times.items()}

    print("\n" + "=" * 50)
    print(f"PIPELINE PROFILE: {pipeline_type}")
    print("=" * 50)
    print(f"Total Time: {avg_time:.2f}s (avg of {iterations} runs)")

    if avg_stages:
        print("\nStage Breakdown:")
        for stage, t in sorted(avg_stages.items(), key=lambda x: -x[1]):
            pct = (t / avg_time) * 100
            bar = '#' * int(pct / 2)
            print(f"  {stage:20s}: {t:6.2f}s ({pct:5.1f}%) {bar}")

    print("=" * 50 + "\n")

    return ProfileResult(
        total_time=avg_time,
        stages=avg_stages,
        memory_peak_mb=0.0  # Would need memory_profiler for this
    )


# =============================================================================
# FEATURE COMPARISON
# =============================================================================

def compare_features(
    features1: Dict[str, float],
    features2: Dict[str, float],
    name1: str = "A",
    name2: str = "B",
    threshold: float = 0.1
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare two feature dictionaries and highlight differences.

    Args:
        features1: First feature dict
        features2: Second feature dict
        name1: Label for first set
        name2: Label for second set
        threshold: Minimum difference to report

    Returns:
        Dict of features with significant differences: {name: (val1, val2, diff)}
    """
    differences = {}

    all_keys = set(features1.keys()) | set(features2.keys())

    print("\n" + "=" * 70)
    print(f"FEATURE COMPARISON: {name1} vs {name2}")
    print("=" * 70)

    for key in sorted(all_keys):
        val1 = features1.get(key, 0.0)
        val2 = features2.get(key, 0.0)

        # Normalize difference by magnitude
        max_val = max(abs(val1), abs(val2), 1e-10)
        rel_diff = abs(val1 - val2) / max_val

        if rel_diff > threshold:
            differences[key] = (val1, val2, rel_diff)

    if differences:
        print(f"\nSignificant differences (>{threshold*100:.0f}% relative):")
        print("-" * 70)
        print(f"{'Feature':30s} | {name1:>12s} | {name2:>12s} | {'Diff':>10s}")
        print("-" * 70)

        for key, (val1, val2, diff) in sorted(differences.items(), key=lambda x: -x[1][2]):
            print(f"{key:30s} | {val1:12.4f} | {val2:12.4f} | {diff*100:9.1f}%")
    else:
        print("\nNo significant differences found!")

    print("=" * 70 + "\n")

    return differences


# =============================================================================
# QUICK ANALYSIS HELPERS
# =============================================================================

def quick_analyze(audio_or_path, sr: int = 22050) -> Dict[str, Any]:
    """
    Quickly analyze audio and return key metrics.

    Args:
        audio_or_path: Either numpy array or file path
        sr: Sample rate (if providing array)

    Returns:
        Dict with key analysis results
    """
    from app.modules.analysis.tasks.base import create_audio_context
    from app.modules.analysis.tasks.feature_extraction import FeatureExtractionTask
    from app.modules.analysis.tasks.zone_classification import ZoneClassificationTask

    # Load audio if path provided
    if isinstance(audio_or_path, (str, Path)):
        import librosa
        y, sr = librosa.load(str(audio_or_path), sr=sr)
    else:
        y = audio_or_path

    # Create context
    ctx = create_audio_context(y, sr)

    # Run tasks
    features_task = FeatureExtractionTask()
    zone_task = ZoneClassificationTask()

    features_result = features_task.execute(ctx)
    zone_result = zone_task.execute(ctx)

    return {
        'duration_sec': ctx.duration_sec,
        'zone': zone_result.zone if zone_result.success else None,
        'zone_confidence': zone_result.confidence if zone_result.success else 0.0,
        'tempo': features_result.features.get('tempo', 0.0) if features_result.success else 0.0,
        'brightness': features_result.features.get('brightness', 0.0) if features_result.success else 0.0,
        'drop_count': features_result.features.get('drop_count', 0) if features_result.success else 0,
        'features': features_result.features if features_result.success else {}
    }


def print_audio_stats(y: np.ndarray, sr: int = 22050) -> None:
    """
    Print basic statistics about an audio array.

    Args:
        y: Audio array
        sr: Sample rate
    """
    print("\n" + "=" * 50)
    print("AUDIO STATISTICS")
    print("=" * 50)
    print(f"Shape: {y.shape}")
    print(f"Dtype: {y.dtype}")
    print(f"Duration: {len(y)/sr:.2f}s")
    print(f"Sample Rate: {sr} Hz")
    print(f"Min: {y.min():.4f}")
    print(f"Max: {y.max():.4f}")
    print(f"Mean: {y.mean():.4f}")
    print(f"Std: {y.std():.4f}")
    print(f"RMS: {np.sqrt(np.mean(y**2)):.4f}")
    print(f"Peak dB: {20*np.log10(np.abs(y).max() + 1e-10):.1f}")
    print("=" * 50 + "\n")


# =============================================================================
# TEST DATA VALIDATION
# =============================================================================

def validate_feature_vector(
    vector: np.ndarray,
    expected_length: int = 79
) -> Tuple[bool, List[str]]:
    """
    Validate a feature vector for ML model input.

    Args:
        vector: Feature vector to validate
        expected_length: Expected number of features

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check type
    if not isinstance(vector, np.ndarray):
        issues.append(f"Expected numpy array, got {type(vector)}")
        return False, issues

    # Check length
    if len(vector) != expected_length:
        issues.append(f"Expected {expected_length} features, got {len(vector)}")

    # Check for NaN
    nan_count = np.isnan(vector).sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")

    # Check for Inf
    inf_count = np.isinf(vector).sum()
    if inf_count > 0:
        issues.append(f"Contains {inf_count} Inf values")

    # Check for extreme values
    extreme_threshold = 1e6
    extreme_count = (np.abs(vector) > extreme_threshold).sum()
    if extreme_count > 0:
        issues.append(f"Contains {extreme_count} extreme values (>{extreme_threshold})")

    return len(issues) == 0, issues


if __name__ == "__main__":
    # Demo usage
    print("DJ Tools Bot - Debug Utilities Demo\n")

    # Generate test audio
    print("Generating synthetic audio...")
    audio_purple, sr = generate_test_audio(duration=10, zone='purple', add_drops=True)
    audio_yellow, _ = generate_test_audio(duration=10, zone='yellow')

    print_audio_stats(audio_purple, sr)

    # Quick analysis
    print("Running quick analysis...")
    result_purple = quick_analyze(audio_purple, sr)
    result_yellow = quick_analyze(audio_yellow, sr)

    print(f"\nPurple zone audio: zone={result_purple['zone']}, tempo={result_purple['tempo']:.1f}")
    print(f"Yellow zone audio: zone={result_yellow['zone']}, tempo={result_yellow['tempo']:.1f}")

    # Compare features
    compare_features(
        result_purple['features'],
        result_yellow['features'],
        name1="Purple",
        name2="Yellow",
        threshold=0.2
    )

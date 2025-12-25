#!/usr/bin/env python3
"""
Validate Detection - Create audio clips for manual validation.

Extracts short audio clips around detected:
- Drops (energy peaks)
- Transitions (mixin/mixout points)
- Segment boundaries

Usage:
    python scripts/validate_detection.py data/dj_sets/josh-baker/boiler-room.m4a

Output:
    validation_clips/
    ├── drops/
    │   ├── drop_001_02m30s.mp3
    │   ├── drop_002_05m15s.mp3
    │   └── ...
    ├── transitions/
    │   ├── trans_001_08m00s.mp3
    │   └── ...
    └── segments/
        ├── seg_001_start_00m00s.mp3
        └── ...
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
import soundfile as sf
from src.core.pipelines.set_analysis import SetAnalysisPipeline, MixingStyle


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m{s:02d}s"


def extract_clip(
    y: np.ndarray,
    sr: int,
    center_sec: float,
    duration_sec: float = 10.0,
    fade_sec: float = 0.5
) -> np.ndarray:
    """
    Extract audio clip centered at given time.

    Args:
        y: Full audio array
        sr: Sample rate
        center_sec: Center time in seconds
        duration_sec: Clip duration
        fade_sec: Fade in/out duration

    Returns:
        Audio clip array
    """
    half_dur = duration_sec / 2
    start_sec = max(0, center_sec - half_dur)
    end_sec = min(len(y) / sr, center_sec + half_dur)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    clip = y[start_sample:end_sample].copy()

    # Apply fade in/out
    fade_samples = int(fade_sec * sr)
    if len(clip) > fade_samples * 2:
        # Fade in
        clip[:fade_samples] *= np.linspace(0, 1, fade_samples)
        # Fade out
        clip[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return clip


def save_clip(clip: np.ndarray, sr: int, path: str):
    """Save audio clip to file."""
    sf.write(path, clip, sr)


def create_validation_clips(
    audio_path: str,
    output_dir: str = "validation_clips",
    clip_duration: float = 10.0,
    max_clips_per_type: int = 20,
):
    """
    Analyze audio and create validation clips.

    Args:
        audio_path: Path to audio file
        output_dir: Output directory for clips
        clip_duration: Duration of each clip in seconds
        max_clips_per_type: Maximum clips per category
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    print(f"=== Validation Clip Generator ===")
    print(f"Input: {audio_path.name}")
    print(f"Output: {output_dir}/")
    print()

    # Create output directories
    drops_dir = output_dir / "drops"
    timeline_dir = output_dir / "timeline"  # Sequential TRACK/TRANSITION clips

    for d in [drops_dir, timeline_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load full audio
    print("Loading audio...")
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration_sec = len(y) / sr
    print(f"Duration: {duration_sec/60:.1f} min")
    print()

    # Run analysis
    print("Analyzing with SMOOTH settings...")
    pipeline = SetAnalysisPipeline(
        mixing_style=MixingStyle.SMOOTH,
        analyze_genres=False,
        verbose=True
    )
    result = pipeline.analyze(str(audio_path))

    print()
    print(f"Detected:")
    print(f"  Drops: {result.total_drops}")
    print(f"  Transitions: {result.n_transitions}")
    print(f"  Segments: {result.n_segments}")
    print()

    # Get detailed drop info from pipeline context
    # Re-run drop detection to get DropCandidate objects
    from src.core.tasks import DropDetectionTask, create_audio_context
    from src.core.config import SetAnalysisConfig, MixingStyle as MS

    cfg = SetAnalysisConfig.for_style(MS.SMOOTH)
    ctx = create_audio_context(y, sr)

    drop_task = DropDetectionTask(
        min_drop_magnitude=cfg.drop_detection.min_drop_magnitude,
        min_confidence=cfg.drop_detection.min_confidence,
        buildup_window_sec=cfg.drop_detection.buildup_window_sec,
        use_multiband=cfg.drop_detection.use_multiband,
    )
    drop_result = drop_task.execute(ctx)

    # === DROPS ===
    print(f"Creating drop clips...")
    drops_info = []

    # Sort by confidence and take top N
    sorted_drops = sorted(drop_result.drops, key=lambda d: d.confidence, reverse=True)

    for i, drop in enumerate(sorted_drops[:max_clips_per_type]):
        time_str = format_time(drop.time_sec)
        filename = f"drop_{i+1:03d}_{time_str}_conf{drop.confidence:.2f}.wav"
        filepath = drops_dir / filename

        clip = extract_clip(y, sr, drop.time_sec, clip_duration)
        save_clip(clip, sr, str(filepath))

        drops_info.append({
            "index": i + 1,
            "time_sec": drop.time_sec,
            "time_str": time_str,
            "confidence": drop.confidence,
            "magnitude": drop.drop_magnitude,
            "buildup_score": drop.buildup_score,
            "file": filename,
        })
        print(f"  [{i+1}/{min(len(sorted_drops), max_clips_per_type)}] {time_str} (conf={drop.confidence:.2f}, mag={drop.drop_magnitude:.2f})")

    # === TIMELINE: Sequential TRACK → TRANSITION → TRACK ===
    print(f"\nCreating timeline clips (TRACK → TRANSITION → TRACK)...")
    timeline_info = []

    for i, seg in enumerate(result.segments):
        seg_type = "TRANSITION" if seg.is_transition_zone else "TRACK"
        time_start = format_time(seg.start_time)
        time_end = format_time(seg.end_time)

        # Filename with zero-padded index for correct sorting
        filename = f"{i+1:02d}_{seg_type}_{time_start}-{time_end}.wav"
        filepath = timeline_dir / filename

        # Extract clip from START of segment (to hear the boundary)
        clip = extract_clip(y, sr, seg.start_time, clip_duration)
        save_clip(clip, sr, str(filepath))

        timeline_info.append({
            "index": i + 1,
            "type": seg_type,
            "start_sec": seg.start_time,
            "end_sec": seg.end_time,
            "duration_sec": seg.duration,
            "start_str": time_start,
            "end_str": time_end,
            "file": filename,
        })
        print(f"  [{i+1:02d}] {seg_type:10} {time_start} - {time_end} ({seg.duration/60:.1f} min)")

    # Save metadata
    metadata = {
        "source_file": str(audio_path),
        "duration_sec": duration_sec,
        "analysis": {
            "total_drops": result.total_drops,
            "total_transitions": result.n_transitions,
            "total_segments": result.n_segments,
            "tracks_count": sum(1 for s in result.segments if not s.is_transition_zone),
            "transition_zones_count": sum(1 for s in result.segments if s.is_transition_zone),
        },
        "config": {
            "min_drop_magnitude": cfg.drop_detection.min_drop_magnitude,
            "min_confidence": cfg.drop_detection.min_confidence,
            "use_multiband": cfg.drop_detection.use_multiband,
        },
        "drops": drops_info,
        "timeline": timeline_info,
    }

    metadata_path = output_dir / "validation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Drops: {len(drops_info)} clips in {drops_dir}/")
    print(f"Timeline: {len(timeline_info)} clips in {timeline_dir}/")
    print(f"  - Tracks: {metadata['analysis']['tracks_count']}")
    print(f"  - Transitions: {metadata['analysis']['transition_zones_count']}")
    print(f"Metadata: {metadata_path}")

    return metadata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_detection.py <audio_file> [output_dir]")
        print()
        print("Example:")
        print("  python scripts/validate_detection.py data/dj_sets/josh-baker/boiler-room.m4a")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "validation_clips"

    create_validation_clips(audio_path, output_dir)
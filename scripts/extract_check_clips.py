#!/usr/bin/env python3
"""Extract check clips at specific times for manual validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
import soundfile as sf


def extract_clips_at_times(audio_path: str, times_sec: list, output_dir: str = "check_clips"):
    """Extract 10-second clips centered at specific times."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)

    for t in times_sec:
        start = max(0, int((t - 5) * sr))
        end = min(len(y), int((t + 5) * sr))
        clip = y[start:end]

        # Apply fade
        fade_samples = int(0.5 * sr)
        if len(clip) > fade_samples * 2:
            clip[:fade_samples] *= np.linspace(0, 1, fade_samples)
            clip[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        mins = int(t // 60)
        secs = int(t % 60)
        filename = f"check_{mins:02d}m{secs:02d}s.wav"
        sf.write(output_dir / filename, clip, sr)
        print(f"  Created: {filename}")


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/dj_sets/josh-baker/boiler-room---josh-baker-boiler-room-london.m4a"

    # Focus on the gap between track 2 (04:00) and track 3 (07:00)
    # Looking for the transition that leads to track 3
    times = [
        240,    # 04:00 - track 2 playing
        270,    # 04:30
        300,    # 05:00 - potential transition zone
        330,    # 05:30
        360,    # 06:00 - potential transition zone
        390,    # 06:30
        420,    # 07:00 - track 3 playing
    ]

    extract_clips_at_times(audio_path, times, "check_clips_transition_to_track3")
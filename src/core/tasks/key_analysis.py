"""
Key Analysis Task - Detect musical key over time.

Uses sliding window approach to track key changes across DJ sets:
- Compute chromagram for each window
- Estimate key using Krumhansl-Schmuckler profiles
- Track dominant key and key changes
- Convert to Camelot notation for DJ-friendly display
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

from .base import AudioContext, TaskResult, BaseTask
from ..primitives import compute_key


# Camelot Wheel mapping
CAMELOT_WHEEL = {
    'C': '8B', 'Cm': '5A',
    'C#': '3B', 'C#m': '12A',
    'D': '10B', 'Dm': '7A',
    'D#': '5B', 'D#m': '2A',
    'E': '12B', 'Em': '9A',
    'F': '7B', 'Fm': '4A',
    'F#': '2B', 'F#m': '11A',
    'G': '9B', 'Gm': '6A',
    'G#': '4B', 'G#m': '1A',
    'A': '11B', 'Am': '8A',
    'A#': '6B', 'A#m': '3A',
    'B': '1B', 'Bm': '10A',
}


def key_to_camelot(key: str) -> str:
    """Convert musical key to Camelot notation."""
    return CAMELOT_WHEEL.get(key, '?')


@dataclass
class KeyAnalysisResult(TaskResult):
    """
    Result of key analysis over time.

    Attributes:
        dominant_key: Most frequent key
        dominant_camelot: Dominant key in Camelot notation
        key_changes: Number of key changes
        key_trajectory: Key sequence over time
        camelot_trajectory: Camelot sequence over time
        key_histogram: Distribution of keys
        key_stability: 0-1 score (1 = single key, 0 = many changes)
    """
    # Inherited from TaskResult
    success: bool = True
    task_name: str = "KeyAnalysis"
    processing_time_sec: float = 0.0
    error: Optional[str] = None

    # Key metrics
    dominant_key: str = "C"
    dominant_camelot: str = "8B"
    key_changes: int = 0
    key_trajectory: List[str] = field(default_factory=list)
    camelot_trajectory: List[str] = field(default_factory=list)
    key_histogram: Dict[str, int] = field(default_factory=dict)
    key_stability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'dominant_key': self.dominant_key,
            'dominant_camelot': self.dominant_camelot,
            'key_changes': int(self.key_changes),
            'key_trajectory': self.key_trajectory,
            'camelot_trajectory': self.camelot_trajectory,
            'key_histogram': {k: int(v) for k, v in self.key_histogram.items()},
            'key_stability': float(self.key_stability),
        })
        return base


class KeyAnalysisTask(BaseTask):
    """
    Analyze musical key distribution across a DJ set.

    Uses sliding window approach:
    1. Divide set into windows (e.g., 60 seconds)
    2. Compute chromagram for each window
    3. Estimate key using Krumhansl-Schmuckler profiles
    4. Track key changes and dominant key
    5. Convert to Camelot notation
    """

    def __init__(
        self,
        window_duration_sec: float = 60.0,  # 1 minute windows
        hop_duration_sec: float = 30.0,     # 50% overlap
    ):
        """
        Initialize key analysis task.

        Args:
            window_duration_sec: Window size for key estimation
            hop_duration_sec: Hop between windows
        """
        super().__init__()
        self.window_duration_sec = window_duration_sec
        self.hop_duration_sec = hop_duration_sec

    @property
    def name(self) -> str:
        return "KeyAnalysis"

    def execute(self, context: AudioContext) -> KeyAnalysisResult:
        """
        Analyze key distribution.

        Args:
            context: Audio context with signal

        Returns:
            KeyAnalysisResult with key metrics
        """
        y = context.y
        sr = context.sr

        # Calculate window parameters
        window_samples = int(self.window_duration_sec * sr)
        hop_samples = int(self.hop_duration_sec * sr)

        # Extract key for each window
        keys = []

        for start in range(0, len(y) - window_samples, hop_samples):
            end = start + window_samples
            y_window = y[start:end]

            try:
                # Compute chromagram
                chroma = librosa.feature.chroma_cqt(
                    y=y_window,
                    sr=sr,
                    hop_length=512
                )

                # Estimate key
                key, confidence = compute_key(chroma)

                # Filter low-confidence detections
                if confidence > 0.3:
                    keys.append(key)

            except Exception:
                # Skip windows where key estimation fails
                continue

        if len(keys) == 0:
            # No valid keys detected - return defaults
            return KeyAnalysisResult()

        # Compute statistics
        key_counts = Counter(keys)
        dominant_key = key_counts.most_common(1)[0][0]
        dominant_camelot = key_to_camelot(dominant_key)

        # Count key changes
        key_changes = sum(1 for i in range(1, len(keys)) if keys[i] != keys[i-1])

        # Key stability (inverse of change rate)
        key_stability = 1.0 - (key_changes / len(keys)) if len(keys) > 0 else 0.0

        # Convert to Camelot
        camelot_trajectory = [key_to_camelot(k) for k in keys]

        return KeyAnalysisResult(
            dominant_key=dominant_key,
            dominant_camelot=dominant_camelot,
            key_changes=key_changes,
            key_trajectory=keys,
            camelot_trajectory=camelot_trajectory,
            key_histogram=dict(key_counts),
            key_stability=key_stability,
        )

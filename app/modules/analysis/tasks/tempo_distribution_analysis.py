"""
Tempo Distribution Analysis Task - Analyze tempo distribution over DJ sets.

Computes tempo over sliding windows to capture:
- Tempo range (min/max BPM)
- Tempo distribution (histogram)
- Dominant tempo
- Tempo variance (tight vs eclectic)

Uses Primitives layer for all mathematical operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import Counter

from .base import AudioContext, TaskResult, BaseTask
# INTERNAL imports from submodules (bypass guard for internal use)
from app.common.primitives.rhythm import compute_onset_strength, compute_tempo


@dataclass
class TempoDistributionAnalysisResult(TaskResult):
    """
    Result of tempo distribution analysis.

    Attributes:
        tempo_mean: Average BPM across set
        tempo_std: Tempo standard deviation
        tempo_min: Minimum BPM
        tempo_max: Maximum BPM
        tempo_range: Max - min BPM
        tempo_histogram: Distribution of BPM values
        dominant_tempo: Most frequent BPM
        tempo_trajectory: BPM over time (per window)
    """
    # Inherited from TaskResult
    success: bool = True
    task_name: str = "TempoDistributionAnalysis"
    processing_time_sec: float = 0.0
    error: Optional[str] = None

    # Tempo distribution metrics
    tempo_mean: float = 0.0
    tempo_std: float = 0.0
    tempo_min: float = 0.0
    tempo_max: float = 0.0
    tempo_range: float = 0.0
    tempo_histogram: Dict[int, int] = field(default_factory=dict)
    dominant_tempo: int = 120
    tempo_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'tempo_mean': float(self.tempo_mean),
            'tempo_std': float(self.tempo_std),
            'tempo_min': float(self.tempo_min),
            'tempo_max': float(self.tempo_max),
            'tempo_range': float(self.tempo_range),
            'tempo_histogram': {int(k): int(v) for k, v in self.tempo_histogram.items()},
            'dominant_tempo': int(self.dominant_tempo),
            'tempo_trajectory': [float(x) for x in self.tempo_trajectory],
        })
        return base


class TempoDistributionAnalysisTask(BaseTask):
    """
    Analyze tempo distribution across a DJ set.

    Uses sliding window approach:
    1. Divide set into windows (e.g., 60 seconds)
    2. Compute tempo for each window
    3. Build histogram and statistics

    All mathematical operations delegated to Primitives layer.
    """

    def __init__(
        self,
        window_duration_sec: float = 60.0,  # 1 minute windows
        hop_duration_sec: float = 30.0,     # 50% overlap
        tempo_min: float = 80.0,
        tempo_max: float = 180.0,
    ):
        """
        Initialize tempo distribution analysis task.

        Args:
            window_duration_sec: Window size for tempo estimation
            hop_duration_sec: Hop between windows
            tempo_min: Minimum expected BPM
            tempo_max: Maximum expected BPM
        """
        super().__init__()
        self.window_duration_sec = window_duration_sec
        self.hop_duration_sec = hop_duration_sec
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max

    @property
    def name(self) -> str:
        return "TempoDistributionAnalysis"

    def execute(self, context: AudioContext) -> TempoDistributionAnalysisResult:
        """
        Analyze tempo distribution.

        Args:
            context: Audio context with signal

        Returns:
            TempoDistributionAnalysisResult with tempo metrics
        """
        y = context.y
        sr = context.sr

        # Calculate window parameters
        window_samples = int(self.window_duration_sec * sr)
        hop_samples = int(self.hop_duration_sec * sr)

        # Extract tempo for each window using primitives (NO PRIOR - independent detection)
        tempos = []
        hop_length = 512

        for start in range(0, len(y) - window_samples, hop_samples):
            end = start + window_samples
            y_window = y[start:end]

            try:
                # Compute onset strength using primitive (uses librosa internally but isolated)
                onset_env = compute_onset_strength(y_window, sr, hop_length=hop_length)

                # Detect tempo using primitive
                tempo = compute_tempo(onset_env, sr=sr, hop_length=hop_length)

                # Filter valid tempos
                if self.tempo_min <= tempo <= self.tempo_max:
                    tempos.append(tempo)

            except Exception:
                # Skip windows where tempo estimation fails
                continue

        if len(tempos) == 0:
            # No valid tempos detected - return defaults
            return TempoDistributionAnalysisResult()

        # Compute statistics (no prior correction - each window detected independently)
        tempos_array = np.array(tempos, dtype=np.float32)

        # Correct octave errors with context-aware approach (VECTORIZED)
        # Step 1: Basic correction (extreme values) - fully vectorized
        corrected_tempos = np.where(
            tempos_array < 70.0, tempos_array * 2,
            np.where(tempos_array > 200.0, tempos_array / 2, tempos_array)
        )

        # Step 2: Context-aware octave correction (vectorized using rolling median)
        if len(corrected_tempos) > 5:
            from scipy.ndimage import median_filter

            # Compute rolling median (excludes current value via filter)
            window_size = 5
            # Use median filter which includes current value
            rolling_median = median_filter(corrected_tempos, size=window_size, mode='nearest')

            # Compute ratios
            ratios = corrected_tempos / (rolling_median + 1e-10)

            # Vectorized octave correction
            # If tempo is ~half of neighbors, double it
            double_mask = (ratios > 0.45) & (ratios < 0.55)
            # If tempo is ~double of neighbors, halve it
            halve_mask = (ratios > 1.9) & (ratios < 2.1)

            corrected_tempos = np.where(double_mask, corrected_tempos * 2,
                                        np.where(halve_mask, corrected_tempos / 2, corrected_tempos))

        tempos_array = corrected_tempos

        # Step 3: Apply median smoothing to remove remaining octave errors
        # Use large window (11 samples) to smooth out noise
        from scipy.signal import medfilt
        if len(tempos_array) > 11:
            tempos_array = medfilt(tempos_array, kernel_size=11)

        # Compute statistics
        tempo_mean = float(np.mean(tempos_array))
        tempo_std = float(np.std(tempos_array))
        tempo_min = float(np.min(tempos_array))
        tempo_max = float(np.max(tempos_array))
        tempo_range = tempo_max - tempo_min

        # Build histogram (round to nearest integer BPM)
        tempos_rounded = [int(round(t)) for t in tempos_array]
        tempo_counts = Counter(tempos_rounded)
        tempo_histogram = dict(tempo_counts)

        # Find dominant tempo
        dominant_tempo = tempo_counts.most_common(1)[0][0]

        return TempoDistributionAnalysisResult(
            tempo_mean=tempo_mean,
            tempo_std=tempo_std,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
            tempo_range=tempo_range,
            tempo_histogram=tempo_histogram,
            dominant_tempo=dominant_tempo,
            tempo_trajectory=list(tempos_array),
        )

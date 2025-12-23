"""
Local Tempo Analysis Task - PLP-based tempo analysis for DJ sets.

Single Responsibility: Analyze LOCAL tempo variations using PLP.
Does NOT call other Tasks - Pipeline orchestrates.

Uses Primitives:
- compute_onset_strength
- compute_plp_tempo
- segment_by_tempo_changes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .base import AudioContext, TaskResult, BaseTask
from app.common.primitives import (
    # NOTE: compute_onset_strength is BLOCKED - use context.stft_cache.get_onset_strength() instead
    compute_plp_tempo,
    segment_by_tempo_changes,
    PLPResult,
    TempoSegment,
)


@dataclass
class LocalTempoResult(TaskResult):
    """
    Result of local tempo analysis.

    Provides frame-by-frame tempo and tempo segments for DJ sets.
    """
    success: bool = True
    task_name: str = "LocalTempoAnalysis"
    processing_time_sec: float = 0.0
    error: Optional[str] = None

    # PLP data (local tempo per frame)
    local_tempo: np.ndarray = field(default_factory=lambda: np.array([]))
    pulse_strength: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))

    # Tempo segments (regions with consistent tempo)
    segments: List[TempoSegment] = field(default_factory=list)

    # Summary statistics
    tempo_mean: float = 120.0
    tempo_std: float = 0.0
    tempo_min: float = 120.0
    tempo_max: float = 120.0
    tempo_range: float = 0.0
    n_segments: int = 0

    def get_tempo_at_time(self, time_sec: float) -> float:
        """Get local tempo at specific time."""
        if len(self.times) == 0:
            return self.tempo_mean
        idx = np.searchsorted(self.times, time_sec)
        idx = min(idx, len(self.local_tempo) - 1)
        return float(self.local_tempo[idx])

    def get_segment_at_time(self, time_sec: float) -> Optional[TempoSegment]:
        """Get tempo segment containing time."""
        for seg in self.segments:
            if seg.start_sec <= time_sec <= seg.end_sec:
                return seg
        return None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'tempo_mean': float(self.tempo_mean),
            'tempo_std': float(self.tempo_std),
            'tempo_min': float(self.tempo_min),
            'tempo_max': float(self.tempo_max),
            'tempo_range': float(self.tempo_range),
            'n_segments': self.n_segments,
            'segments': [
                {
                    'start_sec': seg.start_sec,
                    'end_sec': seg.end_sec,
                    'mean_tempo': seg.mean_tempo,
                    'tempo_std': seg.tempo_std,
                    'confidence': seg.confidence,
                }
                for seg in self.segments
            ],
        })
        return base


class LocalTempoAnalysisTask(BaseTask):
    """
    Analyze local tempo variations using PLP (Predominant Local Pulse).

    Essential for DJ sets where tempo varies (118 → 145 BPM).
    Single track analysis uses global tempo; DJ sets need LOCAL tempo.

    Single Responsibility: Local tempo analysis ONLY.
    Does NOT call other Tasks.

    Uses Primitives:
    - compute_onset_strength (from STFT)
    - compute_plp_tempo (frame-by-frame tempo)
    - segment_by_tempo_changes (tempo regions)
    """

    def __init__(
        self,
        tempo_min: float = 60.0,
        tempo_max: float = 200.0,
        prior_bpm: float = 128.0,
        prior_weight: float = 0.5,
        min_segment_sec: float = 30.0,
        tempo_change_threshold: float = 5.0,
    ):
        """
        Initialize local tempo analysis task.

        Args:
            tempo_min: Minimum expected BPM
            tempo_max: Maximum expected BPM
            prior_bpm: Prior tempo estimate (regularization)
            prior_weight: Weight of prior (0 = no prior)
            min_segment_sec: Minimum segment duration
            tempo_change_threshold: BPM change threshold for segmentation
        """
        super().__init__()
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.prior_bpm = prior_bpm
        self.prior_weight = prior_weight
        self.min_segment_sec = min_segment_sec
        self.tempo_change_threshold = tempo_change_threshold

    @property
    def name(self) -> str:
        return "LocalTempoAnalysis"

    def execute(self, context: AudioContext) -> LocalTempoResult:
        """
        Analyze local tempo from AudioContext.

        Args:
            context: AudioContext with STFT cache

        Returns:
            LocalTempoResult with PLP data and segments
        """
        cache = context.stft_cache
        sr = context.sr
        hop_length = cache.hop_length

        # Step 1: Compute onset strength from STFTCache (librosa-based)
        onset_env = cache.get_onset_strength(aggregate=True)

        # Step 2: Compute PLP (primitive)
        plp_result = compute_plp_tempo(
            onset_env=onset_env,
            sr=sr,
            hop_length=hop_length,
            tempo_min=self.tempo_min,
            tempo_max=self.tempo_max,
            prior_bpm=self.prior_bpm,
            prior_weight=self.prior_weight,
        )

        # Step 3: Segment by tempo changes (primitive)
        segments = segment_by_tempo_changes(
            plp_result=plp_result,
            min_segment_sec=self.min_segment_sec,
            tempo_change_threshold=self.tempo_change_threshold,
        )

        # Compute statistics
        local_tempo = plp_result.local_tempo
        if len(local_tempo) > 0:
            tempo_mean = float(np.mean(local_tempo))
            tempo_std = float(np.std(local_tempo))
            tempo_min = float(np.min(local_tempo))
            tempo_max = float(np.max(local_tempo))
            tempo_range = tempo_max - tempo_min
        else:
            tempo_mean = self.prior_bpm
            tempo_std = 0.0
            tempo_min = self.prior_bpm
            tempo_max = self.prior_bpm
            tempo_range = 0.0

        return LocalTempoResult(
            success=True,
            task_name=self.name,
            processing_time_sec=0.0,
            # PLP data
            local_tempo=local_tempo,
            pulse_strength=plp_result.pulse_strength,
            times=plp_result.times,
            # Segments
            segments=segments,
            # Statistics
            tempo_mean=tempo_mean,
            tempo_std=tempo_std,
            tempo_min=tempo_min,
            tempo_max=tempo_max,
            tempo_range=tempo_range,
            n_segments=len(segments),
        )

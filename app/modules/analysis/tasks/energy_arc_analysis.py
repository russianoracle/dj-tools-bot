"""
Energy Arc Analysis Task - Analyze energy trajectory over DJ sets.

Computes metrics describing how DJ manages energy:
- Opening/peak/closing energy levels
- Arc shape classification
- Energy variance (stability indicator)
- Full energy trajectory

Uses Primitives layer for all mathematical operations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .base import AudioContext, TaskResult, BaseTask
from app.common.primitives import (
    smooth_gaussian,
    normalize_minmax,
)
# INTERNAL import from submodule (bypass guard for internal use)
# This task processes raw audio directly, not through STFTCache
from app.common.primitives.energy import compute_rms_from_audio


@dataclass
class EnergyArcAnalysisResult(TaskResult):
    """
    Result of energy arc analysis.

    Attributes:
        opening_energy: First 5 minutes average RMS (normalized 0-1)
        peak_energy: 95th percentile of RMS
        closing_energy: Last 5 minutes average RMS
        arc_shape: Overall energy trajectory classification
        energy_variance: Std dev of smoothed RMS (stability indicator)
        trajectory: Full energy curve (smoothed)
        opening_to_peak_ratio: opening / peak (how much room to build)
        peak_timing_normalized: When peak occurs (0=start, 1=end)
    """
    # Inherited from TaskResult (need defaults because they come after required fields)
    success: bool = True
    task_name: str = "EnergyArcAnalysis"
    processing_time_sec: float = 0.0
    error: Optional[str] = None

    # Energy arc specific fields
    opening_energy: float = 0.0
    peak_energy: float = 0.0
    closing_energy: float = 0.0
    arc_shape: str = "plateau"
    energy_variance: float = 0.0
    trajectory: Optional[np.ndarray] = None
    opening_to_peak_ratio: float = 0.0
    peak_timing_normalized: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'opening_energy': float(self.opening_energy),
            'peak_energy': float(self.peak_energy),
            'closing_energy': float(self.closing_energy),
            'arc_shape': self.arc_shape,
            'energy_variance': float(self.energy_variance),
            'opening_to_peak_ratio': float(self.opening_to_peak_ratio),
            'peak_timing_normalized': float(self.peak_timing_normalized),
            'trajectory': self.trajectory.tolist() if isinstance(self.trajectory, np.ndarray) else self.trajectory,
        })
        return base


class EnergyArcAnalysisTask(BaseTask):
    """
    Analyze energy arc of a DJ set.

    Extracts RMS energy, smooths it, and computes arc metrics.
    All mathematical operations delegated to Primitives layer.
    """

    def __init__(
        self,
        frame_length: int = 110250,     # ~5 sec at 22050 Hz
        hop_length: int = 55125,         # ~2.5 sec hop (50% overlap)
        smooth_sigma_sec: float = 30.0,  # Gaussian smoothing (seconds)
        opening_duration_sec: float = 300.0,  # First 5 minutes
        closing_duration_sec: float = 300.0,  # Last 5 minutes
    ):
        """
        Initialize energy arc analysis task.

        Args:
            frame_length: Frame length for RMS calculation (samples)
            hop_length: Hop between frames (samples)
            smooth_sigma_sec: Gaussian smoothing sigma (seconds)
            opening_duration_sec: Duration to consider as "opening"
            closing_duration_sec: Duration to consider as "closing"
        """
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.smooth_sigma_sec = smooth_sigma_sec
        self.opening_duration_sec = opening_duration_sec
        self.closing_duration_sec = closing_duration_sec

    @property
    def name(self) -> str:
        return "EnergyArcAnalysis"

    def execute(self, context: AudioContext) -> EnergyArcAnalysisResult:
        """
        Analyze energy arc.

        Args:
            context: Audio context with signal

        Returns:
            EnergyArcAnalysisResult with arc metrics
        """
        y = context.y
        sr = context.sr

        # 1. Compute RMS energy using Primitives
        rms = compute_rms_from_audio(
            y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        # 2. Normalize to [0, 1] using Primitives
        rms_normalized = normalize_minmax(rms)

        # 3. Smooth with Gaussian filter using Primitives
        # Convert sigma from seconds to frame indices
        time_per_frame = self.hop_length / sr
        sigma_frames = self.smooth_sigma_sec / time_per_frame
        rms_smooth = smooth_gaussian(rms_normalized, sigma=sigma_frames)

        # 4. Compute metrics
        duration_sec = len(y) / sr
        frames_per_sec = 1.0 / time_per_frame

        # Opening energy (first N minutes or 25% of set, whichever is smaller)
        opening_frames = int(self.opening_duration_sec * frames_per_sec)
        opening_frames = min(opening_frames, len(rms_smooth) // 4)
        opening_energy = float(np.mean(rms_smooth[:opening_frames]))

        # Closing energy (last N minutes or 25% of set)
        closing_frames = int(self.closing_duration_sec * frames_per_sec)
        closing_frames = min(closing_frames, len(rms_smooth) // 4)
        closing_energy = float(np.mean(rms_smooth[-closing_frames:]))

        # Peak energy (95th percentile)
        peak_energy = float(np.percentile(rms_smooth, 95))

        # Energy variance (stability)
        energy_variance = float(np.std(rms_smooth))

        # Peak timing (when does peak occur?)
        peak_idx = np.argmax(rms_smooth)
        peak_timing_normalized = peak_idx / len(rms_smooth)

        # Opening to peak ratio
        opening_to_peak_ratio = opening_energy / (peak_energy + 1e-8)

        # Classify arc shape
        arc_shape = self._classify_arc_shape(
            opening_energy, peak_energy, closing_energy,
            peak_timing_normalized, energy_variance
        )

        return EnergyArcAnalysisResult(
            opening_energy=opening_energy,
            peak_energy=peak_energy,
            closing_energy=closing_energy,
            arc_shape=arc_shape,
            energy_variance=energy_variance,
            trajectory=rms_smooth,
            opening_to_peak_ratio=opening_to_peak_ratio,
            peak_timing_normalized=peak_timing_normalized,
        )

    def _classify_arc_shape(
        self,
        opening: float,
        peak: float,
        closing: float,
        peak_timing: float,
        variance: float
    ) -> str:
        """
        Classify energy arc shape.

        Rules:
        - chaotic: High variance (>0.25)
        - plateau: Sustained high energy, low variance
        - crescendo: Builds up, peaks near end
        - peak_and_fade: Peak in middle/early, energy drops later

        Args:
            opening: Opening energy
            peak: Peak energy
            closing: Closing energy
            peak_timing: Peak timing (0=start, 1=end)
            variance: Energy variance

        Returns:
            Arc shape classification
        """
        # Chaotic: high variance
        if variance > 0.25:
            return "chaotic"

        # Crescendo: builds up, peaks near end (check first to avoid plateau false positive)
        if closing > opening * 1.3 and peak_timing > 0.6:
            return "crescendo"

        # Peak and fade: peak early/middle, significant drop afterward
        # Check if peak is significantly higher than both opening and closing
        if peak_timing < 0.7 and (peak - closing) > 0.10 and (peak - opening) > 0.10:
            return "peak_and_fade"

        # Plateau: sustained high energy, low variance
        if opening > 0.6 and closing > 0.6 and variance < 0.15:
            return "plateau"

        # Moderate plateau: moderate energy with low variance
        if variance < 0.12 and abs(opening - closing) < 0.15:
            return "plateau"

        # Default: plateau (conservative choice)
        return "plateau"

"""
Core module configuration.

Centralized configuration for all core tasks and pipelines.
Supports presets for different DJ mixing styles.
"""

from dataclasses import dataclass, field
from enum import Enum


class MixingStyle(Enum):
    """DJ mixing style presets."""
    SMOOTH = "smooth"      # Long blends, minimal energy changes (techno, minimal)
    STANDARD = "standard"  # Typical club mixing
    HARD = "hard"          # Quick cuts, high energy changes (bass, dubstep)


@dataclass
class TransitionConfig:
    """Configuration for transition detection."""
    min_transition_gap_sec: float = 30.0
    energy_threshold: float = 0.25
    bass_weight: float = 0.5
    smooth_sigma: float = 5.0
    detect_filters: bool = True
    filter_velocity_threshold: float = 500.0
    # Peak detection percentile (lower = more sensitive)
    peak_percentile: float = 85.0
    # Window to merge nearby mixin/mixout points into single transition zone
    transition_merge_window_sec: float = 90.0
    # Timbral weight for mixin detection (0=energy-based, 1=timbral-only)
    # For smooth mixing, new tracks fade in via fader (timbral change)
    # not via energy/bass increase
    timbral_weight: float = 0.0
    # Context filter parameters
    verbose: bool = False
    drop_window_sec: float = 10.0
    drop_confidence_threshold: float = 0.3
    boundary_tolerance_sec: float = 15.0

    @classmethod
    def for_style(cls, style: MixingStyle) -> 'TransitionConfig':
        """Get config preset for mixing style."""
        presets = {
            # CALIBRATED with F-beta(0.5) + context filters
            # Calibration: F-beta=0.277, P=24.4%, R=61.0% (WITHOUT filters in calibration loop)
            # Context filters (drop + segmentation) applied in production execute() only
            MixingStyle.SMOOTH: cls(
                min_transition_gap_sec=50.5,
                energy_threshold=0.171,
                bass_weight=0.42,
                smooth_sigma=3.6,
                filter_velocity_threshold=200.0,
                peak_percentile=90.7,
                transition_merge_window_sec=30.0,
                timbral_weight=0.95,
                verbose=False,
                drop_window_sec=10.0,
                drop_confidence_threshold=0.3,
                boundary_tolerance_sec=15.0,
            ),
            MixingStyle.STANDARD: cls(
                min_transition_gap_sec=30.0,
                energy_threshold=0.25,
                bass_weight=0.5,
                smooth_sigma=5.0,
                filter_velocity_threshold=500.0,
                peak_percentile=85.0,
                transition_merge_window_sec=90.0,
                verbose=False,
                drop_window_sec=10.0,
                drop_confidence_threshold=0.3,
                boundary_tolerance_sec=15.0,
            ),
            MixingStyle.HARD: cls(
                min_transition_gap_sec=15.0,
                energy_threshold=0.4,
                bass_weight=0.4,
                smooth_sigma=3.0,
                filter_velocity_threshold=800.0,
                peak_percentile=92.0,
                transition_merge_window_sec=60.0,  # Shorter transitions in hard styles
                verbose=False,
                drop_window_sec=8.0,  # Tighter window for hard styles
                drop_confidence_threshold=0.4,
                boundary_tolerance_sec=12.0,
            ),
        }
        return presets[style]


@dataclass
class DropDetectionConfig:
    """Configuration for drop detection."""
    min_drop_magnitude: float = 0.25
    min_confidence: float = 0.4
    buildup_window_sec: float = 2.0
    use_multiband: bool = True

    @classmethod
    def for_style(cls, style: MixingStyle) -> 'DropDetectionConfig':
        """Get config preset for mixing style."""
        presets = {
            MixingStyle.SMOOTH: cls(
                min_drop_magnitude=0.12,  # Lower threshold for subtle drops
                min_confidence=0.20,      # Much lower - smooth music has no buildups
                buildup_window_sec=4.0,   # Longer buildup window
                use_multiband=False,      # Multiband compresses dynamics too much
            ),
            MixingStyle.STANDARD: cls(
                min_drop_magnitude=0.25,
                min_confidence=0.4,
                buildup_window_sec=2.0,
            ),
            MixingStyle.HARD: cls(
                min_drop_magnitude=0.35,
                min_confidence=0.5,
                buildup_window_sec=1.5,
            ),
        }
        return presets[style]


@dataclass
class SegmentationConfig:
    """Configuration for track segmentation."""
    min_track_duration: float = 60.0       # Minimum TRACK (solo) duration
    min_transition_duration: float = 30.0  # Minimum TRANSITION (overlap) duration
    min_segment_for_genre: float = 30.0

    @classmethod
    def for_style(cls, style: MixingStyle) -> 'SegmentationConfig':
        """Get config preset for mixing style."""
        presets = {
            # SMOOTH: Tracks can have short solo periods (1 min+), transitions 30-60 sec
            MixingStyle.SMOOTH: cls(
                min_track_duration=60.0,       # 1 min - allows short solo periods
                min_transition_duration=30.0,  # Transitions can be shorter
                min_segment_for_genre=30.0,
            ),
            MixingStyle.STANDARD: cls(
                min_track_duration=60.0,
                min_transition_duration=30.0,
                min_segment_for_genre=30.0,
            ),
            MixingStyle.HARD: cls(
                min_track_duration=90.0,
                min_transition_duration=15.0,  # Hard cuts = short transitions
                min_segment_for_genre=30.0,
            ),
        }
        return presets.get(style, cls())


@dataclass
class SetAnalysisConfig:
    """Configuration for set analysis pipeline."""
    sr: int = 22050
    timeline_resolution: float = 1.0
    analyze_genres: bool = False
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    drop_detection: DropDetectionConfig = field(default_factory=DropDetectionConfig)

    @classmethod
    def for_style(cls, style: MixingStyle, analyze_genres: bool = False) -> 'SetAnalysisConfig':
        """Get config preset for mixing style."""
        return cls(
            analyze_genres=analyze_genres,
            transition=TransitionConfig.for_style(style),
            segmentation=SegmentationConfig.for_style(style),
            drop_detection=DropDetectionConfig.for_style(style),
        )


# Default configurations
DEFAULT_TRANSITION = TransitionConfig()
DEFAULT_SEGMENTATION = SegmentationConfig()
DEFAULT_SET_ANALYSIS = SetAnalysisConfig()

# Style presets
SMOOTH_MIXING = SetAnalysisConfig.for_style(MixingStyle.SMOOTH)
STANDARD_MIXING = SetAnalysisConfig.for_style(MixingStyle.STANDARD)
HARD_MIXING = SetAnalysisConfig.for_style(MixingStyle.HARD)

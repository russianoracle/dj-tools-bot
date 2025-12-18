"""
Domain Models for Unified Cache System.

These dataclasses represent the domain entities that are cached.
All models have to_dict() and from_dict() for serialization.

Domain entities:
- CachedSetAnalysis: DJ set analysis (transitions, drops, segments)
- CachedTrackAnalysis: Individual track (bpm, key, energy)
- CachedDJProfile: Aggregated DJ style profile
- CachedFeatures: Extracted audio features for reuse
- CachedDrop: Single drop event
- CachedTransition: Single transition (mixin/mixout)
- CachedSegment: Track segment within a set
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class DropType(Enum):
    """Type of drop detected."""
    FULL = "full"
    ENERGY = "energy"
    BASS = "bass"
    BUILD_DROP = "build_drop"


class TransitionType(Enum):
    """Type of transition detected."""
    MIXIN = "mixin"
    MIXOUT = "mixout"


@dataclass
class CachedDrop:
    """A cached drop event."""
    time_sec: float
    confidence: float
    drop_type: str  # DropType name
    energy_delta: float = 0.0
    bass_delta: float = 0.0
    buildup_duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'drop_type': self.drop_type,
            'energy_delta': self.energy_delta,
            'bass_delta': self.bass_delta,
            'buildup_duration_sec': self.buildup_duration_sec,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedDrop':
        return cls(
            time_sec=d['time_sec'],
            confidence=d['confidence'],
            drop_type=d.get('drop_type', 'FULL'),
            energy_delta=d.get('energy_delta', 0.0),
            bass_delta=d.get('bass_delta', 0.0),
            buildup_duration_sec=d.get('buildup_duration_sec', 0.0),
        )


@dataclass
class CachedTransition:
    """A cached transition event (mixin or mixout)."""
    time_sec: float
    confidence: float
    transition_type: str  # TransitionType name
    energy_change: float = 0.0
    bass_change: float = 0.0

    def to_dict(self) -> dict:
        return {
            'time_sec': self.time_sec,
            'confidence': self.confidence,
            'transition_type': self.transition_type,
            'energy_change': self.energy_change,
            'bass_change': self.bass_change,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedTransition':
        return cls(
            time_sec=d['time_sec'],
            confidence=d['confidence'],
            transition_type=d.get('transition_type', 'MIXIN'),
            energy_change=d.get('energy_change', 0.0),
            bass_change=d.get('bass_change', 0.0),
        )


@dataclass
class CachedSegment:
    """A cached segment (track) within a DJ set."""
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    avg_energy: float = 0.0
    zone: Optional[str] = None  # YELLOW/GREEN/PURPLE

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'start_sec': self.start_sec,
            'end_sec': self.end_sec,
            'duration_sec': self.duration_sec,
            'avg_energy': self.avg_energy,
            'zone': self.zone,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedSegment':
        return cls(
            index=d.get('index', d.get('segment_index', 0)),
            start_sec=d.get('start_sec', d.get('start_time', 0.0)),
            end_sec=d.get('end_sec', d.get('end_time', 0.0)),
            duration_sec=d.get('duration_sec', d.get('duration', 0.0)),
            avg_energy=d.get('avg_energy', 0.0),
            zone=d.get('zone'),
        )


@dataclass
class CachedSetAnalysis:
    """
    Cached DJ set analysis.

    Contains all analysis results for a DJ set:
    - Transitions (mixin/mixout points)
    - Drops (energy drops)
    - Segments (track boundaries)
    """
    file_path: str
    file_name: str
    duration_sec: float

    # Transitions
    n_transitions: int = 0
    transition_times: List[float] = field(default_factory=list)
    transition_density: float = 0.0  # per hour
    transitions: List[CachedTransition] = field(default_factory=list)

    # Drops
    total_drops: int = 0
    drop_density: float = 0.0  # per hour
    drops: List[CachedDrop] = field(default_factory=list)

    # Segments (tracks)
    n_segments: int = 0
    segments: List[CachedSegment] = field(default_factory=list)
    avg_segment_duration_sec: float = 0.0

    # Metadata
    processing_time_sec: float = 0.0
    analysis_version: str = "2.0"

    def to_dict(self) -> dict:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'duration_sec': self.duration_sec,
            'n_transitions': self.n_transitions,
            'transition_times': self.transition_times,
            'transition_density': self.transition_density,
            'transitions': [t.to_dict() for t in self.transitions],
            'total_drops': self.total_drops,
            'drop_density': self.drop_density,
            'drops': [d.to_dict() for d in self.drops],
            'n_segments': self.n_segments,
            'segments': [s.to_dict() for s in self.segments],
            'avg_segment_duration_sec': self.avg_segment_duration_sec,
            'processing_time_sec': self.processing_time_sec,
            'analysis_version': self.analysis_version,
            'success': True,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedSetAnalysis':
        # Parse transitions
        transitions = []
        for t in d.get('transitions', []):
            if isinstance(t, dict):
                transitions.append(CachedTransition.from_dict(t))

        # Parse drops
        drops = []
        for drop in d.get('drops', []):
            if isinstance(drop, dict):
                drops.append(CachedDrop.from_dict(drop))

        # Parse segments
        segments = []
        for s in d.get('segments', []):
            if isinstance(s, dict):
                segments.append(CachedSegment.from_dict(s))

        return cls(
            file_path=d.get('file_path', ''),
            file_name=d.get('file_name', ''),
            duration_sec=d.get('duration_sec', 0.0),
            n_transitions=d.get('n_transitions', 0),
            transition_times=d.get('transition_times', []),
            transition_density=d.get('transition_density', 0.0),
            transitions=transitions,
            total_drops=d.get('total_drops', 0),
            drop_density=d.get('drop_density', 0.0),
            drops=drops,
            n_segments=d.get('n_segments', len(segments)),
            segments=segments,
            avg_segment_duration_sec=d.get('avg_segment_duration_sec', 0.0),
            processing_time_sec=d.get('processing_time_sec', 0.0),
            analysis_version=d.get('analysis_version', '1.0'),
        )

    @property
    def duration_hours(self) -> float:
        """Duration in hours."""
        return self.duration_sec / 3600.0

    @property
    def drops_per_hour(self) -> float:
        """Drops per hour."""
        if self.duration_hours > 0:
            return self.total_drops / self.duration_hours
        return 0.0

    @property
    def transitions_per_hour(self) -> float:
        """Transitions per hour."""
        if self.duration_hours > 0:
            return self.n_transitions / self.duration_hours
        return 0.0


@dataclass
class CachedTrackAnalysis:
    """
    Cached individual track analysis.

    Contains:
    - BPM, key, energy level
    - Zone classification
    - Audio features
    """
    file_path: str
    file_name: str
    duration_sec: float

    # Core analysis
    bpm: float = 0.0
    key: str = ""
    energy_level: float = 0.0
    zone: str = ""  # YELLOW/GREEN/PURPLE
    zone_confidence: float = 0.0

    # Features
    features: Dict[str, float] = field(default_factory=dict)

    # Metadata
    processing_time_sec: float = 0.0
    analysis_version: str = "2.0"

    def to_dict(self) -> dict:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'duration_sec': self.duration_sec,
            'bpm': self.bpm,
            'key': self.key,
            'energy_level': self.energy_level,
            'zone': self.zone,
            'zone_confidence': self.zone_confidence,
            'features': self.features,
            'processing_time_sec': self.processing_time_sec,
            'analysis_version': self.analysis_version,
            'success': True,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedTrackAnalysis':
        return cls(
            file_path=d.get('file_path', ''),
            file_name=d.get('file_name', ''),
            duration_sec=d.get('duration_sec', 0.0),
            bpm=d.get('bpm', 0.0),
            key=d.get('key', ''),
            energy_level=d.get('energy_level', 0.0),
            zone=d.get('zone', ''),
            zone_confidence=d.get('zone_confidence', 0.0),
            features=d.get('features', {}),
            processing_time_sec=d.get('processing_time_sec', 0.0),
            analysis_version=d.get('analysis_version', '1.0'),
        )


@dataclass
class CachedDJProfile:
    """
    Cached aggregated DJ style profile.

    Contains:
    - Drop patterns (frequency, types)
    - Transition patterns (frequency, lengths)
    - Energy arc preferences
    - Source set references
    """
    dj_name: str

    # Drop patterns
    avg_drops_per_hour: float = 0.0
    drop_type_distribution: Dict[str, float] = field(default_factory=dict)
    avg_drop_confidence: float = 0.0

    # Transition patterns
    avg_transitions_per_hour: float = 0.0
    avg_track_duration_sec: float = 0.0
    transition_length_distribution: Dict[str, float] = field(default_factory=dict)

    # Energy patterns
    energy_arc_type: str = ""  # build, maintain, wave
    avg_energy_level: float = 0.0
    energy_variance: float = 0.0

    # Source data
    n_sets_analyzed: int = 0
    total_hours_analyzed: float = 0.0
    set_file_paths: List[str] = field(default_factory=list)

    # Metadata
    created_at: float = 0.0
    updated_at: float = 0.0
    profile_version: str = "2.0"

    def to_dict(self) -> dict:
        return {
            'dj_name': self.dj_name,
            'avg_drops_per_hour': self.avg_drops_per_hour,
            'drop_type_distribution': self.drop_type_distribution,
            'avg_drop_confidence': self.avg_drop_confidence,
            'avg_transitions_per_hour': self.avg_transitions_per_hour,
            'avg_track_duration_sec': self.avg_track_duration_sec,
            'transition_length_distribution': self.transition_length_distribution,
            'energy_arc_type': self.energy_arc_type,
            'avg_energy_level': self.avg_energy_level,
            'energy_variance': self.energy_variance,
            'n_sets_analyzed': self.n_sets_analyzed,
            'total_hours_analyzed': self.total_hours_analyzed,
            'set_file_paths': self.set_file_paths,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'profile_version': self.profile_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CachedDJProfile':
        return cls(
            dj_name=d.get('dj_name', ''),
            avg_drops_per_hour=d.get('avg_drops_per_hour', 0.0),
            drop_type_distribution=d.get('drop_type_distribution', {}),
            avg_drop_confidence=d.get('avg_drop_confidence', 0.0),
            avg_transitions_per_hour=d.get('avg_transitions_per_hour', 0.0),
            avg_track_duration_sec=d.get('avg_track_duration_sec', 0.0),
            transition_length_distribution=d.get('transition_length_distribution', {}),
            energy_arc_type=d.get('energy_arc_type', ''),
            avg_energy_level=d.get('avg_energy_level', 0.0),
            energy_variance=d.get('energy_variance', 0.0),
            n_sets_analyzed=d.get('n_sets_analyzed', 0),
            total_hours_analyzed=d.get('total_hours_analyzed', 0.0),
            set_file_paths=d.get('set_file_paths', []),
            created_at=d.get('created_at', 0.0),
            updated_at=d.get('updated_at', 0.0),
            profile_version=d.get('profile_version', '1.0'),
        )


@dataclass
class CachedFeatures:
    """
    Cached extracted audio features for reuse.

    Stores expensive-to-compute features that can be reused
    across different analysis tasks (transitions, drops, segmentation).

    Features are stored as numpy arrays, serialized to .npy files.
    """
    file_path: str
    file_hash: str

    # Audio metadata
    sr: int = 22050
    hop_length: int = 512
    duration_sec: float = 0.0

    # Energy features
    rms: Optional[np.ndarray] = None
    bass_energy: Optional[np.ndarray] = None

    # Spectral features
    centroid: Optional[np.ndarray] = None
    rolloff: Optional[np.ndarray] = None
    brightness: Optional[np.ndarray] = None

    # Novelty features
    novelty: Optional[np.ndarray] = None
    timbral_novelty: Optional[np.ndarray] = None
    chroma_novelty: Optional[np.ndarray] = None

    # Velocity (for filter detection)
    centroid_velocity: Optional[np.ndarray] = None

    # Beat/tempo
    tempo: float = 0.0
    beat_times: Optional[np.ndarray] = None

    # Metadata
    created_at: float = 0.0
    feature_version: str = "1.0"

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names (vectorized)."""
        all_names = np.array(['rms', 'bass_energy', 'centroid', 'rolloff', 'brightness',
                              'novelty', 'timbral_novelty', 'chroma_novelty',
                              'centroid_velocity', 'beat_times'])
        available_mask = np.array([getattr(self, n, None) is not None for n in all_names])
        return all_names[available_mask].tolist()

    def has_features(self, feature_names: List[str]) -> bool:
        """Check if all requested features are available (vectorized)."""
        names_arr = np.array(feature_names)
        available_mask = np.array([getattr(self, n, None) is not None for n in names_arr])
        return bool(np.all(available_mask))

    def to_dict(self) -> dict:
        """Convert to dict (arrays stored separately as .npy files)."""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'sr': self.sr,
            'hop_length': self.hop_length,
            'duration_sec': self.duration_sec,
            'tempo': self.tempo,
            'created_at': self.created_at,
            'feature_version': self.feature_version,
            'available_features': self.get_feature_names(),
        }

    @classmethod
    def from_dict(cls, d: dict, arrays: Dict[str, np.ndarray] = None) -> 'CachedFeatures':
        """Create from dict and optional numpy arrays."""
        arrays = arrays or {}
        return cls(
            file_path=d.get('file_path', ''),
            file_hash=d.get('file_hash', ''),
            sr=d.get('sr', 22050),
            hop_length=d.get('hop_length', 512),
            duration_sec=d.get('duration_sec', 0.0),
            tempo=d.get('tempo', 0.0),
            created_at=d.get('created_at', 0.0),
            feature_version=d.get('feature_version', '1.0'),
            # Arrays loaded separately
            rms=arrays.get('rms'),
            bass_energy=arrays.get('bass_energy'),
            centroid=arrays.get('centroid'),
            rolloff=arrays.get('rolloff'),
            brightness=arrays.get('brightness'),
            novelty=arrays.get('novelty'),
            timbral_novelty=arrays.get('timbral_novelty'),
            chroma_novelty=arrays.get('chroma_novelty'),
            centroid_velocity=arrays.get('centroid_velocity'),
            beat_times=arrays.get('beat_times'),
        )

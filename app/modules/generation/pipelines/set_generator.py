"""
Set Generator Pipeline - Generate DJ set plans from profiles.

Two-phase workflow:
1. generate_plan() - Quick plan from Rekordbox metadata (no ML)
2. verify_and_optimize() - Full ML analysis + transition optimization

Uses:
- DJ profiles from CacheRepository
- Tracks from pyrekordbox
- TrackCompatibilityTask for analysis
- TransitionScore for optimization
"""

import logging
import time
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum

from app.core.connectors import CacheRepository
from .track_compatibility import (
    TrackCompatibilityPipeline,
    TrackCompatibilityResult,
)
from app.modules.analysis.tasks.track_compatibility import (
    TrackAnalysis,
    compute_track_transition_score,
)
from app.common.primitives.transition_scoring import (
    TransitionScore,
    compute_transition_score,
    score_harmonic_compatibility,
    score_harmonic_progression,
    score_bpm_compatibility,
)

logger = logging.getLogger(__name__)


# ============== Set Phase Definitions ==============

class SetPhase(Enum):
    """Phases of a DJ set based on energy arc."""
    OPENING = "opening"       # Warm-up, low energy
    BUILD = "build"           # Rising energy
    PEAK = "peak"             # High energy, main room
    CLOSING = "closing"       # Cool down


@dataclass
class PhaseConfig:
    """Configuration for a set phase."""
    phase: SetPhase
    energy_min: float         # Minimum energy (0-1)
    energy_max: float         # Maximum energy (0-1)
    bpm_range: Tuple[float, float]  # BPM range for this phase
    duration_ratio: float     # What fraction of set this phase takes

    @classmethod
    def default_phases(cls, profile: Optional[Dict] = None) -> List['PhaseConfig']:
        """
        Create default phase configs, optionally based on DJ profile.

        Uses DJ profile for:
        - tempo_distribution: BPM ranges for each phase
        - energy_arc: Energy levels for each phase (opening, peak, closing)

        Args:
            profile: DJ profile dict (from CacheRepository)

        Returns:
            List of PhaseConfig for all phases
        """
        # Default BPM ranges
        base_bpm_min = 122
        base_bpm_max = 135

        # Default energy levels (normalized 0-1)
        opening_energy = 0.35
        peak_energy = 0.85
        closing_energy = 0.45

        # Adjust from profile if available
        if profile:
            # === Tempo Distribution ===
            tempo = profile.get('tempo_distribution', {})
            if tempo:
                dominant = int(tempo.get('dominant_tempo', 128))
                # Use dominant tempo as center, with reasonable range
                # Profile min/max can be extreme outliers, so we use dominant ± 8-10 BPM
                base_bpm_min = dominant - 8
                base_bpm_max = dominant + 6

            # === Energy Arc ===
            energy_arc = profile.get('energy_arc', {})
            if energy_arc:
                # Use DJ's actual energy levels from profiled sets
                opening_energy = float(energy_arc.get('opening_energy', 0.35))
                peak_energy = float(energy_arc.get('peak_energy', 0.85))
                closing_energy = float(energy_arc.get('closing_energy', 0.45))

                # Ensure reasonable bounds
                opening_energy = max(0.1, min(0.6, opening_energy))
                peak_energy = max(0.5, min(1.0, peak_energy))
                closing_energy = max(0.2, min(0.7, closing_energy))

        # Build energy: interpolate between opening and peak
        build_energy_min = opening_energy + 0.1
        build_energy_max = (opening_energy + peak_energy) / 2 + 0.1

        return [
            PhaseConfig(
                phase=SetPhase.OPENING,
                energy_min=max(0.1, opening_energy - 0.15),
                energy_max=opening_energy + 0.15,
                bpm_range=(base_bpm_min - 6, base_bpm_min + 4),
                duration_ratio=0.15,
            ),
            PhaseConfig(
                phase=SetPhase.BUILD,
                energy_min=build_energy_min,
                energy_max=build_energy_max,
                bpm_range=(base_bpm_min, base_bpm_max - 4),
                duration_ratio=0.30,
            ),
            PhaseConfig(
                phase=SetPhase.PEAK,
                energy_min=peak_energy - 0.2,
                energy_max=1.0,
                bpm_range=(base_bpm_max - 6, base_bpm_max + 4),
                duration_ratio=0.40,
            ),
            PhaseConfig(
                phase=SetPhase.CLOSING,
                energy_min=max(0.2, closing_energy - 0.15),
                energy_max=closing_energy + 0.15,
                bpm_range=(base_bpm_min - 4, base_bpm_max - 4),
                duration_ratio=0.15,
            ),
        ]


# ============== Set Plan Data Structures ==============

@dataclass
class PlannedTrack:
    """A track planned for the set."""
    # Track identity
    path: str
    title: str
    artist: str

    # From Rekordbox or analysis
    bpm: float
    key: str
    camelot: str
    duration_sec: float

    # Planning metadata
    phase: SetPhase
    position: int              # Position in set (1-indexed)
    cumulative_time_sec: float # Time when this track starts

    # Optional fields (with defaults)
    genre: str = ""            # From Rekordbox metadata
    year: int = 0              # Release year from Rekordbox
    transition_score: Optional[TransitionScore] = None

    # Full analysis (populated in verify phase)
    analysis: Optional[TrackCompatibilityResult] = None

    # Status
    verified: bool = False     # True if ML-analyzed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            'path': self.path,
            'title': self.title,
            'artist': self.artist,
            'bpm': float(self.bpm),
            'key': self.key,
            'camelot': self.camelot,
            'duration_sec': float(self.duration_sec),
            'genre': self.genre,
            'year': self.year,
            'phase': self.phase.value,
            'position': self.position,
            'cumulative_time_sec': float(self.cumulative_time_sec),
            'verified': self.verified,
        }
        if self.transition_score:
            result['transition_score'] = self.transition_score.to_dict()
        if self.analysis:
            result['analysis'] = self.analysis.to_dict()
        return result


@dataclass
class SetPlan:
    """Complete DJ set plan."""
    # Metadata
    dj_name: str
    duration_min: int          # Target duration in minutes
    created_at: float          # Unix timestamp

    # Tracks
    tracks: List[PlannedTrack] = field(default_factory=list)

    # Quality metrics
    total_transition_score: float = 0.0
    avg_transition_score: float = 0.0
    n_transitions: int = 0

    # Status
    status: str = "draft"      # "draft", "verified", "exported"

    # Unique identifier (for caching)
    plan_id: Optional[str] = None

    def __post_init__(self):
        """Generate plan_id if not provided."""
        if self.plan_id is None:
            import uuid
            self.plan_id = str(uuid.uuid4())[:8]

    @property
    def actual_duration_min(self) -> float:
        """Calculate actual set duration from tracks."""
        if not self.tracks:
            return 0.0
        return sum(t.duration_sec for t in self.tracks) / 60.0

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def get_tracks_by_phase(self, phase: SetPhase) -> List[PlannedTrack]:
        """Get all tracks in a specific phase."""
        return [t for t in self.tracks if t.phase == phase]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'plan_id': self.plan_id,
            'dj_name': self.dj_name,
            'duration_min': self.duration_min,
            'actual_duration_min': float(self.actual_duration_min),
            'created_at': self.created_at,
            'tracks': [t.to_dict() for t in self.tracks],
            'total_transition_score': float(self.total_transition_score),
            'avg_transition_score': float(self.avg_transition_score),
            'n_transitions': self.n_transitions,
            'n_tracks': self.n_tracks,
            'status': self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SetPlan':
        """Create SetPlan from dict (e.g., from cache)."""
        tracks = []
        for t in data.get('tracks', []):
            # Reconstruct PlannedTrack
            track = PlannedTrack(
                path=t['path'],
                title=t['title'],
                artist=t['artist'],
                bpm=t['bpm'],
                key=t['key'],
                camelot=t['camelot'],
                duration_sec=t['duration_sec'],
                genre=t.get('genre', ''),
                year=t.get('year', 0),
                phase=SetPhase(t['phase']),
                position=t['position'],
                cumulative_time_sec=t['cumulative_time_sec'],
                verified=t.get('verified', False),
            )
            # Reconstruct TransitionScore if present
            if 'transition_score' in t and t['transition_score']:
                ts = t['transition_score']
                track.transition_score = TransitionScore(
                    harmonic=ts.get('harmonic', 0.5),
                    energy=ts.get('energy', 0.5),
                    drop_conflict=ts.get('drop_conflict', 0.5),
                    spectral=ts.get('spectral', 0.5),
                    genre=ts.get('genre', 0.5),
                    bpm=ts.get('bpm', 0.5),
                )
            tracks.append(track)

        return cls(
            plan_id=data.get('plan_id'),
            dj_name=data['dj_name'],
            duration_min=data['duration_min'],
            created_at=data['created_at'],
            tracks=tracks,
            total_transition_score=data.get('total_transition_score', 0.0),
            avg_transition_score=data.get('avg_transition_score', 0.0),
            n_transitions=data.get('n_transitions', 0),
            status=data.get('status', 'draft'),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ============== Rekordbox Integration ==============

@dataclass
class RekordboxTrack:
    """Track info from Rekordbox database."""
    path: str
    title: str
    artist: str
    bpm: float
    key: str
    duration_sec: float
    rating: int = 0
    play_count: int = 0
    genre: str = ""
    label: str = ""
    year: int = 0

    @property
    def camelot(self) -> str:
        """Convert Rekordbox key to Camelot notation."""
        import re

        # Check if already in Camelot format (e.g., "8A", "12B")
        if re.match(r'^(1[0-2]|[1-9])[AB]$', self.key):
            return self.key

        # Rekordbox uses different key notation
        key_map = {
            'C': '8B', 'Cm': '5A', 'C♯': '3B', 'C♯m': '12A',
            'D': '10B', 'Dm': '7A', 'D♯': '5B', 'D♯m': '2A',
            'E': '12B', 'Em': '9A',
            'F': '7B', 'Fm': '4A', 'F♯': '2B', 'F♯m': '11A',
            'G': '9B', 'Gm': '6A', 'G♯': '4B', 'G♯m': '1A',
            'A': '11B', 'Am': '8A', 'A♯': '6B', 'A♯m': '3A',
            'B': '1B', 'Bm': '10A',
            # Alternative notations
            'Db': '3B', 'Dbm': '12A',
            'Eb': '5B', 'Ebm': '2A',
            'Gb': '2B', 'Gbm': '11A',
            'Ab': '4B', 'Abm': '1A',
            'Bb': '6B', 'Bbm': '3A',
            # Sharp/flat variations
            'C#': '3B', 'C#m': '12A',
            'D#': '5B', 'D#m': '2A',
            'F#': '2B', 'F#m': '11A',
            'G#': '4B', 'G#m': '1A',
            'A#': '6B', 'A#m': '3A',
        }
        return key_map.get(self.key, '?')


def load_rekordbox_tracks(require_local_file: bool = False) -> List[RekordboxTrack]:
    """
    Load tracks from LOCAL Rekordbox database copy in project directory.

    SECURITY: Only reads from rekordbox_bak_*/master.db in project directory.
    Never accesses production Rekordbox database.

    Args:
        require_local_file: If True, only return tracks with existing local files.
                           If False, return all tracks (metadata-only for planning).

    Returns:
        List of RekordboxTrack objects
    """
    try:
        from pyrekordbox import Rekordbox6Database

        # SECURITY: Only use local DB copy from project directory
        project_root = Path(__file__).parent.parent.parent.parent  # src/core/pipelines -> project root

        # Find latest rekordbox backup directory (in project root or data/)
        # Look for directories matching rekordbox_bak_* pattern
        backup_dirs = []
        for search_dir in [project_root, project_root / "data"]:
            if search_dir.exists():
                backup_dirs.extend(search_dir.glob("rekordbox_bak_*"))

        # Filter to directories with master.db
        valid_backups = [d for d in backup_dirs if (d / "master.db").exists()]

        if not valid_backups:
            logger.error("No Rekordbox backup found!")
            logger.error("Copy your Rekordbox database to data/rekordbox_bak_YYYYMMDD/")
            logger.error("Required file: master.db")
            return []

        # Use the most recent backup (sorted by name = date)
        local_db_dir = sorted(valid_backups, key=lambda p: p.name)[-1]
        local_db_path = local_db_dir / "master.db"

        # Verify path is within project directory (security check)
        try:
            local_db_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            logger.error(f"Security: DB path {local_db_path} is outside project directory")
            return []

        logger.info(f"Using LOCAL Rekordbox DB: {local_db_path}")

        # Rekordbox 6 database is encrypted with SQLCipher
        # The key is publicly known (same for all installations)
        # See: https://github.com/dylanljones/pyrekordbox/discussions/97
        RB6_DB_KEY = "402fd482c38817c35ffa8ffb8c7d93143b749e7d315df7a81732a1ff43608497"

        db = Rekordbox6Database(db_dir=local_db_dir, key=RB6_DB_KEY)
        tracks = []
        skipped_no_path = 0
        skipped_not_found = 0

        for content in db.get_content():
            try:
                # Get file path - try multiple attributes
                # OrgFolderPath contains original local path
                # rb_LocalFolderPath may also have local path
                # FolderPath contains Rekordbox internal path
                path = None
                path_exists = False

                # Try OrgFolderPath first (original import path)
                if content.OrgFolderPath:
                    path = content.OrgFolderPath
                    path_exists = Path(path).exists()

                # Fall back to rb_LocalFolderPath
                if not path and content.rb_LocalFolderPath:
                    path = content.rb_LocalFolderPath
                    path_exists = Path(path).exists()

                # Use FolderPath as last resort (for metadata-only mode)
                if not path and content.FolderPath:
                    # Skip streaming tracks (soundcloud:, etc.)
                    if not content.FolderPath.startswith(('soundcloud:', 'spotify:', 'http')):
                        path = content.FolderPath
                        path_exists = False

                # Skip tracks without any path
                if not path:
                    skipped_no_path += 1
                    continue

                # If require_local_file, skip non-existing files
                if require_local_file and not path_exists:
                    skipped_not_found += 1
                    continue

                # Get key name safely
                key_name = ""
                if content.Key:
                    try:
                        key_name = content.Key.ScaleName
                    except Exception:
                        key_name = str(content.Key) if content.Key else ""

                # Get year safely
                year = 0
                if hasattr(content, 'Year') and content.Year:
                    try:
                        year = int(content.Year)
                    except (ValueError, TypeError):
                        pass

                # Extract track info
                track = RekordboxTrack(
                    path=path,
                    title=content.Title or Path(path).stem,
                    artist=content.Artist.Name if content.Artist else "Unknown",
                    bpm=float(content.BPM or 0) / 100,  # Rekordbox stores BPM * 100
                    key=key_name,
                    duration_sec=float(content.Length or 0),
                    rating=int(content.Rating or 0),
                    play_count=int(content.DJPlayCount or 0),
                    genre=content.Genre.Name if content.Genre else "",
                    label=content.Label.Name if content.Label else "",
                    year=year,
                )
                tracks.append(track)

            except Exception as e:
                logger.debug(f"Failed to load track: {e}")
                continue

        logger.info(
            f"Loaded {len(tracks)} tracks from Rekordbox "
            f"(skipped: {skipped_no_path} no path, {skipped_not_found} not found)"
        )
        return tracks

    except ImportError:
        logger.warning(
            "pyrekordbox not installed - using empty library. "
            "For training/development: pip install -r requirements-training.txt"
        )
        return []
    except Exception as e:
        logger.error(f"Failed to load Rekordbox database: {e}")
        return []


def filter_tracks_for_phase(
    tracks: List[RekordboxTrack],
    phase_config: PhaseConfig,
) -> List[RekordboxTrack]:
    """
    Filter tracks suitable for a set phase (VECTORIZED).

    Uses numpy broadcasting for BPM filtering with half/double tempo support.

    Args:
        tracks: All available tracks
        phase_config: Phase configuration with BPM/energy requirements

    Returns:
        Filtered list of tracks, float32 optimized
    """
    if not tracks:
        return []

    bpm_min, bpm_max = phase_config.bpm_range

    # Vectorized BPM extraction
    bpms = np.array([t.bpm for t in tracks], dtype=np.float32)

    # Valid BPM mask (> 0)
    valid_bpm = bpms > 0

    # Check BPM in range OR half tempo in range OR double tempo in range
    in_range = (bpms >= bpm_min) & (bpms <= bpm_max)
    half_in_range = (bpms * 2 >= bpm_min) & (bpms * 2 <= bpm_max)
    double_in_range = (bpms / 2 >= bpm_min) & (bpms / 2 <= bpm_max)

    # Combined mask
    mask = valid_bpm & (in_range | half_in_range | double_in_range)

    # Get indices where mask is True
    indices = np.flatnonzero(mask)

    # Return filtered tracks
    return [tracks[i] for i in indices]


# ============== Set Generator Pipeline ==============

class SetGeneratorPipeline:
    """
    Generate DJ set plans based on DJ profiles.

    NOTE: This is a Domain Service, NOT a Pipeline in the architectural sense.
    It does not inherit from Pipeline because:
    - It works with a collection of tracks, not a single audio file
    - It uses generate_plan()/verify_and_optimize() API, not run(context)
    - It orchestrates other Pipelines (TrackCompatibilityPipeline) internally

    Two-phase workflow:
    1. generate_plan() - Quick draft from Rekordbox (seconds)
    2. verify_and_optimize() - Full ML analysis (minutes)

    DJ Profile Influence:
    - tempo_distribution: BPM ranges per phase
    - energy_arc: Energy levels per phase (opening/peak/closing)
    - key_analysis: Preferred keys (Camelot wheel)
    """

    def __init__(
        self,
        cache_repo: Optional[CacheRepository] = None,
        rekordbox_tracks: Optional[List[RekordboxTrack]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize set generator.

        Args:
            cache_repo: CacheRepository for DJ profiles and track analysis
            rekordbox_tracks: Pre-loaded Rekordbox tracks (or will load automatically)
            seed: Random seed for reproducible results (None = random each time)
        """
        self.cache = cache_repo or CacheRepository()
        self._rekordbox_tracks = rekordbox_tracks
        self._track_pipeline = None  # Lazy-loaded
        self._rng = random.Random(seed)  # Separate RNG for reproducibility
        self._preferred_keys: Dict[str, float] = {}  # Camelot -> weight (0-1)

    @property
    def rekordbox_tracks(self) -> List[RekordboxTrack]:
        """Load Rekordbox tracks on first access."""
        if self._rekordbox_tracks is None:
            self._rekordbox_tracks = load_rekordbox_tracks()
        return self._rekordbox_tracks

    @property
    def track_pipeline(self) -> TrackCompatibilityPipeline:
        """Get or create track analysis pipeline."""
        if self._track_pipeline is None:
            self._track_pipeline = TrackCompatibilityPipeline(
                cache=self.cache,
                include_genre=True,
            )
        return self._track_pipeline

    def generate_plan(
        self,
        dj_name: str,
        duration_min: int,
        profile: Optional[Dict] = None,
    ) -> SetPlan:
        """
        Phase 1: Generate quick set plan from Rekordbox metadata.

        No ML analysis - just BPM/key matching based on DJ profile.

        DJ Profile Influence:
        - tempo_distribution: BPM ranges per phase
        - energy_arc: Energy levels per phase
        - key_analysis: Preferred keys (tracks in DJ's favorite keys get bonus)

        Args:
            dj_name: DJ name to base style on
            duration_min: Target set duration in minutes
            profile: DJ profile dict (or will load from cache)

        Returns:
            SetPlan with status="draft"
        """
        logger.info(f"Generating set plan for {dj_name} ({duration_min} min)...")

        # Load DJ profile
        if profile is None:
            profile = self._load_dj_profile(dj_name)

        # Extract preferred keys from profile
        self._extract_preferred_keys(profile)

        # Get phase configurations
        phases = PhaseConfig.default_phases(profile)

        # Load tracks from Rekordbox
        all_tracks = self.rekordbox_tracks
        if not all_tracks:
            logger.warning("No tracks available from Rekordbox")
            return SetPlan(
                dj_name=dj_name,
                duration_min=duration_min,
                created_at=time.time(),
                status="draft",
            )

        # Build set by phases
        planned_tracks = []
        cumulative_time = 0.0
        target_duration_sec = duration_min * 60

        # GLOBAL track usage - prevent duplicates across entire set
        used_paths: set = set()
        used_artists: set = set()  # Track recently used artists

        for phase_config in phases:
            # Calculate time budget for this phase
            phase_duration_sec = target_duration_sec * phase_config.duration_ratio

            # Filter tracks for this phase (excluding already used)
            phase_tracks = [
                t for t in filter_tracks_for_phase(all_tracks, phase_config)
                if t.path not in used_paths
            ]

            if not phase_tracks:
                logger.warning(f"No tracks found for phase {phase_config.phase.value}")
                continue

            # Select tracks for phase
            selected = self._select_tracks_for_phase(
                phase_tracks,
                phase_duration_sec,
                planned_tracks[-1] if planned_tracks else None,
                used_artists,
            )

            # Add to plan
            for i, track in enumerate(selected):
                if cumulative_time >= target_duration_sec:
                    break

                planned = PlannedTrack(
                    path=track.path,
                    title=track.title,
                    artist=track.artist,
                    bpm=track.bpm,
                    key=track.key,
                    camelot=track.camelot,
                    duration_sec=track.duration_sec,
                    genre=track.genre,
                    year=track.year,
                    phase=phase_config.phase,
                    position=len(planned_tracks) + 1,
                    cumulative_time_sec=cumulative_time,
                )
                planned_tracks.append(planned)
                cumulative_time += track.duration_sec

                # Mark as used globally
                used_paths.add(track.path)
                used_artists.add(track.artist)

        # Calculate basic transition scores (harmonic only, quick)
        self._calculate_basic_scores(planned_tracks)

        # Create plan
        plan = SetPlan(
            dj_name=dj_name,
            duration_min=duration_min,
            created_at=time.time(),
            tracks=planned_tracks,
            status="draft",
        )

        # Calculate aggregate metrics
        self._update_plan_metrics(plan)

        logger.info(
            f"Draft plan created: {plan.n_tracks} tracks, "
            f"{plan.actual_duration_min:.1f} min, "
            f"avg score: {plan.avg_transition_score:.2f}"
        )

        return plan

    def verify_and_optimize(
        self,
        plan: SetPlan,
        reorder: bool = True,
        progress_callback=None,
        stage_callback=None,
    ) -> SetPlan:
        """
        Phase 2: Verify plan with full ML analysis and optimize transitions.

        Uses TrackCompatibilityPipeline for analysis (proper architecture).

        Args:
            plan: Draft plan from generate_plan()
            reorder: Whether to reorder tracks for better transitions
            progress_callback: Optional callback(current, total, track_title, stage)
                              stage: "analyzing", "cached", "done", "error", "optimizing", "complete"
            stage_callback: Optional callback(track_num, total_tracks, stage_num, total_stages, stage_name)
                           Called after each pipeline stage within track analysis

        Returns:
            SetPlan with status="verified"
        """
        import warnings
        import os
        import sys

        # Suppress noisy warnings
        warnings.filterwarnings("ignore", message=".*aifc.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*audioop.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*sunau.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

        # Suppress TensorFlow/MLIR logs
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '3')

        # Flag to track first non-cached analysis (triggers MLIR init)
        first_analysis = [True]  # Use list for closure mutation

        # Analyze each track
        analyzed_tracks = []
        total = plan.n_tracks

        for i, planned in enumerate(plan.tracks):
            track_title = planned.title[:30]

            # Create stage callback wrapper for this track
            track_stage_callback = None
            if stage_callback:
                def track_stage_callback(stage_num, total_stages, stage_name):
                    stage_callback(i + 1, total, stage_num, total_stages, stage_name)

            # Run analysis (pipeline handles caching internally)
            try:
                if progress_callback:
                    progress_callback(i + 1, total, track_title, "analyzing")

                # First analysis triggers MLIR init - suppress stderr at OS level
                if first_analysis[0]:
                    # Redirect stderr at file descriptor level (catches C++ output)
                    stderr_fd = sys.stderr.fileno()
                    old_stderr_fd = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    os.close(devnull)
                    try:
                        analysis = self.track_pipeline.analyze(
                            planned.path,
                            stage_callback=track_stage_callback
                        )
                    finally:
                        os.dup2(old_stderr_fd, stderr_fd)
                        os.close(old_stderr_fd)
                    first_analysis[0] = False
                else:
                    analysis = self.track_pipeline.analyze(
                        planned.path,
                        stage_callback=track_stage_callback
                    )

                planned.analysis = analysis
                planned.verified = analysis.source != "error"

                # Update from analysis
                if planned.verified:
                    planned.bpm = analysis.bpm
                    planned.camelot = analysis.camelot

                stage = "cached" if analysis.source == "cache" else "done"
                if progress_callback:
                    progress_callback(i + 1, total, track_title, stage)

            except Exception as e:
                logger.warning(f"Failed to analyze {planned.title}: {e}")
                planned.verified = False
                if progress_callback:
                    progress_callback(i + 1, total, track_title, "error")

            analyzed_tracks.append(planned)

        plan.tracks = analyzed_tracks

        # Report optimization stage
        if progress_callback:
            progress_callback(total, total, "Optimizing order...", "optimizing")

        # Reorder for optimal transitions
        if reorder and len(plan.tracks) > 2:
            plan.tracks = self._optimize_order(plan.tracks)

        # Recalculate transition scores with full data
        self._calculate_full_scores(plan.tracks)

        # Update metrics
        plan.status = "verified"
        self._update_plan_metrics(plan)

        if progress_callback:
            progress_callback(total, total, "Complete", "complete")

        # Auto-save verified plan
        self.save_plan(plan)

        return plan

    # ============== Plan Persistence ==============

    def save_plan(self, plan: SetPlan):
        """
        Save set plan to cache.

        Args:
            plan: SetPlan to save
        """
        self.cache.save_set_plan(plan.plan_id, plan.to_dict())
        logger.debug(f"Saved plan {plan.plan_id} ({plan.n_tracks} tracks)")

    def load_plan(self, plan_id: str) -> Optional[SetPlan]:
        """
        Load set plan from cache.

        Args:
            plan_id: Plan identifier

        Returns:
            SetPlan or None if not found
        """
        data = self.cache.get_set_plan(plan_id)
        if data:
            return SetPlan.from_dict(data)
        return None

    def list_plans(self, dj_name: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        List cached set plans.

        Args:
            dj_name: Filter by DJ name (optional)
            limit: Maximum plans to return

        Returns:
            List of plan metadata dicts
        """
        if dj_name:
            return self.cache.get_set_plans_by_dj(dj_name, limit)
        return self.cache.get_all_set_plans(limit)

    def delete_plan(self, plan_id: str):
        """Delete a set plan from cache."""
        self.cache.delete_set_plan(plan_id)
        logger.debug(f"Deleted plan {plan_id}")

    def export_to_rekordbox(
        self,
        plan: SetPlan,
        playlist_name: str,
    ) -> bool:
        """
        Export set plan to Rekordbox playlist.

        Args:
            plan: Verified set plan
            playlist_name: Name for the new playlist

        Returns:
            True if successful
        """
        try:
            # SECURITY: No direct Rekordbox DB access for export
            # Export as M3U which can be imported into Rekordbox manually

            logger.info(f"Exporting {plan.n_tracks} tracks to playlist '{playlist_name}'")

            # For now, export as M3U (more universally supported)
            m3u_path = Path.home() / f"{playlist_name}.m3u8"
            with open(m3u_path, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                for track in plan.tracks:
                    f.write(f"#EXTINF:{int(track.duration_sec)},{track.artist} - {track.title}\n")
                    f.write(f"{track.path}\n")

            logger.info(f"Exported playlist to {m3u_path}")
            plan.status = "exported"
            return True

        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return False

    # ============== Private Methods ==============

    def _load_dj_profile(self, dj_name: str) -> Optional[Dict]:
        """Load DJ profile from cache."""
        try:
            # Get all profiles and find matching one
            profiles = self.cache.get_all_dj_profiles()
            for p in profiles:
                if p['dj_name'].lower() == dj_name.lower():
                    # Load full profile
                    return self.cache.get_dj_profile(dj_name, [])
            logger.warning(f"No profile found for DJ: {dj_name}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load profile: {e}")
            return None

    def _extract_preferred_keys(self, profile: Optional[Dict]):
        """
        Extract preferred keys from DJ profile's key_analysis (VECTORIZED).

        Converts key_histogram to normalized weights for Camelot keys.
        DJ's most-used keys get higher weights (0-1).

        Args:
            profile: DJ profile dict
        """
        self._preferred_keys = {}

        if not profile:
            return

        key_analysis = profile.get('key_analysis', {})
        if not key_analysis:
            return

        # key_histogram: {"Am": 15, "Dm": 10, "Gm": 8, ...}
        key_histogram = key_analysis.get('key_histogram', {})
        if not key_histogram:
            return

        # Convert keys to Camelot notation
        key_to_camelot = {
            'C': '8B', 'Cm': '5A', 'C#': '3B', 'C#m': '12A',
            'D': '10B', 'Dm': '7A', 'D#': '5B', 'D#m': '2A',
            'E': '12B', 'Em': '9A',
            'F': '7B', 'Fm': '4A', 'F#': '2B', 'F#m': '11A',
            'G': '9B', 'Gm': '6A', 'G#': '4B', 'G#m': '1A',
            'A': '11B', 'Am': '8A', 'A#': '6B', 'A#m': '3A',
            'B': '1B', 'Bm': '10A',
        }

        # Vectorized: extract keys and counts as arrays
        keys = np.array(list(key_histogram.keys()))
        counts = np.array(list(key_histogram.values()), dtype=np.float32)

        # Map keys to camelot codes (vectorized via list comp, then array)
        camelot_codes = np.array([key_to_camelot.get(k, None) for k in keys])

        # Filter out None values
        valid_mask = camelot_codes != np.array(None)
        valid_camelots = camelot_codes[valid_mask]
        valid_counts = counts[valid_mask]

        if len(valid_camelots) == 0:
            return

        # Aggregate counts per camelot key using numpy unique
        unique_camelots, inverse = np.unique(valid_camelots, return_inverse=True)
        aggregated_counts = np.bincount(inverse, weights=valid_counts).astype(np.float32)

        # Normalize to weights (0-1)
        max_count = np.max(aggregated_counts)
        weights = 0.5 + 0.5 * (aggregated_counts / max_count)

        # Build dict
        self._preferred_keys = dict(zip(unique_camelots, weights.tolist()))

        logger.debug(f"Preferred keys: {self._preferred_keys}")

    def _select_tracks_for_phase(
        self,
        available: List[RekordboxTrack],
        duration_budget_sec: float,
        last_track: Optional[PlannedTrack],
        recently_used_artists: set,
    ) -> List[RekordboxTrack]:
        """
        Select tracks for a phase with smart ordering.

        Selection criteria (weighted):
        1. Harmonic compatibility (Camelot wheel movement)
        2. Artist diversity (avoid back-to-back same artist)
        3. Rating (prefer higher-rated tracks)
        4. BPM progression (gradual changes)

        Args:
            available: Tracks filtered for this phase
            duration_budget_sec: How much time to fill
            last_track: Last track from previous phase (for continuity)
            recently_used_artists: Artists used in last N tracks

        Returns:
            Selected tracks for this phase
        """
        selected = []
        remaining_duration = duration_budget_sec
        used_paths = set()
        phase_artists = set()  # Artists used in THIS phase

        # First track: find best match to last track
        if last_track and available:
            best_first = self._find_best_next_track(
                last_track.camelot,
                last_track.bpm,
                last_track.artist,
                available,
                used_paths,
                recently_used_artists | phase_artists,
            )
            if best_first:
                selected.append(best_first)
                remaining_duration -= best_first.duration_sec
                used_paths.add(best_first.path)
                phase_artists.add(best_first.artist)
        elif available:
            # No last track - pick highest rated with good key
            sorted_by_rating = sorted(available, key=lambda t: -t.rating)
            best_first = sorted_by_rating[0]
            selected.append(best_first)
            remaining_duration -= best_first.duration_sec
            used_paths.add(best_first.path)
            phase_artists.add(best_first.artist)

        # Fill remaining time
        while remaining_duration > 0 and available:
            last = selected[-1] if selected else None
            if not last:
                break

            # Find candidates not yet used
            candidates = [t for t in available if t.path not in used_paths]
            if not candidates:
                break

            # Select next track with composite scoring
            next_track = self._find_best_next_track(
                last.camelot,
                last.bpm,
                last.artist,
                candidates,
                used_paths,
                recently_used_artists | phase_artists,
            )

            if next_track:
                selected.append(next_track)
                remaining_duration -= next_track.duration_sec
                used_paths.add(next_track.path)
                phase_artists.add(next_track.artist)
            else:
                break

        return selected

    def _find_best_next_track(
        self,
        from_camelot: str,
        from_bpm: float,
        from_artist: str,
        candidates: List[RekordboxTrack],
        used_paths: set,
        recently_used_artists: set,
        top_n: int = 5,
    ) -> Optional[RekordboxTrack]:
        """
        Find best next track using composite scoring with randomization.

        Scores all candidates, then randomly picks from top N to add variety.

        Scoring weights:
        - Harmonic compatibility: 35%
        - Artist diversity: 20%
        - Rating: 15%
        - BPM continuity: 15%
        - Key preference (from DJ profile): 15%

        Args:
            top_n: Pick randomly from top N candidates (default: 5)
        """
        if not candidates:
            return None

        scored_tracks = []

        for track in candidates:
            if track.path in used_paths:
                continue

            # 1. Harmonic PROGRESSION score (0-1)
            # Use progression score to encourage movement around Camelot wheel
            harmonic = score_harmonic_progression(from_camelot, track.camelot)

            # 2. Artist diversity (0-1)
            # Penalize same artist, especially if recently used
            if track.artist == from_artist:
                artist_score = 0.0  # Never back-to-back same artist
            elif track.artist in recently_used_artists:
                artist_score = 0.5  # Mild penalty for recent artist
            else:
                artist_score = 1.0  # Full score for new artist

            # 3. Rating score (0-1)
            rating_score = track.rating / 5.0 if track.rating > 0 else 0.5

            # 4. BPM continuity (0-1)
            bpm_diff = abs(track.bpm - from_bpm)
            if bpm_diff <= 2:
                bpm_score = 1.0  # Perfect
            elif bpm_diff <= 4:
                bpm_score = 0.8  # Good
            elif bpm_diff <= 6:
                bpm_score = 0.5  # Acceptable
            else:
                bpm_score = 0.2  # Large jump

            # 5. Key preference from DJ profile (0-1)
            # Tracks in DJ's favorite keys get bonus
            key_pref_score = self._preferred_keys.get(track.camelot, 0.5)

            # Composite score
            score = (
                0.35 * harmonic +
                0.20 * artist_score +
                0.15 * rating_score +
                0.15 * bpm_score +
                0.15 * key_pref_score
            )

            scored_tracks.append((track, score))

        if not scored_tracks:
            return None

        # Sort by score descending
        scored_tracks.sort(key=lambda x: -x[1])

        # Pick randomly from top N (weighted by score)
        top_candidates = scored_tracks[:top_n]

        if len(top_candidates) == 1:
            return top_candidates[0][0]

        # Weighted random selection - higher scores more likely
        weights = [s[1] for s in top_candidates]
        total = sum(weights)
        if total == 0:
            return top_candidates[0][0]

        # Normalize weights
        weights = [w / total for w in weights]

        # Random selection
        r = self._rng.random()
        cumulative = 0
        for (track, _), weight in zip(top_candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return track

        return top_candidates[0][0]

    def _calculate_basic_scores(self, tracks: List[PlannedTrack]):
        """Calculate basic transition scores (harmonic + BPM only, VECTORIZED)."""
        if len(tracks) < 2:
            return

        # Extract arrays for vectorized computation
        camelots = np.array([t.camelot for t in tracks])
        bpms = np.array([t.bpm for t in tracks], dtype=np.float32)

        # Vectorized score computation using numpy broadcasting
        # Shift arrays to get prev/curr pairs
        prev_camelots = camelots[:-1]
        curr_camelots = camelots[1:]
        prev_bpms = bpms[:-1]
        curr_bpms = bpms[1:]

        # Compute harmonic and BPM scores for all transitions at once
        # (score_harmonic_compatibility is not vectorizable, but we batch the data)
        n_transitions = len(tracks) - 1
        harmonic_scores = np.array([
            score_harmonic_compatibility(prev_camelots[i], curr_camelots[i])
            for i in range(n_transitions)
        ], dtype=np.float32)

        bpm_scores = np.array([
            score_bpm_compatibility(prev_bpms[i], curr_bpms[i])
            for i in range(n_transitions)
        ], dtype=np.float32)

        # Assign scores to tracks (vectorized loop-free assignment)
        for i in range(n_transitions):
            tracks[i + 1].transition_score = TransitionScore(
                harmonic=float(harmonic_scores[i]),
                energy=0.5,  # Unknown without analysis
                drop_conflict=0.5,
                spectral=0.5,
                genre=0.5,
                bpm=float(bpm_scores[i]),
            )

    def _calculate_full_scores(self, tracks: List[PlannedTrack]):
        """Calculate full transition scores using analysis data."""
        for i in range(1, len(tracks)):
            prev = tracks[i - 1]
            curr = tracks[i]

            if prev.analysis and curr.analysis:
                # Convert TrackCompatibilityResult to TrackAnalysis for scoring
                prev_ta = self._result_to_analysis(prev.analysis)
                curr_ta = self._result_to_analysis(curr.analysis)
                curr.transition_score = compute_track_transition_score(prev_ta, curr_ta)
            elif prev.verified or curr.verified:
                # Partial data - use what we have
                self._calculate_partial_score(prev, curr)

    def _result_to_analysis(self, result: TrackCompatibilityResult) -> TrackAnalysis:
        """Convert TrackCompatibilityResult to TrackAnalysis for scoring."""
        return TrackAnalysis(
            path=result.path,
            filename=result.filename,
            duration_sec=result.duration_sec,
            bpm=result.bpm,
            key=result.key,
            camelot=result.camelot,
            dj_category=result.dj_category,
            genre_confidence=result.genre_confidence,
            intro_energy=result.intro_energy,
            outro_energy=result.outro_energy,
            peak_energy=result.peak_energy,
            drop_times=result.drop_times,
            drop_count=result.drop_count,
            best_mix_in=result.best_mix_in,
            best_mix_out=result.best_mix_out,
            spectral_centroid_mean=result.spectral_centroid_mean,
            beat_times=result.beat_times,
            bar_boundaries=result.bar_boundaries,
            phrase_boundaries=result.phrase_boundaries,
            beat_duration_sec=result.beat_duration_sec,
            bar_duration_sec=result.bar_duration_sec,
            phrase_duration_sec=result.phrase_duration_sec,
            grid_calibrated=result.grid_calibrated,
            calibration_confidence=result.calibration_confidence,
            analysis_time_sec=result.analysis_time_sec,
            source=result.source,
        )

    def _calculate_partial_score(
        self,
        prev: PlannedTrack,
        curr: PlannedTrack,
    ):
        """Calculate transition score with partial data."""
        harmonic = score_harmonic_compatibility(prev.camelot, curr.camelot)
        bpm = score_bpm_compatibility(prev.bpm, curr.bpm)

        # Use analysis data if available
        if prev.analysis and curr.analysis:
            energy = 0.7  # Trust the analysis
            drop_conflict = 0.6
            spectral = 0.6
            genre = 0.6
        else:
            energy = 0.5
            drop_conflict = 0.5
            spectral = 0.5
            genre = 0.5

        curr.transition_score = TransitionScore(
            harmonic=harmonic,
            energy=energy,
            drop_conflict=drop_conflict,
            spectral=spectral,
            genre=genre,
            bpm=bpm,
        )

    def _optimize_order(
        self,
        tracks: List[PlannedTrack],
    ) -> List[PlannedTrack]:
        """
        Reorder tracks to maximize transition quality.

        Uses greedy algorithm: for each position, pick the track
        that gives the best transition from the previous track.
        Respects phase boundaries.
        """
        if len(tracks) <= 2:
            return tracks

        # Group by phase
        phases = {}
        for track in tracks:
            if track.phase not in phases:
                phases[track.phase] = []
            phases[track.phase].append(track)

        # Optimize within each phase
        optimized = []
        for phase in [SetPhase.OPENING, SetPhase.BUILD, SetPhase.PEAK, SetPhase.CLOSING]:
            if phase not in phases:
                continue

            phase_tracks = phases[phase]

            # Get starting point (last track from previous phase)
            if optimized:
                prev_track = optimized[-1]
                phase_tracks = self._greedy_order(
                    phase_tracks, prev_track.camelot, prev_track.bpm
                )
            else:
                # First phase - start with lowest BPM
                phase_tracks = sorted(phase_tracks, key=lambda t: t.bpm)

            optimized.extend(phase_tracks)

        # Update positions
        cumulative_time = 0.0
        for i, track in enumerate(optimized):
            track.position = i + 1
            track.cumulative_time_sec = cumulative_time
            cumulative_time += track.duration_sec

        return optimized

    def _greedy_order(
        self,
        tracks: List[PlannedTrack],
        start_camelot: str,
        start_bpm: float,
    ) -> List[PlannedTrack]:
        """Greedy ordering within a phase for best transitions."""
        if len(tracks) <= 1:
            return tracks

        ordered = []
        remaining = list(tracks)
        current_camelot = start_camelot
        current_bpm = start_bpm

        while remaining:
            # Find best next track
            best_idx = 0
            best_score = -1

            for i, track in enumerate(remaining):
                harm_score = score_harmonic_compatibility(current_camelot, track.camelot)
                bpm_score = score_bpm_compatibility(current_bpm, track.bpm)
                combined = 0.7 * harm_score + 0.3 * bpm_score

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            # Add best track
            best_track = remaining.pop(best_idx)
            ordered.append(best_track)
            current_camelot = best_track.camelot
            current_bpm = best_track.bpm

        return ordered

    def _update_plan_metrics(self, plan: SetPlan):
        """Update plan aggregate metrics."""
        scores = [
            t.transition_score.total
            for t in plan.tracks
            if t.transition_score
        ]

        plan.n_transitions = len(scores)
        plan.total_transition_score = sum(scores)
        plan.avg_transition_score = np.mean(scores) if scores else 0.0

    # ============== Iterative Optimization ==============

    def find_weak_transitions(
        self,
        plan: SetPlan,
        threshold: float = 0.6,
    ) -> List[Tuple[int, PlannedTrack, PlannedTrack, TransitionScore]]:
        """
        Find transitions below quality threshold (VECTORIZED).

        Uses numpy for score filtering and sorting.

        Args:
            plan: Set plan to analyze
            threshold: Score threshold (transitions below this are "weak")

        Returns:
            List of (position, track_from, track_to, score) tuples, sorted worst-first
        """
        if len(plan.tracks) < 2:
            return []

        # Extract transition scores as numpy array
        scores = np.array([
            t.transition_score.total if t.transition_score else 1.0
            for t in plan.tracks[1:]  # Skip first track (no incoming transition)
        ], dtype=np.float32)

        # Find indices where score < threshold
        weak_mask = scores < threshold
        weak_indices = np.flatnonzero(weak_mask) + 1  # +1 to get actual position

        if len(weak_indices) == 0:
            return []

        # Get weak scores for sorting
        weak_scores = scores[weak_mask]

        # Sort by score ascending (worst first)
        sort_order = np.argsort(weak_scores)
        sorted_indices = weak_indices[sort_order]

        # Build result list
        return [
            (int(i), plan.tracks[i - 1], plan.tracks[i], plan.tracks[i].transition_score)
            for i in sorted_indices
        ]

    def find_replacement_candidates(
        self,
        plan: SetPlan,
        position: int,
        max_candidates: int = 5,
        min_harmonic_score: float = 0.6,
        prefer_rated: bool = True,
    ) -> List[Tuple[RekordboxTrack, TransitionScore, TransitionScore]]:
        """
        Find replacement candidates for a track at given position.

        Searches the FULL Rekordbox library (not cache) for tracks that:
        1. Fit the phase BPM range
        2. Have good harmonic compatibility with neighbors
        3. Are not already in the set

        Args:
            plan: Current set plan
            position: Position of track to replace (1-indexed)
            max_candidates: Maximum candidates to return
            min_harmonic_score: Minimum harmonic compatibility score (0-1)
            prefer_rated: Prefer tracks with higher Rekordbox rating

        Returns:
            List of (track, score_from_prev, score_to_next) tuples
        """
        if position < 1 or position > len(plan.tracks):
            return []

        # Get context tracks
        track_to_replace = plan.tracks[position - 1]
        prev_track = plan.tracks[position - 2] if position > 1 else None
        next_track = plan.tracks[position] if position < len(plan.tracks) else None

        # Tracks already in the plan
        used_paths = {t.path for t in plan.tracks}

        # Artists already used (to prefer diversity)
        used_artists = {t.artist for t in plan.tracks}

        # Get phase constraints - wider range for more candidates
        phase = track_to_replace.phase
        phase_tracks_in_plan = [t for t in plan.tracks if t.phase == phase]
        phase_bpm_min = min(t.bpm for t in phase_tracks_in_plan) - 6
        phase_bpm_max = max(t.bpm for t in phase_tracks_in_plan) + 6

        # Find candidates from FULL Rekordbox library
        candidates = []
        for track in self.rekordbox_tracks:
            # Skip already used
            if track.path in used_paths:
                continue

            # Skip tracks with no BPM/key
            if track.bpm <= 0 or track.camelot == '?':
                continue

            # Phase BPM constraint
            if not (phase_bpm_min <= track.bpm <= phase_bpm_max):
                continue

            # Pre-filter by harmonic compatibility (saves computation)
            if prev_track:
                harm_in = score_harmonic_compatibility(prev_track.camelot, track.camelot)
                if harm_in < min_harmonic_score:
                    continue
            if next_track:
                harm_out = score_harmonic_compatibility(track.camelot, next_track.camelot)
                if harm_out < min_harmonic_score:
                    continue

            # Calculate full scores for this candidate
            score_from_prev = None
            score_to_next = None

            if prev_track:
                score_from_prev = TransitionScore(
                    harmonic=score_harmonic_compatibility(prev_track.camelot, track.camelot),
                    energy=0.5,  # No ML analysis - use neutral
                    drop_conflict=0.5,
                    spectral=0.5,
                    genre=0.5,
                    bpm=score_bpm_compatibility(prev_track.bpm, track.bpm),
                )

            if next_track:
                score_to_next = TransitionScore(
                    harmonic=score_harmonic_compatibility(track.camelot, next_track.camelot),
                    energy=0.5,
                    drop_conflict=0.5,
                    spectral=0.5,
                    genre=0.5,
                    bpm=score_bpm_compatibility(track.bpm, next_track.bpm),
                )

            # Calculate combined score
            new_score_in = score_from_prev.total if score_from_prev else 0.5
            new_score_out = score_to_next.total if score_to_next else 0.5
            avg_score = (new_score_in + new_score_out) / 2

            # Bonus for diversity (new artist)
            diversity_bonus = 0.05 if track.artist not in used_artists else 0

            # Bonus for rating
            rating_bonus = (track.rating / 5.0) * 0.1 if prefer_rated and track.rating > 0 else 0

            # Combined ranking score
            ranking_score = avg_score + diversity_bonus + rating_bonus

            candidates.append((
                track,
                score_from_prev,
                score_to_next,
                avg_score,
                ranking_score,
            ))

        # Sort by ranking score (best first)
        candidates.sort(key=lambda x: -x[4])

        # Return top candidates
        return [(c[0], c[1], c[2]) for c in candidates[:max_candidates]]

    def replace_track(
        self,
        plan: SetPlan,
        position: int,
        new_track: RekordboxTrack,
    ) -> SetPlan:
        """
        Replace track at position with new track.

        Args:
            plan: Current set plan
            position: Position to replace (1-indexed)
            new_track: New track to insert

        Returns:
            Updated plan
        """
        if position < 1 or position > len(plan.tracks):
            return plan

        old_track = plan.tracks[position - 1]

        # Create new PlannedTrack
        new_planned = PlannedTrack(
            path=new_track.path,
            title=new_track.title,
            artist=new_track.artist,
            bpm=new_track.bpm,
            key=new_track.key,
            camelot=new_track.camelot,
            duration_sec=new_track.duration_sec,
            genre=new_track.genre,
            year=new_track.year,
            phase=old_track.phase,
            position=position,
            cumulative_time_sec=old_track.cumulative_time_sec,
        )

        # Replace in list
        plan.tracks[position - 1] = new_planned

        # Recalculate affected transition scores
        if position > 1:
            prev = plan.tracks[position - 2]
            new_planned.transition_score = TransitionScore(
                harmonic=score_harmonic_compatibility(prev.camelot, new_planned.camelot),
                energy=0.5,
                drop_conflict=0.5,
                spectral=0.5,
                genre=0.5,
                bpm=score_bpm_compatibility(prev.bpm, new_planned.bpm),
            )

        if position < len(plan.tracks):
            next_track = plan.tracks[position]
            next_track.transition_score = TransitionScore(
                harmonic=score_harmonic_compatibility(new_planned.camelot, next_track.camelot),
                energy=0.5,
                drop_conflict=0.5,
                spectral=0.5,
                genre=0.5,
                bpm=score_bpm_compatibility(new_planned.bpm, next_track.bpm),
            )

        # Recalculate cumulative times
        cumulative = 0.0
        for track in plan.tracks:
            track.cumulative_time_sec = cumulative
            cumulative += track.duration_sec

        # Update metrics
        self._update_plan_metrics(plan)

        return plan

    def interactive_optimize(
        self,
        plan: SetPlan,
        threshold: float = 0.6,
        max_iterations: int = 10,
        auto_accept_threshold: float = 0.85,
        callback=None,
    ) -> SetPlan:
        """
        Iteratively optimize plan by replacing weak transitions.

        Args:
            plan: Set plan to optimize
            threshold: Score threshold for "weak" transitions
            max_iterations: Maximum replacements to attempt
            auto_accept_threshold: Automatically accept if new score >= this
            callback: Optional callback(weak_idx, weak_total, position, old_track, candidates)
                     Should return: index of selected candidate (0-based), -1 to skip, None to stop

        Returns:
            Optimized plan
        """
        iterations = 0
        skipped_positions = set()  # Track positions we've already tried and skipped

        while iterations < max_iterations:
            # Find weak transitions (excluding already skipped ones)
            weak = [
                (pos, prev, curr, score)
                for pos, prev, curr, score in self.find_weak_transitions(plan, threshold)
                if pos not in skipped_positions
            ]
            if not weak:
                break  # All transitions are good or skipped

            # Process worst transition
            pos, prev_track, curr_track, score = weak[0]

            # Find replacement candidates
            candidates = self.find_replacement_candidates(plan, pos)
            if not candidates:
                # No good candidates - skip this position
                skipped_positions.add(pos)
                continue

            # Let callback decide
            if callback:
                selection = callback(
                    weak_idx=iterations + 1,
                    weak_total=len(weak),
                    position=pos,
                    old_track=curr_track,
                    old_score=score,
                    prev_track=prev_track,
                    candidates=candidates,
                )

                if selection is None:
                    break  # User wants to stop
                elif selection == -1:
                    # Skip this transition
                    skipped_positions.add(pos)
                    continue
                else:
                    # Replace with selected candidate
                    new_track, _, _ = candidates[selection]
                    plan = self.replace_track(plan, pos, new_track)
            else:
                # Auto mode - accept best candidate if it improves
                best_track, score_in, score_out = candidates[0]
                new_avg = ((score_in.total if score_in else 0.5) +
                          (score_out.total if score_out else 0.5)) / 2

                # Accept if new score meets threshold or significantly improves
                if new_avg >= auto_accept_threshold or new_avg > score.total + 0.1:
                    plan = self.replace_track(plan, pos, best_track)
                    logger.debug(f"Auto-replaced track at pos {pos}: {score.total:.2f} -> {new_avg:.2f}")
                else:
                    # Not good enough - skip this position
                    skipped_positions.add(pos)
                    logger.debug(f"Skipped pos {pos}: best candidate {new_avg:.2f} < threshold {auto_accept_threshold:.2f}")

            iterations += 1

        return plan

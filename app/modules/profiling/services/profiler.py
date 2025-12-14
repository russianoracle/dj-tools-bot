"""
ProfilingService - DJ style profiling and metrics extraction.

This service wraps the DJ profiling stages and provides a clean API
for extracting DJ style characteristics from sets.

Usage:
    from app.modules import ProfilingService

    service = ProfilingService()

    # Profile single set
    metrics = service.profile_set("set.mp3")

    # Profile DJ across multiple sets
    profile = service.profile_dj("Josh Baker", "/path/to/sets/")

    # Get cached profile
    profile = service.get_profile("Josh Baker")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from ..core.pipelines import (
    Pipeline,
    PipelineContext,
    LoadAudioStage,
    ComputeSTFTStage,
)
from ..core.pipelines.dj_profiling import (
    DJProfilingStage,
    DJProfileMetrics,
    EnergyArcProfilingStage,
    DropPatternProfilingStage,
    TempoDistributionProfilingStage,
)
from ..core.cache import CacheRepository, ICacheStatusProvider

logger = logging.getLogger(__name__)


@dataclass
class DJProfile:
    """
    Aggregated DJ profile from multiple sets.

    Combines metrics from individual sets into a DJ-level profile
    with statistical aggregation.
    """
    dj_name: str
    n_sets_analyzed: int
    total_duration_hours: float

    # Aggregated metrics
    avg_energy_arc_shape: str  # "building", "plateau", "journey", etc.
    avg_drops_per_hour: float
    tempo_range: tuple  # (min, max)
    dominant_tempo: int
    preferred_keys: List[str]

    # Style classification
    mixing_style: str  # "smooth", "hard", "technical", etc.
    energy_profile: str  # "peak_time", "warm_up", "deep", etc.

    # Individual set metrics
    set_metrics: List[DJProfileMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dj_name': self.dj_name,
            'n_sets_analyzed': self.n_sets_analyzed,
            'total_duration_hours': self.total_duration_hours,
            'avg_energy_arc_shape': self.avg_energy_arc_shape,
            'avg_drops_per_hour': self.avg_drops_per_hour,
            'tempo_range': self.tempo_range,
            'dominant_tempo': self.dominant_tempo,
            'preferred_keys': self.preferred_keys,
            'mixing_style': self.mixing_style,
            'energy_profile': self.energy_profile,
            'set_metrics': [m.to_dict() for m in self.set_metrics],
        }


@dataclass
class BatchProfilingResult:
    """Result of batch profiling operation."""
    dj_name: str
    total_sets: int
    processed: int
    cached: int
    failed: int
    profile: Optional[DJProfile]
    errors: Dict[str, str]


class ProfilingService:
    """
    Unified service for DJ profiling.

    Architecture:
        - Uses ICacheStatusProvider for READ-ONLY cache status queries (UI)
        - Delegates profiling to Pipelines (which may handle cache internally)

    Provides:
    - Single set profiling (extract metrics)
    - Batch profiling (aggregate across sets)
    - Profile management (save/load/query)
    """

    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus', '.ogg'}

    def __init__(
        self,
        cache_status: Optional[ICacheStatusProvider] = None,
        cache_dir: str = "cache",
        sr: int = 22050,
    ):
        """
        Initialize profiling service.

        Args:
            cache_status: Read-only cache status provider (for UI queries)
            cache_dir: Cache directory path
            sr: Sample rate for analysis
        """
        self.cache_dir = cache_dir
        self.sr = sr

        # Read-only cache status provider
        if cache_status is not None:
            self._cache_status = cache_status
        else:
            self._cache_status = CacheRepository(cache_dir)

    @property
    def cache(self) -> ICacheStatusProvider:
        """Cache status provider (read-only)."""
        return self._cache_status

    # ==================== CACHE STATUS (Read-only for UI) ====================

    def is_profile_cached(self, dj_name: str) -> bool:
        """Check if DJ profile is cached. For UI display."""
        return self._cache_status.exists_profile(dj_name)

    def list_profiles(self) -> List[str]:
        """Get list of cached DJ profiles. For UI display."""
        return self._cache_status.list_profiles()

    # ==================== SINGLE SET PROFILING ====================

    def profile_set(
        self,
        path: str,
        include_energy_arc: bool = True,
        include_drop_pattern: bool = True,
        include_tempo: bool = True,
        include_key: bool = True,
        include_genre: bool = False,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> DJProfileMetrics:
        """
        Profile a single DJ set.

        Args:
            path: Path to audio file
            include_energy_arc: Extract energy arc metrics
            include_drop_pattern: Extract drop pattern metrics
            include_tempo: Extract tempo distribution
            include_key: Extract key analysis
            include_genre: Extract genre (requires essentia)
            use_cache: Use cached results
            verbose: Print detailed progress

        Returns:
            DJProfileMetrics with all requested metrics
        """
        # Create profiling pipeline
        stages = [
            LoadAudioStage(sr=self.sr),
            ComputeSTFTStage(),
            DJProfilingStage(
                include_energy_arc=include_energy_arc,
                include_drop_pattern=include_drop_pattern,
                include_tempo_distribution=include_tempo,
                include_key_analysis=include_key,
                include_genre=include_genre,
            ),
        ]

        pipeline = Pipeline(stages, name="DJProfiling")

        # Setup context
        context = PipelineContext(input_path=path, cache_dir=self.cache_dir)

        # Run pipeline
        if verbose:
            logger.info(f"Profiling: {path}")

        context = pipeline.run(context)

        # Extract metrics
        metrics = context.get_result('dj_profile', DJProfileMetrics())

        return metrics

    # ==================== BATCH PROFILING ====================

    def profile_dj(
        self,
        dj_name: str,
        folder: str,
        recursive: bool = True,
        include_genre: bool = False,
        use_cache: bool = True,
        force: bool = False,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        verbose: bool = False,
    ) -> BatchProfilingResult:
        """
        Profile a DJ across all sets in a folder.

        Aggregates metrics from individual sets into a DJ-level profile.

        Args:
            dj_name: DJ name for the profile
            folder: Path to folder with DJ sets
            recursive: Recursively scan subfolders
            include_genre: Include genre analysis
            use_cache: Use cached results
            force: Force re-analysis
            on_progress: Callback(done, total, current_file)
            verbose: Print detailed progress

        Returns:
            BatchProfilingResult with aggregated profile
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder}")

        # Find audio files
        files = self._find_audio_files(folder_path, recursive)
        if not files:
            return BatchProfilingResult(
                dj_name=dj_name,
                total_sets=0,
                processed=0,
                cached=0,
                failed=0,
                profile=None,
                errors={}
            )

        all_metrics: List[DJProfileMetrics] = []
        errors: Dict[str, str] = {}
        processed = 0
        cached = 0

        total = len(files)
        for i, file_path in enumerate(files, 1):
            abs_path = str(file_path.absolute())

            if on_progress:
                on_progress(i, total, abs_path)

            try:
                metrics = self.profile_set(
                    abs_path,
                    include_genre=include_genre,
                    use_cache=use_cache,
                    verbose=verbose,
                )
                all_metrics.append(metrics)
                processed += 1

            except Exception as e:
                logger.error(f"Error profiling {file_path.name}: {e}")
                errors[abs_path] = str(e)

        # Aggregate into DJ profile
        profile = None
        if all_metrics:
            profile = self._aggregate_profile(dj_name, all_metrics)

            # Save to cache
            if use_cache:
                self.cache.save_dj_profile(dj_name, profile.to_dict())

        return BatchProfilingResult(
            dj_name=dj_name,
            total_sets=total,
            processed=processed,
            cached=cached,
            failed=len(errors),
            profile=profile,
            errors=errors
        )

    # ==================== PROFILE MANAGEMENT ====================

    def get_profile(self, dj_name: str) -> Optional[DJProfile]:
        """
        Get cached DJ profile.

        Args:
            dj_name: DJ name

        Returns:
            DJProfile if found, None otherwise
        """
        cached = self.cache.get_dj_profile(dj_name)
        if cached:
            return self._dict_to_profile(cached)
        return None

    def list_profiles(self) -> List[str]:
        """
        List all cached DJ profiles.

        Returns:
            List of DJ names with profiles
        """
        return self.cache.get_all_dj_profiles()

    def delete_profile(self, dj_name: str):
        """Delete a DJ profile from cache."""
        self.cache.invalidate_by_dj(dj_name)

    # ==================== UTILITY METHODS ====================

    def _find_audio_files(self, folder: Path, recursive: bool) -> List[Path]:
        """Find all audio files in folder."""
        files = []
        pattern = '**/*' if recursive else '*'

        for f in folder.glob(pattern):
            if f.is_file() and f.suffix.lower() in self.AUDIO_EXTENSIONS:
                files.append(f)

        return sorted(files)

    def _aggregate_profile(
        self,
        dj_name: str,
        metrics: List[DJProfileMetrics]
    ) -> DJProfile:
        """
        Aggregate individual set metrics into DJ profile (VECTORIZED).

        Uses numpy operations with float32 arrays for M2 optimization.
        """
        import numpy as np

        # Vectorized extraction using numpy (float32 arrays for M2 optimization)
        energy_shapes = np.array([m.energy_arc.arc_shape for m in metrics if m.energy_arc])
        durations = np.array([m.energy_arc.duration_min / 60.0 for m in metrics if m.energy_arc], dtype=np.float32)
        drops_per_hour = np.array([m.drop_pattern.drops_per_hour for m in metrics if m.drop_pattern], dtype=np.float32)
        tempos = np.array([m.tempo_distribution.tempo_mean for m in metrics if m.tempo_distribution], dtype=np.float32)

        # Extract keys for aggregation (vectorized comprehension)
        keys = np.array([
            m.key_analysis.key for m in metrics
            if m.key_analysis and hasattr(m.key_analysis, 'key')
        ])

        total_duration = float(durations.sum()) if len(durations) > 0 else 0.0

        # Aggregate using numpy (vectorized)
        avg_drops = float(drops_per_hour.mean()) if len(drops_per_hour) > 0 else 0.0
        tempo_range = (float(tempos.min()), float(tempos.max())) if len(tempos) > 0 else (0, 0)
        dominant_tempo = int(np.median(tempos)) if len(tempos) > 0 else 0

        # Most common energy shape (numpy unique + argmax)
        if len(energy_shapes) > 0:
            unique_shapes, counts = np.unique(energy_shapes, return_counts=True)
            avg_shape = unique_shapes[np.argmax(counts)]
        else:
            avg_shape = "unknown"

        # Most common keys (top 3) - vectorized numpy unique + argsort
        if len(keys) > 0:
            unique_keys, key_counts = np.unique(keys, return_counts=True)
            # Vectorized top-k: argsort descending, take first 3
            top_indices = np.argsort(key_counts)[::-1][:3]
            preferred_keys = unique_keys[top_indices].tolist()
        else:
            preferred_keys = []

        # Classify mixing style
        mixing_style = self._classify_mixing_style(avg_drops, tempo_range)
        energy_profile = self._classify_energy_profile(avg_drops, avg_shape)

        return DJProfile(
            dj_name=dj_name,
            n_sets_analyzed=len(metrics),
            total_duration_hours=total_duration,
            avg_energy_arc_shape=avg_shape,
            avg_drops_per_hour=avg_drops,
            tempo_range=tempo_range,
            dominant_tempo=dominant_tempo,
            preferred_keys=preferred_keys,
            mixing_style=mixing_style,
            energy_profile=energy_profile,
            set_metrics=metrics,
        )

    def _classify_mixing_style(
        self,
        drops_per_hour: float,
        tempo_range: tuple
    ) -> str:
        """Classify DJ mixing style."""
        tempo_variance = tempo_range[1] - tempo_range[0] if tempo_range[1] > 0 else 0

        if drops_per_hour > 10:
            return "hard"
        elif tempo_variance > 20:
            return "eclectic"
        elif drops_per_hour < 3:
            return "smooth"
        else:
            return "balanced"

    def _classify_energy_profile(
        self,
        drops_per_hour: float,
        arc_shape: str
    ) -> str:
        """Classify DJ energy profile."""
        if drops_per_hour > 8:
            return "peak_time"
        elif arc_shape == "building":
            return "warm_up"
        elif arc_shape in ("plateau", "minimal_arc"):
            return "deep"
        else:
            return "journey"

    def _dict_to_profile(self, data: dict) -> DJProfile:
        """Convert cached dict to DJProfile object."""
        return DJProfile(
            dj_name=data.get('dj_name', ''),
            n_sets_analyzed=data.get('n_sets_analyzed', 0),
            total_duration_hours=data.get('total_duration_hours', 0.0),
            avg_energy_arc_shape=data.get('avg_energy_arc_shape', ''),
            avg_drops_per_hour=data.get('avg_drops_per_hour', 0.0),
            tempo_range=tuple(data.get('tempo_range', (0, 0))),
            dominant_tempo=data.get('dominant_tempo', 0),
            preferred_keys=data.get('preferred_keys', []),
            mixing_style=data.get('mixing_style', ''),
            energy_profile=data.get('energy_profile', ''),
            set_metrics=[],  # Don't restore individual metrics from cache
        )

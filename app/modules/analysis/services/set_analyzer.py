"""
AnalysisService - Unified interface for audio analysis.

This service wraps the SetAnalysisPipeline and TrackAnalysisPipeline,
providing a clean API for CLI scripts and interactive menus.

Architecture:
    - Services use ICacheStatusProvider for READ-ONLY cache status (UI display)
    - Pipelines handle cache-aside pattern internally (read + write)
    - Services delegate analysis to Pipelines, which handle caching

Usage:
    from app.modules import create_analysis_service

    service = create_analysis_service()

    # Check cache status (for UI)
    if service.is_set_cached("set.mp3"):
        print("✓ Cached")

    # Analyze (Pipeline handles caching internally)
    result = service.analyze_set("set.mp3")

    # Force re-analysis
    result = service.analyze_set("set.mp3", force=True)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from app.common.logging import get_logger
from ..core.pipelines import (
    SetAnalysisPipeline,
    SetAnalysisResult,
    TrackAnalysisPipeline,
    TrackAnalysisResult,
    MixingStyle,
    PipelineContext,
)
from ..core.cache import CacheRepository, ICacheStatusProvider, CacheStats


# ============== Progress Tracking (Local Implementation) ==============

class IntegratedProgressCallback:
    """Callback interface for progress tracking."""

    def __call__(self, progress: float, stage: str, message: str) -> None:
        """Called with progress updates."""
        pass


class IntegratedProgressTracker:
    """
    Track progress across multiple pipeline stages.

    Usage:
        tracker = IntegratedProgressTracker(
            stage_weights={'LoadAudioStage': 0.1, 'ComputeSTFTStage': 0.2},
            callback=my_callback
        )
        tracker.start_stage('LoadAudioStage')
        tracker.complete_stage()
    """

    def __init__(
        self,
        stage_weights: Dict[str, float],
        callback: Optional[Callable[[float, str, str], None]] = None
    ):
        self.stage_weights = stage_weights
        self.callback = callback
        self.completed_weight = 0.0
        self.current_stage = ""

    def start_stage(self, stage_name: str) -> None:
        """Mark stage as started."""
        self.current_stage = stage_name
        if self.callback:
            progress = self.completed_weight * 100
            self.callback(progress, stage_name, f"Starting {stage_name}...")

    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        if self.current_stage in self.stage_weights:
            self.completed_weight += self.stage_weights[self.current_stage]
        if self.callback:
            progress = min(self.completed_weight * 100, 100.0)
            self.callback(progress, self.current_stage, f"Completed {self.current_stage}")


class BatchProgressDisplay:
    """Display for batch processing progress."""

    def __init__(self):
        self.total = 0
        self.current = 0

    def set_total(self, total: int) -> None:
        self.total = total

    def update(self, current: int, status: str = "") -> None:
        self.current = current
        # Use structured logging instead of print
        logger = get_logger(__name__)
        logger.info(f"Progress: [{current}/{self.total}] {status}")

logger = get_logger(__name__)


# Type aliases for callbacks
ProgressCallback = Callable[[float, str, str], None]  # (progress, stage, message)
FileCompleteCallback = Callable[[int, int, 'SetAnalysisResult'], None]  # (done, total, result)


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis operation."""
    total_files: int
    processed: int
    cached: int
    failed: int
    results: List[SetAnalysisResult]
    errors: Dict[str, str]  # file_path -> error message

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed + self.cached) / self.total_files


class AnalysisService:
    """
    Unified service for audio analysis.

    Architecture:
        - Uses ICacheStatusProvider for READ-ONLY cache status queries (UI)
        - Delegates analysis to Pipelines (which handle cache-aside internally)
        - Does NOT write to cache directly

    Provides consistent API for:
    - Single track analysis (TrackAnalysisPipeline)
    - DJ set analysis (SetAnalysisPipeline)
    - Batch processing with progress tracking
    - Cache status queries (read-only)
    """

    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus', '.ogg'}

    def __init__(
        self,
        cache_status: Optional[ICacheStatusProvider] = None,
        cache_dir: str = "cache",
        sr: int = 22050,
    ):
        """
        Initialize analysis service.

        Args:
            cache_status: Read-only cache status provider (for UI queries)
            cache_dir: Cache directory path (used if cache_status not provided)
            sr: Sample rate for analysis
        """
        self.cache_dir = cache_dir
        self.sr = sr

        # Read-only cache status provider
        if cache_status is not None:
            self._cache_status = cache_status
        else:
            # Fallback: create CacheRepository (implements ICacheStatusProvider)
            self._cache_status = CacheRepository(cache_dir)

    # ==================== CACHE STATUS (Read-only for UI) ====================

    def is_set_cached(self, path: str) -> bool:
        """Check if set analysis is cached. For UI display."""
        return self._cache_status.exists_set(path)

    def is_track_cached(self, path: str) -> bool:
        """Check if track analysis is cached. For UI display."""
        return self._cache_status.exists_track(path)

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics. For UI display."""
        return self._cache_status.get_cache_stats()

    def get_folder_cache_status(
        self,
        folder: str,
        mode: str = 'set',
        recursive: bool = True
    ) -> Dict[str, bool]:
        """
        Get cache status for all files in folder. For file browser UI.

        Returns:
            Dict mapping file path to cache status (True = cached)
        """
        return self._cache_status.get_folder_status(folder, mode, recursive)

    @property
    def cache(self) -> ICacheStatusProvider:
        """Cache status provider (read-only)."""
        return self._cache_status

    # ==================== TRACK ANALYSIS ====================

    def analyze_track(
        self,
        path: str,
        use_cache: bool = True,
        force: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TrackAnalysisResult:
        """
        Analyze a single track (3-7 minutes).

        Pipeline handles cache-aside internally.

        Args:
            path: Path to audio file
            use_cache: Use cache (default True)
            force: Force re-analysis (default False)
            on_progress: Progress callback (progress, stage, message)

        Returns:
            TrackAnalysisResult with features and classification
        """
        pipeline = TrackAnalysisPipeline(sr=self.sr)

        # Pipeline handles cache-aside internally
        return pipeline.analyze(path, use_cache=use_cache, force=force)

    # ==================== SET ANALYSIS ====================

    def analyze_set(
        self,
        path: str,
        mixing_style: Optional[MixingStyle] = None,
        analyze_genres: bool = False,
        use_ml_drops: bool = True,
        ml_drop_threshold: float = 0.5,
        use_cache: bool = True,
        force: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        verbose: bool = False,
    ) -> SetAnalysisResult:
        """
        Analyze a DJ set (30min - 2hr+).

        Pipeline handles cache-aside internally.

        Args:
            path: Path to audio file
            mixing_style: DJ mixing style (SMOOTH, STANDARD, HARD)
            analyze_genres: Enable genre analysis per segment
            use_ml_drops: Use ML model for drop detection
            ml_drop_threshold: ML detection threshold (0.5-0.9)
            use_cache: Use cache (default True)
            force: Force re-analysis (default False)
            on_progress: Progress callback (progress, stage, message)
            verbose: Print detailed stage-by-stage progress

        Returns:
            SetAnalysisResult with transitions, drops, segments
        """
        # Create pipeline
        pipeline = SetAnalysisPipeline(
            mixing_style=mixing_style,
            analyze_genres=analyze_genres,
            use_ml_drops=use_ml_drops,
            ml_drop_threshold=ml_drop_threshold,
            sr=self.sr,
            verbose=verbose,
        )

        # Setup progress tracker if callback provided
        if on_progress:
            tracker = IntegratedProgressTracker(
                stage_weights={
                    'LoadAudioStage': 0.10,
                    'ComputeSTFTStage': 0.15,
                    'DetectTransitionsStage': 0.20,
                    'SegmentTracksStage': 0.05,
                    'LaplacianSegmentationStage': 0.05,
                    'ComputeBeatGridStage': 0.10,
                    'DetectAllDropsStage': 0.20,
                    'DetectDropsMLStage': 0.20,
                    'BuildTimelineStage': 0.05,
                    'AnalyzeSegmentGenresStage': 0.10,
                },
                callback=on_progress
            )

            original_callback = pipeline.on_stage_complete

            def progress_callback(stage_name: str, context: PipelineContext):
                tracker.start_stage(stage_name)
                tracker.complete_stage()
                if original_callback:
                    original_callback(stage_name, context)

            pipeline.on_stage_complete = progress_callback

        # Pipeline handles cache-aside internally
        return pipeline.analyze(path, use_cache=use_cache, force=force)

    # ==================== BATCH ANALYSIS ====================

    def analyze_batch(
        self,
        folder: str,
        recursive: bool = True,
        mixing_style: Optional[MixingStyle] = None,
        analyze_genres: bool = False,
        use_ml_drops: bool = True,
        ml_drop_threshold: float = 0.5,
        use_cache: bool = True,
        force: bool = False,
        on_file_start: Optional[Callable[[int, int, str], None]] = None,
        on_file_complete: Optional[FileCompleteCallback] = None,
        on_progress: Optional[ProgressCallback] = None,
        display: Optional[BatchProgressDisplay] = None,
    ) -> BatchAnalysisResult:
        """
        Batch analyze all audio files in a folder.

        Args:
            folder: Path to folder with audio files
            recursive: Recursively scan subfolders
            mixing_style: DJ mixing style preset
            analyze_genres: Enable genre analysis
            use_ml_drops: Use ML for drop detection
            ml_drop_threshold: ML detection threshold
            use_cache: Use cached results
            force: Force re-analysis of all files
            on_file_start: Callback(index, total, path) when starting file
            on_file_complete: Callback(index, total, result) when file done
            on_progress: Progress callback for current file
            display: BatchProgressDisplay for static UI updates

        Returns:
            BatchAnalysisResult with all results and statistics
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder}")

        # Find audio files
        files = self._find_audio_files(folder_path, recursive)
        if not files:
            return BatchAnalysisResult(
                total_files=0,
                processed=0,
                cached=0,
                failed=0,
                results=[],
                errors={}
            )

        # Deduplicate (same set, different format)
        files = self._deduplicate_files(files)

        # Separate cached vs uncached (vectorized using read-only cache status)
        import numpy as np
        files_arr = np.array(files, dtype=object)

        if not force and use_cache:
            # Vectorized cache status check
            cache_mask = np.array([self.is_set_cached(str(f)) for f in files_arr])
            cached_files = files_arr[cache_mask].tolist()
            uncached_files = files_arr[~cache_mask].tolist()
        else:
            cached_files = []
            uncached_files = files

        results: List[SetAnalysisResult] = []
        errors: Dict[str, str] = {}
        processed = 0

        # Load cached results via pipeline (which handles cache-aside)
        for f in cached_files:
            # Pipeline.analyze() will return from cache
            result = self.analyze_set(
                str(f),
                mixing_style=mixing_style,
                analyze_genres=analyze_genres,
                use_ml_drops=use_ml_drops,
                ml_drop_threshold=ml_drop_threshold,
                use_cache=True,
                force=False,
            )
            results.append(result)
            if display:
                display.log_cached(f.name)

        # Process uncached files
        total = len(files)
        for i, file_path in enumerate(uncached_files, 1):
            abs_path = str(file_path.absolute())
            file_index = len(cached_files) + i

            # Notify start
            if on_file_start:
                on_file_start(file_index, total, abs_path)
            if display:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                display.start_file(file_path.name, size_mb=file_size_mb)

            try:
                # Create per-file progress callback
                file_progress = None
                if display:
                    file_progress = lambda p, s, m: display.update_file_progress(p, s, m)
                elif on_progress:
                    file_progress = on_progress

                result = self.analyze_set(
                    abs_path,
                    mixing_style=mixing_style,
                    analyze_genres=analyze_genres,
                    use_ml_drops=use_ml_drops,
                    ml_drop_threshold=ml_drop_threshold,
                    use_cache=use_cache,
                    force=force,
                    on_progress=file_progress,
                )

                results.append(result)
                processed += 1

                # Notify complete
                if on_file_complete:
                    on_file_complete(file_index, total, result)
                if display:
                    display.update_batch(done=processed, cached=len(cached_files))
                    display.log_result(
                        file_path.name,
                        transitions=result.n_transitions,
                        drops=result.total_drops,
                        time_sec=result.processing_time_sec
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                errors[abs_path] = str(e)
                if display:
                    display.log_result(file_path.name, error=str(e)[:30])

        # Complete display
        if display:
            display.complete()

        return BatchAnalysisResult(
            total_files=total,
            processed=processed,
            cached=len(cached_files),
            failed=len(errors),
            results=results,
            errors=errors
        )

    # ==================== UTILITY METHODS ====================

    def _find_audio_files(self, folder: Path, recursive: bool) -> List[Path]:
        """Find all audio files in folder (VECTORIZED filtering)."""
        import numpy as np
        pattern = '**/*' if recursive else '*'

        # Collect all matching paths at once
        all_files = list(folder.glob(pattern))
        if not all_files:
            return []

        # Vectorized filtering: check is_file and extension in one pass
        paths_arr = np.array(all_files, dtype=object)
        is_file_mask = np.array([f.is_file() for f in paths_arr])
        ext_mask = np.array([f.suffix.lower() in self.AUDIO_EXTENSIONS for f in paths_arr])

        # Combined mask
        valid_mask = is_file_mask & ext_mask
        valid_files = paths_arr[valid_mask].tolist()

        return sorted(valid_files)

    def _deduplicate_files(self, files: List[Path]) -> List[Path]:
        """
        Remove duplicate sets (same content, different format) - VECTORIZED.

        Priority: FLAC > WAV > M4A > MP3 > AAC
        Uses numpy for efficient grouping and selection.
        """
        import re
        import numpy as np

        if not files:
            return []

        # Vectorized name normalization
        def normalize_name(f: Path) -> str:
            name = f.stem.lower()
            name = re.sub(r'[_-]?(320|256|192|128|hq|lq|\(\d+\))$', '', name)
            return re.sub(r'[-_\s]+', '-', name)

        files_arr = np.array(files, dtype=object)
        names = np.array([normalize_name(f) for f in files_arr])

        # Get unique names and their groups
        unique_names = np.unique(names)

        # Format priority for vectorized selection
        format_priority = {'.flac': 0, '.wav': 1, '.m4a': 2, '.mp3': 3, '.aac': 4, '.opus': 5, '.ogg': 6}
        priorities = np.array([format_priority.get(f.suffix.lower(), 99) for f in files_arr])

        # For each unique name, find the file with lowest priority (best format)
        result = []
        for name in unique_names:
            group_mask = names == name
            group_priorities = priorities[group_mask]
            group_files = files_arr[group_mask]
            best_idx = np.argmin(group_priorities)
            result.append(group_files[best_idx])

        return sorted(result)

    def get_cached_sets(self, folder: str, recursive: bool = True) -> Dict[str, bool]:
        """
        Get cache status for all files in folder.

        Returns:
            Dict mapping file path to cache status (True = cached)
        """
        return self.get_folder_cache_status(folder, mode='set', recursive=recursive)

    def clear_cache(self, path: Optional[str] = None):
        """
        Clear cache.

        Note: This requires write access, so only works if cache_status
        is actually a CacheRepository (not just ICacheStatusProvider).

        Args:
            path: If provided, clear only this file's cache.
                  If None, clear all set analysis cache.
        """
        if not isinstance(self._cache_status, CacheRepository):
            raise RuntimeError("Cannot clear cache: read-only cache provider")

        if path:
            self._cache_status.invalidate_set(path)
        else:
            self._cache_status.clear_sets()

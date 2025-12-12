"""
DJ Analysis Session - Coordinates set analysis and profile building.

Keeps domains separate:
- Set analysis: src/core/pipelines/set_analysis.py
- Profile model: src/core/models/dj_style_profile.py
- Cache storage: src/core/pipelines/cache_manager.py (predictions.db)
- Session coordination: this module
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SessionProgress:
    """Current session progress."""
    total_sets: int = 0
    analyzed_sets: int = 0
    cached_sets: int = 0
    failed_sets: int = 0
    current_set: str = ""
    profile_updates: int = 0
    is_complete: bool = False


class DJAnalysisSession:
    """
    Coordinates DJ set analysis with incremental profile building.

    Analysis and profile building run in parallel:
    - Main thread: analyzes sets one by one
    - Background thread: updates profile after each successful analysis

    All data stored in predictions.db (SQLite):
    - Set analysis results: set_analysis_results table
    - DJ profiles: dj_profiles table

    Usage:
        session = DJAnalysisSession("Nina Kraviz")

        # Analyze sets with incremental profile updates
        results, profile = session.run(set_paths, mixing_style=MixingStyle.SMOOTH)

        # Or with progress callback
        def on_progress(progress: SessionProgress):
            print(f"Progress: {progress.analyzed_sets}/{progress.total_sets}")

        results, profile = session.run(set_paths, on_progress=on_progress)

        # List all cached profiles
        profiles = DJAnalysisSession.list_profiles()
    """

    def __init__(
        self,
        dj_name: str,
        cache_dir: str = "~/.mood-classifier/cache",
    ):
        """
        Args:
            dj_name: DJ name for profile
            cache_dir: Cache directory (predictions.db location)
        """
        self.dj_name = dj_name
        self.cache_dir = cache_dir

        # State
        self._progress = SessionProgress()
        self._results: List[Any] = []  # SetAnalysisResult
        self._set_paths: List[str] = []  # For cache key
        self._profile = None
        self._profile_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._profile_future: Optional[Future] = None
        self._cache_manager = None

    @property
    def progress(self) -> SessionProgress:
        """Current progress."""
        return self._progress

    @property
    def profile(self):
        """Current profile (may be partial during analysis)."""
        with self._profile_lock:
            return self._profile

    @staticmethod
    def list_profiles(cache_dir: str = "~/.mood-classifier/cache") -> List[Dict[str, Any]]:
        """
        List all cached DJ profiles.

        Returns:
            List of profile metadata: [{'dj_name': ..., 'n_sets': ..., 'total_hours': ...}, ...]
        """
        from ..core.pipelines.cache_manager import CacheManager
        cache = CacheManager(cache_dir)
        return cache.get_all_dj_profiles()

    @staticmethod
    def load_profile(dj_name: str, cache_dir: str = "~/.mood-classifier/cache"):
        """
        Load cached DJ profile by name.

        Args:
            dj_name: DJ name

        Returns:
            DJStyleProfile or None
        """
        from ..core.pipelines.cache_manager import CacheManager
        from ..core.models.dj_style_profile import DJStyleProfile

        cache = CacheManager(cache_dir)
        conn = __import__('sqlite3').connect(str(cache.predictions_db))
        cursor = conn.cursor()
        cursor.execute(
            'SELECT profile_json FROM dj_profiles WHERE dj_name = ?',
            (dj_name,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            import json
            data = json.loads(row[0])
            return DJStyleProfile.from_dict(data) if hasattr(DJStyleProfile, 'from_dict') else data
        return None

    def run(
        self,
        set_paths: List[str],
        analyze_genres: bool = True,
        force: bool = False,
        mixing_style = None,  # Optional[MixingStyle]
        verbose: bool = False,
        show_progress: bool = True,
        on_progress: Optional[Callable[[SessionProgress], None]] = None,
    ):
        """
        Run analysis session with incremental profile building.

        Args:
            set_paths: Paths to DJ set audio files
            analyze_genres: Enable genre analysis per segment
            force: Force re-analysis (ignore cache)
            mixing_style: MixingStyle preset (SMOOTH/STANDARD/HARD)
            verbose: Print detailed step-by-step progress
            show_progress: Print progress to console
            on_progress: Callback for progress updates

        Returns:
            Tuple[List[SetAnalysisResult], DJStyleProfile]
        """
        # Import here to avoid circular imports
        from ..core.pipelines.set_analysis import SetBatchAnalyzer
        from ..core.pipelines.cache_manager import CacheManager
        from ..core.models.dj_style_profile import DJStyleProfile
        import os

        # Initialize
        self._progress = SessionProgress(total_sets=len(set_paths))
        self._results = []
        self._set_paths = [os.path.abspath(p) for p in set_paths]
        self._profile = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_manager = CacheManager(self.cache_dir)

        analyzer = SetBatchAnalyzer(cache_dir=self.cache_dir)

        def on_set_complete(done: int, total: int, path: str, result):
            """Called after each set analysis."""
            self._progress.analyzed_sets = done
            self._progress.current_set = Path(path).name

            if result.success:
                self._results.append(result)
                # Trigger profile update in background
                self._update_profile_async(DJStyleProfile)
            else:
                self._progress.failed_sets += 1

            if on_progress:
                on_progress(self._progress)

        if show_progress:
            print(f"\n=== DJ Analysis Session: {self.dj_name} ===")
            print(f"Sets to analyze: {len(set_paths)}")
            print(f"Storage: predictions.db\n")

        # Run analysis
        results = analyzer.analyze_sets(
            set_paths=set_paths,
            analyze_genres=analyze_genres,
            force=force,
            mixing_style=mixing_style,
            verbose=verbose,
            show_progress=show_progress,
            on_progress=on_set_complete,
        )

        # Wait for final profile update
        if self._profile_future:
            self._profile_future.result()

        self._executor.shutdown(wait=True)
        self._progress.is_complete = True

        # Final profile
        with self._profile_lock:
            final_profile = self._profile

        if show_progress and final_profile:
            print(f"\n=== Profile Complete ===")
            print(final_profile.summary())
            print(f"\nSaved to: predictions.db (dj_profiles table)")

        return results, final_profile

    def _update_profile_async(self, DJStyleProfile):
        """Update profile in background thread."""
        # Wait for previous update if running
        if self._profile_future and not self._profile_future.done():
            try:
                self._profile_future.result(timeout=0.1)
            except:
                pass  # Continue anyway

        # Submit new update
        results_copy = list(self._results)  # Copy to avoid race
        self._profile_future = self._executor.submit(
            self._build_profile, DJStyleProfile, results_copy
        )

    def _build_profile(self, DJStyleProfile, results: List):
        """Build profile from current results (runs in background)."""
        try:
            set_dicts = [r.to_dict() for r in results if r.success]
            if not set_dicts:
                return

            # Convert to format expected by from_set_analyses
            formatted_sets = []
            for s in set_dicts:
                formatted = {
                    'file_path': s['file_path'],
                    'file_name': s['file_name'],
                    'duration_min': s['duration_sec'] / 60,
                    'n_transitions': s['n_transitions'],
                    'transition_density': s['transition_density'],
                    'n_segments': s['n_segments'],
                    'total_drops': s['total_drops'],
                    'drop_density': s['drop_density'],
                }

                # Segment stats
                segments = s.get('segments', [])
                if segments:
                    durations = [seg['duration'] / 60 for seg in segments]
                    import numpy as np
                    formatted['segment_duration_mean'] = float(np.mean(durations))
                    formatted['segment_duration_std'] = float(np.std(durations))
                    formatted['segment_duration_min'] = float(np.min(durations))
                    formatted['segment_duration_max'] = float(np.max(durations))

                # Genre stats
                gd = s.get('genre_distribution')
                if gd:
                    formatted['primary_genre'] = gd.get('primary_category', '')
                    formatted['genre_diversity'] = gd.get('genre_diversity', 0)
                    formatted['n_unique_subgenres'] = gd.get('n_unique_subgenres', 0)
                    formatted['genre_transitions'] = gd.get('genre_transitions', 0)
                    formatted['top_subgenres'] = gd.get('top_subgenres', [])
                    formatted['mood_tags'] = gd.get('mood_tags', {})

                formatted_sets.append(formatted)

            profile = DJStyleProfile.from_set_analyses(
                dj_name=self.dj_name,
                set_results=formatted_sets
            )

            with self._profile_lock:
                self._profile = profile
                self._progress.profile_updates += 1

                # Save to database
                if self._cache_manager:
                    self._cache_manager.save_dj_profile(
                        dj_name=self.dj_name,
                        profile=profile.to_dict(),
                        set_paths=self._set_paths
                    )

            logger.info(f"Profile updated: {len(formatted_sets)} sets -> predictions.db")

        except Exception as e:
            logger.error(f"Profile update failed: {e}")
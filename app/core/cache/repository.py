"""
Cache Repository - Unified Unit of Work for all caching operations.

Single entry point for ALL cache operations in the project.
Encapsulates CacheManager and provides domain-aware API.

Domains:
- Sets (DJ set analysis)
- Tracks (individual track analysis)
- DJ Profiles (aggregated style)
- Features (ML training features)
- Derived Features (calibration features)
- STFT (spectral data)
- Predictions (zone classifications)

Usage:
    from app.core.cache import CacheRepository

    # Singleton access (recommended)
    repo = CacheRepository.get_instance()

    # Domain operations
    cached_set = repo.get_set("/path/to/set.mp3")
    repo.save_set(analysis)

    # Feature operations (for training/calibration)
    features = repo.get_features(file_hash)
    repo.save_features(file_hash, features)

    # Granular invalidation
    repo.invalidate_set("/path/to/set.mp3")
    repo.invalidate_by_directory("/path/to/dj/")
    repo.invalidate_by_dj("Josh Baker")

Architecture:
    CacheRepository (public API, Unit of Work)
        └── CacheManager (internal implementation, NOT exported)

    CacheRepository implements ICacheStatusProvider for read-only queries.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from app.common.logging import get_logger
from .models import (
    CachedSetAnalysis,
    CachedTrackAnalysis,
    CachedDJProfile,
)
from .interfaces import ICacheStatusProvider, CacheStats
from app.modules.analysis.pipelines.cache_manager import CacheManager

if TYPE_CHECKING:
    from ..primitives import STFTCache

logger = get_logger(__name__)


class CacheRepository(ICacheStatusProvider):
    """
    Unified Unit of Work for all caching operations.

    Single entry point for ALL cache operations in the project.
    Encapsulates CacheManager (internal) and provides domain-aware API.

    Storage:
    - SQLite (predictions.db): structured data (sets, tracks, profiles, predictions)
    - Pickle files (features/): ML training features
    - NumPy files (stft/): STFT matrices, derived features

    Thread Safety:
    - SQLite operations are thread-safe (separate connections)
    - File operations use atomic writes where possible

    Usage:
        repo = CacheRepository.get_instance()  # Singleton
        # or
        repo = CacheRepository("cache")  # Custom path
    """

    _instance: Optional['CacheRepository'] = None

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache repository.

        Args:
            cache_dir: Directory for cache storage (relative or absolute)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self._manager = CacheManager(str(self.cache_dir))

    @classmethod
    def get_instance(cls, cache_dir: Optional[str] = None) -> 'CacheRepository':
        """
        Get singleton instance.

        Args:
            cache_dir: Cache directory path. If None, uses DATA_DIR environment variable
                      (default: /data in production, 'cache' in development)
        """
        if cls._instance is None:
            if cache_dir is None:
                # Use DATA_DIR from environment (volume-backed in production)
                cache_dir = os.getenv('DATA_DIR', 'cache')
            cls._instance = cls(cache_dir)
        return cls._instance

    # ============== Set Analysis ==============

    def get_set(self, file_path: str) -> Optional[CachedSetAnalysis]:
        """
        Get cached set analysis by file path.

        Args:
            file_path: Path to audio file

        Returns:
            CachedSetAnalysis or None if not cached/outdated
        """
        file_path = os.path.abspath(file_path)
        cached_dict = self._manager.get_set_analysis(file_path)

        if cached_dict:
            try:
                return CachedSetAnalysis.from_dict(cached_dict)
            except Exception as e:
                logger.warning(f"Failed to parse cached set: {e}")
                return None
        return None

    def save_set(self, analysis: CachedSetAnalysis) -> bool:
        """
        Save set analysis to cache.

        Args:
            analysis: CachedSetAnalysis to cache

        Returns:
            True if saved successfully
        """
        try:
            file_path = os.path.abspath(analysis.file_path)
            self._manager.save_set_analysis(file_path, analysis.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to save set cache: {e}")
            return False

    def save_set_dict(self, file_path: str, result_dict: Dict) -> bool:
        """
        Save set analysis from dictionary.

        Convenience method for backward compatibility.

        Args:
            file_path: Path to audio file
            result_dict: Analysis result as dictionary

        Returns:
            True if saved successfully
        """
        try:
            file_path = os.path.abspath(file_path)
            self._manager.save_set_analysis(file_path, result_dict)
            return True
        except Exception as e:
            logger.error(f"Failed to save set cache: {e}")
            return False

    def invalidate_set(self, file_path: str):
        """Remove set analysis from cache."""
        file_path = os.path.abspath(file_path)
        self._manager.invalidate_set_analysis(file_path)

    def get_all_cached_sets(self) -> List[str]:
        """Get paths of all cached sets."""
        return self._manager.get_cached_set_paths()

    # ============== Track Analysis ==============

    def get_track(self, file_path: str) -> Optional[CachedTrackAnalysis]:
        """
        Get cached track analysis by file path.

        Args:
            file_path: Path to audio file

        Returns:
            CachedTrackAnalysis or None if not cached/outdated
        """
        file_path = os.path.abspath(file_path)
        cached_dict = self._manager.get_track_analysis(file_path)

        if cached_dict:
            try:
                return CachedTrackAnalysis.from_dict(cached_dict)
            except Exception as e:
                logger.warning(f"Failed to parse cached track: {e}")
                return None
        return None

    def save_track(self, analysis: CachedTrackAnalysis) -> bool:
        """
        Save track analysis to cache.

        Args:
            analysis: CachedTrackAnalysis to cache

        Returns:
            True if saved successfully
        """
        try:
            file_path = os.path.abspath(analysis.file_path)
            self._manager.save_track_analysis(file_path, analysis.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to save track cache: {e}")
            return False

    def save_track_dict(self, file_path: str, result_dict: Dict) -> bool:
        """
        Save track analysis from dictionary.

        Args:
            file_path: Path to audio file
            result_dict: Analysis result as dictionary

        Returns:
            True if saved successfully
        """
        try:
            file_path = os.path.abspath(file_path)
            self._manager.save_track_analysis(file_path, result_dict)
            return True
        except Exception as e:
            logger.error(f"Failed to save track cache: {e}")
            return False

    def invalidate_track(self, file_path: str):
        """Remove track analysis from cache."""
        file_path = os.path.abspath(file_path)
        self._manager.invalidate_track_analysis(file_path)

    def get_all_cached_tracks(self) -> List[str]:
        """Get paths of all cached tracks."""
        return self._manager.get_cached_track_paths()

    # ============== DJ Profiles ==============

    def get_dj_profile(self, dj_name: str) -> Optional[CachedDJProfile]:
        """
        Get cached DJ profile by name.

        Args:
            dj_name: DJ name (case-insensitive)

        Returns:
            CachedDJProfile or None if not cached
        """
        cached_dict = self._manager.get_dj_profile(dj_name.lower())

        if cached_dict:
            try:
                return CachedDJProfile.from_dict(cached_dict)
            except Exception as e:
                logger.warning(f"Failed to parse cached profile: {e}")
                return None
        return None

    def save_dj_profile(self, profile: CachedDJProfile) -> bool:
        """
        Save DJ profile to cache.

        Args:
            profile: CachedDJProfile to cache

        Returns:
            True if saved successfully
        """
        try:
            profile.updated_at = time.time()
            if profile.created_at == 0:
                profile.created_at = profile.updated_at

            self._manager.save_dj_profile(
                profile.dj_name.lower(),
                profile.to_dict(),
                profile.set_file_paths
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save DJ profile: {e}")
            return False

    def get_all_dj_profiles(self) -> List[str]:
        """Get names of all cached DJ profiles."""
        return self._manager.get_dj_profile_names()

    def get_all_dj_profiles_info(self) -> List[Dict]:
        """Get metadata of all cached DJ profiles."""
        return self._manager.get_all_dj_profiles()

    def save_dj_profile_dict(
        self,
        dj_name: str,
        profile_dict: Dict,
        set_paths: List[str]
    ) -> bool:
        """
        Save DJ profile from dictionary.

        Args:
            dj_name: DJ name
            profile_dict: Profile data as dictionary
            set_paths: List of set file paths

        Returns:
            True if saved successfully
        """
        try:
            self._manager.save_dj_profile(dj_name, profile_dict, set_paths)
            return True
        except Exception as e:
            logger.error(f"Failed to save DJ profile: {e}")
            return False

    # ============== Statistics ==============

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        base_stats = self._manager.get_cache_stats()
        set_stats = self._manager.get_set_analysis_stats()

        return {
            **base_stats,
            'set_count': set_stats.get('count', 0),
        }

    def clear_all(self):
        """Clear all cached data."""
        self._manager.clear_all()

    def cleanup(self, max_age_days: int = 30):
        """Remove old cache entries."""
        self._manager.cleanup_old_entries(max_age_days)

    # ============== File Hash Utilities ==============

    def compute_file_hash(self, file_path: str) -> str:
        """Compute hash for a file (for cache key generation)."""
        return self._manager.compute_file_hash(file_path)

    def is_file_changed(self, file_path: str, cached_hash: str) -> bool:
        """Check if file has changed since caching."""
        current_hash = self.compute_file_hash(file_path)
        return current_hash != cached_hash

    # ============== Granular Invalidation ==============

    def invalidate_by_directory(self, directory: str) -> int:
        """
        Invalidate all cached sets in a directory.

        Args:
            directory: Directory path (all sets under this path will be invalidated)

        Returns:
            Number of invalidated entries
        """
        directory = os.path.abspath(directory)
        cached_paths = self.get_all_cached_sets()
        count = 0

        for path in cached_paths:
            if path.startswith(directory):
                self.invalidate_set(path)
                count += 1

        logger.info(f"Invalidated {count} cache entries in {directory}")
        return count

    def invalidate_by_dj(self, dj_name: str) -> int:
        """
        Invalidate DJ profile and all associated set analyses.

        Args:
            dj_name: DJ name to invalidate

        Returns:
            Number of invalidated entries (1 profile + N sets)
        """
        dj_name_lower = dj_name.lower()
        count = 0

        # Get profile to find associated sets
        profile = self.get_dj_profile(dj_name)
        if profile and profile.set_file_paths:
            for set_path in profile.set_file_paths:
                self.invalidate_set(set_path)
                count += 1

        # Delete profile
        self._manager.delete_dj_profile(dj_name_lower)
        count += 1

        logger.info(f"Invalidated DJ profile '{dj_name}' and {count - 1} associated sets")
        return count

    def invalidate_older_than(self, days: int) -> int:
        """
        Invalidate cache entries older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of invalidated entries
        """
        self._manager.cleanup_old_entries(days)
        logger.info(f"Cleaned up cache entries older than {days} days")
        return 0  # cleanup_old_entries doesn't return count

    def clear_sets(self):
        """Clear all set analysis cache (keeps tracks and profiles)."""
        self._manager.clear_set_analysis_cache()
        logger.info("Cleared all set analysis cache")

    def clear_predictions(self):
        """Clear all zone predictions (keeps sets and profiles)."""
        self._manager.clear_predictions()
        logger.info("Cleared all zone predictions")

    # ============== STFT Cache ==============

    def get_stft(self, file_hash: str) -> Optional['STFTCache']:
        """
        Get cached STFT data.

        Args:
            file_hash: File hash (from compute_file_hash)

        Returns:
            STFTCache or None
        """
        return self._manager.get_stft(file_hash)

    def save_stft(self, file_hash: str, cache: 'STFTCache'):
        """
        Save STFT data to cache.

        Args:
            file_hash: File hash
            cache: STFTCache object
        """
        self._manager.save_stft(file_hash, cache)

    # ============== Features (Training) ==============

    def get_features(self, file_hash: str) -> Optional[np.ndarray]:
        """
        Get cached training features.

        Args:
            file_hash: File hash

        Returns:
            Feature array or None
        """
        return self._manager.get_features(file_hash)

    def save_features(self, file_hash: str, features: np.ndarray):
        """
        Save training features to cache.

        Args:
            file_hash: File hash
            features: Feature array
        """
        self._manager.save_features(file_hash, features)

    def get_features_dict(self, file_hash: str) -> Optional[Dict[str, float]]:
        """Get cached features as dictionary."""
        return self._manager.get_features_dict(file_hash)

    def save_features_dict(self, file_hash: str, features: Dict[str, float]):
        """Save features dictionary to cache."""
        self._manager.save_features_dict(file_hash, features)

    # ============== Derived Features (Calibration) ==============

    def get_derived_feature(self, file_hash: str, feature_name: str) -> Optional[np.ndarray]:
        """
        Get single derived feature array.

        Args:
            file_hash: File hash
            feature_name: Feature name (e.g., 'rms', 'spectral_flux')

        Returns:
            Feature array or None
        """
        return self._manager.get_derived_feature(file_hash, feature_name)

    def save_derived_feature(self, file_hash: str, feature_name: str, data: np.ndarray):
        """
        Save single derived feature array.

        Args:
            file_hash: File hash
            feature_name: Feature name
            data: Feature array
        """
        self._manager.save_derived_feature(file_hash, feature_name, data)

    def get_derived_features_batch(
        self,
        file_hash: str,
        feature_names: List[str]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get multiple derived features at once.

        Args:
            file_hash: File hash
            feature_names: List of feature names

        Returns:
            Dict of feature_name -> array, or None if any missing
        """
        return self._manager.get_derived_features_batch(file_hash, feature_names)

    def save_derived_features_batch(self, file_hash: str, features: Dict[str, np.ndarray]):
        """
        Save multiple derived features at once.

        Args:
            file_hash: File hash
            features: Dict of feature_name -> array
        """
        self._manager.save_derived_features_batch(file_hash, features)

    def has_all_derived_features(self, file_hash: str, feature_names: List[str]) -> bool:
        """
        Check if all specified derived features are cached.

        Args:
            file_hash: File hash
            feature_names: List of feature names to check

        Returns:
            True if all features are cached
        """
        return self._manager.has_all_derived_features(file_hash, feature_names)

    def get_derived_feature_metadata(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get metadata for derived features (sr, hop_length, etc.)."""
        return self._manager.get_derived_feature_metadata(file_hash)

    def save_derived_feature_metadata(self, file_hash: str, metadata: Dict[str, Any]):
        """Save metadata for derived features."""
        self._manager.save_derived_feature_metadata(file_hash, metadata)

    # ============== Predictions ==============

    def get_prediction(self, file_hash: str) -> Optional[Tuple[str, float, Dict]]:
        """
        Get cached zone prediction.

        Args:
            file_hash: File hash

        Returns:
            Tuple of (zone, confidence, features) or None
        """
        return self._manager.get_prediction(file_hash)

    def save_prediction(
        self,
        file_path: str,
        file_hash: str,
        zone: str,
        confidence: float,
        features: Dict
    ):
        """
        Save zone prediction to cache.

        Args:
            file_path: Path to audio file
            file_hash: File hash
            zone: Predicted zone (yellow/green/purple)
            confidence: Prediction confidence
            features: Feature dictionary
        """
        self._manager.save_prediction(file_path, file_hash, zone, confidence, features)

    # ============== Properties ==============

    @property
    def stft_dir(self) -> Path:
        """Directory for STFT and derived feature files."""
        return self._manager.stft_dir

    @property
    def features_dir(self) -> Path:
        """Directory for training feature files."""
        return self._manager.features_dir

    # ============== ICacheStatusProvider Implementation ==============

    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.opus', '.ogg'}

    def exists_set(self, path: str) -> bool:
        """Check if set analysis is cached."""
        return self.get_set(path) is not None

    def exists_track(self, path: str) -> bool:
        """Check if track analysis is cached."""
        return self.get_track(path) is not None

    def exists_profile(self, dj_name: str) -> bool:
        """Check if DJ profile is cached."""
        return self.get_dj_profile(dj_name) is not None

    def get_cache_stats(self) -> CacheStats:
        """Get overall cache statistics (ICacheStatusProvider interface)."""
        stats = self.get_stats()

        # Calculate total size
        total_size_mb = 0.0
        try:
            # SQLite database size
            db_path = self.cache_dir / "predictions.db"
            if db_path.exists():
                total_size_mb += db_path.stat().st_size / (1024 * 1024)

            # STFT directory size
            if self.stft_dir.exists():
                for f in self.stft_dir.glob('**/*'):
                    if f.is_file():
                        total_size_mb += f.stat().st_size / (1024 * 1024)

            # Features directory size
            if self.features_dir.exists():
                for f in self.features_dir.glob('**/*'):
                    if f.is_file():
                        total_size_mb += f.stat().st_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Error calculating cache size: {e}")

        return CacheStats(
            set_count=stats.get('set_count', 0),
            track_count=stats.get('track_count', 0),
            profile_count=len(self.get_all_dj_profiles()),
            total_size_mb=round(total_size_mb, 2),
        )

    def get_folder_status(
        self,
        folder: str,
        mode: str = 'set',
        recursive: bool = True
    ) -> Dict[str, bool]:
        """
        Get cache status for all audio files in a folder.

        Optimized: Uses single SQL query instead of N queries.

        Args:
            folder: Path to folder
            mode: 'set' or 'track'
            recursive: Scan subfolders

        Returns:
            Dict mapping absolute file path to cache status (True = cached)
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            return {}

        pattern = '**/*' if recursive else '*'
        files = [
            f for f in folder_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.AUDIO_EXTENSIONS
        ]

        # Get all cached paths in one query (O(1) instead of O(n))
        if mode == 'set':
            cached_set = set(self.get_all_cached_sets())
        else:
            cached_set = set(self.get_all_cached_tracks())

        # Fast set lookup: O(1) per file
        result = {}
        for f in files:
            abs_path = str(f.absolute())
            result[abs_path] = abs_path in cached_set

        return result

    def list_cached_sets(self) -> List[str]:
        """Get paths of all cached sets."""
        return self.get_all_cached_sets()

    def list_profiles(self) -> List[str]:
        """Get names of all cached DJ profiles."""
        return self.get_all_dj_profiles()

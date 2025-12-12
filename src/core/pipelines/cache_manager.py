"""
Cache Manager - Manage cached analysis results.

Provides caching for:
- STFT computations (memory-mapped)
- Extracted features (pickle)
- ML predictions (sqlite)
"""

import os
import pickle
import hashlib
import sqlite3
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import json

from ..primitives import STFTCache


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    file_hash: str
    created_at: float
    size_bytes: int
    data_type: str


class CacheManager:
    """
    Manage cached results for audio analysis.

    Caches:
    - STFT (memory-mapped numpy files)
    - Features (pickle files)
    - Predictions (SQLite database)

    Usage:
        cache = CacheManager("cache/")

        # Check cache
        features = cache.get_features(file_hash)
        if features is None:
            features = extract_features(audio)
            cache.save_features(file_hash, features)

        # Invalidate
        cache.invalidate(file_hash)

        # Clear all
        cache.clear_all()
    """

    def __init__(
        self,
        cache_dir: str = "~/.mood-classifier/cache",
        max_size_gb: float = 10.0,
        enable_stft_cache: bool = True,
        enable_feature_cache: bool = True,
        enable_prediction_cache: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Root directory for cache files
            max_size_gb: Maximum cache size in GB
            enable_stft_cache: Cache STFT computations
            enable_feature_cache: Cache feature extractions
            enable_prediction_cache: Cache ML predictions
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        self.enable_stft = enable_stft_cache
        self.enable_features = enable_feature_cache
        self.enable_predictions = enable_prediction_cache

        # Create subdirectories
        self.stft_dir = self.cache_dir / "stft"
        self.features_dir = self.cache_dir / "features"
        self.predictions_db = self.cache_dir / "predictions.db"

        self._ensure_dirs()
        self._init_predictions_db()

    def _ensure_dirs(self):
        """Create cache directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stft_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)

    def _init_predictions_db(self):
        """Initialize SQLite database for predictions and set analysis."""
        if not self.enable_predictions:
            return

        conn = sqlite3.connect(str(self.predictions_db))
        cursor = conn.cursor()

        # Original predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                file_hash TEXT PRIMARY KEY,
                zone TEXT,
                confidence REAL,
                zone_scores TEXT,
                created_at REAL
            )
        ''')

        # Set analysis results table (DJ sets - transitions, drops, segments)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS set_analysis_results (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_mtime REAL,
                result_json TEXT NOT NULL,
                created_at REAL
            )
        ''')

        # Track analysis table (individual tracks - bpm, key, energy)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS track_analysis (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_mtime REAL,
                result_json TEXT NOT NULL,
                created_at REAL
            )
        ''')

        # Path index for fast lookup by file path (for sets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS path_index (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                file_mtime REAL
            )
        ''')

        # Track path index for fast lookup by file path (for tracks)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS track_path_index (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                file_mtime REAL
            )
        ''')

        # NEW: DJ profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dj_profiles (
                dj_name TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                set_hashes TEXT NOT NULL,
                n_sets INTEGER,
                total_hours REAL,
                created_at REAL,
                updated_at REAL
            )
        ''')

        # NEW: Set plans table (generated tracklists)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS set_plans (
                plan_id TEXT PRIMARY KEY,
                dj_name TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                n_tracks INTEGER,
                duration_min INTEGER,
                avg_score REAL,
                status TEXT,
                created_at REAL,
                updated_at REAL
            )
        ''')

        # Create indexes
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_set_plans_dj ON set_plans(dj_name)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_set_results_path ON set_analysis_results(file_path)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_path_index_hash ON path_index(file_hash)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_track_analysis_path ON track_analysis(file_path)'
        )
        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_track_path_index_hash ON track_path_index(file_hash)'
        )

        conn.commit()
        conn.close()

    # ============== File Hashing ==============

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of audio file.

        Uses first 1MB of file for speed.
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read first 1MB
            data = f.read(1024 * 1024)
            hasher.update(data)
            # Add file size
            f.seek(0, 2)
            hasher.update(str(f.tell()).encode())
        return hasher.hexdigest()

    # ============== STFT Cache ==============

    def get_stft(self, file_hash: str) -> Optional[STFTCache]:
        """Get cached STFT."""
        if not self.enable_stft:
            return None

        meta_path = self.stft_dir / f"{file_hash}_meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Load memory-mapped arrays
            S = np.load(self.stft_dir / f"{file_hash}_S.npy", mmap_mode='r')
            S_db = np.load(self.stft_dir / f"{file_hash}_S_db.npy", mmap_mode='r')
            phase = np.load(self.stft_dir / f"{file_hash}_phase.npy", mmap_mode='r')
            freqs = np.load(self.stft_dir / f"{file_hash}_freqs.npy")
            times = np.load(self.stft_dir / f"{file_hash}_times.npy")

            return STFTCache(
                S=S,
                S_db=S_db,
                phase=phase,
                freqs=freqs,
                times=times,
                sr=meta['sr'],
                hop_length=meta['hop_length'],
                n_fft=meta['n_fft']
            )
        except Exception:
            return None

    def save_stft(self, file_hash: str, cache: STFTCache):
        """Save STFT to cache."""
        if not self.enable_stft:
            return

        try:
            # Save arrays
            np.save(self.stft_dir / f"{file_hash}_S.npy", cache.S)
            np.save(self.stft_dir / f"{file_hash}_S_db.npy", cache.S_db)
            np.save(self.stft_dir / f"{file_hash}_phase.npy", cache.phase)
            np.save(self.stft_dir / f"{file_hash}_freqs.npy", cache.freqs)
            np.save(self.stft_dir / f"{file_hash}_times.npy", cache.times)

            # Save metadata
            meta = {
                'sr': cache.sr,
                'hop_length': cache.hop_length,
                'n_fft': cache.n_fft
            }
            with open(self.stft_dir / f"{file_hash}_meta.json", 'w') as f:
                json.dump(meta, f)
        except Exception:
            pass  # Silently fail on cache errors

    # ============== Features Cache ==============

    def get_features(self, file_hash: str) -> Optional[np.ndarray]:
        """Get cached features."""
        if not self.enable_features:
            return None

        path = self.features_dir / f"{file_hash}.pkl"
        if not path.exists():
            return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def save_features(self, file_hash: str, features: np.ndarray):
        """Save features to cache."""
        if not self.enable_features:
            return

        try:
            path = self.features_dir / f"{file_hash}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass

    def get_features_dict(self, file_hash: str) -> Optional[Dict[str, float]]:
        """Get cached features as dictionary."""
        path = self.features_dir / f"{file_hash}_dict.pkl"
        if not path.exists():
            return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def save_features_dict(self, file_hash: str, features: Dict[str, float]):
        """Save features dictionary to cache."""
        if not self.enable_features:
            return

        try:
            path = self.features_dir / f"{file_hash}_dict.pkl"
            with open(path, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass

    # ============== Derived Features Cache (for calibration) ==============

    def get_derived_feature(self, file_hash: str, feature_name: str) -> Optional[np.ndarray]:
        """
        Get cached derived feature by name.

        Used for calibration to avoid re-computing features that don't depend on parameters.

        Args:
            file_hash: File hash from compute_file_hash()
            feature_name: Feature name (e.g., 'rms', 'novelty', 'bass_energy')

        Returns:
            numpy array or None if not cached
        """
        if not self.enable_stft:
            return None

        path = self.stft_dir / f"{file_hash}_{feature_name}.npy"
        if not path.exists():
            return None

        try:
            # Load without mmap to avoid file descriptor leaks
            # Copy is needed anyway for safe usage across boundaries
            return np.load(path, mmap_mode=None)
        except Exception:
            return None

    def save_derived_feature(self, file_hash: str, feature_name: str, data: np.ndarray):
        """
        Save derived feature to cache.

        Args:
            file_hash: File hash from compute_file_hash()
            feature_name: Feature name (e.g., 'rms', 'novelty', 'bass_energy')
            data: numpy array to cache
        """
        if not self.enable_stft:
            return

        try:
            path = self.stft_dir / f"{file_hash}_{feature_name}.npy"
            np.save(path, np.ascontiguousarray(data))
        except Exception:
            pass

    def get_derived_features_batch(
        self,
        file_hash: str,
        feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Load multiple derived features at once.

        Args:
            file_hash: File hash
            feature_names: List of feature names to load

        Returns:
            Dict of feature_name -> numpy array (only includes found features)
        """
        result = {}
        for name in feature_names:
            data = self.get_derived_feature(file_hash, name)
            if data is not None:
                result[name] = np.array(data)  # Copy from mmap
        return result

    def save_derived_features_batch(
        self,
        file_hash: str,
        features: Dict[str, np.ndarray]
    ):
        """
        Save multiple derived features at once.

        Args:
            file_hash: File hash
            features: Dict of feature_name -> numpy array
        """
        for name, data in features.items():
            self.save_derived_feature(file_hash, name, data)

    def has_all_derived_features(self, file_hash: str, feature_names: List[str]) -> bool:
        """Check if all specified derived features are cached."""
        for name in feature_names:
            path = self.stft_dir / f"{file_hash}_{name}.npy"
            if not path.exists():
                return False
        return True

    def get_derived_feature_metadata(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for cached derived features.

        Args:
            file_hash: File hash

        Returns:
            Metadata dict or None if not cached
        """
        metadata_path = self.stft_dir / f"{file_hash}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_derived_feature_metadata(self, file_hash: str, metadata: Dict[str, Any]):
        """
        Save metadata for derived features.

        Args:
            file_hash: File hash
            metadata: Metadata dict (sr, hop_length, duration_sec, etc.)
        """
        metadata_path = self.stft_dir / f"{file_hash}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    # ============== Predictions Cache ==============

    def get_prediction(self, file_hash: str) -> Optional[Tuple[str, float, Dict]]:
        """
        Get cached prediction.

        Returns:
            Tuple of (zone, confidence, zone_scores) or None
        """
        if not self.enable_predictions:
            return None

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute(
                'SELECT zone, confidence, zone_scores FROM predictions WHERE file_hash = ?',
                (file_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return row[0], row[1], json.loads(row[2])
            return None
        except Exception:
            return None

    def save_prediction(
        self,
        file_hash: str,
        zone: str,
        confidence: float,
        zone_scores: Dict[str, float]
    ):
        """Save prediction to cache."""
        if not self.enable_predictions:
            return

        try:
            import time
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO predictions
                (file_hash, zone, confidence, zone_scores, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_hash, zone, confidence, json.dumps(zone_scores), time.time()))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ============== Cache Management ==============

    def invalidate(self, file_hash: str):
        """Invalidate all cache entries for a file."""
        # STFT files
        for suffix in ['_S.npy', '_S_db.npy', '_phase.npy', '_freqs.npy', '_times.npy', '_meta.json']:
            path = self.stft_dir / f"{file_hash}{suffix}"
            if path.exists():
                path.unlink()

        # Features
        for suffix in ['.pkl', '_dict.pkl']:
            path = self.features_dir / f"{file_hash}{suffix}"
            if path.exists():
                path.unlink()

        # Predictions
        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE file_hash = ?', (file_hash,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def clear_all(self):
        """Clear all cached data."""
        import shutil

        if self.stft_dir.exists():
            shutil.rmtree(self.stft_dir)
        if self.features_dir.exists():
            shutil.rmtree(self.features_dir)
        if self.predictions_db.exists():
            self.predictions_db.unlink()

        self._ensure_dirs()
        self._init_predictions_db()

    def get_cache_size(self) -> int:
        """Get total cache size in bytes.

        Uses os.scandir for O(1) per-file stat calls instead of rglob.
        """
        import os

        def _dir_size(path: str) -> int:
            total = 0
            try:
                with os.scandir(path) as entries:
                    for entry in entries:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += _dir_size(entry.path)
            except (PermissionError, OSError):
                pass
            return total

        return _dir_size(str(self.cache_dir))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stft_count = len(list(self.stft_dir.glob('*_meta.json')))
        feature_count = len(list(self.features_dir.glob('*.pkl')))

        prediction_count = 0
        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM predictions')
            prediction_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass

        return {
            'cache_dir': str(self.cache_dir),
            'total_size_mb': self.get_cache_size() / (1024 * 1024),
            'stft_count': stft_count,
            'feature_count': feature_count,
            'prediction_count': prediction_count,
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }

    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove cache entries older than max_age_days."""
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)

        # Clean files by modification time
        for path in self.cache_dir.rglob('*'):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()

        # Clean predictions and set analysis
        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE created_at < ?', (cutoff,))
            cursor.execute('DELETE FROM set_analysis_results WHERE created_at < ?', (cutoff,))
            # Clean orphaned path_index entries
            cursor.execute('''
                DELETE FROM path_index WHERE file_hash NOT IN
                (SELECT file_hash FROM set_analysis_results)
            ''')
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ============== Set Analysis Cache (NEW) ==============

    def get_set_analysis(self, file_path: str) -> Optional[Dict]:
        """
        Get cached SetAnalysisResult by file path.

        Checks if file has changed since caching (via mtime + hash).

        Args:
            file_path: Absolute path to audio file

        Returns:
            result dict (SetAnalysisResult.to_dict()) or None if not cached/changed
        """
        if not self.enable_predictions:
            return None

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # 1. Lookup path → hash
            cursor.execute(
                'SELECT file_hash, file_mtime FROM path_index WHERE file_path = ?',
                (file_path,)
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            cached_hash, cached_mtime = row

            # 2. Check if file changed (fast mtime check first)
            try:
                current_mtime = os.path.getmtime(file_path)
                if abs(current_mtime - cached_mtime) > 0.01:
                    # mtime changed, verify hash
                    current_hash = self.compute_file_hash(file_path)
                    if current_hash != cached_hash:
                        conn.close()
                        return None  # File changed, need re-analysis
            except OSError:
                conn.close()
                return None  # File doesn't exist or not accessible

            # 3. Get result from set_analysis_results
            cursor.execute(
                'SELECT result_json FROM set_analysis_results WHERE file_hash = ?',
                (cached_hash,)
            )
            result_row = cursor.fetchone()
            conn.close()

            if result_row:
                return json.loads(result_row[0])
            return None

        except Exception:
            return None

    def save_set_analysis(self, file_path: str, result: Dict):
        """
        Save SetAnalysisResult to cache.

        Stores both the result and path→hash mapping for fast lookup.

        Args:
            file_path: Absolute path to audio file
            result: SetAnalysisResult.to_dict()
        """
        if not self.enable_predictions:
            return

        try:
            file_hash = self.compute_file_hash(file_path)
            file_mtime = os.path.getmtime(file_path)

            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # Atomic transaction
            cursor.execute('BEGIN')
            try:
                # Update path_index
                cursor.execute('''
                    INSERT OR REPLACE INTO path_index (file_path, file_hash, file_mtime)
                    VALUES (?, ?, ?)
                ''', (file_path, file_hash, file_mtime))

                # Update set_analysis_results
                cursor.execute('''
                    INSERT OR REPLACE INTO set_analysis_results
                    (file_hash, file_path, file_mtime, result_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (file_hash, file_path, file_mtime, json.dumps(result), time.time()))

                cursor.execute('COMMIT')
            except Exception:
                cursor.execute('ROLLBACK')
                raise
            finally:
                conn.close()

        except Exception:
            pass  # Silently fail on cache errors

    def get_cached_set_paths(self) -> List[str]:
        """Get all cached set analysis file paths."""
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM path_index')
            paths = [row[0] for row in cursor.fetchall()]
            conn.close()
            return paths
        except Exception:
            return []

    def get_cached_track_paths(self) -> List[str]:
        """Get all cached track analysis file paths."""
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            # Join track_analysis with path_index to get file paths
            cursor.execute('''
                SELECT pi.file_path
                FROM track_analysis ta
                JOIN path_index pi ON ta.file_hash = pi.file_hash
            ''')
            paths = [row[0] for row in cursor.fetchall()]
            conn.close()
            return paths
        except Exception:
            return []

    def invalidate_set_analysis(self, file_path: str):
        """
        Remove set analysis cache for a specific file.

        Args:
            file_path: Absolute path to audio file
        """
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            cursor.execute(
                'SELECT file_hash FROM path_index WHERE file_path = ?',
                (file_path,)
            )
            row = cursor.fetchone()

            if row:
                file_hash = row[0]
                cursor.execute(
                    'DELETE FROM path_index WHERE file_path = ?',
                    (file_path,)
                )
                cursor.execute(
                    'DELETE FROM set_analysis_results WHERE file_hash = ?',
                    (file_hash,)
                )
                conn.commit()

            conn.close()
        except Exception:
            pass

    def clear_set_analysis_cache(self):
        """Clear all set analysis cache entries."""
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM set_analysis_results')
            cursor.execute('DELETE FROM path_index')
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_set_analysis_stats(self) -> Dict[str, Any]:
        """Get set analysis cache statistics."""
        if not self.enable_predictions:
            return {'count': 0}

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM set_analysis_results')
            count = cursor.fetchone()[0]
            conn.close()
            return {'count': count}
        except Exception:
            return {'count': 0}

    # ============== Track Analysis Cache ==============
    # Separate from set_analysis for individual track analysis (bpm, key, energy)

    def get_track_analysis(self, file_path: str) -> Optional[Dict]:
        """
        Get cached track analysis (bpm, key, energy, etc.).

        Validates that file hasn't changed via mtime + hash.

        Args:
            file_path: Absolute path to audio file

        Returns:
            TrackAnalysis.to_dict() or None if not cached/outdated
        """
        if not self.enable_predictions:
            return None

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # 1. Look up file_hash via path
            cursor.execute(
                'SELECT file_hash, file_mtime FROM track_path_index WHERE file_path = ?',
                (file_path,)
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            cached_hash, cached_mtime = row

            # 2. Check if file changed (fast mtime check first)
            try:
                current_mtime = os.path.getmtime(file_path)
                if abs(current_mtime - cached_mtime) > 0.01:
                    # mtime changed, verify hash
                    current_hash = self.compute_file_hash(file_path)
                    if current_hash != cached_hash:
                        conn.close()
                        return None  # File changed, need re-analysis
            except OSError:
                conn.close()
                return None  # File doesn't exist or not accessible

            # 3. Get result from track_analysis
            cursor.execute(
                'SELECT result_json FROM track_analysis WHERE file_hash = ?',
                (cached_hash,)
            )
            result_row = cursor.fetchone()
            conn.close()

            if result_row:
                return json.loads(result_row[0])
            return None

        except Exception:
            return None

    def save_track_analysis(self, file_path: str, result: Dict):
        """
        Save TrackAnalysis to cache.

        Stores both the result and path→hash mapping for fast lookup.

        Args:
            file_path: Absolute path to audio file
            result: TrackAnalysis.to_dict()
        """
        if not self.enable_predictions:
            return

        try:
            file_hash = self.compute_file_hash(file_path)
            file_mtime = os.path.getmtime(file_path)

            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # Atomic transaction
            cursor.execute('BEGIN')
            try:
                # Update track_path_index
                cursor.execute('''
                    INSERT OR REPLACE INTO track_path_index (file_path, file_hash, file_mtime)
                    VALUES (?, ?, ?)
                ''', (file_path, file_hash, file_mtime))

                # Update track_analysis
                cursor.execute('''
                    INSERT OR REPLACE INTO track_analysis
                    (file_hash, file_path, file_mtime, result_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (file_hash, file_path, file_mtime, json.dumps(result), time.time()))

                cursor.execute('COMMIT')
            except Exception:
                cursor.execute('ROLLBACK')
                raise
            finally:
                conn.close()

        except Exception:
            pass  # Silently fail on cache errors

    def invalidate_track_analysis(self, file_path: str):
        """
        Remove track analysis cache for a specific file.

        Args:
            file_path: Absolute path to audio file
        """
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            cursor.execute(
                'SELECT file_hash FROM track_path_index WHERE file_path = ?',
                (file_path,)
            )
            row = cursor.fetchone()

            if row:
                file_hash = row[0]
                cursor.execute(
                    'DELETE FROM track_path_index WHERE file_path = ?',
                    (file_path,)
                )
                cursor.execute(
                    'DELETE FROM track_analysis WHERE file_hash = ?',
                    (file_hash,)
                )
                conn.commit()

            conn.close()
        except Exception:
            pass

    def clear_track_analysis_cache(self):
        """Clear all track analysis cache entries."""
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM track_analysis')
            cursor.execute('DELETE FROM track_path_index')
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_track_analysis_stats(self) -> Dict[str, Any]:
        """Get track analysis cache statistics."""
        if not self.enable_predictions:
            return {'count': 0}

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM track_analysis')
            count = cursor.fetchone()[0]
            conn.close()
            return {'count': count}
        except Exception:
            return {'count': 0}

    # ============== DJ Profile Cache ==============

    def get_dj_profile(
        self,
        dj_name: str,
        set_paths: Optional[List[str]] = None,
        skip_validation: bool = False
    ) -> Optional[Dict]:
        """
        Get cached DJ profile if it exists and is up-to-date.

        Checks if all underlying sets are still the same (via hashes).

        Args:
            dj_name: DJ name
            set_paths: List of set paths used for this profile (optional)
            skip_validation: If True, skip hash validation and return cached profile

        Returns:
            Profile dict or None if not cached/outdated
        """
        if not self.enable_predictions:
            return None

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # 1. Get cached profile
            cursor.execute(
                'SELECT profile_json, set_hashes FROM dj_profiles WHERE dj_name = ?',
                (dj_name,)
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            profile_json, cached_hashes_json = row

            # Skip validation if requested or no paths provided
            if skip_validation or not set_paths:
                conn.close()
                return json.loads(profile_json)

            cached_hashes = set(json.loads(cached_hashes_json))

            # 2. Compute current hashes for provided paths
            current_hashes = set()
            for path in set_paths:
                abs_path = os.path.abspath(path)
                try:
                    file_hash = self.compute_file_hash(abs_path)
                    current_hashes.add(file_hash)
                except Exception:
                    conn.close()
                    return None  # Can't compute hash, invalidate

            # 3. Check if sets changed
            if cached_hashes != current_hashes:
                conn.close()
                return None  # Sets changed, need rebuild

            conn.close()
            return json.loads(profile_json)

        except Exception:
            return None

    def save_dj_profile(
        self,
        dj_name: str,
        profile: Dict,
        set_paths: List[str]
    ):
        """
        Save DJ profile to cache.

        Args:
            dj_name: DJ name
            profile: DJStyleProfile.to_dict() or asdict(profile)
            set_paths: List of set paths used to build this profile
        """
        if not self.enable_predictions:
            return

        try:
            # Compute hashes for all sets
            set_hashes = []
            for path in set_paths:
                abs_path = os.path.abspath(path)
                file_hash = self.compute_file_hash(abs_path)
                set_hashes.append(file_hash)

            n_sets = profile.get('n_sets_analyzed', len(set_paths))
            total_hours = profile.get('total_duration_hours', 0.0)

            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO dj_profiles
                (dj_name, profile_json, set_hashes, n_sets, total_hours, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                dj_name,
                json.dumps(profile),
                json.dumps(set_hashes),
                n_sets,
                total_hours,
                time.time(),
                time.time()
            ))

            conn.commit()
            conn.close()

        except Exception:
            pass

    def get_all_dj_profiles(self) -> List[Dict[str, Any]]:
        """Get list of all cached DJ profiles (metadata only)."""
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT dj_name, n_sets, total_hours, updated_at
                FROM dj_profiles
                ORDER BY updated_at DESC
            ''')
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'dj_name': row[0],
                    'n_sets': row[1],
                    'total_hours': row[2],
                    'updated_at': row[3]
                }
                for row in rows
            ]
        except Exception:
            return []

    def delete_dj_profile(self, dj_name: str):
        """
        Delete a DJ profile from cache.

        Args:
            dj_name: DJ name (lowercase)
        """
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM dj_profiles WHERE dj_name = ?', (dj_name.lower(),))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def clear_predictions(self):
        """Clear all zone predictions from cache (keeps sets and profiles)."""
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions')
            conn.commit()
            conn.close()
        except Exception:
            pass

    def add_set_to_dj_profile(
        self,
        dj_name: str,
        set_path: str,
        set_profile: Dict
    ):
        """
        Add a single set to DJ profile (incremental update).

        If DJ profile doesn't exist, creates new one.
        If set already exists in profile, updates it.

        Args:
            dj_name: DJ name
            set_path: Absolute path to audio file
            set_profile: Profile data for this set (from SetAnalysisResult)
        """
        if not self.enable_predictions:
            return

        try:
            abs_path = os.path.abspath(set_path)
            file_hash = self.compute_file_hash(abs_path)

            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            # Get existing profile
            cursor.execute(
                'SELECT profile_json, set_hashes, n_sets, total_hours, created_at FROM dj_profiles WHERE dj_name = ?',
                (dj_name,)
            )
            row = cursor.fetchone()

            if row:
                # Update existing profile
                existing_profile = json.loads(row[0])
                existing_hashes = json.loads(row[1])
                created_at = row[4]

                # Check if this set is already in profile
                if file_hash in existing_hashes:
                    # Update existing set - remove old, add new
                    pass  # Hash already present, just update profile
                else:
                    # Add new set
                    existing_hashes.append(file_hash)

                # Update aggregated profile (use latest set's profile as base)
                # In future: proper aggregation across all sets
                existing_profile.update({
                    'n_sets_analyzed': len(existing_hashes),
                })

                # Recalculate total hours
                set_duration = set_profile.get('duration_sec', 0)
                if file_hash not in json.loads(row[1]):
                    # New set - add duration
                    total_hours = row[3] + (set_duration / 3600)
                else:
                    total_hours = row[3]

                existing_profile['total_duration_hours'] = total_hours

                cursor.execute('''
                    UPDATE dj_profiles
                    SET profile_json = ?, set_hashes = ?, n_sets = ?, total_hours = ?, updated_at = ?
                    WHERE dj_name = ?
                ''', (
                    json.dumps(existing_profile),
                    json.dumps(existing_hashes),
                    len(existing_hashes),
                    total_hours,
                    time.time(),
                    dj_name
                ))
            else:
                # Create new profile
                set_duration = set_profile.get('duration_sec', 0)
                total_hours = set_duration / 3600

                new_profile = set_profile.copy()
                new_profile['n_sets_analyzed'] = 1
                new_profile['total_duration_hours'] = total_hours

                cursor.execute('''
                    INSERT INTO dj_profiles
                    (dj_name, profile_json, set_hashes, n_sets, total_hours, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dj_name,
                    json.dumps(new_profile),
                    json.dumps([file_hash]),
                    1,
                    total_hours,
                    time.time(),
                    time.time()
                ))

            conn.commit()
            conn.close()

        except Exception:
            pass

    def invalidate_dj_profile(self, dj_name: str):
        """Remove cached DJ profile."""
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM dj_profiles WHERE dj_name = ?', (dj_name,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ============== Set Plans (Generated Tracklists) ==============

    def get_set_plan(self, plan_id: str) -> Optional[Dict]:
        """
        Get cached set plan by ID.

        Args:
            plan_id: Unique plan identifier

        Returns:
            Plan dict or None if not found
        """
        if not self.enable_predictions:
            return None

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute(
                'SELECT plan_json FROM set_plans WHERE plan_id = ?',
                (plan_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None
        except Exception:
            return None

    def get_set_plans_by_dj(self, dj_name: str, limit: int = 10) -> List[Dict]:
        """
        Get all set plans for a DJ.

        Args:
            dj_name: DJ name
            limit: Maximum plans to return

        Returns:
            List of plan dicts (without full track analysis)
        """
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT plan_id, dj_name, n_tracks, duration_min, avg_score, status, created_at, updated_at
                FROM set_plans
                WHERE dj_name = ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (dj_name, limit))
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'plan_id': row[0],
                    'dj_name': row[1],
                    'n_tracks': row[2],
                    'duration_min': row[3],
                    'avg_score': row[4],
                    'status': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                }
                for row in rows
            ]
        except Exception:
            return []

    def get_all_set_plans(self, limit: int = 50) -> List[Dict]:
        """
        Get all cached set plans (metadata only).

        Returns:
            List of plan metadata dicts
        """
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT plan_id, dj_name, n_tracks, duration_min, avg_score, status, created_at, updated_at
                FROM set_plans
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'plan_id': row[0],
                    'dj_name': row[1],
                    'n_tracks': row[2],
                    'duration_min': row[3],
                    'avg_score': row[4],
                    'status': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                }
                for row in rows
            ]
        except Exception:
            return []

    def save_set_plan(self, plan_id: str, plan_dict: Dict):
        """
        Save set plan to cache.

        Args:
            plan_id: Unique plan identifier
            plan_dict: Full plan dict (from SetPlan.to_dict())
        """
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            now = time.time()
            dj_name = plan_dict.get('dj_name', 'Unknown')
            n_tracks = plan_dict.get('n_tracks', 0)
            duration_min = plan_dict.get('duration_min', 0)
            avg_score = plan_dict.get('avg_transition_score', 0.0)
            status = plan_dict.get('status', 'draft')

            cursor.execute('''
                INSERT OR REPLACE INTO set_plans
                (plan_id, dj_name, plan_json, n_tracks, duration_min, avg_score, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM set_plans WHERE plan_id = ?), ?), ?)
            ''', (
                plan_id,
                dj_name,
                json.dumps(plan_dict),
                n_tracks,
                duration_min,
                avg_score,
                status,
                plan_id,
                now,
                now,
            ))

            conn.commit()
            conn.close()
        except Exception:
            pass

    def delete_set_plan(self, plan_id: str):
        """Delete a set plan from cache."""
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM set_plans WHERE plan_id = ?', (plan_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def clear_set_plans(self, dj_name: Optional[str] = None):
        """
        Clear set plans from cache.

        Args:
            dj_name: If provided, only clear plans for this DJ
        """
        if not self.enable_predictions:
            return

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            if dj_name:
                cursor.execute('DELETE FROM set_plans WHERE dj_name = ?', (dj_name,))
            else:
                cursor.execute('DELETE FROM set_plans')

            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_dj_profile_names(self) -> List[str]:
        """Get list of all cached DJ profile names."""
        if not self.enable_predictions:
            return []

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()
            cursor.execute('SELECT dj_name FROM dj_profiles ORDER BY dj_name')
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except Exception:
            return []

    def get_set_plans_stats(self) -> Dict[str, Any]:
        """Get statistics about cached set plans."""
        if not self.enable_predictions:
            return {'total_plans': 0}

        try:
            conn = sqlite3.connect(str(self.predictions_db))
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM set_plans')
            total = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT dj_name) FROM set_plans')
            unique_djs = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(avg_score) FROM set_plans WHERE status = "verified"')
            avg_score = cursor.fetchone()[0] or 0.0

            cursor.execute('SELECT status, COUNT(*) FROM set_plans GROUP BY status')
            status_counts = dict(cursor.fetchall())

            conn.close()

            return {
                'total_plans': total,
                'unique_djs': unique_djs,
                'avg_score': avg_score,
                'by_status': status_counts,
            }
        except Exception:
            return {'total_plans': 0}

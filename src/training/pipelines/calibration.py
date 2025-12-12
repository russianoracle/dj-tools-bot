"""
Calibration Pipeline - Optimize TransitionDetection parameters.

Uses SetAnalysisPipeline for loading + CacheRepository for feature caching.
After first run, all features are cached and optimization is ~100x faster.

Usage:
    from src.training.pipelines import CalibrationPipeline
    from src.core.config import MixingStyle

    # Calibrate SMOOTH style
    pipeline = CalibrationPipeline(
        mixing_style=MixingStyle.SMOOTH,
        verbose=True
    )

    result = pipeline.calibrate(
        audio_paths=['mix1.m4a', 'mix2.m4a'],
        ground_truth={'mix1': [60, 180, 300], 'mix2': [90, 210]},
        max_iter=100
    )

    print(f"Best F1: {result.best_f1}")
    print(f"Optimal params: {result.optimal_params}")
"""

import json
import sys
import time
import signal
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from scipy.optimize import differential_evolution

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.core.pipelines.base import Pipeline, PipelineStage, PipelineContext, LoadAudioStage, ComputeSTFTStage
from src.core.cache import CacheRepository  # Unified cache access
from src.core.tasks import TransitionDetectionTask
from src.core.config import MixingStyle, TransitionConfig

logger = logging.getLogger(__name__)

# Feature names that are cached (don't depend on task parameters)
CACHED_FEATURE_NAMES = [
    'rms', 'bass_energy', 'novelty', 'timbral_novelty',
    'chroma_novelty', 'centroid', 'rolloff', 'centroid_velocity',
    'drops', 'seg_boundaries'  # Context filters: cache once, reuse for all calibrations
]


@dataclass
class CachedSetFeatures:
    """Cached features for one set."""
    set_id: str
    file_path: str
    features: Dict[str, np.ndarray]
    ground_truth: np.ndarray
    sr: int
    hop_length: int
    duration_sec: float

    @property
    def n_ground_truth(self) -> int:
        return len(self.ground_truth)


@dataclass
class CalibrationResult:
    """Result of calibration optimization."""
    success: bool
    best_f1: float
    precision: float
    recall: float
    optimal_params: Dict[str, float]
    iterations: int
    optimization_time_sec: float
    n_sets: int
    total_ground_truth: int
    mixing_style: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'best_f1': self.best_f1,
            'precision': self.precision,
            'recall': self.recall,
            'optimal_params': self.optimal_params,
            'iterations': self.iterations,
            'optimization_time_sec': self.optimization_time_sec,
            'n_sets': self.n_sets,
            'total_ground_truth': self.total_ground_truth,
            'mixing_style': self.mixing_style,
            'error': self.error,
        }

    def to_config_snippet(self) -> str:
        """Generate config.py snippet for these parameters."""
        p = self.optimal_params
        return f"""MixingStyle.{self.mixing_style.upper()}: cls(
    min_transition_gap_sec={p['min_transition_gap_sec']:.1f},
    energy_threshold={p['energy_threshold']:.3f},
    bass_weight={p['bass_weight']:.2f},
    smooth_sigma={p['smooth_sigma']:.1f},
    filter_velocity_threshold=200.0,
    peak_percentile={p['peak_percentile']:.1f},
    transition_merge_window_sec=30.0,
    timbral_weight={p['timbral_weight']:.2f},
),"""


class LoadAndCacheFeaturesStage(PipelineStage):
    """
    Stage that loads audio and caches transition detection features.

    Uses CacheRepository for persistent caching:
    - First run: loads audio, computes STFT + features, saves to cache
    - Subsequent runs: loads features from cache (~0.05s vs ~15s)

    Architecture:
    - On cache miss: uses LoadAudioStage + ComputeSTFTStage for consistency
    - On cache hit: loads features directly from cache

    Progress output (verbose=True):
        [1/5] mix2073: cache HIT (8 GT) [0.04s]
        [2/5] mix5024: computing...
              → loading audio... [2.3s]
              → computing STFT... [8.1s]
              → extracting features... [3.8s]
              → saving to cache... [0.2s]
              Total: [14.4s] (cached)
    """

    def __init__(self, cache_repo: CacheRepository, sr: int = 22050, verbose: bool = False, num_workers: int = 1):
        self.cache_repo = cache_repo
        self.sr = sr
        self.verbose = verbose
        self.num_workers = num_workers

        # Create reusable stages for cache misses (consistency with base architecture)
        self._load_audio_stage = LoadAudioStage(sr=sr, mono=True)
        self._compute_stft_stage = ComputeSTFTStage()

        # Thread-safe locks for progress tracking
        self._progress_lock = Lock()
        self._stats_lock = Lock()

    def _load_one_set(self, audio_path: str, ground_truth_dict: Dict, pbar=None) -> Optional[CachedSetFeatures]:
        """Load features for one set (thread-safe)."""
        path = Path(audio_path)
        set_id = path.stem

        # DIAGNOSTIC: File info
        file_size_mb = path.stat().st_size / 1e6
        if self.verbose:
            print(f"\n[START] {set_id} ({file_size_mb:.1f} MB)", flush=True)

        # Get ground truth for this set
        gt = ground_truth_dict.get(set_id, [])
        if isinstance(gt, list):
            gt = np.array(gt, dtype=np.float64)

        if len(gt) < 2:
            if self.verbose:
                print(f"[SKIP] {set_id}: insufficient GT ({len(gt)} transitions)", flush=True)
            if pbar:
                pbar.write(f"  {set_id}: skipping (only {len(gt)} GT)")
            return None

        # Compute file hash
        file_hash = self.cache_repo.compute_file_hash(str(path))

        # Try cache first
        if self.cache_repo.has_all_derived_features(file_hash, CACHED_FEATURE_NAMES):
            if self.verbose:
                print(f"[CACHE] {set_id}: checking cache...", flush=True)

            start = time.time()
            features = self.cache_repo.get_derived_features_batch(file_hash, CACHED_FEATURE_NAMES)

            if self.verbose:
                print(f"[CACHE] {set_id}: loaded from cache [{time.time() - start:.2f}s]", flush=True)

            # Load metadata
            metadata_path = self.cache_repo.stft_dir / f"{file_hash}_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            elapsed = time.time() - start
            if pbar:
                pbar.set_postfix_str(f"{set_id}: ✅ cache HIT ({len(gt)} GT) [{elapsed:.2f}s]")
                pbar.refresh()

            return CachedSetFeatures(
                set_id=set_id,
                file_path=str(path),
                features=features,
                ground_truth=gt,
                sr=metadata['sr'],
                hop_length=metadata['hop_length'],
                duration_sec=metadata['duration_sec'],
            )

        # Cache miss - compute features
        if self.verbose:
            print(f"[COMPUTE] {set_id}: cache MISS, computing...", flush=True)

        start = time.time()
        if pbar:
            pbar.set_postfix_str(f"{set_id}: loading audio...")
            pbar.refresh()

        # Create temporary context for this file
        temp_context = PipelineContext(input_path=str(path))

        # Stage 1: Load audio
        if self.verbose:
            print(f"  → Loading audio...", flush=True)

        stage_start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*PySoundFile.*')
            warnings.filterwarnings('ignore', category=FutureWarning, message='.*__audioread_load.*')
            temp_context = self._load_audio_stage.process(temp_context)

        elapsed = time.time() - stage_start
        if self.verbose:
            print(f"  → Audio loaded [{elapsed:.1f}s]", flush=True)

        if pbar:
            pbar.set_postfix_str(f"{set_id}: audio loaded [{elapsed:.1f}s], computing STFT...")
            pbar.refresh()

        # Stage 2: Compute STFT
        if self.verbose:
            print(f"  → Computing STFT...", flush=True)

        stage_start = time.time()
        temp_context = self._compute_stft_stage.process(temp_context)

        elapsed = time.time() - stage_start
        if self.verbose:
            print(f"  → STFT done [{elapsed:.1f}s]", flush=True)

        if pbar:
            pbar.set_postfix_str(f"{set_id}: STFT done [{elapsed:.1f}s], extracting features...")
            pbar.refresh()

        # Get AudioContext from pipeline
        ctx = temp_context.audio_context
        if ctx is None:
            raise ValueError(f"Failed to create AudioContext for {path}")

        # Stage 3: Compute raw features
        if self.verbose:
            print(f"  → Extracting features (drops, segmentation, etc.)...", flush=True)

        stage_start = time.time()
        features = TransitionDetectionTask.compute_raw_features(ctx)

        elapsed = time.time() - stage_start
        if self.verbose:
            print(f"  → Features extracted [{elapsed:.1f}s]", flush=True)

        if pbar:
            pbar.set_postfix_str(f"{set_id}: features extracted [{elapsed:.1f}s], saving...")
            pbar.refresh()

        # Stage 4: Save to cache
        # Separate arrays and list features (drops, seg_boundaries)
        feature_arrays = {}
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                feature_arrays[k] = v
            elif isinstance(v, list) and k in ['drops', 'seg_boundaries']:
                # Cache list features as-is (CacheManager handles serialization)
                feature_arrays[k] = v
        self.cache_repo.save_derived_features_batch(file_hash, feature_arrays)

        # Save metadata
        metadata = {
            'sr': features['sr'],
            'hop_length': features['hop_length'],
            'duration_sec': features['duration_sec'],
        }
        metadata_path = self.cache_repo.stft_dir / f"{file_hash}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        total_elapsed = time.time() - start

        if self.verbose:
            print(f"[DONE] {set_id}: completed in {total_elapsed:.1f}s total", flush=True)

        if pbar:
            pbar.set_postfix_str(f"{set_id}: ✅ DONE [{total_elapsed:.1f}s total]")
            pbar.refresh()

        return CachedSetFeatures(
            set_id=set_id,
            file_path=str(path),
            features=features,
            ground_truth=gt,
            sr=metadata['sr'],
            hop_length=metadata['hop_length'],
            duration_sec=metadata['duration_sec'],
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        """Load features for all sets, using cache when available."""
        audio_paths = context.config.get('audio_paths', [])
        ground_truth = context.config.get('ground_truth', {})

        cached_sets: List[CachedSetFeatures] = []
        load_start = time.time()

        if self.num_workers > 1:
            # Parallel loading with progress bar
            if HAS_TQDM and self.verbose:
                pbar = tqdm(total=len(audio_paths), desc="Loading sets", unit="set")
            else:
                pbar = None

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._load_one_set, audio_path, ground_truth, pbar): audio_path
                    for audio_path in audio_paths
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        cached_sets.append(result)
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.close()

        else:
            # Sequential loading (original behavior)
            for i, audio_path in enumerate(audio_paths):
                path = Path(audio_path)
                set_id = path.stem

                # Get ground truth for this set
                gt = ground_truth.get(set_id, [])
                if isinstance(gt, list):
                    gt = np.array(gt, dtype=np.float64)

                if len(gt) < 2:
                    if self.verbose:
                        print(f"  {set_id}: skipping (only {len(gt)} GT)")
                    continue

                # Compute file hash
                file_hash = self.cache_repo.compute_file_hash(str(path))

                # Try cache first
                if self.cache_repo.has_all_derived_features(file_hash, CACHED_FEATURE_NAMES):
                    start = time.time()
                    features = self.cache_repo.get_derived_features_batch(file_hash, CACHED_FEATURE_NAMES)

                    # Load metadata
                    metadata_path = self.cache_repo.stft_dir / f"{file_hash}_metadata.json"
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    elapsed = time.time() - start
                    if self.verbose:
                        print(f"  [{i+1}/{len(audio_paths)}] {set_id}: cache HIT ({len(gt)} GT) [{elapsed:.2f}s]")

                else:
                    # Cache miss - use standard pipeline stages for consistency
                    start = time.time()
                    if self.verbose:
                        print(f"  [{i+1}/{len(audio_paths)}] {set_id}: computing...")
                        print(f"      → loading audio...", end="", flush=True)

                    # Create temporary context for this file
                    temp_context = PipelineContext(input_path=str(path))

                    # Stage 1: Load audio
                    stage_start = time.time()
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, message='.*PySoundFile.*')
                        warnings.filterwarnings('ignore', category=FutureWarning, message='.*__audioread_load.*')
                        temp_context = self._load_audio_stage.process(temp_context)
                    load_elapsed = time.time() - stage_start
                    if self.verbose:
                        print(f" [{load_elapsed:.1f}s]")
                        print(f"      → computing STFT...", end="", flush=True)

                    # Stage 2: Compute STFT
                    stage_start = time.time()
                    temp_context = self._compute_stft_stage.process(temp_context)
                    stft_elapsed = time.time() - stage_start
                    if self.verbose:
                        print(f" [{stft_elapsed:.1f}s]")
                        print(f"      → extracting features...", end="", flush=True)

                    # Get AudioContext from pipeline
                    ctx = temp_context.audio_context
                    if ctx is None:
                        raise ValueError(f"Failed to create AudioContext for {path}")

                    # Stage 3: Compute raw features
                    stage_start = time.time()
                    features = TransitionDetectionTask.compute_raw_features(ctx)
                    features_elapsed = time.time() - stage_start
                    if self.verbose:
                        print(f" [{features_elapsed:.1f}s]")
                        print(f"      → saving to cache...", end="", flush=True)

                    # Stage 4: Save to cache
                    stage_start = time.time()
                    feature_arrays = {k: v for k, v in features.items() if isinstance(v, np.ndarray)}
                    self.cache_repo.save_derived_features_batch(file_hash, feature_arrays)

                    # Save metadata
                    metadata = {
                        'sr': features['sr'],
                        'hop_length': features['hop_length'],
                        'duration_sec': features['duration_sec'],
                    }
                    metadata_path = self.cache_repo.stft_dir / f"{file_hash}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    cache_elapsed = time.time() - stage_start

                    elapsed = time.time() - start
                    if self.verbose:
                        print(f" [{cache_elapsed:.1f}s]")
                        print(f"      Total: [{elapsed:.1f}s] (cached)")

                cached_sets.append(CachedSetFeatures(
                    set_id=set_id,
                    file_path=str(path),
                    features=features,
                    ground_truth=gt,
                    sr=metadata['sr'],
                    hop_length=metadata['hop_length'],
                    duration_sec=metadata['duration_sec'],
                ))

        load_elapsed = time.time() - load_start
        total_gt = sum(cs.n_ground_truth for cs in cached_sets)

        # Count cache hits/misses
        cache_hits = sum(1 for cs in cached_sets)  # Simplified for now
        cache_misses = 0

        if self.verbose:
            print(f"\nLoaded {len(cached_sets)} sets in {load_elapsed:.1f}s")
            if self.num_workers > 1:
                print(f"  Parallel workers: {self.num_workers}")

        context.set_result('cached_sets', cached_sets)
        context.set_result('total_ground_truth', total_gt)
        context.set_result('load_time', load_elapsed)

        return context


class OptimizeTransitionParamsStage(PipelineStage):
    """
    Stage that optimizes TransitionDetectionTask parameters.

    Uses differential_evolution with cached features for fast iteration.
    """

    def __init__(
        self,
        mixing_style: MixingStyle = MixingStyle.SMOOTH,
        max_iter: int = 100,
        tolerance_sec: float = 30.0,
        verbose: bool = False,
        on_progress: Optional[Callable[[int, float, Dict], None]] = None
    ):
        self.mixing_style = mixing_style
        self.max_iter = max_iter
        self.tolerance_sec = tolerance_sec
        self.verbose = verbose
        self.on_progress = on_progress
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self._interrupted:
            raise KeyboardInterrupt("Force quit")
        self._interrupted = True
        if self.verbose:
            print("\n\nInterrupted! Returning best result so far...")

    def process(self, context: PipelineContext) -> PipelineContext:
        """Run optimization."""
        cached_sets = context.get_result('cached_sets', [])

        if not cached_sets:
            context.set_result('calibration_result', CalibrationResult(
                success=False,
                best_f1=0,
                precision=0,
                recall=0,
                optimal_params={},
                iterations=0,
                optimization_time_sec=0,
                n_sets=0,
                total_ground_truth=0,
                mixing_style=self.mixing_style.value,
                error="No sets loaded"
            ))
            return context

        # Setup signal handler
        old_handler = signal.signal(signal.SIGINT, self._signal_handler)

        # Get starting point from current config
        cfg = TransitionConfig.for_style(self.mixing_style)
        x0 = np.array([
            cfg.min_transition_gap_sec,
            cfg.energy_threshold,
            cfg.bass_weight,
            cfg.smooth_sigma,
            cfg.peak_percentile,
            cfg.timbral_weight,
        ])

        # Optimization bounds
        bounds = [
            (30.0, 60.0),   # min_transition_gap_sec
            (0.05, 0.3),    # energy_threshold
            (0.2, 0.8),     # bass_weight
            (3.0, 10.0),    # smooth_sigma
            (70.0, 95.0),   # peak_percentile
            (0.0, 1.0),     # timbral_weight
        ]

        # Objective function state
        n_calls = [0]
        best_f1 = [0.0]
        best_params = [None]
        best_precision = [0.0]
        best_recall = [0.0]

        def objective(params):
            if self._interrupted:
                raise StopIteration("User interrupted")

            n_calls[0] += 1

            tp, fp, fn = self._run_detection(cached_sets, params)
            f1, precision, recall = self._compute_metrics(tp, fp, fn)

            # F-beta with beta=0.5 (prioritize precision over recall)
            # For DJ annotation: false positives (spurious transitions) worse than false negatives
            beta = 0.5
            if (precision + recall) == 0:
                f_beta = 0.0
            else:
                f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

            if f_beta > best_f1[0]:  # Using best_f1 var name for compatibility
                best_f1[0] = f_beta
                best_params[0] = params.copy()
                best_precision[0] = precision
                best_recall[0] = recall

                if self.verbose:
                    # Clear current line and show improvement on NEW line
                    sys.stdout.write(f"\r{' ' * 120}\r")
                    sys.stdout.write(f"✨ NEW BEST! F-beta={f_beta:.3f} (P={precision:.1%}, R={recall:.1%})\n")
                    sys.stdout.flush()

                if self.on_progress:
                    self.on_progress(n_calls[0], f_beta, {
                        'min_transition_gap_sec': params[0],
                        'energy_threshold': params[1],
                        'bass_weight': params[2],
                        'smooth_sigma': params[3],
                        'peak_percentile': params[4],
                        'timbral_weight': params[5],
                    })

            # Compact progress update (always show after potential NEW BEST message)
            if self.verbose:
                # Progress bar - SHORT (limit to 100% to avoid overflow)
                progress = min(n_calls[0] / self.max_iter, 1.0)
                bar_width = 10
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)

                # Compact metrics
                perc = min(int(progress * 100), 100)

                # Build status line - COMPACT (max 70 chars)
                status = f"\r[{n_calls[0]:>3}] {perc:>3}% {bar} │ F={f_beta:.2f} P={precision:.0%} R={recall:.0%} │ Best={best_f1[0]:.2f}"

                # Clear rest of line and write
                sys.stdout.write(status + ' ' * 10)
                sys.stdout.flush()

            return -f_beta  # Minimize negative F-beta

        if self.verbose:
            print(f"\n╔═══════════════════════════════════════════════════════════╗")
            print(f"║  Calibrating {self.mixing_style.value.upper()} style (max {self.max_iter} iterations)")
            print(f"╚═══════════════════════════════════════════════════════════╝")

        start_time = time.time()

        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=self.max_iter,
                disp=False,
                workers=1,  # Sequential (nested function can't be pickled for multiprocessing)
                updating='immediate',  # Sequential updates
                strategy='best1bin',  # Faster convergence
                polish=True,
                seed=42,
                atol=0.002,  # Relaxed tolerance for faster convergence
                tol=0.002,
            )
            final_params = result.x
            success = result.success
        except (StopIteration, KeyboardInterrupt):
            final_params = best_params[0] if best_params[0] is not None else x0
            success = True  # We have a result

        elapsed = time.time() - start_time

        # Restore signal handler
        signal.signal(signal.SIGINT, old_handler)

        # Build result
        optimal_params = {
            'min_transition_gap_sec': float(final_params[0]),
            'energy_threshold': float(final_params[1]),
            'bass_weight': float(final_params[2]),
            'smooth_sigma': float(final_params[3]),
            'peak_percentile': float(final_params[4]),
            'timbral_weight': float(final_params[5]),
        }

        calibration_result = CalibrationResult(
            success=success,
            best_f1=best_f1[0],
            precision=best_precision[0],
            recall=best_recall[0],
            optimal_params=optimal_params,
            iterations=n_calls[0],
            optimization_time_sec=elapsed,
            n_sets=len(cached_sets),
            total_ground_truth=context.get_result('total_ground_truth', 0),
            mixing_style=self.mixing_style.value,
        )

        context.set_result('calibration_result', calibration_result)

        if self.verbose:
            sys.stdout.write("\n")  # Move to new line after progress bar
            sys.stdout.write(f"\n✅ Optimization complete in {elapsed:.1f}s ({n_calls[0]} iterations)\n")
            sys.stdout.write(f"   Final Best: F-beta={best_f1[0]:.3f} (P={best_precision[0]:.1%}, R={best_recall[0]:.1%})\n")
            sys.stdout.flush()

        return context

    def _run_detection(self, cached_sets: List[CachedSetFeatures], params: np.ndarray) -> Tuple[int, int, int]:
        """Run detection on all cached sets (vectorized)."""
        min_gap, energy_thr, bass_w, sigma, percentile, timbral_w = params

        # Vectorized approach: compute all detection results in parallel
        # Using list comprehension for minimal overhead
        results = []

        # Create single task instance (reuse for all sets - faster)
        task = TransitionDetectionTask(
            min_transition_gap_sec=min_gap,
            energy_threshold=energy_thr,
            bass_weight=bass_w,
            smooth_sigma=sigma,
            peak_percentile=percentile,
            timbral_weight=timbral_w,
            detect_filters=True,
            filter_velocity_threshold=200.0,
            transition_merge_window_sec=30.0,
        )

        # Process all sets (silent - progress shown in main loop)
        for cs in cached_sets:
            result = task.execute_from_features(
                rms=cs.features['rms'],
                bass_energy=cs.features['bass_energy'],
                novelty=cs.features['novelty'],
                timbral_novelty=cs.features['timbral_novelty'],
                chroma_novelty=cs.features['chroma_novelty'],
                centroid=cs.features['centroid'],
                rolloff=cs.features['rolloff'],
                centroid_velocity=cs.features['centroid_velocity'],
                sr=cs.sr,
                hop_length=cs.hop_length,
                duration_sec=cs.duration_sec,
                drops=cs.features.get('drops'),
                seg_boundaries=cs.features.get('seg_boundaries'),
            )

            detected = np.array([m.time_sec for m in result.mixins]) if result.mixins else np.array([])
            results.append((cs.ground_truth, detected))

        # Vectorized evaluation of all results
        tp_fp_fn = [self._evaluate(gt, det) for gt, det in results]

        # Sum using numpy for speed
        tp_fp_fn_array = np.array(tp_fp_fn)
        total_tp, total_fp, total_fn = tp_fp_fn_array.sum(axis=0)

        return int(total_tp), int(total_fp), int(total_fn)

    def _evaluate(self, gt: np.ndarray, detected: np.ndarray) -> Tuple[int, int, int]:
        """Greedy matching evaluation."""
        if len(gt) == 0:
            return 0, len(detected), 0
        if len(detected) == 0:
            return 0, 0, len(gt)

        dist = np.abs(detected[:, np.newaxis] - gt[np.newaxis, :])
        matched_gt = np.zeros(len(gt), dtype=bool)
        matched_det = np.zeros(len(detected), dtype=bool)

        min_dists = dist.min(axis=1)
        sorted_det = np.argsort(min_dists)

        for i in sorted_det:
            if matched_det[i]:
                continue
            valid = ~matched_gt & (dist[i] <= self.tolerance_sec)
            if valid.any():
                valid_dists = np.where(valid, dist[i], np.inf)
                j = np.argmin(valid_dists)
                matched_det[i] = True
                matched_gt[j] = True

        tp = int(matched_gt.sum())
        fp = len(detected) - int(matched_det.sum())
        fn = len(gt) - tp

        return tp, fp, fn

    def _compute_metrics(self, tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        """Compute F1, precision, recall."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall


class CalibrationPipeline(Pipeline):
    """
    Pipeline for calibrating TransitionDetection parameters.

    Two-phase approach:
    1. Load all sets and cache features (uses SetAnalysis caching)
    2. Run optimization on cached data (~30ms per iteration)

    Usage:
        pipeline = CalibrationPipeline(mixing_style=MixingStyle.SMOOTH, verbose=True)

        result = pipeline.calibrate(
            audio_paths=['set1.m4a', 'set2.m4a'],
            ground_truth={'set1': [60, 180], 'set2': [90, 210]},
            max_iter=100
        )
    """

    def __init__(
        self,
        mixing_style: MixingStyle = MixingStyle.SMOOTH,
        max_iter: int = 100,
        tolerance_sec: float = 30.0,
        sr: int = 22050,
        verbose: bool = False,
        cache_dir: str = "cache",
        num_workers: int = 1,
    ):
        self.mixing_style = mixing_style
        self.max_iter = max_iter
        self.tolerance_sec = tolerance_sec
        self.sr = sr
        self.verbose = verbose
        self.num_workers = num_workers

        # Initialize cache repository (single entry point for all cache operations)
        self.cache_repo = CacheRepository(cache_dir)

        # Build stages
        stages = [
            LoadAndCacheFeaturesStage(
                cache_repo=self.cache_repo,
                sr=sr,
                verbose=verbose,
                num_workers=num_workers
            ),
            OptimizeTransitionParamsStage(
                mixing_style=mixing_style,
                max_iter=max_iter,
                tolerance_sec=tolerance_sec,
                verbose=verbose
            ),
        ]

        super().__init__(stages, name="CalibrationPipeline")

    def calibrate(
        self,
        audio_paths: List[str],
        ground_truth: Dict[str, List[float]],
        max_iter: Optional[int] = None
    ) -> CalibrationResult:
        """
        Run calibration.

        Args:
            audio_paths: Paths to DJ set audio files
            ground_truth: {set_id: [transition_times_in_seconds]}
            max_iter: Override max iterations

        Returns:
            CalibrationResult with optimal parameters
        """
        if max_iter:
            # Update optimization stage
            for stage in self.stages:
                if isinstance(stage, OptimizeTransitionParamsStage):
                    stage.max_iter = max_iter

        context = PipelineContext(
            input_path="calibration",
            config={
                'audio_paths': audio_paths,
                'ground_truth': ground_truth,
            }
        )

        if self.verbose:
            print("=" * 60)
            print(f"Calibration Pipeline - {self.mixing_style.value}")
            print("=" * 60)
            print()
            print("Phase 1: Load sets (with caching)")

        context = self.run(context)

        result = context.get_result('calibration_result')

        if self.verbose and result:
            print()
            print("╔═══════════════════════════════════════════════════════════╗")
            print("║               CALIBRATION RESULTS                         ║")
            print("╠═══════════════════════════════════════════════════════════╣")
            print(f"║  F-beta: {result.best_f1:.3f}  │  Precision: {result.precision:>5.1%}  │  Recall: {result.recall:>5.1%}  ║")
            print("╠═══════════════════════════════════════════════════════════╣")
            print("║  Optimal Parameters:                                      ║")
            for k, v in result.optimal_params.items():
                param_display = k.replace('_', ' ').title()[:35]
                print(f"║    {param_display:<35} {v:>8.3f}      ║")
            print("╚═══════════════════════════════════════════════════════════╝")
            print()
            print("📋 Config snippet for src/core/config.py:")
            print(result.to_config_snippet())

        return result

    def test_current_config(
        self,
        audio_paths: List[str],
        ground_truth: Dict[str, List[float]]
    ) -> Tuple[float, float, float]:
        """
        Test current config without optimization.

        Returns:
            (f1, precision, recall)
        """
        # Just run loading stage
        context = PipelineContext(
            input_path="test",
            config={
                'audio_paths': audio_paths,
                'ground_truth': ground_truth,
            }
        )

        load_stage = self.stages[0]
        context = load_stage.process(context)

        cached_sets = context.get_result('cached_sets', [])

        # Get current config params
        cfg = TransitionConfig.for_style(self.mixing_style)
        params = np.array([
            cfg.min_transition_gap_sec,
            cfg.energy_threshold,
            cfg.bass_weight,
            cfg.smooth_sigma,
            cfg.peak_percentile,
            cfg.timbral_weight,
        ])

        optimize_stage = self.stages[1]
        tp, fp, fn = optimize_stage._run_detection(cached_sets, params)
        f1, precision, recall = optimize_stage._compute_metrics(tp, fp, fn)

        if self.verbose:
            print(f"\nCurrent {self.mixing_style.value} config:")
            print(f"  F1: {f1:.3f}")
            print(f"  Precision: {precision:.1%}")
            print(f"  Recall: {recall:.1%}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")

        return f1, precision, recall
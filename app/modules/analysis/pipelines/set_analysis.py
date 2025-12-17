"""
Set Analysis Pipeline - Analyze DJ sets (long recordings).

Includes transition detection and track segmentation.
"""

import gc
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path

import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

from .base import (
    Pipeline, PipelineContext, PipelineStage,
    LoadAudioStage, ComputeSTFTStage
)
from .cache_manager import CacheManager
from app.modules.analysis.tasks import (
    FeatureExtractionTask,
    DropDetectionTask,
    TransitionDetectionTask,
    GenreAnalysisTask,
    SegmentationTask,
    SegmentBoundary,
    create_audio_context,
)
from ..config import (
    SetAnalysisConfig, TransitionConfig, MixingStyle,
    DEFAULT_SET_ANALYSIS,
)

# Re-export for convenience
__all__ = [
    'SetAnalysisPipeline', 'SetAnalysisResult', 'SegmentInfo',
    'SegmentGenre', 'SetGenreDistribution', 'AnalyzeSegmentGenresStage',
    'MixingStyle', 'SetBatchAnalyzer',
]


@dataclass
class SegmentGenre:
    """Genre analysis result for a segment."""
    genre: str = ""
    subgenre: str = ""
    dj_category: str = ""
    confidence: float = 0.0
    all_styles: List[Tuple[str, float]] = field(default_factory=list)
    mood_tags: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'genre': self.genre,
            'subgenre': self.subgenre,
            'dj_category': self.dj_category,
            'confidence': self.confidence,
            'all_styles': self.all_styles[:5],
            'mood_tags': self.mood_tags[:5],
        }


@dataclass
class SegmentInfo:
    """Information about a segment (potential track) in the set."""
    start_time: float
    end_time: float
    duration: float
    segment_index: int
    zone: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    genre: Optional[SegmentGenre] = None
    is_transition_zone: bool = False  # True if this is a mixin-mixout overlap zone

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'segment_index': self.segment_index,
            'zone': self.zone,
            'features': self.features,
            'genre': self.genre.to_dict() if self.genre else None,
            'is_transition_zone': self.is_transition_zone,
        }


@dataclass
class SetGenreDistribution:
    """
    Aggregated genre distribution across the entire set.

    Computed from segment-level genre analysis.
    """
    # Primary DJ category
    primary_category: str = ""
    primary_category_ratio: float = 0.0

    # Distribution by DJ category (sums to 1.0)
    category_distribution: Dict[str, float] = field(default_factory=dict)

    # Top subgenres with ratios
    top_subgenres: List[Tuple[str, float]] = field(default_factory=list)

    # Diversity metrics
    genre_diversity: float = 0.0      # Shannon entropy
    n_unique_subgenres: int = 0

    # Genre flow
    genre_flow: List[str] = field(default_factory=list)  # Sequence of categories
    genre_transitions: int = 0         # Number of category changes

    # Aggregated mood tags
    mood_tags: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_category': self.primary_category,
            'primary_category_ratio': self.primary_category_ratio,
            'category_distribution': self.category_distribution,
            'top_subgenres': self.top_subgenres[:10],
            'genre_diversity': self.genre_diversity,
            'n_unique_subgenres': self.n_unique_subgenres,
            'genre_flow': self.genre_flow,
            'genre_transitions': self.genre_transitions,
            'mood_tags': dict(sorted(self.mood_tags.items(), key=lambda x: -x[1])[:10]),
        }


@dataclass
class SetAnalysisResult:
    """
    Complete analysis result for a DJ set.
    """
    file_path: str
    file_name: str
    duration_sec: float

    # Transitions
    n_transitions: int
    transition_times: List[Tuple[float, float]]
    transition_density: float

    # Segments
    n_segments: int
    segments: List[SegmentInfo]

    # Drops (across whole set)
    total_drops: int
    drop_density: float

    # Timeline
    energy_timeline: Optional[List[float]] = None

    # Genre distribution
    genre_distribution: Optional[SetGenreDistribution] = None

    # Metadata
    processing_time_sec: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'duration_sec': self.duration_sec,
            'n_transitions': self.n_transitions,
            'transition_times': self.transition_times,
            'transition_density': self.transition_density,
            'n_segments': self.n_segments,
            'segments': [s.to_dict() for s in self.segments],
            'total_drops': self.total_drops,
            'drop_density': self.drop_density,
            'genre_distribution': self.genre_distribution.to_dict() if self.genre_distribution else None,
            'processing_time_sec': self.processing_time_sec,
            'success': self.success,
            'error': self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DetectTransitionsStage(PipelineStage):
    """Stage that detects mixin/mixout transitions."""

    def __init__(self, config: Optional[TransitionConfig] = None):
        cfg = config or TransitionConfig()
        self.task = TransitionDetectionTask(
            min_transition_gap_sec=cfg.min_transition_gap_sec,
            energy_threshold=cfg.energy_threshold,
            bass_weight=cfg.bass_weight,
            smooth_sigma=cfg.smooth_sigma,
            detect_filters=cfg.detect_filters,
            filter_velocity_threshold=cfg.filter_velocity_threshold,
            peak_percentile=cfg.peak_percentile,
            transition_merge_window_sec=cfg.transition_merge_window_sec,
            timbral_weight=cfg.timbral_weight,  # For smooth mixing: prioritize timbral novelty
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('transitions', result)
        return context


class SegmentTracksStage(PipelineStage):
    """Stage that segments the set into tracks based on transitions."""

    def __init__(
        self,
        min_track_duration: float = 60.0,
        min_transition_duration: float = 30.0
    ):
        """
        Initialize segmentation stage.

        Args:
            min_track_duration: Minimum TRACK (solo) duration in seconds
            min_transition_duration: Minimum TRANSITION (overlap) duration in seconds
        """
        self.min_track_duration = min_track_duration
        self.min_transition_duration = min_transition_duration

    def process(self, context: PipelineContext) -> PipelineContext:
        transitions = context.get_result('transitions')
        duration = context.results.get('_duration', 0.0)

        if transitions is None:
            context.set_result('segments', [])
            return context

        segments = []

        # Collect all time points with their types
        # (time, type) where type is 'start', 'mixin', 'mixout', 'end'
        time_points = [(0.0, 'start')]
        for t in transitions.transitions:
            time_points.append((t.mixin.time_sec, 'mixin'))
            time_points.append((t.mixout.time_sec, 'mixout'))
        time_points.append((duration, 'end'))

        # Sort by time
        time_points = sorted(time_points, key=lambda x: x[0])

        # Create segments between points
        # A segment is a transition_zone if it's between mixin and mixout
        for i in range(len(time_points) - 1):
            start_time, start_type = time_points[i]
            end_time, end_type = time_points[i + 1]

            seg_duration = end_time - start_time

            # Segment is transition zone if it starts with mixin (mixin -> mixout)
            is_transition = (start_type == 'mixin')

            # Apply different minimum durations for tracks vs transitions
            min_duration = self.min_transition_duration if is_transition else self.min_track_duration

            if seg_duration >= min_duration:
                segments.append(SegmentInfo(
                    start_time=start_time,
                    end_time=end_time,
                    duration=seg_duration,
                    segment_index=len(segments),
                    is_transition_zone=is_transition,
                ))

        context.set_result('segments', segments)
        return context


class DetectAllDropsStage(PipelineStage):
    """Stage that detects drops across the whole set."""

    def __init__(
        self,
        min_drop_magnitude: float = 0.3,
        min_confidence: float = 0.5,
        buildup_window_sec: float = 2.0,
        use_multiband: bool = True
    ):
        self.task = DropDetectionTask(
            min_drop_magnitude=min_drop_magnitude,
            min_confidence=min_confidence,
            buildup_window_sec=buildup_window_sec,
            use_multiband=use_multiband
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('drops', result)
        return context


class LaplacianSegmentationStage(PipelineStage):
    """
    Stage that segments the set using Laplacian spectral clustering.

    Uses SegmentationTask (based on McFee & Ellis 2014) for better
    track boundary detection in smooth mixing styles.

    This replaces or augments SegmentTracksStage for DJ sets.
    """

    def __init__(
        self,
        min_segment_sec: float = 120.0,
        max_segment_sec: float = 480.0,
        merge_threshold_sec: float = 60.0,
    ):
        """
        Args:
            min_segment_sec: Minimum segment duration
            max_segment_sec: Maximum segment duration
            merge_threshold_sec: Merge boundaries closer than this
        """
        self.task = SegmentationTask(
            min_segment_sec=min_segment_sec,
            max_segment_sec=max_segment_sec,
            merge_threshold_sec=merge_threshold_sec,
        )

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            raise ValueError("AudioContext required")

        result = self.task.execute(context.audio_context)
        context.set_result('laplacian_segmentation', result)

        if not result.success:
            logger.warning(f"Laplacian segmentation failed: {result.error}")
            return context

        # Convert boundaries to SegmentInfo
        duration = context.results.get('_duration', 0.0)
        boundary_times = result.boundary_times + [duration]

        segments = []
        for i, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
            seg_duration = end - start
            if seg_duration >= 30.0:  # Min 30 sec
                segments.append(SegmentInfo(
                    start_time=start,
                    end_time=end,
                    duration=seg_duration,
                    segment_index=len(segments)
                ))

        context.set_result('segments', segments)
        logger.info(f"Laplacian segmentation: {len(segments)} segments (tempo={result.tempo:.1f} BPM)")
        return context


class BuildTimelineStage(PipelineStage):
    """Stage that builds energy timeline for visualization."""

    def __init__(self, resolution_sec: float = 1.0):
        """
        Initialize timeline stage.

        Args:
            resolution_sec: Time resolution in seconds
        """
        self.resolution_sec = resolution_sec

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            return context

        import numpy as np
        from app.common.primitives import smooth_gaussian

        sr = context.audio_context.sr
        hop_length = context.audio_context.hop_length

        # Compute RMS energy via STFTCache (librosa-based)
        rms = context.audio_context.stft_cache.get_rms()
        # get_rms returns 2D array (1, n_frames), flatten it
        if rms.ndim > 1:
            rms = rms.flatten()
        rms_smooth = smooth_gaussian(rms, sigma=5.0)

        # Downsample to target resolution
        frame_duration = hop_length / sr
        target_frames = int(context.results.get('_duration', 0) / self.resolution_sec)

        if target_frames > 0 and len(rms_smooth) > 0:
            # Resample
            indices = np.linspace(0, len(rms_smooth) - 1, target_frames).astype(int)
            timeline = rms_smooth[indices]
            # Normalize
            timeline = (timeline - np.min(timeline)) / (np.max(timeline) - np.min(timeline) + 1e-10)
            context.set_result('energy_timeline', timeline.tolist())

        return context


class AnalyzeSegmentGenresStage(PipelineStage):
    """
    Stage that analyzes genre for each segment (track) in the set.

    For each segment:
    1. Extract audio slice from full audio
    2. Create AudioContext for the segment (without file_path)
    3. Run GenreAnalysisTask (uses predict_from_array)
    4. Store result in SegmentInfo.genre

    Also computes aggregated SetGenreDistribution.
    """

    def __init__(self, min_segment_duration: float = 30.0):
        """
        Args:
            min_segment_duration: Minimum segment duration for genre analysis (seconds)
        """
        self.min_segment_duration = min_segment_duration
        self.task = GenreAnalysisTask(top_n=10)

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.audio_context is None:
            return context

        segments: List[SegmentInfo] = context.get_result('segments', [])
        if not segments:
            logger.info("No segments to analyze")
            return context

        y = context.audio_context.y
        sr = context.audio_context.sr

        # Vectorized: compute segment boundaries
        seg_times = np.array([(s.start_time, s.end_time, s.duration) for s in segments])
        valid_mask = seg_times[:, 2] >= self.min_segment_duration
        valid_indices = np.where(valid_mask)[0]

        logger.info(f"Analyzing genres for {len(valid_indices)}/{len(segments)} segments")
        segment_genres: List[SegmentGenre] = []

        for i, idx in enumerate(valid_indices):
            segment = segments[idx]
            start_sample = int(segment.start_time * sr)
            end_sample = int(segment.end_time * sr)
            segment_y = y[start_sample:end_sample]

            if len(segment_y) < sr * 10:
                continue

            logger.info(f"  Segment {i+1}/{len(valid_indices)}: {segment.start_time/60:.1f}-{segment.end_time/60:.1f}min")
            try:
                segment_ctx = create_audio_context(segment_y, sr)
                result = self.task.execute(segment_ctx)

                if result.success:
                    segment.genre = SegmentGenre(
                        genre=result.genre,
                        subgenre=result.subgenre,
                        dj_category=result.dj_category,
                        confidence=result.confidence,
                        all_styles=result.all_styles,
                        mood_tags=result.mood_tags,
                    )
                    segment_genres.append(segment.genre)
                    logger.info(f"    -> {result.genre} ({result.dj_category}) conf={result.confidence:.2f}")
                else:
                    logger.warning(f"    -> Genre analysis failed: {result.error}")
            except Exception as e:
                logger.error(f"    -> Exception: {e}")

        logger.info(f"Genre analysis complete: {len(segment_genres)} genres extracted")
        if segment_genres:
            genre_dist = self._compute_distribution(segment_genres)
            context.set_result('genre_distribution', genre_dist)

        return context

    def _compute_distribution(self, segment_genres: List[SegmentGenre]) -> SetGenreDistribution:
        """Compute aggregated genre distribution (numpy-optimized)."""
        dist = SetGenreDistribution()

        # Extract arrays
        categories = np.array([sg.dj_category for sg in segment_genres if sg.dj_category])
        subgenres_list = [sg.genre for sg in segment_genres if sg.genre]

        if len(categories) == 0:
            return dist

        # Vectorized: category counts using np.unique
        unique_cats, cat_counts = np.unique(categories, return_counts=True)
        total = len(categories)
        dist.category_distribution = {cat: int(cnt) / total for cat, cnt in zip(unique_cats, cat_counts)}

        # Primary category (max count)
        max_idx = np.argmax(cat_counts)
        dist.primary_category = unique_cats[max_idx]
        dist.primary_category_ratio = float(cat_counts[max_idx]) / total

        # Subgenres
        subgenre_counts = Counter(subgenres_list)
        n_subgenres = len(subgenres_list)
        dist.top_subgenres = [(sg, cnt / n_subgenres) for sg, cnt in subgenre_counts.most_common(10)]
        dist.n_unique_subgenres = len(subgenre_counts)

        # Shannon entropy (vectorized)
        probs = cat_counts / total
        probs = probs[probs > 0]
        dist.genre_diversity = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0

        # Genre flow
        dist.genre_flow = categories.tolist()

        # Vectorized: count transitions (where category[i] != category[i-1])
        dist.genre_transitions = int(np.sum(categories[1:] != categories[:-1]))

        # Aggregate mood tags (vectorized per tag)
        all_moods: Dict[str, List[float]] = {}
        for sg in segment_genres:
            for tag, conf in sg.mood_tags:
                if tag not in all_moods:
                    all_moods[tag] = []
                all_moods[tag].append(conf)

        dist.mood_tags = {tag: float(np.mean(confs)) for tag, confs in all_moods.items()}

        return dist


# Human-readable stage names for verbose output
STAGE_DESCRIPTIONS = {
    'LoadAudioStage': 'Loading audio file',
    'ComputeSTFTStage': 'Computing spectrogram (STFT)',
    'DetectTransitionsStage': 'Detecting transitions',
    'LaplacianSegmentationStage': 'Segmenting tracks (Laplacian)',
    'SegmentTracksStage': 'Segmenting tracks',
    'DetectAllDropsStage': 'Detecting drops',
    'BuildTimelineStage': 'Building energy timeline',
    'AnalyzeSegmentGenresStage': 'Analyzing genres per segment',
}


class SetAnalysisPipeline(Pipeline):
    """
    DJ Set analysis pipeline.

    Stages:
    1. LoadAudio - Load audio file
    2. ComputeSTFT - Compute spectrogram
    3. DetectTransitions - Find mixin/mixout points
    4. LaplacianSegmentation (SMOOTH) / SegmentTracks (other) - Segment into tracks
    5. DetectAllDrops - Find drops
    6. BuildTimeline - Create energy timeline
    7. AnalyzeSegmentGenres - Genre per segment (optional)

    Usage:
        # Standard mixing
        pipeline = SetAnalysisPipeline(analyze_genres=True)

        # Smooth techno mixing (NK style) - uses Laplacian segmentation
        pipeline = SetAnalysisPipeline(mixing_style=MixingStyle.SMOOTH)

        result = pipeline.analyze("dj_set.mp3")

        # Verbose mode (prints step-by-step progress)
        pipeline = SetAnalysisPipeline(verbose=True)
        result = pipeline.analyze("dj_set.mp3")
    """

    def __init__(
        self,
        mixing_style: Optional[MixingStyle] = None,
        config: Optional[SetAnalysisConfig] = None,
        analyze_genres: bool = False,
        use_laplacian_segmentation: bool = False,  # Default: transition-based (n_segments = n_transitions + 1)
        sr: int = 22050,
        verbose: bool = False,
    ):
        """
        Initialize set analysis pipeline.

        Args:
            mixing_style: DJ mixing style preset (SMOOTH, STANDARD, HARD)
            config: Full configuration (overrides mixing_style)
            analyze_genres: Enable genre analysis per segment
            use_laplacian_segmentation: Use Laplacian (spectral clustering) for segmentation
            sr: Sample rate
            verbose: Print detailed step-by-step progress to stdout
        """
        self.verbose = verbose

        # Resolve config
        if config is not None:
            cfg = config
        elif mixing_style is not None:
            cfg = SetAnalysisConfig.for_style(mixing_style, analyze_genres)
        else:
            cfg = DEFAULT_SET_ANALYSIS
            cfg.analyze_genres = analyze_genres

        # Choose segmentation method
        # Default: use transition-based segmentation (segments = transitions + 1)
        # Laplacian: only when explicitly requested (for structure analysis)
        use_laplacian = use_laplacian_segmentation

        stages = [
            LoadAudioStage(sr=sr),
            ComputeSTFTStage(),
            DetectTransitionsStage(config=cfg.transition),
        ]

        if use_laplacian:
            stages.append(LaplacianSegmentationStage(
                min_segment_sec=cfg.segmentation.min_track_duration,
                max_segment_sec=480.0,
                merge_threshold_sec=60.0,
            ))
        else:
            # Transition-based: segments are tracks between transitions
            # Different minimums: tracks min 1.5min, transitions min 30sec
            stages.append(SegmentTracksStage(
                min_track_duration=cfg.segmentation.min_track_duration,
                min_transition_duration=cfg.segmentation.min_transition_duration
            ))

        stages.extend([
            DetectAllDropsStage(
                min_drop_magnitude=cfg.drop_detection.min_drop_magnitude,
                min_confidence=cfg.drop_detection.min_confidence,
                buildup_window_sec=cfg.drop_detection.buildup_window_sec,
                use_multiband=cfg.drop_detection.use_multiband,
            ),
            BuildTimelineStage(resolution_sec=cfg.timeline_resolution),
        ])

        if cfg.analyze_genres or analyze_genres:
            stages.append(AnalyzeSegmentGenresStage(
                min_segment_duration=cfg.segmentation.min_segment_for_genre
            ))

        # Setup verbose callback
        on_stage_complete = None
        if verbose:
            on_stage_complete = self._verbose_stage_callback

        super().__init__(stages, name="SetAnalysis", on_stage_complete=on_stage_complete)

    def _verbose_stage_callback(self, stage_name: str, context: PipelineContext):
        """Print stage completion with timing and details."""
        elapsed = context.results.get(f'_stage_{stage_name}_time', 0.0)
        description = STAGE_DESCRIPTIONS.get(stage_name, stage_name)

        # Add relevant details based on stage
        details = ""
        if stage_name == 'LoadAudioStage':
            duration = context.results.get('_duration', 0)
            details = f" ({duration/60:.1f} min)"
        elif stage_name == 'DetectTransitionsStage':
            transitions = context.get_result('transitions')
            if transitions:
                details = f" ({transitions.n_transitions} transitions)"
        elif stage_name in ('LaplacianSegmentationStage', 'SegmentTracksStage'):
            segments = context.get_result('segments', [])
            details = f" ({len(segments)} segments)"
        elif stage_name == 'DetectAllDropsStage':
            drops = context.get_result('drops')
            if drops:
                details = f" ({drops.n_drops} drops)"

        print(f"    [{elapsed:.1f}s] {description}{details}")

    def analyze(self, path: str) -> SetAnalysisResult:
        """
        Analyze a DJ set.

        Args:
            path: Path to audio file

        Returns:
            SetAnalysisResult with all analysis data
        """
        context = PipelineContext(input_path=path)

        try:
            context = self.run(context)

            transitions = context.get_result('transitions')
            segments = context.get_result('segments', [])
            drops = context.get_result('drops')
            genre_dist = context.get_result('genre_distribution')

            return SetAnalysisResult(
                file_path=path,
                file_name=Path(path).name,
                duration_sec=context.results.get('_duration', 0.0),
                n_transitions=transitions.n_transitions if transitions else 0,
                transition_times=transitions.get_transition_times() if transitions else [],
                transition_density=transitions.transition_density if transitions else 0.0,
                n_segments=len(segments),
                segments=segments,
                total_drops=drops.n_drops if drops else 0,
                drop_density=drops.drop_density if drops else 0.0,
                energy_timeline=context.get_result('energy_timeline'),
                genre_distribution=genre_dist,
                processing_time_sec=context.results.get('_total_time', 0.0),
                success=True
            )
        except Exception as e:
            return SetAnalysisResult(
                file_path=path,
                file_name=Path(path).name,
                duration_sec=0.0,
                n_transitions=0,
                transition_times=[],
                transition_density=0.0,
                n_segments=0,
                segments=[],
                total_drops=0,
                drop_density=0.0,
                processing_time_sec=0.0,
                success=False,
                error=str(e)
            )
        finally:
            # Cleanup STFTCache to prevent memory leak on long audio files
            if context.audio_context and hasattr(context.audio_context, 'stft_cache'):
                context.audio_context.stft_cache.clear_feature_cache()


class SetBatchAnalyzer:
    """
    Пакетный анализ DJ сетов с автоматическим checkpoint через CacheManager.

    Независим от DJStyleExtractor — только анализ, без построения профиля.
    Результаты автоматически кэшируются в predictions.db.

    Usage:
        analyzer = SetBatchAnalyzer()

        # Проверить статус
        status = analyzer.get_status(set_paths)
        print(f"Cached: {status['cached']}, Pending: {status['pending']}")

        # Запустить анализ (с автоматическим checkpoint)
        results = analyzer.analyze_sets(set_paths, analyze_genres=True)

        # Если прервано (Ctrl+C) — просто запустить снова:
        results = analyzer.analyze_sets(set_paths)  # продолжит с checkpoint
    """

    def __init__(
        self,
        cache_dir: str = "~/.mood-classifier/cache",
        sr: int = 22050
    ):
        """
        Args:
            cache_dir: Директория для кэша (predictions.db)
            sr: Sample rate для анализа
        """
        self.cache_manager = CacheManager(cache_dir)
        self.sr = sr

    def analyze_sets(
        self,
        set_paths: List[str],
        analyze_genres: bool = True,
        force: bool = False,
        on_progress: Optional[Callable[[int, int, str, 'SetAnalysisResult'], None]] = None,
        show_progress: bool = True,
        verbose: bool = False,
        mixing_style: Optional[MixingStyle] = None
    ) -> List['SetAnalysisResult']:
        """
        Анализирует список сетов с автоматическим checkpoint.

        Каждый успешно обработанный файл сразу сохраняется в кэш.
        При прерывании (Ctrl+C) и повторном запуске — продолжит с места остановки.

        Args:
            set_paths: Пути к аудио файлам сетов
            analyze_genres: Включить жанровый анализ (медленно)
            force: Принудительно переанализировать все файлы (игнорировать кэш)
            on_progress: Callback(done, total, path, result)
            show_progress: Печатать прогресс в консоль
            verbose: Печатать детальный прогресс по шагам (для каждого stage)
            mixing_style: Стиль микса (SMOOTH/STANDARD/HARD) - влияет на чувствительность
                         SMOOTH - для плавного techno/minimal микса
                         STANDARD - стандартный клубный микс
                         HARD - ��ыстрые переходы (bass/dubstep)

        Returns:
            List[SetAnalysisResult] для всех сетов
        """
        total = len(set_paths)

        # Vectorized: convert all paths to absolute
        abs_paths = np.array([os.path.abspath(p) for p in set_paths])

        # Vectorized: check cache status for all paths at once
        if not force:
            cache_hits = np.array([
                self.cache_manager.get_set_analysis(p) for p in abs_paths
            ], dtype=object)
            cached_mask = np.array([c is not None for c in cache_hits])
        else:
            cache_hits = np.array([None] * total, dtype=object)
            cached_mask = np.zeros(total, dtype=bool)

        cached_count = int(np.sum(cached_mask))
        pending_indices = np.where(~cached_mask)[0]

        # Pre-allocate results array
        results: List[Optional[SetAnalysisResult]] = [None] * total

        # Vectorized: load cached results
        cached_indices = np.where(cached_mask)[0]
        for idx in cached_indices:
            result = self._result_from_dict(cache_hits[idx])
            results[idx] = result
            if show_progress:
                print(f"[{idx + 1}/{total}] {Path(set_paths[idx]).name}: cached")
            if on_progress:
                on_progress(idx + 1, total, set_paths[idx], result)

        # Create pipeline once
        pipeline = SetAnalysisPipeline(
            sr=self.sr,
            analyze_genres=analyze_genres,
            verbose=verbose,
            mixing_style=mixing_style
        )

        # Process pending files
        for idx in pending_indices:
            path = set_paths[idx]
            abs_path = abs_paths[idx]
            i = idx + 1

            if show_progress:
                print(f"[{i}/{total}] Analyzing: {Path(path).name}")

            try:
                result = pipeline.analyze(abs_path)

                # СРАЗУ сохранить в кэш (checkpoint)
                self.cache_manager.save_set_analysis(abs_path, result.to_dict())

                if show_progress:
                    status = "OK" if result.success else "FAIL"
                    print(f"  -> {result.n_segments} segments, {result.total_drops} drops ({status})")

            except Exception as e:
                result = SetAnalysisResult(
                    file_path=abs_path,
                    file_name=Path(path).name,
                    duration_sec=0,
                    n_transitions=0,
                    transition_times=[],
                    transition_density=0,
                    n_segments=0,
                    segments=[],
                    total_drops=0,
                    drop_density=0,
                    success=False,
                    error=str(e)
                )
                if show_progress:
                    print(f"  -> ERROR: {e}")

            results[idx] = result

            if on_progress:
                on_progress(i, total, path, result)

            gc.collect()

        if show_progress:
            new_count = len(pending_indices)
            print(f"\nDone: {total} sets ({cached_count} from cache, {new_count} analyzed)")

        return results

    def get_status(self, set_paths: List[str]) -> Dict[str, Any]:
        """
        Проверить статус: сколько файлов уже в кэше.

        Args:
            set_paths: Список путей

        Returns:
            {
                'total': int,
                'cached': int,
                'pending': int,
                'pending_paths': List[str]
            }
        """
        # Vectorized: check all paths
        abs_paths = np.array([os.path.abspath(p) for p in set_paths])
        cached_mask = np.array([
            self.cache_manager.get_set_analysis(p) is not None for p in abs_paths
        ])

        cached_count = int(np.sum(cached_mask))
        pending_paths = list(np.array(set_paths)[~cached_mask])

        return {
            'total': len(set_paths),
            'cached': cached_count,
            'pending': len(pending_paths),
            'pending_paths': pending_paths
        }

    def clear_cache(self, set_paths: Optional[List[str]] = None):
        """
        Очистить кэш для указанных файлов или всех.

        Args:
            set_paths: Конкретные файлы для очистки (None = все)
        """
        if set_paths:
            abs_paths = [os.path.abspath(p) for p in set_paths]
            for abs_path in abs_paths:
                self.cache_manager.invalidate_set_analysis(abs_path)
        else:
            self.cache_manager.clear_set_analysis_cache()

    def _result_from_dict(self, data: Dict) -> 'SetAnalysisResult':
        """
        Восстановить SetAnalysisResult из сохранённого dict.

        Реконструирует вложенные dataclasses (SegmentInfo, SegmentGenre, etc.)
        """
        # Vectorized: reconstruct segments
        raw_segments = data.get('segments', [])
        segments = [
            SegmentInfo(
                start_time=s['start_time'],
                end_time=s['end_time'],
                duration=s['duration'],
                segment_index=s['segment_index'],
                zone=s.get('zone'),
                features=s.get('features'),
                genre=SegmentGenre(
                    genre=s['genre'].get('genre', ''),
                    subgenre=s['genre'].get('subgenre', ''),
                    dj_category=s['genre'].get('dj_category', ''),
                    confidence=s['genre'].get('confidence', 0.0),
                    all_styles=s['genre'].get('all_styles', []),
                    mood_tags=s['genre'].get('mood_tags', []),
                ) if s.get('genre') else None,
                is_transition_zone=s.get('is_transition_zone', False),
            )
            for s in raw_segments
        ]

        # Reconstruct genre_distribution
        genre_dist = None
        gd = data.get('genre_distribution')
        if gd:
            genre_dist = SetGenreDistribution(
                primary_category=gd.get('primary_category', ''),
                primary_category_ratio=gd.get('primary_category_ratio', 0.0),
                category_distribution=gd.get('category_distribution', {}),
                top_subgenres=gd.get('top_subgenres', []),
                genre_diversity=gd.get('genre_diversity', 0.0),
                n_unique_subgenres=gd.get('n_unique_subgenres', 0),
                genre_flow=gd.get('genre_flow', []),
                genre_transitions=gd.get('genre_transitions', 0),
                mood_tags=gd.get('mood_tags', {}),
            )

        return SetAnalysisResult(
            file_path=data['file_path'],
            file_name=data['file_name'],
            duration_sec=data['duration_sec'],
            n_transitions=data['n_transitions'],
            transition_times=data['transition_times'],
            transition_density=data['transition_density'],
            n_segments=data['n_segments'],
            segments=segments,
            total_drops=data['total_drops'],
            drop_density=data['drop_density'],
            energy_timeline=data.get('energy_timeline'),
            genre_distribution=genre_dist,
            processing_time_sec=data.get('processing_time_sec', 0.0),
            success=data.get('success', True),
            error=data.get('error'),
        )

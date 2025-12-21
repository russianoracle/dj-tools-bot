"""
M2 Batch Processor - Parallel processing optimized for Apple Silicon.

Uses ProcessPoolExecutor for CPU-bound parallel processing.
"""

import os
import gc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Iterator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .track_analysis import TrackAnalysisPipeline, TrackAnalysisResult
from app.common.logging import get_logger

logger = get_logger(__name__)


# Set environment variables for M2 optimization
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')


@dataclass
class BatchResult:
    """
    Result of batch processing.

    Contains all individual results plus aggregate statistics.
    """
    results: List[TrackAnalysisResult]
    total_files: int
    successful: int
    failed: int
    total_time_sec: float
    avg_time_per_file: float

    # Zone distribution
    zone_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_files': self.total_files,
            'successful': self.successful,
            'failed': self.failed,
            'total_time_sec': self.total_time_sec,
            'avg_time_per_file': self.avg_time_per_file,
            'zone_counts': self.zone_counts,
            'results': [r.to_dict() for r in self.results]
        }

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.successful / self.total_files


def _process_single_file(args: tuple) -> TrackAnalysisResult:
    """
    Worker function for parallel processing.

    This runs in a separate process.
    """
    path, model_path, sr = args

    # Create pipeline in this process
    pipeline = TrackAnalysisPipeline(
        model_path=model_path,
        sr=sr
    )

    return pipeline.analyze(path)


class M2BatchProcessor:
    """
    Parallel batch processor optimized for Apple Silicon M2.

    Optimizations:
    - ProcessPoolExecutor with 4 workers (M2 performance cores)
    - Chunked processing for memory efficiency
    - Progress tracking
    - Error resilience

    Usage:
        pipeline = TrackAnalysisPipeline()
        processor = M2BatchProcessor(pipeline, workers=4)

        # Process directory
        results = processor.process_directory("/music/folder")

        # Process file list
        results = processor.process_files(file_list)

        # With progress callback
        def on_progress(done, total, result):
            print(f"{done}/{total}: {result.file_name}")

        results = processor.process_files(files, on_progress=on_progress)
    """

    def __init__(
        self,
        pipeline: Optional[TrackAnalysisPipeline] = None,
        workers: int = 4,
        chunk_size: int = 10,
        model_path: Optional[str] = None,
        sr: int = 22050
    ):
        """
        Initialize batch processor.

        Args:
            pipeline: Pipeline instance (creates default if None)
            workers: Number of parallel workers (default: 4 for M2)
            chunk_size: Files per chunk (for memory management)
            model_path: Path to zone classification model
            sr: Sample rate
        """
        self.pipeline = pipeline or TrackAnalysisPipeline(model_path=model_path, sr=sr)
        self.workers = workers
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.sr = sr

    def process_files(
        self,
        paths: List[str],
        on_progress: Optional[Callable[[int, int, TrackAnalysisResult], None]] = None,
        show_progress: bool = True
    ) -> BatchResult:
        """
        Process multiple files in parallel.

        Args:
            paths: List of audio file paths
            on_progress: Callback(done_count, total, result)
            show_progress: Print progress to stdout

        Returns:
            BatchResult with all results and statistics
        """
        start_time = time.time()
        results = []
        total = len(paths)
        done = 0

        # Process in chunks to manage memory
        for chunk_paths in self._chunked(paths, self.chunk_size):
            chunk_results = self._process_chunk(chunk_paths)

            for result in chunk_results:
                results.append(result)
                done += 1

                if on_progress:
                    on_progress(done, total, result)

                if show_progress:
                    status = "OK" if result.success else "FAIL"
                    logger.info("Track processed", data={"done": done, "total": total, "file": result.file_name, "zone": result.zone, "status": status})

            # Memory cleanup between chunks
            gc.collect()

        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        zone_counts = {}
        for r in results:
            if r.success:
                zone_counts[r.zone] = zone_counts.get(r.zone, 0) + 1

        return BatchResult(
            results=results,
            total_files=total,
            successful=successful,
            failed=failed,
            total_time_sec=total_time,
            avg_time_per_file=total_time / total if total > 0 else 0,
            zone_counts=zone_counts
        )

    def process_directory(
        self,
        directory: str,
        pattern: str = "*.mp3",
        recursive: bool = True,
        on_progress: Optional[Callable[[int, int, TrackAnalysisResult], None]] = None
    ) -> BatchResult:
        """
        Process all files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for files
            recursive: Search recursively
            on_progress: Progress callback

        Returns:
            BatchResult
        """
        dir_path = Path(directory)

        if recursive:
            paths = list(dir_path.rglob(pattern))
        else:
            paths = list(dir_path.glob(pattern))

        # Also include other audio formats
        for ext in ['*.wav', '*.flac', '*.m4a', '*.aac']:
            if ext != pattern:
                if recursive:
                    paths.extend(dir_path.rglob(ext))
                else:
                    paths.extend(dir_path.glob(ext))

        paths = [str(p) for p in paths]
        logger.info("Audio files discovered", data={"count": len(paths), "directory": directory})

        return self.process_files(paths, on_progress=on_progress)

    def _process_chunk(self, paths: List[str]) -> List[TrackAnalysisResult]:
        """Process a chunk of files in parallel."""
        if self.workers <= 1:
            # Sequential processing
            return [self.pipeline.analyze(p) for p in paths]

        # Parallel processing
        args_list = [(p, self.model_path, self.sr) for p in paths]

        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {
                executor.submit(_process_single_file, args): args[0]
                for args in args_list
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result
                    results.append(TrackAnalysisResult(
                        file_path=path,
                        file_name=Path(path).name,
                        duration_sec=0.0,
                        zone='unknown',
                        zone_confidence=0.0,
                        zone_scores={},
                        features={},
                        feature_count=0,
                        drop_count=0,
                        drop_density=0.0,
                        drop_times=[],
                        processing_time_sec=0.0,
                        success=False,
                        error=str(e)
                    ))

        return results

    def _chunked(self, lst: List, size: int) -> Iterator[List]:
        """Split list into chunks."""
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    def process_single(self, path: str) -> TrackAnalysisResult:
        """
        Process a single file (convenience method).

        Args:
            path: Audio file path

        Returns:
            TrackAnalysisResult
        """
        return self.pipeline.analyze(path)

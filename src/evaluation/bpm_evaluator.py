"""BPM accuracy evaluation against ground truth data."""

import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

from ..audio import AudioLoader, FeatureExtractor
from ..utils import get_logger, get_config

logger = get_logger(__name__)


@dataclass
class TrackResult:
    """Result of BPM detection for a single track."""

    # File info
    file_path: str
    track_title: str = ""
    artist: str = ""
    album: str = ""

    # BPM data
    expected_bpm: float = 0.0
    detected_bpm: float = 0.0
    tempo_confidence: float = 0.0

    # Error analysis
    error: float = 0.0
    error_type: str = "unknown"  # exact, close, octave_high, octave_low, large_error, failed
    octave_correction: str = "none"  # none, 2x, 0.5x

    # Processing info
    processing_time: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total_tracks: int = 0
    successful: int = 0
    failed: int = 0

    # Accuracy metrics
    exact_matches: int = 0  # ±2 BPM
    close_matches: int = 0  # ±5 BPM
    octave_errors: int = 0  # 2x or 0.5x
    large_errors: int = 0   # >5 BPM

    # Statistics
    mean_absolute_error: float = 0.0
    median_error: float = 0.0
    accuracy_rate: float = 0.0  # % within ±2 BPM
    octave_error_rate: float = 0.0

    # Processing performance
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0

    # Distribution
    bpm_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_tracks': self.total_tracks,
            'successful': self.successful,
            'failed': self.failed,
            'exact_matches': self.exact_matches,
            'close_matches': self.close_matches,
            'octave_errors': self.octave_errors,
            'large_errors': self.large_errors,
            'mean_absolute_error': self.mean_absolute_error,
            'median_error': self.median_error,
            'accuracy_rate': self.accuracy_rate,
            'octave_error_rate': self.octave_error_rate,
            'average_processing_time': self.average_processing_time,
            'total_processing_time': self.total_processing_time,
            'bpm_distribution': self.bpm_distribution
        }


class BPMAccuracyEvaluator:
    """Evaluates BPM detection accuracy against ground truth data."""

    def __init__(self, test_data_path: str, config: Any = None):
        """
        Initialize evaluator.

        Args:
            test_data_path: Path to test data file (TSV format)
            config: Configuration object (uses global config if None)
        """
        self.test_data_path = Path(test_data_path)
        self.config = config or get_config()

        # Initialize audio loader with sample rate from config
        sample_rate = self.config.get('audio.sample_rate', 22050)
        self.loader = AudioLoader(sample_rate=sample_rate)
        self.extractor = FeatureExtractor(config=self.config)

        logger.info(f"Initialized BPM evaluator with test data: {self.test_data_path}")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from TSV file.

        Expected format (UTF-16 TSV with header):
        Artwork | BPM | Key | Time | Bitrate | Track Title | Artist | Date Added | Album | Path

        Returns:
            List of dictionaries with test track information
        """
        logger.info(f"Loading test data from {self.test_data_path}")

        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_data_path}")

        tracks = []

        with open(self.test_data_path, 'r', encoding='utf-16') as f:
            # Read TSV file
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                try:
                    # Parse BPM (may have decimals like 131.00)
                    bpm_str = row.get('BPM', '').strip()
                    if not bpm_str:
                        logger.warning(f"Skipping row - no BPM: {row.get('Track Title', 'Unknown')}")
                        continue

                    bpm = float(bpm_str)

                    # Get file path (try multiple column names)
                    file_path = (row.get('Location', '').strip() or
                                row.get('Path', '').strip() or
                                row.get('путь', '').strip())
                    if not file_path:
                        logger.warning(f"Skipping row - no path: {row.get('Track Title', 'Unknown')}")
                        continue

                    track_info = {
                        'path': file_path,
                        'bpm': bpm,
                        'track_title': row.get('Track Title', '').strip(),
                        'artist': row.get('Artist', '').strip(),
                        'album': row.get('Album', '').strip(),
                        'key': row.get('Key', '').strip(),
                        'time': row.get('Time', '').strip()
                    }

                    tracks.append(track_info)

                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing row: {e} - {row}")
                    continue

        logger.info(f"Loaded {len(tracks)} tracks from test data")
        return tracks

    def validate_files(self, test_data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Check which files exist and which are missing.

        Args:
            test_data: List of track dictionaries from load_test_data()

        Returns:
            Tuple of (found_tracks, missing_tracks)
        """
        logger.info("Validating file paths...")

        found = []
        missing = []

        for track in test_data:
            file_path = Path(track['path'])

            if file_path.exists():
                found.append(track)
            else:
                missing.append(track)
                logger.warning(f"File not found: {file_path}")

        logger.info(f"Found {len(found)} files, missing {len(missing)} files")

        return found, missing

    def extract_bpm_from_tracks(self, tracks: List[Dict[str, Any]],
                                 progress_callback: Optional[callable] = None) -> List[TrackResult]:
        """
        Extract BPM from all tracks.

        Args:
            tracks: List of track dictionaries
            progress_callback: Optional callback function(current, total, track_name)

        Returns:
            List of TrackResult objects
        """
        logger.info(f"Starting BPM extraction for {len(tracks)} tracks...")

        results = []
        total = len(tracks)

        for idx, track in enumerate(tracks, 1):
            file_path = track['path']
            expected_bpm = track['bpm']

            if progress_callback:
                progress_callback(idx, total, Path(file_path).name)

            logger.info(f"[{idx}/{total}] Processing: {Path(file_path).name}")

            result = TrackResult(
                file_path=file_path,
                track_title=track.get('track_title', ''),
                artist=track.get('artist', ''),
                album=track.get('album', ''),
                expected_bpm=expected_bpm
            )

            try:
                start_time = time.time()

                # Load and extract features
                y, sr = self.loader.load(file_path)
                features = self.extractor.extract(y, sr)

                processing_time = time.time() - start_time

                # Store results
                result.detected_bpm = features.tempo
                result.tempo_confidence = features.tempo_confidence
                result.processing_time = processing_time
                result.success = True

                logger.info(f"  Detected: {features.tempo:.1f} BPM (confidence: {features.tempo_confidence:.0%})")
                logger.info(f"  Expected: {expected_bpm:.1f} BPM")
                logger.info(f"  Processing time: {processing_time:.2f}s")

            except Exception as e:
                logger.error(f"  Failed to process {file_path}: {e}")
                result.success = False
                result.error_message = str(e)
                result.error_type = "failed"

            results.append(result)

        logger.info("BPM extraction complete")
        return results

    def compare_results(self, results: List[TrackResult]) -> List[TrackResult]:
        """
        Compare detected BPM against expected values and classify errors.

        Args:
            results: List of TrackResult objects from extract_bpm_from_tracks()

        Returns:
            Updated list of TrackResult objects with error analysis
        """
        logger.info("Analyzing BPM detection errors...")

        for result in results:
            if not result.success:
                continue

            detected = result.detected_bpm
            expected = result.expected_bpm

            # Calculate errors for different octave scenarios
            error_normal = abs(detected - expected)
            error_half = abs(detected / 2 - expected)
            error_double = abs(detected * 2 - expected)

            # Find minimum error
            min_error = min(error_normal, error_half, error_double)
            result.error = min_error

            # Determine octave correction
            if error_normal == min_error:
                result.octave_correction = "none"
            elif error_half == min_error:
                result.octave_correction = "0.5x"
            else:
                result.octave_correction = "2x"

            # Classify error type
            if min_error <= 2:
                result.error_type = "exact"
            elif min_error <= 5:
                result.error_type = "close"
            elif result.octave_correction != "none":
                if result.octave_correction == "2x":
                    result.error_type = "octave_low"
                else:
                    result.error_type = "octave_high"
            else:
                result.error_type = "large_error"

            logger.debug(f"{result.track_title}: error={min_error:.1f} BPM, "
                        f"type={result.error_type}, octave={result.octave_correction}")

        return results

    def calculate_metrics(self, results: List[TrackResult]) -> EvaluationMetrics:
        """
        Calculate aggregated evaluation metrics.

        Args:
            results: List of TrackResult objects

        Returns:
            EvaluationMetrics object
        """
        logger.info("Calculating evaluation metrics...")

        metrics = EvaluationMetrics()
        metrics.total_tracks = len(results)

        successful_results = [r for r in results if r.success]
        metrics.successful = len(successful_results)
        metrics.failed = metrics.total_tracks - metrics.successful

        if not successful_results:
            logger.warning("No successful results to calculate metrics")
            return metrics

        # Count error types
        for result in successful_results:
            if result.error_type == "exact":
                metrics.exact_matches += 1
            elif result.error_type == "close":
                metrics.close_matches += 1
            elif result.error_type in ["octave_high", "octave_low"]:
                metrics.octave_errors += 1
            elif result.error_type == "large_error":
                metrics.large_errors += 1

        # Calculate statistics
        errors = [r.error for r in successful_results]
        metrics.mean_absolute_error = sum(errors) / len(errors)
        metrics.median_error = sorted(errors)[len(errors) // 2]

        # Rates
        metrics.accuracy_rate = (metrics.exact_matches / metrics.successful) * 100
        metrics.octave_error_rate = (metrics.octave_errors / metrics.successful) * 100

        # Processing times
        processing_times = [r.processing_time for r in successful_results]
        metrics.total_processing_time = sum(processing_times)
        metrics.average_processing_time = metrics.total_processing_time / len(processing_times)

        # BPM distribution
        metrics.bpm_distribution = self._calculate_bpm_distribution(successful_results)

        logger.info(f"Metrics calculated: {metrics.accuracy_rate:.1f}% accuracy, "
                   f"MAE={metrics.mean_absolute_error:.1f} BPM")

        return metrics

    def _calculate_bpm_distribution(self, results: List[TrackResult]) -> Dict[str, int]:
        """Calculate BPM range distribution."""
        ranges = {
            '60-90': 0,
            '90-110': 0,
            '110-128': 0,
            '128-140': 0,
            '140-160': 0,
            '160-180': 0,
            '180+': 0
        }

        for result in results:
            bpm = result.expected_bpm

            if bpm < 90:
                ranges['60-90'] += 1
            elif bpm < 110:
                ranges['90-110'] += 1
            elif bpm < 128:
                ranges['110-128'] += 1
            elif bpm < 140:
                ranges['128-140'] += 1
            elif bpm < 160:
                ranges['140-160'] += 1
            elif bpm < 180:
                ranges['160-180'] += 1
            else:
                ranges['180+'] += 1

        return ranges

    def run_evaluation(self, export_csv: Optional[str] = None,
                      show_errors: bool = False,
                      progress_callback: Optional[callable] = None) -> Tuple[List[TrackResult], EvaluationMetrics]:
        """
        Run complete evaluation cycle.

        Args:
            export_csv: Optional path to export detailed results CSV
            show_errors: If True, print detailed error analysis
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (results, metrics)
        """
        logger.info("="*70)
        logger.info("Starting BPM Accuracy Evaluation")
        logger.info("="*70)

        # Load test data
        test_data = self.load_test_data()

        # Validate files
        found_tracks, missing_tracks = self.validate_files(test_data)

        if missing_tracks:
            logger.warning(f"{len(missing_tracks)} files are missing")
            if show_errors:
                for track in missing_tracks:
                    logger.warning(f"  Missing: {track['path']}")

        if not found_tracks:
            raise ValueError("No valid files found for evaluation")

        # Extract BPM
        results = self.extract_bpm_from_tracks(found_tracks, progress_callback)

        # Compare results
        results = self.compare_results(results)

        # Calculate metrics
        metrics = self.calculate_metrics(results)

        # Export to CSV if requested
        if export_csv:
            self.export_to_csv(results, export_csv)

        # Show errors if requested
        if show_errors:
            self.print_error_details(results)

        return results, metrics

    def export_to_csv(self, results: List[TrackResult], output_path: str):
        """
        Export detailed results to CSV.

        Args:
            results: List of TrackResult objects
            output_path: Path to output CSV file
        """
        logger.info(f"Exporting results to {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'File Path', 'Track Title', 'Artist', 'Album',
                'Expected BPM', 'Detected BPM', 'Error', 'Error Type',
                'Octave Correction', 'Confidence', 'Processing Time',
                'Success', 'Error Message'
            ])

            # Data
            for result in results:
                writer.writerow([
                    result.file_path,
                    result.track_title,
                    result.artist,
                    result.album,
                    f"{result.expected_bpm:.2f}",
                    f"{result.detected_bpm:.2f}" if result.success else 'N/A',
                    f"{result.error:.2f}" if result.success else 'N/A',
                    result.error_type,
                    result.octave_correction,
                    f"{result.tempo_confidence:.2%}" if result.success else 'N/A',
                    f"{result.processing_time:.2f}s",
                    'Yes' if result.success else 'No',
                    result.error_message
                ])

        logger.info(f"Results exported to {output_path}")

    def print_report(self, metrics: EvaluationMetrics, results: List[TrackResult]):
        """
        Print evaluation report to console.

        Args:
            metrics: EvaluationMetrics object
            results: List of TrackResult objects
        """
        print("\n" + "="*70)
        print("BPM ACCURACY EVALUATION REPORT")
        print("="*70)

        # Overall statistics
        print(f"\nTotal tracks: {metrics.total_tracks}")
        print(f"Successfully processed: {metrics.successful}")
        print(f"Failed: {metrics.failed}")

        if metrics.successful == 0:
            print("\nNo successful results to report.")
            return

        # Accuracy metrics
        print(f"\n--- Accuracy Metrics ---")
        print(f"Exact matches (±2 BPM): {metrics.exact_matches}/{metrics.successful} ({metrics.accuracy_rate:.1f}%)")
        print(f"Close matches (±5 BPM): {metrics.close_matches}/{metrics.successful}")
        print(f"Octave errors: {metrics.octave_errors}/{metrics.successful} ({metrics.octave_error_rate:.1f}%)")
        print(f"Large errors (>5 BPM): {metrics.large_errors}/{metrics.successful}")

        # Error statistics
        print(f"\n--- Error Statistics ---")
        print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error:.2f} BPM")
        print(f"Median Error: {metrics.median_error:.2f} BPM")

        # Performance
        print(f"\n--- Processing Performance ---")
        print(f"Total processing time: {metrics.total_processing_time:.1f}s")
        print(f"Average time per track: {metrics.average_processing_time:.2f}s")

        # BPM distribution
        print(f"\n--- BPM Distribution ---")
        for range_name, count in sorted(metrics.bpm_distribution.items()):
            if count > 0:
                print(f"{range_name} BPM: {count} tracks")

        print("\n" + "="*70)

    def print_error_details(self, results: List[TrackResult], top_n: int = 10):
        """
        Print detailed error analysis.

        Args:
            results: List of TrackResult objects
            top_n: Number of worst errors to display
        """
        print(f"\n--- Top {top_n} Largest Errors ---")

        # Sort by error (descending)
        successful = [r for r in results if r.success]
        sorted_results = sorted(successful, key=lambda r: r.error, reverse=True)

        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. {Path(result.file_path).name}")
            print(f"   Expected: {result.expected_bpm:.1f} BPM")
            print(f"   Detected: {result.detected_bpm:.1f} BPM")
            print(f"   Error: {result.error:.1f} BPM ({result.error_type})")
            if result.octave_correction != "none":
                print(f"   Octave correction: {result.octave_correction}")
            print(f"   Confidence: {result.tempo_confidence:.0%}")

#!/usr/bin/env python3
"""
CLI tool for evaluating BPM detection accuracy against test dataset.

Usage:
    python scripts/evaluate_bpm_accuracy.py tests/test_data.txt
    python scripts/evaluate_bpm_accuracy.py tests/test_data.txt --export results.csv
    python scripts/evaluate_bpm_accuracy.py tests/test_data.txt --show-errors
    python scripts/evaluate_bpm_accuracy.py tests/test_data.txt --export results.csv --show-errors
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import BPMAccuracyEvaluator
from src.utils import get_logger

logger = get_logger(__name__)


def progress_callback(current: int, total: int, track_name: str):
    """Print progress to console."""
    percent = (current / total) * 100
    print(f"[{current}/{total}] ({percent:.1f}%) Processing: {track_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate BPM detection accuracy against test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'test_data',
        type=str,
        help='Path to test data file (TSV format with BPM and file paths)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export detailed results to CSV file'
    )

    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Show detailed error analysis (top 10 worst errors)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate test data file
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}", file=sys.stderr)
        sys.exit(1)

    print("="*70)
    print("BPM Accuracy Evaluator")
    print("="*70)
    print(f"Test data: {test_data_path}")
    if args.export:
        print(f"Export CSV: {args.export}")
    print()

    try:
        # Initialize evaluator
        evaluator = BPMAccuracyEvaluator(test_data_path=str(test_data_path))

        # Run evaluation
        callback = None if args.quiet else progress_callback
        results, metrics = evaluator.run_evaluation(
            export_csv=args.export,
            show_errors=args.show_errors,
            progress_callback=callback
        )

        # Print report
        evaluator.print_report(metrics, results)

        # Success message
        print("\nEvaluation complete!")
        if args.export:
            print(f"Detailed results exported to: {args.export}")

        # Exit with non-zero if accuracy is below 70%
        if metrics.accuracy_rate < 70:
            print(f"\nWARNING: Accuracy ({metrics.accuracy_rate:.1f}%) is below 70% threshold")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Evaluation failed")
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

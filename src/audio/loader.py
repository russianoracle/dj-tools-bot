"""Audio file loading and validation."""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import soundfile as sf

from ..utils import get_logger

logger = get_logger(__name__)


class AudioLoader:
    """Handles audio file loading with validation and error handling."""

    SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.mp4', '.ogg'}

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio loader.

        Args:
            sample_rate: Target sample rate for audio loading
        """
        self.sample_rate = sample_rate

    @classmethod
    def is_supported_format(cls, file_path: str) -> bool:
        """
        Check if file format is supported.

        Args:
            file_path: Path to audio file

        Returns:
            True if format is supported
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_FORMATS

    def load(
        self,
        file_path: str,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            file_path: Path to audio file
            duration: Duration to load in seconds (None = entire file)
            offset: Start offset in seconds

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            RuntimeError: If file cannot be loaded
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Validate format
        if not self.is_supported_format(str(file_path)):
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            logger.info(f"Loading audio: {file_path.name}")

            # Load audio with librosa
            y, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                duration=duration,
                offset=offset,
                mono=True  # Convert to mono
            )

            logger.info(
                f"Loaded {file_path.name}: {len(y)/sr:.2f}s, "
                f"{sr}Hz, {y.shape}"
            )

            return y, sr

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise RuntimeError(f"Failed to load audio file: {e}") from e

    def get_duration(self, file_path: str) -> float:
        """
        Get audio file duration without loading entire file.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If duration cannot be determined
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Use soundfile for fast duration detection
            info = sf.info(str(file_path))
            return info.duration

        except Exception as e:
            # Fallback to librosa
            try:
                duration = librosa.get_duration(path=str(file_path))
                return duration
            except Exception as e2:
                logger.error(f"Failed to get duration for {file_path}: {e2}")
                raise RuntimeError(f"Failed to get audio duration: {e2}") from e2

    def validate_file(self, file_path: str) -> bool:
        """
        Validate that audio file can be loaded.

        Args:
            file_path: Path to audio file

        Returns:
            True if file is valid and loadable
        """
        try:
            # Try to load first 1 second
            self.load(file_path, duration=1.0)
            return True
        except Exception as e:
            logger.warning(f"File validation failed for {file_path}: {e}")
            return False

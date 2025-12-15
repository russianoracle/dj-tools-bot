"""Audio file discovery utilities.

This module consolidates file discovery functions that were duplicated
across batch processing scripts.
"""

from pathlib import Path
from typing import List, Optional, Set, Union

# Supported audio formats
AUDIO_EXTENSIONS: Set[str] = {
    '.mp3',
    '.wav',
    '.flac',
    '.m4a',
    '.aac',
    '.ogg',
    '.opus',
    '.aiff',
    '.wma',
}


def find_audio_files(
    path: Union[str, Path],
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
    sort: bool = True,
) -> List[Path]:
    """Find all audio files in path.

    Args:
        path: File path or directory to search
        extensions: Set of extensions to match (default: AUDIO_EXTENSIONS)
        recursive: If True, search subdirectories
        sort: If True, sort results by path

    Returns:
        List of Path objects for found audio files

    Examples:
        >>> files = find_audio_files('/path/to/music')
        >>> files = find_audio_files('/path/to/music', extensions={'.mp3', '.wav'})
        >>> files = find_audio_files('/path/to/track.mp3')  # Returns single file
    """
    path = Path(path)
    extensions = extensions or AUDIO_EXTENSIONS

    # Normalize extensions to lowercase with leading dot
    extensions = {
        ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
        for ext in extensions
    }

    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        return []

    if not path.is_dir():
        return []

    files: List[Path] = []
    search_func = path.rglob if recursive else path.glob

    for ext in extensions:
        # Match both lowercase and uppercase extensions
        files.extend(search_func(f'*{ext}'))
        files.extend(search_func(f'*{ext.upper()}'))

    # Remove duplicates and sort
    files = list(set(files))
    if sort:
        files.sort()

    return files


def find_audio_files_with_filter(
    path: Union[str, Path],
    name_pattern: Optional[str] = None,
    min_size_mb: Optional[float] = None,
    max_size_mb: Optional[float] = None,
    extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """Find audio files with additional filtering.

    Args:
        path: File path or directory to search
        name_pattern: Glob pattern for filename (e.g., '*remix*')
        min_size_mb: Minimum file size in MB
        max_size_mb: Maximum file size in MB
        extensions: Set of extensions to match

    Returns:
        List of Path objects matching criteria
    """
    files = find_audio_files(path, extensions=extensions)

    if name_pattern:
        files = [f for f in files if Path(f.name).match(name_pattern)]

    if min_size_mb is not None:
        min_bytes = min_size_mb * 1024 * 1024
        files = [f for f in files if f.stat().st_size >= min_bytes]

    if max_size_mb is not None:
        max_bytes = max_size_mb * 1024 * 1024
        files = [f for f in files if f.stat().st_size <= max_bytes]

    return files


def get_relative_path(file_path: Path, base_path: Path) -> str:
    """Get relative path string from base.

    Args:
        file_path: Full path to file
        base_path: Base directory

    Returns:
        Relative path string, or filename if not relative
    """
    try:
        return str(file_path.relative_to(base_path))
    except ValueError:
        return file_path.name


def estimate_total_duration(files: List[Path]) -> float:
    """Estimate total duration from file sizes.

    Rough estimate: 1 MB ~= 1 minute for MP3 @ 128kbps

    Args:
        files: List of audio file paths

    Returns:
        Estimated total duration in seconds
    """
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    return total_mb * 60  # Rough MP3 estimate

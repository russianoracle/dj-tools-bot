"""
Metadata adapters for reading/writing classification to audio files.

Supports: MP3, M4A, MP4, FLAC, WAV
"""

from .reader import MetadataReader
from .writer import MetadataWriter

__all__ = ['MetadataReader', 'MetadataWriter']
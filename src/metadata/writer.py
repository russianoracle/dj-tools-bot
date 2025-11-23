"""Write classification results to audio file metadata."""

import shutil
from pathlib import Path
from typing import Optional
from mutagen import File
from mutagen.id3 import ID3, COMM, TXXX, TIT1
from mutagen.mp4 import MP4
from mutagen.flac import FLAC

from ..classification.classifier import ClassificationResult, EnergyZone
from ..utils import get_logger, get_config

logger = get_logger(__name__)


class MetadataWriter:
    """Writes classification results to audio file metadata."""

    def __init__(self, config: Any = None):
        """
        Initialize metadata writer.

        Args:
            config: Configuration object
        """
        if config is None:
            config = get_config()

        self.config = config
        self.use_comment = config.get('metadata.use_comment_field', True)
        self.use_grouping = config.get('metadata.use_grouping_field', True)
        self.use_genre = config.get('metadata.use_genre_field', False)
        self.custom_field = config.get('metadata.custom_field_name', 'ENERGYZONE')
        self.create_backup = config.get('metadata.create_backup', True)
        self.backup_suffix = config.get('metadata.backup_suffix', '.backup')

    def write(self, file_path: str, result: ClassificationResult) -> bool:
        """
        Write classification result to file metadata.

        Args:
            file_path: Path to audio file
            result: Classification result to write

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Create backup if enabled
        if self.create_backup:
            self._create_backup(file_path)

        try:
            # Determine file type and write accordingly
            suffix = file_path.suffix.lower()

            if suffix == '.mp3':
                return self._write_mp3(file_path, result)
            elif suffix in ['.m4a', '.mp4']:
                return self._write_mp4(file_path, result)
            elif suffix == '.flac':
                return self._write_flac(file_path, result)
            elif suffix == '.wav':
                return self._write_wav(file_path, result)
            else:
                logger.warning(f"Unsupported format for metadata writing: {suffix}")
                return False

        except Exception as e:
            logger.error(f"Failed to write metadata to {file_path}: {e}")
            return False

    def _create_backup(self, file_path: Path):
        """Create backup of original file."""
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)

        if not backup_path.exists():
            try:
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

    def _format_zone_text(self, result: ClassificationResult) -> str:
        """Format zone information as text."""
        return (
            f"{result.zone.emoji} Energy Zone: {result.zone.display_name} "
            f"(Confidence: {result.confidence:.1%}, Method: {result.method})"
        )

    def _write_mp3(self, file_path: Path, result: ClassificationResult) -> bool:
        """Write metadata to MP3 file."""
        try:
            audio = ID3(file_path)
        except:
            # Create new ID3 tag if doesn't exist
            audio = ID3()

        zone_text = self._format_zone_text(result)

        # Write to comment field
        if self.use_comment:
            audio.delall('COMM')
            audio.add(COMM(encoding=3, lang='eng', desc='', text=zone_text))

        # Write to grouping field (TIT1)
        if self.use_grouping:
            audio.delall('TIT1')
            audio.add(TIT1(encoding=3, text=result.zone.display_name))

        # Write custom field
        audio.delall(self.custom_field)
        audio.add(TXXX(encoding=3, desc=self.custom_field, text=str(result.zone.value)))

        audio.save(file_path)
        logger.info(f"Wrote metadata to MP3: {file_path.name}")
        return True

    def _write_mp4(self, file_path: Path, result: ClassificationResult) -> bool:
        """Write metadata to M4A/MP4 file."""
        audio = MP4(file_path)

        zone_text = self._format_zone_text(result)

        # Write to comment field
        if self.use_comment:
            audio['\xa9cmt'] = zone_text

        # Write to grouping field
        if self.use_grouping:
            audio['\xa9grp'] = result.zone.display_name

        # Write custom field
        audio['----:com.apple.iTunes:' + self.custom_field] = str(result.zone.value).encode('utf-8')

        audio.save()
        logger.info(f"Wrote metadata to M4A/MP4: {file_path.name}")
        return True

    def _write_flac(self, file_path: Path, result: ClassificationResult) -> bool:
        """Write metadata to FLAC file."""
        audio = FLAC(file_path)

        zone_text = self._format_zone_text(result)

        # FLAC uses Vorbis comments
        if self.use_comment:
            audio['COMMENT'] = zone_text

        if self.use_grouping:
            audio['GROUPING'] = result.zone.display_name

        # Custom field
        audio[self.custom_field] = str(result.zone.value)

        audio.save()
        logger.info(f"Wrote metadata to FLAC: {file_path.name}")
        return True

    def _write_wav(self, file_path: Path, result: ClassificationResult) -> bool:
        """Write metadata to WAV file."""
        try:
            # WAV files can have ID3 or INFO tags
            audio = File(file_path)

            if audio is None:
                logger.warning(f"Cannot write metadata to WAV: {file_path.name}")
                return False

            zone_text = self._format_zone_text(result)

            # Try to add tags
            if hasattr(audio, 'tags'):
                if self.use_comment:
                    audio.tags['COMMENT'] = zone_text
                if self.use_grouping:
                    audio.tags['GROUPING'] = result.zone.display_name
                audio.tags[self.custom_field] = str(result.zone.value)

                audio.save()
                logger.info(f"Wrote metadata to WAV: {file_path.name}")
                return True
            else:
                logger.warning(f"WAV file has no tag support: {file_path.name}")
                return False

        except Exception as e:
            logger.error(f"Failed to write WAV metadata: {e}")
            return False

"""Read classification results from audio file metadata."""

from pathlib import Path
from typing import Any, Optional
from mutagen import File

from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EnergyZone(str, Enum):
    """Energy zone classification for DJ tracks."""
    YELLOW = "yellow"  # Rest zone - low energy
    GREEN = "green"    # Transitional - medium energy
    PURPLE = "purple"  # Hits/energy - high energy

    @property
    def emoji(self) -> str:
        return {"yellow": "🟨", "green": "🟩", "purple": "🟪"}[self.value]

    @property
    def display_name(self) -> str:
        return self.value.capitalize()


def get_config():
    """Get default config dict."""
    return {
        'metadata.custom_field_name': 'ENERGYZONE',
        'metadata.use_comment_field': True,
        'metadata.use_grouping_field': True,
        'metadata.use_genre_field': False,
        'metadata.create_backup': True,
        'metadata.backup_suffix': '.backup',
    }


class MetadataReader:
    """Reads classification results from audio file metadata."""

    def __init__(self, config: Any = None):
        """
        Initialize metadata reader.

        Args:
            config: Configuration object
        """
        if config is None:
            config = get_config()

        self.config = config
        self.custom_field = config.get('metadata.custom_field_name', 'ENERGYZONE')

    def read_zone(self, file_path: str) -> Optional[EnergyZone]:
        """
        Read energy zone from file metadata.

        Args:
            file_path: Path to audio file

        Returns:
            EnergyZone if found, None otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            audio = File(file_path)

            if audio is None or audio.tags is None:
                logger.debug(f"No tags found in {file_path.name}")
                return None

            # Try to read custom field
            zone_value = self._get_tag_value(audio, self.custom_field)

            if zone_value:
                # Convert string to EnergyZone
                try:
                    return EnergyZone(zone_value.lower())
                except ValueError:
                    logger.warning(f"Invalid zone value: {zone_value}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Failed to read metadata from {file_path}: {e}")
            return None

    def _get_tag_value(self, audio, field_name: str) -> Optional[str]:
        """
        Get tag value from audio file.

        Args:
            audio: Mutagen audio file object
            field_name: Tag field name

        Returns:
            Tag value as string, or None if not found
        """
        tags = audio.tags

        # Try different tag access methods
        try:
            # Direct access
            if field_name in tags:
                value = tags[field_name]
                if isinstance(value, list):
                    return str(value[0])
                return str(value)

            # ID3 custom field (TXXX)
            if hasattr(tags, 'getall'):
                txxx_tags = tags.getall('TXXX')
                for tag in txxx_tags:
                    if hasattr(tag, 'desc') and tag.desc == field_name:
                        return str(tag.text[0]) if tag.text else None

            # MP4 custom field
            mp4_key = f'----:com.apple.iTunes:{field_name}'
            if mp4_key in tags:
                value = tags[mp4_key]
                if isinstance(value, list) and value:
                    return value[0].decode('utf-8') if isinstance(value[0], bytes) else str(value[0])

            return None

        except Exception as e:
            logger.debug(f"Error reading tag {field_name}: {e}")
            return None

    def has_classification(self, file_path: str) -> bool:
        """
        Check if file already has classification.

        Args:
            file_path: Path to audio file

        Returns:
            True if file has energy zone classification
        """
        return self.read_zone(file_path) is not None

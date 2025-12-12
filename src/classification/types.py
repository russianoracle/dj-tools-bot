"""Type definitions for classification module."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio.extractors import AudioFeatures


class EnergyZone(Enum):
    """Energy zone classifications."""
    YELLOW = "yellow"
    GREEN = "green"
    PURPLE = "purple"
    UNCERTAIN = "uncertain"

    def __str__(self):
        return self.value

    @property
    def emoji(self):
        """Get emoji for zone."""
        emoji_map = {
            EnergyZone.YELLOW: "🟨",
            EnergyZone.GREEN: "🟩",
            EnergyZone.PURPLE: "🟪",
            EnergyZone.UNCERTAIN: "❓"
        }
        return emoji_map[self]

    @property
    def display_name(self):
        """Get display name for zone."""
        name_map = {
            EnergyZone.YELLOW: "Yellow (Rest)",
            EnergyZone.GREEN: "Green (Transition)",
            EnergyZone.PURPLE: "Purple (Energy)",
            EnergyZone.UNCERTAIN: "Uncertain"
        }
        return name_map[self]


@dataclass
class ClassificationResult:
    """Result of energy zone classification."""
    zone: EnergyZone
    confidence: float
    features: 'AudioFeatures'
    method: str  # 'rule-based' or 'ml'

    def __str__(self):
        return f"{self.zone.emoji} {self.zone.display_name} (confidence: {self.confidence:.2%}, method: {self.method})"

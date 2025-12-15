"""
Job schemas - Job status and result types.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from enum import Enum


class JobState(str, Enum):
    """Job state enum."""
    PENDING = "PENDING"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"


@dataclass
class JobStatus:
    """Job status information."""
    state: JobState
    progress: int = 0
    status: str = ""
    result: Optional[Dict[str, Any]] = None

    @property
    def emoji(self) -> str:
        """Get emoji for state."""
        return {
            JobState.PENDING: "⏳",
            JobState.PROGRESS: "🔄",
            JobState.SUCCESS: "✅",
            JobState.FAILURE: "❌",
            JobState.ERROR: "❓",
        }.get(self.state, "❓")


@dataclass
class JobResult:
    """Analysis result from completed job."""
    job_id: str
    tracks: List[Dict] = None
    drops: List[Dict] = None
    transitions: List[Dict] = None
    genres: Dict[str, float] = None
    duration_seconds: float = 0.0

    @classmethod
    def from_dict(cls, job_id: str, data: Dict) -> 'JobResult':
        """Create from dictionary."""
        return cls(
            job_id=job_id,
            tracks=data.get('tracks', []),
            drops=data.get('drops', []),
            transitions=data.get('transitions', []),
            genres=data.get('genres', {}),
            duration_seconds=data.get('duration_seconds', 0.0),
        )

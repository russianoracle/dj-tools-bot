"""Time and value formatting utilities.

This module consolidates formatting functions that were duplicated
across 9+ script files.
"""

from typing import Optional


def format_time(seconds: float, include_hours: bool = True) -> str:
    """Format seconds as MM:SS or HH:MM:SS.

    Args:
        seconds: Time in seconds (can be float)
        include_hours: If True, include hours when > 0

    Returns:
        Formatted time string

    Examples:
        >>> format_time(65.5)
        '01:05'
        >>> format_time(3665.5)
        '01:01:05'
        >>> format_time(3665.5, include_hours=False)
        '61:05'
    """
    if seconds < 0:
        return "-" + format_time(-seconds, include_hours)

    total_secs = int(seconds)
    mins, secs = divmod(total_secs, 60)

    if include_hours and mins >= 60:
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    return f"{mins:02d}:{secs:02d}"


def format_time_range(start_sec: float, end_sec: float) -> str:
    """Format a time range as 'MM:SS - MM:SS'.

    Args:
        start_sec: Start time in seconds
        end_sec: End time in seconds

    Returns:
        Formatted time range string
    """
    return f"{format_time(start_sec)} - {format_time(end_sec)}"


def format_duration(seconds: float) -> str:
    """Format duration with appropriate units.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string

    Examples:
        >>> format_duration(0.5)
        '500ms'
        >>> format_duration(65)
        '1m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"

    total_secs = int(seconds)

    if total_secs < 60:
        return f"{total_secs}s"

    mins, secs = divmod(total_secs, 60)

    if mins < 60:
        if secs:
            return f"{mins}m {secs}s"
        return f"{mins}m"

    hours, mins = divmod(mins, 60)
    parts = [f"{hours}h"]
    if mins:
        parts.append(f"{mins}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts)


def format_bpm(bpm: float, precision: int = 1) -> str:
    """Format BPM value.

    Args:
        bpm: Beats per minute
        precision: Decimal places (default 1)

    Returns:
        Formatted BPM string
    """
    return f"{bpm:.{precision}f} BPM"


def format_percent(value: float, precision: int = 1) -> str:
    """Format value as percentage.

    Args:
        value: Value between 0 and 1
        precision: Decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def format_confidence(confidence: float) -> str:
    """Format confidence value with visual indicator.

    Args:
        confidence: Confidence value 0-1

    Returns:
        Formatted string with indicator
    """
    pct = confidence * 100
    if confidence >= 0.8:
        indicator = "HIGH"
    elif confidence >= 0.5:
        indicator = "MED"
    else:
        indicator = "LOW"
    return f"{pct:.0f}% ({indicator})"

"""Small utilities shared across modules."""

from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Return the current local timestamp as an ISO 8601 string (seconds)."""
    return datetime.now().isoformat(timespec="seconds")


def pct(value: float) -> str:
    """Format a 0.0-1.0 similarity value as a human-readable percent."""
    return f"{value * 100:.1f}%"

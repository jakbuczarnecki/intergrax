# © Artur Czarnecki. All rights reserved.

"""Deterministic chronological ordering for memory timestamps (MEM-ENT-6R2 / MEM-ENT-7R)."""

from __future__ import annotations

from datetime import datetime


def memory_timestamps_same_awareness(left: datetime, right: datetime) -> bool:
    """Return True when both timestamps are aware or both are naive."""
    return (left.tzinfo is not None) == (right.tzinfo is not None)


def naive_chronological_ordinal(dt: datetime) -> float:
    """Host-independent sort key for naive datetimes (no ``timestamp()``)."""
    seconds_since_midnight = (
        dt.hour * 3600.0
        + dt.minute * 60.0
        + dt.second
        + dt.microsecond / 1_000_000.0
    )
    return dt.toordinal() * 86400.0 + seconds_since_midnight


def memory_chronological_ordinal(dt: datetime) -> float:
    """Sortable instant key: aware uses ``timestamp()``, naive uses ordinal arithmetic."""
    if dt.tzinfo is not None:
        return dt.timestamp()
    return naive_chronological_ordinal(dt)

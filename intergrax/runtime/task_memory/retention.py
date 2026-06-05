# © Artur Czarnecki. All rights reserved.

"""Retention helpers for memory stores (Phase MEM-6.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def is_record_expired(updated_at_utc: str, *, retention_days: int | None) -> bool:
    """Return True when ``updated_at_utc`` is older than ``retention_days``."""
    if retention_days is None or retention_days < 1:
        return False
    try:
        updated = datetime.fromisoformat(updated_at_utc)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    return updated < cutoff

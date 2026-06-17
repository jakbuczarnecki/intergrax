# © Artur Czarnecki. All rights reserved.

"""Temporal validity helpers for LTM entries (MEM-DEPTH-5.2 enforcement)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.utils.time_provider import SystemTimeProvider


def _parse_iso_timestamp(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_memory_entry_active(
    entry: UserProfileMemoryEntry,
    *,
    as_of: datetime | None = None,
) -> bool:
    """Return True when entry is not deleted and within valid_from / valid_until window."""
    if entry.deleted:
        return False
    now = as_of or SystemTimeProvider.utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    valid_from = _parse_iso_timestamp(entry.valid_from or "")
    if valid_from is not None and now < valid_from:
        return False

    valid_until = _parse_iso_timestamp(entry.valid_until or "")
    if valid_until is not None and now >= valid_until:
        return False
    return True


def filter_active_memory_entries(
    entries: list[UserProfileMemoryEntry],
    *,
    as_of: datetime | None = None,
) -> list[UserProfileMemoryEntry]:
    return [entry for entry in entries if is_memory_entry_active(entry, as_of=as_of)]

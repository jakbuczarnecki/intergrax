# © Artur Czarnecki. All rights reserved.

"""Task priority for queue broker adapters (ORCH-MAINT-03)."""

from __future__ import annotations

from enum import IntEnum


class TaskPriority(IntEnum):
    """Lower numeric value = higher scheduling priority."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

    @classmethod
    def coerce(cls, raw: int | str | TaskPriority | None) -> TaskPriority:
        if raw is None:
            return cls.NORMAL
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, int):
            return cls(raw)
        normalized = str(raw).strip().upper()
        return cls[normalized]

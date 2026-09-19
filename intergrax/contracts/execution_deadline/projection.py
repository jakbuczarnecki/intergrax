# © Artur Czarnecki. All rights reserved.

"""Process-local execution deadline projection derived from durable authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ExecutionDeadlineProjection:
    """Immutable view of effective deadline for the active execution scope."""

    deadline_at_utc: datetime | None
    remaining_seconds: float
    is_expired: bool
    global_deadline_monotonic: float | None

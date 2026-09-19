# © Artur Czarnecki. All rights reserved.

"""Process-local execution deadline projection derived from durable authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ExecutionDeadlineProjection:
    """Immutable view of effective deadline for the active execution scope.

    ``remaining_seconds`` and ``is_expired`` are bind-time snapshots for diagnostics
    and observability only. Runtime pre-effect admission must use live monotonic
    evaluation (see ``intergrax.runtime.execution.live_deadline_evaluator``).
    """

    deadline_at_utc: datetime | None
    remaining_seconds: float
    is_expired: bool
    global_deadline_monotonic: float | None

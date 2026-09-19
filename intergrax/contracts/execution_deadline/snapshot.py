# © Artur Czarnecki. All rights reserved.

"""Durable execution deadline authority snapshot (HARNESS-02 ADR1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import RunId

EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExecutionDeadlineAuthoritySnapshot:
    """Immutable durable deadline authority for one run."""

    schema_version: int
    run_id: RunId
    deadline_at_utc: datetime | None
    authority_created_at_utc: datetime
    policy_max_wall_time_seconds: float | None

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("unsupported execution deadline authority schema version")
        if self.deadline_at_utc is not None and self.deadline_at_utc.tzinfo is None:
            raise ValueError("deadline_at_utc must be timezone-aware")
        if self.authority_created_at_utc.tzinfo is None:
            raise ValueError("authority_created_at_utc must be timezone-aware")
        if self.policy_max_wall_time_seconds is not None and self.policy_max_wall_time_seconds <= 0:
            raise ValueError("policy_max_wall_time_seconds must be positive when set")

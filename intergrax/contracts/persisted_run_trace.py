# © Artur Czarnecki. All rights reserved.

"""Neutral persisted execution run trace models (tool harness / replay consumers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RunStats:
    duration_ms: int
    llm_usage: Mapping[str, object]


@dataclass(frozen=True)
class RunError:
    error_type: str
    message: str


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    session_id: str
    user_id: str
    tenant_id: str
    started_at_utc: str
    stats: RunStats
    error: RunError | None = None


@dataclass(frozen=True)
class PersistedRun:
    metadata: RunMetadata
    events: list[Mapping[str, object]]


@dataclass(frozen=True)
class RunSummary:
    """Lightweight run row for harness list operations."""

    run_id: str
    tenant_id: str
    user_id: str
    session_id: str
    started_at_utc: str
    duration_ms: int
    event_count: int


__all__ = [
    "PersistedRun",
    "RunError",
    "RunMetadata",
    "RunStats",
    "RunSummary",
]

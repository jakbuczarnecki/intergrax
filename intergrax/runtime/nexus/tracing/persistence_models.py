# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent


@dataclass(frozen=True)
class SerializedTraceEvent:
    event_id: str
    run_id: str
    seq: int
    ts_utc: str
    level: str
    component: str
    step: str
    message: str
    payload_schema_id: Optional[str]
    payload_schema_version: Optional[int]
    payload: Optional[Dict[str, Any]]
    tags: Dict[str, Any]

    @classmethod
    def from_trace_event(cls, event: TraceEvent) -> "SerializedTraceEvent":
        data = event.to_dict()
        return cls(
            event_id=data["event_id"],
            run_id=data["run_id"],
            seq=data["seq"],
            ts_utc=data["ts_utc"],
            level=data["level"],
            component=data["component"],
            step=data["step"],
            message=data["message"],
            payload_schema_id=data.get("payload_schema_id"),
            payload_schema_version=data.get("payload_schema_version"),
            payload=data.get("payload"),
            tags=data.get("tags", {}),
        )


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    session_id: str
    user_id: str
    tenant_id: str
    started_at_utc: str
    stats: RunStats
    error: Optional[RunError]=None


@dataclass(frozen=True)
class RunStats:
    duration_ms: int
    llm_usage: Dict[str, Any]


@dataclass(frozen=True)
class RunError:
    error_type: RuntimeErrorCode
    message: str


@dataclass(frozen=True)
class PersistedRun:
    metadata: RunMetadata
    events: List[Dict[str, Any]]  # Serialized TraceEvent dicts


class RunTraceWriter(Protocol):
    def append_event(self, event: TraceEvent) -> None:
        ...

    def finalize_run(self, run_id: str, metadata: RunMetadata) -> None:
        ...


class RunTraceReader(Protocol):
    def read_run(self, run_id: str) -> PersistedRun:
        ...

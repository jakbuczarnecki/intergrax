# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations  # noqa: I001

from abc import ABC, abstractmethod  # noqa: F401
from dataclasses import dataclass
from typing import Any, Dict, List, Optional  # noqa: F401, UP035
from intergrax.runtime.nexus.artifacts.models import TraceArtifactRef
from intergrax.contracts.tracing import TraceEvent


@dataclass(frozen=True)
class SerializedArtifactRef:
    artifact_id: str
    kind: str
    size_bytes: int

    @classmethod
    def from_ref(cls, ref: TraceArtifactRef) -> SerializedArtifactRef:
        return cls(
            artifact_id=ref.artifact_id,
            kind=ref.kind,
            size_bytes=ref.size_bytes,
        )


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
    payload_schema_id: str | None
    payload_schema_version: int | None
    payload: dict[str, Any] | None
    tags: dict[str, Any]
    artifact_refs: list[SerializedArtifactRef]

    @classmethod
    def from_trace_event(cls, event: TraceEvent) -> SerializedTraceEvent:
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
            artifact_refs=[SerializedArtifactRef.from_ref(r) for r in event.artifact_refs],
        )


from intergrax.contracts.persisted_run_trace import (  # noqa: E402
    PersistedRun,  # noqa: F401
    RunError,  # noqa: F401
    RunMetadata,  # noqa: F401
    RunStats,  # noqa: F401
    RunSummary,  # noqa: F401
)
from intergrax.contracts.run_trace_store import (  # noqa: E402
    RunTraceReader,  # noqa: F401
    RunTraceStore,  # noqa: F401
    RunTraceWriter,  # noqa: F401
)

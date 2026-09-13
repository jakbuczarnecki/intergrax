# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plane B trace read-model contracts (Harness Observability Spine)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any
import uuid

from intergrax.contracts.tracing.diagnostics import DiagnosticPayload


class TraceLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceComponent(str, Enum):
    RUNTIME = "runtime"
    ENGINE = "engine"
    PIPELINE = "pipeline"
    STEP = "step"
    POLICY = "policy"
    TOOLS = "tools"
    WEBSEARCH = "websearch"
    RAG = "rag"
    MEMORY = "memory"
    PLANNER = "planner"
    CRITIC = "critic"
    CODECRAFT = "codecraft"


@dataclass(frozen=True, slots=True)
class TraceArtifactRef:
    """Lightweight artifact pointer embedded in trace events (retrieval via artifact store)."""

    artifact_id: str
    kind: str
    size_bytes: int


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    run_id: str
    seq: int
    ts_utc: str

    level: TraceLevel
    component: TraceComponent
    step: str
    message: str

    payload: DiagnosticPayload | None = None

    tags: dict[str, Any] = field(default_factory=dict)

    artifact_refs: tuple[TraceArtifactRef, ...] = ()

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-safe serialization for notebooks/tests/log export.

        Notes:
        - Enums are serialized to their `.value`.
        - payload is exported as:
            - payload_schema_id / payload_schema_version computed from payload classmethods
            - payload = payload.to_dict()
        - tags is kept as-is (must be JSON-safe by convention).
        """
        payload_schema_id: str | None = None
        payload_schema_version: int | None = None
        payload_dict: dict[str, Any] | None = None

        if self.payload is not None:
            payload_schema_id = self.payload.__class__.schema_id()
            payload_schema_version = self.payload.__class__.schema_version()
            payload_dict = self.payload.to_dict()

        artifact_refs = [
            {
                "artifact_id": r.artifact_id,
                "kind": r.kind,
                "size_bytes": r.size_bytes,
            }
            for r in self.artifact_refs
        ]

        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts_utc": self.ts_utc,
            "level": self.level.value,
            "component": self.component.value,
            "step": self.step,
            "message": self.message,
            "payload_schema_id": payload_schema_id,
            "payload_schema_version": payload_schema_version,
            "payload": payload_dict,
            "tags": self.tags,
            "artifact_refs": artifact_refs,
        }

    def with_redacted_payload(self) -> TraceEvent:
        """Copy with :meth:`DiagnosticPayload.redact` applied when a payload is present."""
        if self.payload is None:
            return self
        return replace(self, payload=self.payload.redact())


@dataclass(frozen=True)
class ToolCallTrace:
    """
    Typed runtime artifact describing a single executed tool call.

    Notes:
    - This is NOT a DiagnosticPayload (not emitted to trace_events directly).
    - It is used to build RuntimeAnswer.tool_calls (API-facing).
    - Keep fields JSON-friendly and stable.
    """

    tool_name: str
    arguments: dict[str, Any]
    output_preview: str | None
    success: bool
    error_message: str | None
    raw_trace: dict[str, Any]

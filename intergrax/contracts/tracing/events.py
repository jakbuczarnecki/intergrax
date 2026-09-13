# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plane B trace read-model contracts (Harness Observability Spine)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import uuid

from intergrax.contracts.tracing.diagnostics import DiagnosticPayload
from intergrax.contracts.tracing.values import (
    TraceObject,
    TraceValue,
    normalize_trace_object,
    normalize_trace_tags,
)


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

    tags: TraceObject = field(default_factory=dict)

    artifact_refs: tuple[TraceArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", normalize_trace_tags(self.tags))

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def to_dict(self) -> TraceObject:
        """
        JSON-safe serialization for notebooks/tests/log export.

        Notes:
        - Enums are serialized to their `.value`.
        - payload is exported as:
            - payload_schema_id / payload_schema_version computed from payload classmethods
            - payload = payload.serialized_dict()
        - tags are validated JSON-safe trace attributes.
        """
        payload_schema_id: str | None = None
        payload_schema_version: int | None = None
        payload_dict: TraceObject | None = None

        if self.payload is not None:
            payload_schema_id = self.payload.__class__.schema_id()
            payload_schema_version = self.payload.__class__.schema_version()
            payload_dict = self.payload.serialized_dict()

        artifact_refs: list[TraceValue] = [
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
            "tags": dict(self.tags),
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
    - Public for runtime/API consumers; import from ``intergrax.contracts.tracing``
      or the Nexus compatibility re-export.
    """

    tool_name: str
    arguments: TraceObject
    output_preview: str | None
    success: bool
    error_message: str | None
    raw_trace: TraceObject

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            normalize_trace_object(self.arguments, field_name="arguments"),
        )
        object.__setattr__(
            self,
            "raw_trace",
            normalize_trace_object(self.raw_trace, field_name="raw_trace"),
        )

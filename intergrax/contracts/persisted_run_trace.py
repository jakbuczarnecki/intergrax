# © Artur Czarnecki. All rights reserved.

"""Neutral persisted execution run trace models (tool harness / replay consumers)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.structured_json_value import (
    StructuredJsonObject,
    normalize_structured_json_object,
)


class PersistedRunErrorCode(str, Enum):
    """Neutral persisted-run failure codes (wire-aligned with execution runtime taxonomy)."""

    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    USER_ERROR = "user_error"
    POLICY_ERROR = "policy_error"
    DEPENDENCY_ERROR = "dependency_error"
    QUALITY_ERROR = "quality_error"
    PERMISSION_ERROR = "permission_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


_PERSISTED_RUN_ERROR_WIRE_BY_CODE = {member.value: member for member in PersistedRunErrorCode}


def parse_persisted_run_error_code(raw: str) -> PersistedRunErrorCode:
    normalized = str(raw or "").strip().lower()
    if not normalized:
        return PersistedRunErrorCode.UNKNOWN
    return _PERSISTED_RUN_ERROR_WIRE_BY_CODE.get(normalized, PersistedRunErrorCode.UNKNOWN)


@dataclass(frozen=True)
class RunStats:
    duration_ms: int
    llm_usage: StructuredJsonObject


@dataclass(frozen=True)
class RunError:
    error_type: PersistedRunErrorCode
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
class PersistedTraceArtifactRef:
    artifact_id: str
    kind: str
    size_bytes: int


@dataclass(frozen=True)
class PersistedTraceEvent:
    event_id: str
    run_id: str
    seq: int
    ts_utc: str
    level: str
    component: str
    step: str
    message: str
    payload: StructuredJsonObject | None = None
    tags: StructuredJsonObject | None = None
    payload_schema_id: str | None = None
    payload_schema_version: int | None = None
    artifact_refs: tuple[PersistedTraceArtifactRef, ...] = ()


@dataclass(frozen=True)
class PersistedRun:
    metadata: RunMetadata
    events: list[PersistedTraceEvent]


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


def run_stats_to_storage_dict(stats: RunStats) -> dict[str, StructuredJsonObject | int]:
    return {
        "duration_ms": stats.duration_ms,
        "llm_usage": dict(stats.llm_usage),
    }


def decode_persisted_run_stats(raw: StructuredJsonObject) -> RunStats:
    duration_raw = raw.get("duration_ms", 0)
    if not isinstance(duration_raw, int) or isinstance(duration_raw, bool):
        raise ValueError("duration_ms must be a non-bool int")
    llm_raw = raw.get("llm_usage", {})
    if not isinstance(llm_raw, Mapping):
        llm_raw = {}
    return RunStats(
        duration_ms=duration_raw,
        llm_usage=normalize_structured_json_object(llm_raw, field_name="llm_usage"),
    )


def run_error_to_storage_dict(error: RunError) -> dict[str, str]:
    return {
        "error_type": error.error_type.value,
        "message": error.message,
    }


def decode_persisted_run_error(raw: StructuredJsonObject) -> RunError:
    error_type_raw = raw.get("error_type", "")
    if isinstance(error_type_raw, PersistedRunErrorCode):
        code = error_type_raw
    else:
        code = parse_persisted_run_error_code(str(error_type_raw))
    message_raw = raw.get("message", "")
    return RunError(error_type=code, message=str(message_raw))


def persisted_trace_event_to_wire(event: PersistedTraceEvent) -> StructuredJsonObject:
    wire: StructuredJsonObject = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "seq": event.seq,
        "ts_utc": event.ts_utc,
        "level": event.level,
        "component": event.component,
        "step": event.step,
        "message": event.message,
        "payload": event.payload,
        "tags": event.tags if event.tags is not None else {},
        "artifact_refs": [
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind,
                "size_bytes": ref.size_bytes,
            }
            for ref in event.artifact_refs
        ],
    }
    if event.payload_schema_id is not None:
        wire["payload_schema_id"] = event.payload_schema_id
    if event.payload_schema_version is not None:
        wire["payload_schema_version"] = event.payload_schema_version
    return wire


def decode_persisted_trace_event(raw: StructuredJsonObject) -> PersistedTraceEvent:
    event_id = raw.get("event_id")
    run_id = raw.get("run_id")
    seq = raw.get("seq")
    if not isinstance(event_id, str) or not isinstance(run_id, str):
        raise ValueError("event_id and run_id must be strings")
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ValueError("seq must be a non-bool int")

    payload: StructuredJsonObject | None = None
    payload_raw = raw.get("payload")
    if payload_raw is not None:
        if not isinstance(payload_raw, Mapping):
            raise ValueError("payload must be a JSON object when present")
        payload = normalize_structured_json_object(payload_raw, field_name="payload")

    tags: StructuredJsonObject | None = None
    tags_raw = raw.get("tags")
    if tags_raw is not None:
        if not isinstance(tags_raw, Mapping):
            raise ValueError("tags must be a JSON object when present")
        tags = normalize_structured_json_object(tags_raw, field_name="tags")

    artifact_refs: list[PersistedTraceArtifactRef] = []
    refs_raw = raw.get("artifact_refs")
    if isinstance(refs_raw, list):
        for item in refs_raw:
            if not isinstance(item, Mapping):
                continue
            artifact_id = item.get("artifact_id")
            kind = item.get("kind")
            size_bytes = item.get("size_bytes")
            if (
                isinstance(artifact_id, str)
                and isinstance(kind, str)
                and isinstance(size_bytes, int)
                and not isinstance(size_bytes, bool)
            ):
                artifact_refs.append(
                    PersistedTraceArtifactRef(
                        artifact_id=artifact_id,
                        kind=kind,
                        size_bytes=size_bytes,
                    )
                )

    payload_schema_id = raw.get("payload_schema_id")
    payload_schema_version = raw.get("payload_schema_version")

    return PersistedTraceEvent(
        event_id=event_id,
        run_id=run_id,
        seq=seq,
        ts_utc=str(raw.get("ts_utc", "")),
        level=str(raw.get("level", "")),
        component=str(raw.get("component", "")),
        step=str(raw.get("step", "")),
        message=str(raw.get("message", "")),
        payload=payload,
        tags=tags,
        payload_schema_id=payload_schema_id if isinstance(payload_schema_id, str) else None,
        payload_schema_version=payload_schema_version
        if isinstance(payload_schema_version, int) and not isinstance(payload_schema_version, bool)
        else None,
        artifact_refs=tuple(artifact_refs),
    )


__all__ = [
    "PersistedRun",
    "PersistedRunErrorCode",
    "PersistedTraceArtifactRef",
    "PersistedTraceEvent",
    "RunError",
    "RunMetadata",
    "RunStats",
    "RunSummary",
    "decode_persisted_run_error",
    "decode_persisted_run_stats",
    "decode_persisted_trace_event",
    "parse_persisted_run_error_code",
    "persisted_trace_event_to_wire",
    "run_error_to_storage_dict",
    "run_stats_to_storage_dict",
]

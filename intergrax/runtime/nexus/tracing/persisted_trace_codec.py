# © Artur Czarnecki. All rights reserved.

"""Nexus-internal adapters between trace persistence and neutral persisted run contracts."""

from __future__ import annotations

from intergrax.contracts.persisted_run_trace import (
    PersistedTraceEvent,
    decode_persisted_trace_event,
    persisted_trace_event_to_wire,
)
from intergrax.runtime.nexus.tracing.persistence_models import SerializedArtifactRef, SerializedTraceEvent


def serialized_trace_event_to_persisted(event: SerializedTraceEvent) -> PersistedTraceEvent:
    wire = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "seq": event.seq,
        "ts_utc": event.ts_utc,
        "level": event.level,
        "component": event.component,
        "step": event.step,
        "message": event.message,
        "payload_schema_id": event.payload_schema_id,
        "payload_schema_version": event.payload_schema_version,
        "payload": event.payload,
        "tags": event.tags,
        "artifact_refs": [
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind,
                "size_bytes": ref.size_bytes,
            }
            for ref in event.artifact_refs
        ],
    }
    return decode_persisted_trace_event(wire)


def persisted_trace_event_to_serialized(event: PersistedTraceEvent) -> SerializedTraceEvent:
    wire = persisted_trace_event_to_wire(event)
    artifact_wire = wire.get("artifact_refs")
    artifact_refs: list[SerializedArtifactRef] = []
    if isinstance(artifact_wire, list):
        for item in artifact_wire:
            if not isinstance(item, dict):
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
                    SerializedArtifactRef(
                        artifact_id=artifact_id,
                        kind=kind,
                        size_bytes=size_bytes,
                    )
                )
    payload = wire.get("payload")
    tags = wire.get("tags")
    return SerializedTraceEvent(
        event_id=str(wire["event_id"]),
        run_id=str(wire["run_id"]),
        seq=int(wire["seq"]),
        ts_utc=str(wire["ts_utc"]),
        level=str(wire["level"]),
        component=str(wire["component"]),
        step=str(wire["step"]),
        message=str(wire["message"]),
        payload_schema_id=wire.get("payload_schema_id") if isinstance(wire.get("payload_schema_id"), str) else None,
        payload_schema_version=wire.get("payload_schema_version")
        if isinstance(wire.get("payload_schema_version"), int) and not isinstance(wire.get("payload_schema_version"), bool)
        else None,
        payload=dict(payload) if isinstance(payload, dict) else None,
        tags=dict(tags) if isinstance(tags, dict) else {},
        artifact_refs=artifact_refs,
    )

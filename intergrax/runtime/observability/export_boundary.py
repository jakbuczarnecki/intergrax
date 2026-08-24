# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Normalized observability export boundary (OBS-EXPORT-1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord, ToolCallRecord
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    ObservabilityArtifactReference,
    SanitizedApplicationObservabilityAttributes,
)
from intergrax.runtime.observability.journal_export import JournalRef

OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA = "observability_export_envelope.v1"

FORBIDDEN_EXPORT_CONTENT_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "completion",
        "message",
        "messages",
        "content",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
        "query",
        "query_text",
        "text",
        "body",
        "input",
        "output",
        "args",
        "arguments",
        "tool_args",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "source_path",
        "absolute_path",
        "file_path",
        "synthesized_content",
        "redacted_input_summary",
    }
)

_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "tool_id",
        "capability",
        "latency_ms",
        "duration_ms",
        "hit_count",
        "error_code",
        "policy_rule_id",
        "args_digest",
        "collection_id",
        "payload_schema_id",
        "schema_id",
        "event_count",
        "parser_trace_count",
        "status",
    }
)

class ExportRecordKind(StrEnum):
    RUNTIME_EVENT = "runtime_event"
    TOOL_CALL = "tool_call"
    RAG_CALL = "rag_call"
    LLM_CALL = "llm_call"
    DIAGNOSTIC = "diagnostic"
    JOURNAL_REF = "journal_ref"
    PROBLEM_SIGNAL = "problem_signal"
    PLATFORM_SIGNAL = "platform_signal"


class ExportStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class ObservabilityExportEnvelope(BaseModel):
    """Vendor-neutral, redacted-by-default observability export record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["observability_export_envelope.v1"] = OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA
    record_kind: ExportRecordKind
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    run_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    capability: str = ""
    tool_id: str = ""
    event_type: str = ""
    status: ExportStatus = ExportStatus.UNKNOWN

    latency_ms: int | None = None
    counts: dict[str, int] = Field(default_factory=dict)

    artifact_ref: str = ""
    sha256: str = ""
    safe_relative_path: str = ""
    schema_id: str = ""

    tenant_id: str = ""
    workspace_id: str = ""

    source_schema_id: str = ""
    correlation_id: str = ""
    event_id: str = ""

    problem_kind: str = ""
    problem_severity: str = ""
    problem_error_code: str = ""

    application_attributes: ApplicationObservabilityAttributes | None = None
    sanitized_application_attributes: SanitizedApplicationObservabilityAttributes | None = None
    causal_evidence_source: CausalEvidenceExportSource | None = None


class RuntimeEventExportSource(BaseModel):
    """Typed runtime-event source for deferred lifecycle wiring (OBS-EXPORT-2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime_event_export_source.v1"] = "runtime_event_export_source.v1"
    event_id: str
    run_id: str
    task_id: str
    event_type: str
    agent_id: str = ""
    tenant_id: str = ""
    correlation_id: str = ""
    safe_payload: dict[str, str | int] = Field(default_factory=dict)


class CausalEvidenceExportSource(BaseModel):
    """Typed causal-evidence source for observability export projection (DIAG-1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["causal_evidence_export_source.v1"] = "causal_evidence_export_source.v1"
    evidence_id: str
    relation_kind: str
    tenant_id: str
    transport_provider: str
    transport_task_id: str
    target_task_id: str
    target_run_id: str
    target_attempt_id: str
    recorded_at: datetime


class PlatformObservabilityExportSource(BaseModel):
    """Typed non-execution platform signal source for observability export (TRACE-1B-HOS-FIX)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["platform_observability_export_source.v1"] = (
        "platform_observability_export_source.v1"
    )
    event_id: str
    source_schema_id: str
    event_type: str
    occurred_at: datetime
    correlation_id: str = ""
    application_attributes: ApplicationObservabilityAttributes | None = None


class GatewayCallExportSource(BaseModel):
    """Typed tool/RAG call source for deferred lifecycle wiring (OBS-EXPORT-2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["gateway_call_export_source.v1"] = "gateway_call_export_source.v1"
    record_kind: Literal[ExportRecordKind.TOOL_CALL, ExportRecordKind.RAG_CALL]
    call_id: str
    run_id: str
    task_id: str = ""
    agent_id: str = ""
    capability: str = ""
    tool_id: str = ""
    collection_id: str = ""
    status: GatewayCallStatus
    latency_ms: int = 0
    hit_count: int = 0
    args_digest: str = ""
    error_code: str | None = None
    policy_rule_id: str | None = None


@runtime_checkable
class ObservabilityExporter(Protocol):
    async def export(self, envelope: ObservabilityExportEnvelope) -> None: ...


class NoOpObservabilityExporter:
    """Safe default exporter — accepts envelopes and performs no I/O."""

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        return None


class InMemoryObservabilityExporter:
    """Collect exported envelopes in order (tests and local diagnostics)."""

    def __init__(self) -> None:
        self.envelopes: list[ObservabilityExportEnvelope] = []

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.envelopes.append(envelope)


TestObservabilityExporter = InMemoryObservabilityExporter


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _gateway_status_to_export(status: GatewayCallStatus) -> ExportStatus:
    mapping = {
        GatewayCallStatus.SUCCEEDED: ExportStatus.SUCCEEDED,
        GatewayCallStatus.FAILED: ExportStatus.FAILED,
        GatewayCallStatus.DENIED: ExportStatus.DENIED,
    }
    return mapping.get(status, ExportStatus.UNKNOWN)


def _extract_safe_payload(payload: object) -> dict[str, str | int]:
    if not isinstance(payload, dict):
        return {}
    safe: dict[str, str | int] = {}
    for key, value in payload.items():
        if key not in _SAFE_RUNTIME_EVENT_PAYLOAD_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value
    return safe


def _status_from_safe_payload(safe_payload: dict[str, str | int]) -> ExportStatus:
    raw = safe_payload.get("status")
    if isinstance(raw, str):
        try:
            return ExportStatus(raw)
        except ValueError:
            return ExportStatus.UNKNOWN
    return ExportStatus.UNKNOWN


def runtime_event_export_source_from_event(event: RuntimeEvent) -> RuntimeEventExportSource:
    safe_payload = _extract_safe_payload(event.payload)
    return RuntimeEventExportSource(
        event_id=event.event_id,
        run_id=event.run_id,
        task_id=event.task_id,
        event_type=event.event_type.value,
        agent_id=event.agent_id or "",
        tenant_id=event.tenant_id or "",
        correlation_id=event.correlation_id,
        safe_payload=safe_payload,
    )


def gateway_call_export_source_from_tool_call(
    record: ToolCallRecord,
    *,
    run_id: str,
    task_id: str = "",
    agent_id: str = "",
    capability: str = "",
) -> GatewayCallExportSource:
    return GatewayCallExportSource(
        record_kind=ExportRecordKind.TOOL_CALL,
        call_id=record.call_id,
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        capability=capability,
        tool_id=record.tool_id,
        status=record.status,
        latency_ms=record.latency_ms,
        args_digest=record.args_digest,
        error_code=record.error_code,
        policy_rule_id=record.policy_rule_id,
    )


def gateway_call_export_source_from_rag_call(
    record: RagCallRecord,
    *,
    run_id: str,
    task_id: str = "",
    agent_id: str = "",
    capability: str = "",
) -> GatewayCallExportSource:
    return GatewayCallExportSource(
        record_kind=ExportRecordKind.RAG_CALL,
        call_id=record.call_id,
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        capability=capability,
        collection_id=record.collection_id,
        status=record.status,
        latency_ms=record.latency_ms,
        hit_count=record.hit_count,
        policy_rule_id=record.policy_rule_id,
    )


def envelope_from_platform_observability_source(
    source: PlatformObservabilityExportSource,
) -> ObservabilityExportEnvelope:
    """Map a non-execution platform observability source to an export envelope."""
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.PLATFORM_SIGNAL,
        recorded_at=_utc_now(),
        event_type=source.event_type,
        status=ExportStatus.UNKNOWN,
        schema_id=source.schema_version,
        source_schema_id=source.source_schema_id,
        correlation_id=source.correlation_id,
        event_id=source.event_id,
        application_attributes=source.application_attributes,
    )


def envelope_from_runtime_event_source(source: RuntimeEventExportSource) -> ObservabilityExportEnvelope:
    safe_payload = source.safe_payload
    counts: dict[str, int] = {}
    for count_key in ("hit_count", "event_count", "parser_trace_count"):
        value = safe_payload.get(count_key)
        if isinstance(value, int):
            counts[count_key] = value

    latency_ms = safe_payload.get("latency_ms")
    if not isinstance(latency_ms, int):
        duration_ms = safe_payload.get("duration_ms")
        latency_ms = duration_ms if isinstance(duration_ms, int) else None

    schema_id = str(safe_payload.get("payload_schema_id") or safe_payload.get("schema_id") or "")

    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        recorded_at=_utc_now(),
        run_id=source.run_id,
        task_id=source.task_id,
        agent_id=source.agent_id,
        capability=str(safe_payload.get("capability") or ""),
        tool_id=str(safe_payload.get("tool_id") or ""),
        event_type=source.event_type,
        status=_status_from_safe_payload(safe_payload),
        latency_ms=latency_ms,
        counts=counts,
        sha256=str(safe_payload.get("args_digest") or ""),
        schema_id=schema_id,
        tenant_id=source.tenant_id,
        source_schema_id="runtime_event.v1",
        correlation_id=source.correlation_id,
        event_id=source.event_id,
    )


def envelope_from_runtime_event(event: RuntimeEvent) -> ObservabilityExportEnvelope:
    return envelope_from_runtime_event_source(runtime_event_export_source_from_event(event))


def envelope_from_gateway_call_source(source: GatewayCallExportSource) -> ObservabilityExportEnvelope:
    counts: dict[str, int] = {}
    if source.hit_count:
        counts["hit_count"] = source.hit_count

    tool_id = source.tool_id
    if source.record_kind is ExportRecordKind.RAG_CALL and not tool_id:
        tool_id = source.collection_id

    return ObservabilityExportEnvelope(
        record_kind=source.record_kind,
        recorded_at=_utc_now(),
        run_id=source.run_id,
        task_id=source.task_id,
        agent_id=source.agent_id,
        capability=source.capability,
        tool_id=tool_id,
        status=_gateway_status_to_export(source.status),
        latency_ms=source.latency_ms or None,
        counts=counts,
        sha256=source.args_digest,
        schema_id=source.schema_version,
        source_schema_id="agent_run_trace.v1",
        event_id=source.call_id,
    )


def envelope_from_tool_call(
    record: ToolCallRecord,
    *,
    run_id: str,
    task_id: str = "",
    agent_id: str = "",
    capability: str = "",
) -> ObservabilityExportEnvelope:
    return envelope_from_gateway_call_source(
        gateway_call_export_source_from_tool_call(
            record,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            capability=capability,
        )
    )


def envelope_from_rag_call(
    record: RagCallRecord,
    *,
    run_id: str,
    task_id: str = "",
    agent_id: str = "",
    capability: str = "",
) -> ObservabilityExportEnvelope:
    return envelope_from_gateway_call_source(
        gateway_call_export_source_from_rag_call(
            record,
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            capability=capability,
        )
    )


def envelope_from_journal_ref(ref: JournalRef) -> ObservabilityExportEnvelope:
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.JOURNAL_REF,
        recorded_at=_utc_now(),
        run_id=ref.run_id,
        tenant_id=ref.tenant_id,
        status=ExportStatus.SUCCEEDED,
        counts={
            "event_count": ref.event_count,
            "parser_trace_count": ref.parser_trace_count,
        },
        schema_id=ref.schema_version,
        source_schema_id=ref.schema_version,
    )


def envelope_with_observability_extensions(
    envelope: ObservabilityExportEnvelope,
    *,
    application_attributes: ApplicationObservabilityAttributes | None = None,
    artifact_ref: ObservabilityArtifactReference | None = None,
) -> ObservabilityExportEnvelope:
    """Attach typed application metadata and artifact references to an export envelope."""
    updates: dict[str, object] = {}
    if application_attributes is not None:
        updates["application_attributes"] = application_attributes
    if artifact_ref is not None:
        updates.update(
            {
                "artifact_ref": artifact_ref.artifact_ref,
                "sha256": artifact_ref.sha256,
                "safe_relative_path": artifact_ref.safe_relative_path,
                "schema_id": artifact_ref.schema_id,
            }
        )
    if not updates:
        return envelope
    return envelope.model_copy(update=updates)


def envelope_is_content_safe(envelope: ObservabilityExportEnvelope) -> bool:
    """Return False when serialized envelope exposes forbidden raw-content field names."""
    serialized = envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        if f'"{key}"' in serialized:
            return False
    return True

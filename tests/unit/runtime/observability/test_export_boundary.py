# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord, ToolCallRecord
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    InMemoryObservabilityExporter,
    NoOpObservabilityExporter,
    ObservabilityExportEnvelope,
    envelope_from_journal_ref,
    envelope_from_rag_call,
    envelope_from_runtime_event,
    envelope_from_tool_call,
    envelope_is_content_safe,
    runtime_event_export_source_from_event,
)
from intergrax.runtime.observability import export_boundary as export_boundary_module
from intergrax.runtime.observability.journal_export import JournalRef

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_BOUNDARY_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "export_boundary.py"

_FORBIDDEN_VENDOR_TOKENS = (
    "langfuse",
    "arize",
    "phoenix",
    "elasticsearch",
    "opentelemetry",
    "otlp",
    "integrations.providers.observability_backend",
    "httpx",
    "requests",
)


@pytest.mark.asyncio
async def test_noop_exporter_succeeds_without_side_effects() -> None:
    exporter = NoOpObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind="runtime_event", run_id="run-1")
    await exporter.export(envelope)


@pytest.mark.asyncio
async def test_in_memory_exporter_stores_envelopes_in_order() -> None:
    exporter = InMemoryObservabilityExporter()
    first = ObservabilityExportEnvelope(record_kind="tool_call", run_id="run-1", tool_id="t1")
    second = ObservabilityExportEnvelope(record_kind="rag_call", run_id="run-1", tool_id="rag.retrieve")

    await exporter.export(first)
    await exporter.export(second)

    assert exporter.envelopes == [first, second]
    assert export_boundary_module.TestObservabilityExporter is InMemoryObservabilityExporter


def test_export_boundary_has_no_vendor_sdk_coupling() -> None:
    source = _EXPORT_BOUNDARY_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, (
            f"export_boundary.py contains forbidden vendor coupling token: {token}"
        )


def test_envelope_from_runtime_event_excludes_raw_payload_content() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        agent_id="agent-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.write_file",
            "latency_ms": 12,
            "prompt": "secret prompt",
            "content": "raw file body",
            "source_path": "C:\\Users\\secret\\project\\file.txt",
        },
    )

    envelope = envelope_from_runtime_event(event)

    assert envelope.tool_id == "workspace.write_file"
    assert envelope.latency_ms == 12
    assert envelope_is_content_safe(envelope)
    serialized = envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


def test_envelope_from_tool_and_rag_calls_are_safe() -> None:
    tool_envelope = envelope_from_tool_call(
        ToolCallRecord(
            call_id="tc-1",
            tool_id="workspace.read_file",
            status=GatewayCallStatus.SUCCEEDED,
            latency_ms=7,
            args_digest="abc123",
        ),
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
    )
    rag_envelope = envelope_from_rag_call(
        RagCallRecord(
            call_id="rc-1",
            collection_id="docs",
            status=GatewayCallStatus.SUCCEEDED,
            latency_ms=4,
            hit_count=3,
        ),
        run_id="run-1",
    )

    assert tool_envelope.record_kind == "tool_call"
    assert rag_envelope.counts["hit_count"] == 3
    assert envelope_is_content_safe(tool_envelope)
    assert envelope_is_content_safe(rag_envelope)


def test_runtime_event_export_source_strips_unsafe_payload_keys() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={"tool_id": "rag.retrieve", "query": "user question", "input": {"path": "/tmp/x"}},
    )
    source = runtime_event_export_source_from_event(event)
    assert source.safe_payload == {"tool_id": "rag.retrieve"}
    assert "query" not in source.safe_payload
    assert "input" not in source.safe_payload


def test_envelope_from_journal_ref_is_safe() -> None:
    ref = JournalRef(
        schema_version="journal.v1",
        run_id="run-1",
        tenant_id="tenant-a",
        event_count=5,
        parser_trace_count=2,
    )
    envelope = envelope_from_journal_ref(ref)
    assert envelope.counts["event_count"] == 5
    assert envelope_is_content_safe(envelope)

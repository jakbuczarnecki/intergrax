# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_JSONL_EXPORTER_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "jsonl_exporter.py"

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


def _read_jsonl_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_writes_one_envelope_as_one_jsonl_line(tmp_path: Path) -> None:
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        event_id="evt-1",
    )

    await exporter.export(envelope)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_appends_multiple_envelopes_in_order(tmp_path: Path) -> None:
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    first = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.TOOL_CALL,
        run_id="run-1",
        event_id="evt-1",
        tool_id="tool-a",
    )
    second = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RAG_CALL,
        run_id="run-1",
        event_id="evt-2",
        tool_id="rag.retrieve",
    )

    await exporter.export(first)
    await exporter.export(second)

    records = _read_jsonl_lines(output_path)
    assert len(records) == 2
    assert records[0]["event_id"] == "evt-1"
    assert records[0]["tool_id"] == "tool-a"
    assert records[1]["event_id"] == "evt-2"
    assert records[1]["tool_id"] == "rag.retrieve"


@pytest.mark.asyncio
async def test_written_json_contains_stable_envelope_metadata_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.TOOL_CALL,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="workspace.write",
        tool_id="workspace.write_file",
        event_type="tool.completed",
        status="succeeded",
        latency_ms=12,
        counts={"hit_count": 2},
        schema_id="agent_run_trace.v1",
        sha256="abc123",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_schema_id="agent_run_trace.v1",
        correlation_id="corr-1",
        event_id="evt-1",
    )

    await exporter.export(envelope)

    record = _read_jsonl_lines(output_path)[0]
    assert record["schema_version"] == "observability_export_envelope.v1"
    assert record["record_kind"] == "tool_call"
    assert record["run_id"] == "run-1"
    assert record["task_id"] == "task-1"
    assert record["agent_id"] == "agent-1"
    assert record["capability"] == "workspace.write"
    assert record["tool_id"] == "workspace.write_file"
    assert record["latency_ms"] == 12
    assert record["counts"] == {"hit_count": 2}
    assert record["schema_id"] == "agent_run_trace.v1"
    assert record["sha256"] == "abc123"
    assert record["tenant_id"] == "tenant-a"
    assert record["workspace_id"] == "workspace-a"
    assert record["event_id"] == "evt-1"
    assert "recorded_at" in record


@pytest.mark.asyncio
async def test_written_json_does_not_contain_raw_sensitive_content(tmp_path: Path) -> None:
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        tool_id="workspace.read_file",
        latency_ms=9,
    )

    await exporter.export(envelope)

    serialized = output_path.read_text(encoding="utf-8")
    assert envelope_is_content_safe(envelope)
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized
    assert str(output_path) not in serialized


@pytest.mark.asyncio
async def test_works_through_try_export_with_enabled_policy(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    event = RuntimeEvent(
        tenant_id="tenant-a",
        agent_id="agent-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.read_file",
            "latency_ms": 9,
            "prompt": "secret prompt",
            "content": "raw body",
            "source_path": "C:\\Users\\secret\\project\\file.txt",
        },
        **runtime_event_test_identity(),
    )
    envelope = envelope_from_runtime_event(event)
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is True
    record = _read_jsonl_lines(output_path)[0]
    assert record["tool_id"] == "workspace.read_file"
    assert record["latency_ms"] == 9
    serialized = output_path.read_text(encoding="utf-8")
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


@pytest.mark.asyncio
async def test_disabled_policy_does_not_write_records(tmp_path: Path) -> None:
    output_path = tmp_path / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=True)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=False),
    )

    assert result.exported is False
    assert not output_path.exists()


def test_jsonl_exporter_has_no_vendor_sdk_coupling() -> None:
    source = _JSONL_EXPORTER_PATH.read_text(encoding="utf-8")
    for token in _FORBIDDEN_VENDOR_TOKENS:
        assert token not in source, f"jsonl_exporter.py contains forbidden vendor coupling token: {token}"


@pytest.mark.asyncio
async def test_write_failure_is_isolated_through_try_export(tmp_path: Path) -> None:
    output_path = tmp_path / "missing-parent" / "export.jsonl"
    exporter = JsonlObservabilityExporter(output_path, create_parent_dirs=False)
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    policy = ObservabilityExportPolicy(enabled=True)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"
    assert not output_path.exists()

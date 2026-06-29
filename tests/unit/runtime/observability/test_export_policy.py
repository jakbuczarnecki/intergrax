# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ExportRecordKind,
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
    envelope_is_content_safe,
)
from intergrax.runtime.observability.export_policy import (
    ExportPolicyResult,
    ObservabilityExportMode,
    ObservabilityExportPolicy,
    apply_observability_export_policy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_POLICY_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "export_policy.py"
_EXPORT_WIRING_PATH = _PROJECT_ROOT / "intergrax" / "runtime" / "observability" / "export_wiring.py"

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


class _FailingObservabilityExporter:
    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        raise RuntimeError("export sink unavailable")


def test_default_policy_is_disabled_and_metadata_only() -> None:
    policy = ObservabilityExportPolicy()
    assert policy.enabled is False
    assert policy.export_content is False
    assert policy.strict_redaction is True
    assert policy.mode is ObservabilityExportMode.DISABLED


def test_default_policy_drops_export_for_runtime_event_envelope() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={"tool_id": "workspace.read_file", "prompt": "secret"},
    )
    envelope = envelope_from_runtime_event(event)
    result = apply_observability_export_policy(envelope)

    assert result.exported is False
    assert result.envelope is None
    assert result.decision is ObservabilityExportMode.DISABLED
    assert envelope_is_content_safe(envelope)
    serialized = envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized


def test_enabled_policy_preserves_safe_operational_metadata() -> None:
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
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.run_id == "run-1"
    assert result.envelope.task_id == "task-1"
    assert result.envelope.agent_id == "agent-1"
    assert result.envelope.capability == "workspace.write"
    assert result.envelope.tool_id == "workspace.write_file"
    assert result.envelope.latency_ms == 12
    assert result.envelope.counts["hit_count"] == 2
    assert result.envelope.schema_id == "agent_run_trace.v1"
    assert result.envelope.sha256 == "abc123"
    assert envelope_is_content_safe(result.envelope)


def test_enabled_policy_hashes_unsafe_path_like_values() -> None:
    unsafe_path = "C:\\Users\\secret\\project\\file.txt"
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        safe_relative_path=unsafe_path,
    )
    policy = ObservabilityExportPolicy(enabled=True)

    result = apply_observability_export_policy(envelope, policy)

    assert result.exported is True
    assert result.envelope is not None
    assert result.envelope.safe_relative_path != unsafe_path
    assert len(result.envelope.safe_relative_path) == 64


@pytest.mark.asyncio
async def test_disabled_policy_does_not_emit_to_in_memory_exporter() -> None:
    exporter = InMemoryObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(envelope, exporter=exporter)

    assert isinstance(result, ExportPolicyResult)
    assert result.exported is False
    assert exporter.envelopes == []


@pytest.mark.asyncio
async def test_exporter_failure_is_isolated_from_caller() -> None:
    exporter = _FailingObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    policy = ObservabilityExportPolicy(enabled=True)

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=policy,
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"


def test_export_policy_and_wiring_have_no_vendor_sdk_coupling() -> None:
    for path in (_EXPORT_POLICY_PATH, _EXPORT_WIRING_PATH):
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_VENDOR_TOKENS:
            assert token not in source, f"{path.name} contains forbidden vendor coupling token: {token}"


@pytest.mark.asyncio
async def test_observability_export_runtime_plugin_exports_sanitized_envelope() -> None:
    exporter = InMemoryObservabilityExporter()
    policy = ObservabilityExportPolicy(enabled=True)
    bus = RuntimeEventBus(record_history=False)
    plugin = make_observability_export_runtime_plugin(exporter=exporter, policy=policy)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        agent_id="agent-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.read_file",
            "latency_ms": 9,
            "prompt": "secret prompt",
            "source_path": "C:\\secret\\file.txt",
        },
    )

    await bus.publish(event)

    assert len(exporter.envelopes) == 1
    exported = exporter.envelopes[0]
    assert exported.tool_id == "workspace.read_file"
    assert exported.latency_ms == 9
    assert envelope_is_content_safe(exported)


@pytest.mark.asyncio
async def test_observability_export_runtime_plugin_survives_exporter_failure() -> None:
    exporter = _FailingObservabilityExporter()
    policy = ObservabilityExportPolicy(enabled=True)
    bus = RuntimeEventBus(record_history=False)
    plugin = make_observability_export_runtime_plugin(exporter=exporter, policy=policy)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={"journal_ref": {"event_count": 1}},
    )

    await bus.publish(event)

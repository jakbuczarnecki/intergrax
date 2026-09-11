# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 — governed evidence export (OBS-03)."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.observability.export_boundary import (
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
    ExportRecordKind,
    envelope_from_runtime_event,
    envelope_is_content_safe,
    envelope_with_observability_extensions,
)
from intergrax.runtime.observability.export_bridge import make_journal_export_runtime_plugin
from intergrax.runtime.observability.journal_export import (
    JOURNAL_EXPORT_SCHEMA_VERSION,
    JournalExportSnapshot,
    build_journal_export_snapshot,
    render_journal_otlp_json,
    serialize_runtime_event,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_JOURNAL_EXPORT_PATH = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "journal_export.py"
_OBSERVABILITY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"
_CANARIES = (
    "SECRET_PROMPT_123",
    "SECRET_TOKEN_456",
    "RAW_TOOL_ARGS_789",
    "PRIVATE_BODY_ABC",
    "DO_NOT_EXPORT_123",
)
_FORBIDDEN_PAYLOAD_KEYS = (
    "prompt",
    "completion",
    "args",
    "token",
    "api_key",
    "input",
    "output",
    "query",
    "body",
)


def _unsafe_payload(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "tool_id": "demo.tool",
        "capability": "demo.cap",
        "latency_ms": 9,
        "duration_ms": 9,
        "hit_count": 2,
        "error_code": "E_DEMO",
        "policy_rule_id": "rule-1",
        "args_digest": "sha256:abc",
        "schema_id": "payload.v1",
        "status": "succeeded",
        "prompt": _CANARIES[0],
        "completion": "leak",
        "args": {"x": 1},
        "token": _CANARIES[1],
        "api_key": "k",
        "input": "in",
        "output": "out",
        "query": "q",
        "body": _CANARIES[3],
        "super_secret_future_field": "sensitive",
        "metadata": {"prompt": _CANARIES[0]},
        "secret_canary": _CANARIES[4],
    }
    base.update(extra)
    return base


def _runtime_event(
    *,
    tenant_id: str = "tenant-r3",
    payload: dict[str, object] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        payload=dict(payload or {}),
        **runtime_event_test_identity(),
    )


def _persisted(run_id: str, tenant_id: str) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
        events=[],
    )


def _serialized_export_surfaces(envelope: ObservabilityExportEnvelope, snapshot: JournalExportSnapshot) -> str:
    otlp = render_journal_otlp_json(snapshot)
    return json.dumps(
        {
            "envelope": envelope.model_dump(mode="json"),
            "snapshot": snapshot.to_dict(),
            "otlp": otlp,
        }
    )


def test_r3_forbidden_payload_fields_absent_from_export() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    envelope = serialize_runtime_event(event)
    blob = envelope.model_dump_json()
    for key in _FORBIDDEN_PAYLOAD_KEYS:
        assert f'"{key}"' not in blob
    for canary in _CANARIES:
        assert canary not in blob


def test_r3_safe_fields_preserved_on_projection() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    envelope = serialize_runtime_event(event)
    assert envelope.tool_id == "demo.tool"
    assert envelope.capability == "demo.cap"
    assert envelope.latency_ms == 9
    assert envelope.counts.get("hit_count") == 2
    assert envelope.sha256 == "sha256:abc"
    assert envelope.schema_id == "payload.v1"
    assert envelope.status.value == "succeeded"


def test_r3_unknown_payload_field_dropped() -> None:
    event = _runtime_event(payload={"tool_id": "t", "future_new_field": "sensitive"})
    source_json = envelope_from_runtime_event(event).model_dump_json()
    assert "future_new_field" not in source_json
    assert "super_secret_future_field" not in source_json


def test_r3_nested_raw_payload_dropped() -> None:
    event = _runtime_event(payload={"tool_id": "t", "metadata": {"prompt": "nested-secret"}})
    blob = serialize_runtime_event(event).model_dump_json()
    assert "nested-secret" not in blob
    assert "metadata" not in blob


def test_r3_no_raw_model_dump_in_journal_export_module() -> None:
    source = _JOURNAL_EXPORT_PATH.read_text(encoding="utf-8")
    assert "event.model_dump(mode=\"json\")" not in source
    assert "RuntimeEvent.model_dump" not in source


def test_r3_journal_snapshot_events_are_typed_envelopes() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-r3"
    store = InMemoryRuntimeEventStore()
    event = _runtime_event(tenant_id=tenant_id, payload=_unsafe_payload()).model_copy(update={"run_id": run_id})
    store.append(event, tenant_id=tenant_id)
    snapshot = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store)
    assert snapshot.schema_version == JOURNAL_EXPORT_SCHEMA_VERSION
    assert len(snapshot.events) == 1
    assert isinstance(snapshot.events[0], ObservabilityExportEnvelope)
    assert all(envelope_is_content_safe(item) for item in snapshot.events)


def test_r3_logging_extra_has_no_raw_canaries() -> None:
    trace_store = InMemoryRunTraceStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    tenant_id = "tenant-r3-log"
    trace_store._events_by_run[run_id] = []
    trace_store.finalize_run(
        run_id,
        RunMetadata(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id="u",
            session_id="s",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
    )
    runtime_store = InMemoryRuntimeEventStore()
    runtime_store.append(
        _runtime_event(tenant_id=tenant_id, payload=_unsafe_payload()).model_copy(
            update={"run_id": run_id, "task_id": task_id, "event_type": RuntimeEventType.TASK_COMPLETED}
        ),
        tenant_id=tenant_id,
    )
    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=trace_store, runtime_event_store=runtime_store)
    plugin.register(bus, HookRegistry(), MagicMock())

    completed = RuntimeEvent(
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        severity=EventSeverity.INFO,
        payload={"journal_ref": {"event_count": 1}},
        timestamp=datetime(2026, 6, 7, 10, 0, 1, tzinfo=timezone.utc),
        correlation_id=run_id,
        **runtime_event_test_identity(task_id=task_id, run_id=run_id),
    )

    with patch("intergrax.runtime.observability.export_bridge.logger") as mock_logger:
        with patch("intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"):
            import asyncio

            asyncio.run(bus.publish(completed))

    extra_blob = json.dumps(mock_logger.info.call_args.kwargs["extra"], default=str)
    for canary in _CANARIES:
        assert canary not in extra_blob
    for key in _FORBIDDEN_PAYLOAD_KEYS:
        assert f'"{key}"' not in extra_blob


def test_r3_otlp_rendering_excludes_forbidden_fields() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    envelope = serialize_runtime_event(event)
    snapshot = JournalExportSnapshot(
        schema_version=JOURNAL_EXPORT_SCHEMA_VERSION,
        journal_schema_version="journal.v1",
        run_id=event.run_id,
        tenant_id=event.tenant_id or "tenant-r3",
        event_count=1,
        parser_trace_count=0,
        events=(envelope,),
        is_complete=True,
        has_continuation=False,
    )
    otlp_blob = json.dumps(render_journal_otlp_json(snapshot))
    for key in _FORBIDDEN_PAYLOAD_KEYS:
        assert key not in otlp_blob
    for canary in _CANARIES:
        assert canary not in otlp_blob
    assert envelope.event_id in otlp_blob


def test_r3_exporter_protocol_accepts_only_envelope() -> None:
    exporter = InMemoryObservabilityExporter()
    envelope = serialize_runtime_event(_runtime_event(payload={"tool_id": "t"}))

    import asyncio

    asyncio.run(exporter.export(envelope))
    assert exporter.envelopes == [envelope]
    assert all(isinstance(item, ObservabilityExportEnvelope) for item in exporter.envelopes)


@runtime_checkable
class _CustomExporter(Protocol):
    async def export(self, envelope: ObservabilityExportEnvelope) -> None: ...


def test_r3_custom_exporter_pluginability() -> None:
    class _Collector:
        def __init__(self) -> None:
            self.items: list[ObservabilityExportEnvelope] = []

        async def export(self, envelope: ObservabilityExportEnvelope) -> None:
            self.items.append(envelope)

    collector = _Collector()
    assert isinstance(collector, _CustomExporter)
    envelope = serialize_runtime_event(_runtime_event(payload={"tool_id": "t"}))
    import asyncio

    asyncio.run(collector.export(envelope))
    assert collector.items[0].record_kind == ExportRecordKind.RUNTIME_EVENT


def test_r3_extension_safety_blocks_forbidden_application_fields_via_policy() -> None:
    from intergrax.runtime.observability.export_attributes import ApplicationObservabilityAttributes
    from intergrax.runtime.observability.export_policy import (
        ObservabilityExportPolicy,
        apply_observability_export_policy,
    )

    class SensitiveAttributes(ApplicationObservabilityAttributes):
        namespace: str = "example"
        operation: str = "example.run"
        prompt: str = "forbidden"

    base = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")
    extended = envelope_with_observability_extensions(
        base,
        application_attributes=SensitiveAttributes(),
    )
    result = apply_observability_export_policy(extended, ObservabilityExportPolicy(enabled=True))
    assert result.envelope is not None
    assert envelope_is_content_safe(result.envelope)


def test_r3_bounded_journal_read_preserved() -> None:
    source = _JOURNAL_EXPORT_PATH.read_text(encoding="utf-8")
    assert "read_run_journal_page" in source
    assert "load_complete_run_journal" not in source


def test_r3_snapshot_stays_within_single_page() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-page"
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    for _ in range(5):
        store.append(
            _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(
                update={"run_id": run_id, "task_id": task_id}
            ),
            tenant_id=tenant_id,
        )
    snapshot = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store, limit=3)
    assert snapshot.event_count == 3
    assert snapshot.is_complete is False
    assert snapshot.has_continuation is True


def test_r3_tenant_preserved_on_envelope() -> None:
    event = _runtime_event(tenant_id="tenant-keep", payload={"tool_id": "t"})
    assert serialize_runtime_event(event).tenant_id == "tenant-keep"


def test_r3_cross_tenant_journal_isolation() -> None:
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    store.append(_runtime_event(tenant_id="tenant-a", payload={"tool_id": "t"}).model_copy(update={"run_id": run_id}), tenant_id="tenant-a")
    snapshot = build_journal_export_snapshot(_persisted(run_id, "tenant-b"), runtime_store=store)
    assert snapshot.event_count == 0


def test_r3_event_and_execution_ids_preserved() -> None:
    event = _runtime_event(payload={"tool_id": "t"})
    envelope = serialize_runtime_event(event)
    assert envelope.event_id == event.event_id
    assert envelope.execution_id == str(event.execution_id)
    assert envelope.attempt_id == str(event.attempt_id)


def test_r3_projection_error_does_not_echo_raw_payload() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    with patch(
        "intergrax.runtime.observability.export_boundary.envelope_is_content_safe",
        return_value=False,
    ):
        with pytest.raises(ValueError) as exc:
            serialize_runtime_event(event)
    message = str(exc.value)
    for canary in _CANARIES:
        assert canary not in message


def test_r3_no_second_export_framework_symbols() -> None:
    hits: list[str] = []
    for path in _OBSERVABILITY_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "JournalSafeEvent" in text or "RuntimeEventExportDTO" in text:
            hits.append(str(path))
    assert hits == []


def test_r3_no_execution_control_from_observability_export_surface() -> None:
    forbidden_calls = (
        "ExecutionRuntime",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
        "ChildExecutionRunner",
    )
    violations: list[str] = []
    for rel in (
        "journal_export.py",
        "export_boundary.py",
        "export_bridge.py",
    ):
        path = _OBSERVABILITY_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden_calls:
                    violations.append(f"{rel}:{node.func.id}")
    assert violations == []


def test_r3_no_reflection_on_safe_projection_symbols() -> None:
    boundary = (_OBSERVABILITY_ROOT / "export_boundary.py").read_text(encoding="utf-8")
    for symbol in (
        "runtime_event_export_source_from_event",
        "envelope_from_runtime_event_source",
        "_extract_safe_payload",
    ):
        block = boundary.split(f"def {symbol}")[1].split("\ndef ")[0]
        assert not re.search(r"\b(getattr|setattr|hasattr)\(", block)


def test_r3_legacy_unsafe_serializer_absent_from_supported_paths() -> None:
    source = _JOURNAL_EXPORT_PATH.read_text(encoding="utf-8")
    assert "def serialize_runtime_event" in source
    assert re.search(r"return\s+event\.model_dump", source) is None


def test_r3_serialization_is_deterministic() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    first = serialize_runtime_event(event).model_dump_json()
    second = serialize_runtime_event(event).model_dump_json()
    assert first == second

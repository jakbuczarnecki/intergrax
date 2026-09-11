# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 Final — governed evidence export qualification and freeze (OBS-03)."""

from __future__ import annotations

import ast
import inspect
import json
import re
import subprocess
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
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import read_run_journal_page
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.observability.export_boundary import (
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ObservabilityExportEnvelope,
    ExportRecordKind,
    envelope_from_runtime_event,
    envelope_is_content_safe,
    runtime_event_export_source_from_event,
)
from intergrax.runtime.observability.export_bridge import make_journal_export_runtime_plugin
from intergrax.runtime.observability.journal_export import (
    JOURNAL_EXPORT_SCHEMA_VERSION,
    JournalExportSnapshot,
    build_journal_export_snapshot,
    render_journal_otlp_json,
    serialize_runtime_event,
)
from testing_support.npsc5f_r3_protected_drift import (
    R3_IMPLEMENTATION_SHA,
    collect_r3_protected_production_drift,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBSERVABILITY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"
_R3_EXPORT_SURFACES = (
    "export_boundary.py",
    "journal_export.py",
    "export_bridge.py",
)

R2_FINAL_SHA = "76c92847f67da22d97943b55896a88c814d7e39d"
R2_IMPLEMENTATION_SHA = "632507420f0ab8360aede43a2740e8fccc44efb4"
R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
NPSC_5F_P0_SHA = "7811371da1069b661987b050a4c9bf42c02bda69"
NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"

_CANARIES = (
    "SECRET_PROMPT_123",
    "SECRET_TOKEN_456",
    "RAW_TOOL_ARGS_789",
    "PRIVATE_BODY_ABC",
    "DO_NOT_EXPORT_123",
    "FUTURE_SECRET_FIELD_X",
)

_FORBIDDEN_CONTROL_PLANE_SYMBOLS = frozenset(
    {
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
        "ExecutionRuntime",
    },
)

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R3 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py"],
    ),
    (
        "R2 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py"],
    ),
    (
        "R1 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "NPSC-5F P0 gate",
        ["tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "Journal export suites",
        ["tests/unit/runtime/observability/test_journal_export.py"],
    ),
    (
        "Export boundary suites",
        [
            "tests/unit/runtime/observability/test_export_boundary.py",
            "tests/unit/runtime/observability/test_export_boundary_contracts.py",
        ],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "NPSC-5E Final",
        ["tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    (
        "R3 drift classifier",
        ["tests/unit/testing_support/test_npsc5f_r3_protected_drift.py"],
    ),
)


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _unsafe_payload(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "tool_id": "demo.tool",
        "capability": "demo.cap",
        "latency_ms": 9,
        "hit_count": 2,
        "schema_id": "payload.v1",
        "status": "succeeded",
        "prompt": _CANARIES[0],
        "completion": "leak",
        "args": {"x": _CANARIES[2]},
        "token": _CANARIES[1],
        "api_key": "k",
        "input": "in",
        "output": "out",
        "query": "q",
        "body": _CANARIES[3],
        "future_super_secret": _CANARIES[5],
        "metadata": {"prompt": _CANARIES[0]},
        "secret_canary": _CANARIES[4],
    }
    base.update(extra)
    return base


def _runtime_event(
    *,
    tenant_id: str = "tenant-r3-final",
    payload: dict[str, object] | None = None,
    traceparent: str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        payload=dict(payload or {}),
        traceparent=traceparent,
        tracestate="vendor=t1",
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


def _assert_canaries_absent(blob: str) -> None:
    for canary in _CANARIES:
        assert canary not in blob


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_r3_final_canonical_predecessor_shas_recorded() -> None:
    assert R3_IMPLEMENTATION_SHA == "0346face3ef68d8f21504822a26f8f45f2384cf9"
    assert R2_FINAL_SHA.startswith("76c9284")
    assert R2_IMPLEMENTATION_SHA.startswith("6325074")
    assert R1_FINAL_SHA.startswith("455c09f")
    assert NPSC_5F_P0_SHA.startswith("7811371")
    assert NPSC_5E_FINAL_SHA.startswith("fabdcfe")


def test_r3_final_no_unqualified_protected_drift_since_implementation() -> None:
    drift = collect_r3_protected_production_drift(_REPO_ROOT)
    assert drift == [], f"R3 protected production drift since implementation: {drift}"


def test_r3_final_journal_export_schema_v2_only() -> None:
    assert JOURNAL_EXPORT_SCHEMA_VERSION == "journal_export.v2"


def test_r3_final_no_supported_journal_export_v1_in_production() -> None:
    obs_root = _REPO_ROOT / "intergrax" / "runtime" / "observability"
    hits: list[str] = []
    for path in obs_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "journal_export.v1" in text:
            hits.append(path.relative_to(_REPO_ROOT).as_posix())
    assert hits == []


def test_r3_final_raw_payload_absent_from_all_export_surfaces() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    source = runtime_event_export_source_from_event(event)
    envelope = serialize_runtime_event(event)
    run_id = event.run_id
    store = InMemoryRuntimeEventStore()
    store.append(event.model_copy(update={"run_id": run_id}), tenant_id=event.tenant_id or "tenant-r3-final")
    snapshot = build_journal_export_snapshot(_persisted(run_id, event.tenant_id or "tenant-r3-final"), runtime_store=store)

    class _Collector:
        def __init__(self) -> None:
            self.items: list[ObservabilityExportEnvelope] = []

        async def export(self, envelope: ObservabilityExportEnvelope) -> None:
            self.items.append(envelope)

    collector = _Collector()
    import asyncio

    asyncio.run(collector.export(envelope))

    surfaces = json.dumps(
        {
            "safe_payload": source.safe_payload,
            "envelope": envelope.model_dump(mode="json"),
            "snapshot": snapshot.to_dict(),
            "otlp": render_journal_otlp_json(snapshot),
            "exporter": [item.model_dump(mode="json") for item in collector.items],
        },
        default=str,
    )
    _assert_canaries_absent(surfaces)
    envelope_json = envelope.model_dump_json()
    for key in ("prompt", "completion", "args", "api_key", "query", "body"):
        assert f'"{key}"' not in envelope_json


def test_r3_final_unknown_field_dropped() -> None:
    event = _runtime_event(payload={"tool_id": "t", "future_super_secret": "sensitive"})
    blob = envelope_from_runtime_event(event).model_dump_json()
    assert "future_super_secret" not in blob


def test_r3_final_nested_secret_dropped() -> None:
    event = _runtime_event(payload={"tool_id": "t", "metadata": {"prompt": "nested-secret"}, "items": [{"token": "x"}]})
    blob = serialize_runtime_event(event).model_dump_json()
    assert "nested-secret" not in blob
    assert "metadata" not in blob


def test_r3_final_safe_fields_preserved() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    envelope = serialize_runtime_event(event)
    assert envelope.tool_id == "demo.tool"
    assert envelope.tenant_id == event.tenant_id
    assert envelope.w3c_traceparent == event.traceparent
    assert envelope.w3c_tracestate == event.tracestate


def test_r3_final_forbidden_structural_key_rejected_by_validator() -> None:
    unsafe = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        counts={"token": 1},
    )
    assert envelope_is_content_safe(unsafe) is False


def test_r3_final_safe_value_substring_does_not_fail_validator() -> None:
    safe = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        correlation_id="prefix-token-suffix",
        sha256="sha256:deadbeef",
    )
    assert envelope_is_content_safe(safe) is True


def test_r3_final_bounded_export_incomplete_large_journal() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-bounded"
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    for _ in range(8):
        store.append(
            _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(
                update={"run_id": run_id, "task_id": task_id},
            ),
            tenant_id=tenant_id,
        )
    snapshot = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store, limit=3)
    assert snapshot.event_count == 3
    assert snapshot.is_complete is False
    assert snapshot.has_continuation is True


def test_r3_final_complete_page_small_journal() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-complete"
    store = InMemoryRuntimeEventStore()
    event = _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(update={"run_id": run_id})
    store.append(event, tenant_id=tenant_id)
    snapshot = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store, limit=10)
    assert snapshot.is_complete is True
    assert snapshot.has_continuation is False


def test_r3_final_export_snapshot_consistent_after_concurrent_append(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "export-snap.db")
    run_id = mint_run_id()
    tenant_id = "tenant-snap"
    task_id = mint_task_id()
    for _ in range(6):
        store.append(
            _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(
                update={"run_id": run_id, "task_id": task_id},
            ),
            tenant_id=tenant_id,
        )
    first = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store, limit=2)
    page = read_run_journal_page(store, tenant_id=tenant_id, run_id=run_id, page_size=2)
    assert first.event_count == len(page.events)
    for _ in range(3):
        store.append(
            _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(
                update={"run_id": run_id, "task_id": task_id},
            ),
            tenant_id=tenant_id,
        )
    second = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store, limit=2)
    assert len(second.events) == 2
    assert [item.event_id for item in second.events] == [item.event_id for item in first.events]
    store.close()


def test_r3_final_identity_and_timestamp_preserved() -> None:
    event = _runtime_event(payload={"tool_id": "t"})
    envelope = serialize_runtime_event(event)
    assert envelope.event_id == event.event_id
    assert envelope.run_id == event.run_id
    assert envelope.task_id == event.task_id
    assert envelope.attempt_id == str(event.attempt_id)
    assert envelope.execution_id == str(event.execution_id)
    assert envelope.tenant_id == event.tenant_id
    assert envelope.parent_event_id == str(event.parent_event_id or "")
    assert envelope.recorded_at == event.timestamp


def test_r3_final_tenant_isolation_on_export() -> None:
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _runtime_event(tenant_id="tenant-a", payload={"tool_id": "t"}).model_copy(update={"run_id": run_id}),
        tenant_id="tenant-a",
    )
    snapshot = build_journal_export_snapshot(_persisted(run_id, "tenant-b"), runtime_store=store)
    assert snapshot.event_count == 0


def test_r3_final_logger_extra_has_zero_canaries() -> None:
    trace_store = InMemoryRunTraceStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    tenant_id = "tenant-log-final"
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
            update={"run_id": run_id, "task_id": task_id, "event_type": RuntimeEventType.TASK_COMPLETED},
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
    _assert_canaries_absent(extra_blob)


def test_r3_final_otlp_renderer_accepts_typed_snapshot_only() -> None:
    sig = inspect.signature(render_journal_otlp_json)
    snapshot_param = sig.parameters["snapshot"]
    assert snapshot_param.annotation in (JournalExportSnapshot, "JournalExportSnapshot")


def test_r3_final_no_runtime_event_model_dump_in_export_surfaces() -> None:
    violations: list[str] = []
    for rel in _R3_EXPORT_SURFACES:
        source = (_OBSERVABILITY_ROOT / rel).read_text(encoding="utf-8")
        if "RuntimeEvent.model_dump" in source:
            violations.append(f"{rel}:RuntimeEvent.model_dump")
        if re.search(r"\bevent\.model_dump\s*\(", source):
            violations.append(f"{rel}:event.model_dump")
    assert violations == []


def test_r3_final_no_second_export_framework_symbols() -> None:
    forbidden_names = ("SafeExportEnvelope", "EvidenceExportRuntime", "SecureJournalExporter")
    hits: list[str] = []
    for path in _OBSERVABILITY_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden_names:
            if name in text:
                hits.append(f"{path.name}:{name}")
    assert hits == []


def test_r3_final_no_execution_control_from_export_surface() -> None:
    violations: list[str] = []
    for rel in _R3_EXPORT_SURFACES:
        path = _OBSERVABILITY_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            symbol = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if symbol in _FORBIDDEN_CONTROL_PLANE_SYMBOLS:
                violations.append(f"{rel}:{node.lineno}:{symbol}")
    assert violations == []


def test_r3_final_snapshot_events_typed_and_immutable() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-typed"
    store = InMemoryRuntimeEventStore()
    store.append(
        _runtime_event(tenant_id=tenant_id, payload={"tool_id": "t"}).model_copy(update={"run_id": run_id}),
        tenant_id=tenant_id,
    )
    snapshot = build_journal_export_snapshot(_persisted(run_id, tenant_id), runtime_store=store)
    assert isinstance(snapshot.events, tuple)
    assert all(isinstance(item, ObservabilityExportEnvelope) for item in snapshot.events)


@runtime_checkable
class _CustomExporter(Protocol):
    async def export(self, envelope: ObservabilityExportEnvelope) -> None: ...


def test_r3_final_custom_exporter_receives_envelope_without_canaries() -> None:
    class Collector:
        def __init__(self) -> None:
            self.items: list[ObservabilityExportEnvelope] = []

        async def export(self, envelope: ObservabilityExportEnvelope) -> None:
            self.items.append(envelope)

    collector = Collector()
    assert isinstance(collector, _CustomExporter)
    envelope = serialize_runtime_event(_runtime_event(payload=_unsafe_payload()))
    import asyncio

    asyncio.run(collector.export(envelope))
    blob = json.dumps([item.model_dump(mode="json") for item in collector.items])
    _assert_canaries_absent(blob)


def test_r3_final_projection_error_does_not_echo_raw_payload() -> None:
    event = _runtime_event(payload=_unsafe_payload())
    with patch(
        "intergrax.runtime.observability.export_boundary.envelope_is_content_safe",
        return_value=False,
    ):
        with pytest.raises(ValueError) as exc:
            serialize_runtime_event(event)
    message = str(exc.value)
    _assert_canaries_absent(message)


def test_r3_final_no_false_full_snapshot_marketing_in_export_module() -> None:
    source = (_OBSERVABILITY_ROOT / "journal_export.py").read_text(encoding="utf-8")
    assert not re.search(r"\bfull run export\b", source, flags=re.IGNORECASE)
    assert not re.search(r"\bcomplete journal export\b", source, flags=re.IGNORECASE)


@pytest.mark.gate
def test_npsc5f_r3_final_qualification_gate() -> None:
    assert R3_IMPLEMENTATION_SHA == "0346face3ef68d8f21504822a26f8f45f2384cf9"
    assert FORBIDDEN_EXPORT_CONTENT_FIELDS.issuperset({"prompt", "token", "tool_args"})

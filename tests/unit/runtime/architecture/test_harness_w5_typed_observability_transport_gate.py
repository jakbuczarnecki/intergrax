# © Artur Czarnecki. All rights reserved.

"""HARNESS-W5-R1 — typed observability transport architecture gate."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import get_type_hints

import pytest

from intergrax.contracts.event_delivery import (
    EventDeliveryPolicy,
    ObservabilityExportPayload,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.contracts.observability_export import OtlpTransportPort
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    OtlpEventExportSink,
    RuntimeEventExportSink,
    runtime_event_to_deliverable,
)
from intergrax.runtime.observability.exporters.distributed.collector_transport import (
    CollectorTransport,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_TRANSPORT_PARAM_NAMES = frozenset(
    {"object", "Any", "RuntimeEvent", "ObservabilityExportEnvelope"},
)
_PROBE_NAMES = frozenset({"isinstance", "getattr", "hasattr"})


def _read_ast(rel: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / rel).read_text(encoding="utf-8"))


def _annotation_name(node: ast.expr | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return ast.unparse(node)


def _protocol_export_param(tree: ast.Module, class_name: str) -> str:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "export":
                for arg in item.args.args:
                    if arg.arg != "self":
                        return _annotation_name(arg.annotation)
    raise AssertionError(f"{class_name}.export parameter not found")


def _class_export_param(tree: ast.Module, class_name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "export":
                    for arg in item.args.args:
                        if arg.arg == "self":
                            continue
                        return _annotation_name(arg.annotation)
    raise AssertionError(f"{class_name}.export not found")


def _export_body_uses_probes(tree: ast.Module, class_name: str) -> list[str]:
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "export":
                continue
            for sub in ast.walk(item):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    if sub.func.id in _PROBE_NAMES:
                        violations.append(f"{class_name}.export uses {sub.func.id}")
                if isinstance(sub, ast.Attribute) and sub.attr == "ignore":
                    violations.append(f"{class_name}.export type ignore")
    return violations


def _contracts_imports_runtime(rel: str) -> list[str]:
    tree = _read_ast(rel)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                bad.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    bad.append(alias.name)
    return bad


def _otlp_sink_passes_payload_directly() -> bool:
    tree = _read_ast(
        "intergrax/runtime/observability/event_delivery/otlp_event_export_sink.py",
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "export":
            continue
        if not isinstance(node.func.value, ast.Attribute):
            continue
        if node.func.value.attr != "_transport":
            continue
        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return node.args[0].id == "payload"
    return False


def test_g1_otlp_transport_port_export_parameter() -> None:
    tree = _read_ast("intergrax/contracts/observability_export.py")
    param = _protocol_export_param(tree, "OtlpTransportPort")
    assert param == "ObservabilityExportPayload"
    assert param not in _FORBIDDEN_TRANSPORT_PARAM_NAMES


def test_g2_otlp_transport_export_parameter() -> None:
    tree = _read_ast("intergrax/runtime/observability/exporters/otlp/otlp_transport.py")
    assert _class_export_param(tree, "OtlpTransport") == "ObservabilityExportPayload"


def test_g3_collector_transport_export_parameter() -> None:
    tree = _read_ast(
        "intergrax/runtime/observability/exporters/distributed/collector_transport.py",
    )
    assert (
        _class_export_param(tree, "CollectorTransport") == "ObservabilityExportPayload"
    )


def test_g4_transport_export_bodies_no_dynamic_probing() -> None:
    otlp_tree = _read_ast(
        "intergrax/runtime/observability/exporters/otlp/otlp_transport.py"
    )
    collector_tree = _read_ast(
        "intergrax/runtime/observability/exporters/distributed/collector_transport.py",
    )
    violations = _export_body_uses_probes(otlp_tree, "OtlpTransport")
    violations.extend(_export_body_uses_probes(collector_tree, "CollectorTransport"))
    assert violations == []


def test_g5_otlp_event_export_sink_passes_payload_to_transport() -> None:
    assert _otlp_sink_passes_payload_directly()


def test_g6_contracts_observability_export_no_runtime_imports() -> None:
    assert (
        _contracts_imports_runtime("intergrax/contracts/observability_export.py") == []
    )


def test_runtime_hints_match_observability_export_payload() -> None:
    assert get_type_hints(OtlpTransport.export)["payload"] is ObservabilityExportPayload
    assert (
        get_type_hints(CollectorTransport.export)["payload"]
        is ObservabilityExportPayload
    )


@pytest.mark.asyncio
async def test_replaceable_recording_transport_receives_canonical_payload() -> None:
    class RecordingTransport(OtlpTransportPort):
        def __init__(self) -> None:
            self.received: list[ObservabilityExportPayload] = []

        def export(self, payload: ObservabilityExportPayload) -> None:
            self.received.append(payload)

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    transport = RecordingTransport()
    bridge = RuntimeEventExportSink(OtlpEventExportSink(transport=transport))
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=8))
    bus = RuntimeEventBus(event_sink=bounded)
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.STEP_EXECUTION,
        event_kind="qualification.w5r1.replaceability",
        **runtime_event_test_identity(),
    )
    expected = runtime_event_to_deliverable(event).export_payload
    await bus.publish(event)
    await asyncio.sleep(0.05)
    assert len(transport.received) == 1
    received = transport.received[0]
    assert received is expected or received == expected
    for field in (
        "event_id",
        "kind",
        "run_id",
        "task_id",
        "attempt_id",
        "execution_id",
        "agent_id",
        "tenant_id",
        "correlation_id",
        "parent_event_id",
        "w3c_traceparent",
        "w3c_tracestate",
    ):
        assert getattr(received, field) == getattr(expected, field)
    bus.close()

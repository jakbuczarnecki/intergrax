# © Artur Czarnecki. All rights reserved.

"""P2-003-D2-R3 — observability export-boundary runtime and facade hard-contract proofs."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord, ToolCallRecord
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    GatewayCallExportSource,
    envelope_from_journal_ref,
    gateway_call_export_source_from_rag_call,
    gateway_call_export_source_from_tool_call,
    runtime_event_export_source_from_event,
)
from intergrax.runtime.observability.journal_export import JournalRef
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBSERVABILITY_INIT = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "__init__.py"

_FACADE_PUBLIC_API_MATRIX = (
    "from intergrax.runtime.observability import ObservabilityEmitter",
    "from intergrax.runtime.observability import CausalEvidencePersistence",
    "from intergrax.runtime.observability import InMemoryObservabilityExporter",
    "from intergrax.runtime.observability import ProblemReporter",
    "from intergrax.runtime.observability import ExportStatus",
    "from intergrax.runtime.observability import GatewayCallExportSource",
)

_REPRESENTATIVE_ALL_SYMBOLS = (
    "TraceScope",
    "ExportStatus",
    "GatewayCallExportSource",
    "ObservabilityExportEnvelope",
    "ProblemReporter",
    "ObservabilityEmitter",
    "InMemoryObservabilityExporter",
    "CausalEvidencePersistence",
    "envelope_from_tool_call",
    "runtime_event_export_source_from_event",
    "JournalRef",
    "apply_observability_export_policy",
    "PayloadSchemaRegistry",
)


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _call_matches_forbidden_dispatch(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "getattr":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        if isinstance(func.value, ast.Name) and func.value.id == "importlib":
            return True
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
        sub = node.func.value
        if isinstance(sub.func, ast.Name) and sub.func.id in {"locals", "globals", "vars"}:
            return True
    return False


def _collect_forbidden_dispatch_calls(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if _call_matches_forbidden_dispatch(node):
            lines.append(node.lineno)
    return lines


def test_observability_facade_has_no_string_dynamic_dispatch() -> None:
    tree = ast.parse(_OBSERVABILITY_INIT.read_text(encoding="utf-8"), filename=str(_OBSERVABILITY_INIT))
    violations = _collect_forbidden_dispatch_calls(tree)
    assert not violations, f"forbidden facade dispatch at lines: {violations}"


@pytest.mark.parametrize("statement", _FACADE_PUBLIC_API_MATRIX)
def test_observability_facade_public_api_subprocess(statement: str) -> None:
    completed = _run_import_subprocess(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_observability_representative_public_exports_resolve() -> None:
    import intergrax.runtime.observability as obs

    for symbol in _REPRESENTATIVE_ALL_SYMBOLS:
        assert symbol in obs.__all__
        resolved = getattr(obs, symbol)
        assert resolved is not None


def test_gateway_call_export_source_construction_schema_and_validation() -> None:
    source = GatewayCallExportSource(
        record_kind=ExportRecordKind.TOOL_CALL,
        call_id="call-1",
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="workspace.read",
        tool_id="workspace.read_file",
        status=GatewayCallStatus.SUCCEEDED,
        latency_ms=11,
        args_digest="digest-1",
    )
    schema = GatewayCallExportSource.model_json_schema()
    assert "GatewayCallStatus" in schema.get("$defs", {})
    assert schema["properties"]["status"]["$ref"].endswith("GatewayCallStatus")

    validated = GatewayCallExportSource.model_validate(source.model_dump())
    assert validated.status is GatewayCallStatus.SUCCEEDED
    assert validated.model_dump()["latency_ms"] == 11


def test_tool_call_projection_preserves_semantics() -> None:
    record = ToolCallRecord(
        call_id="tc-1",
        tool_id="workspace.read_file",
        status=GatewayCallStatus.FAILED,
        latency_ms=9,
        args_digest="abc123",
        error_code="tool_timeout",
        policy_rule_id="rule-7",
    )
    source = gateway_call_export_source_from_tool_call(
        record,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="workspace.read",
    )
    assert source.record_kind is ExportRecordKind.TOOL_CALL
    assert source.status is GatewayCallStatus.FAILED
    assert source.tool_id == "workspace.read_file"
    assert source.latency_ms == 9
    assert source.error_code == "tool_timeout"
    assert source.policy_rule_id == "rule-7"


def test_rag_call_projection_preserves_semantics() -> None:
    record = RagCallRecord(
        call_id="rc-1",
        collection_id="docs",
        status=GatewayCallStatus.DENIED,
        latency_ms=4,
        hit_count=2,
        policy_rule_id="rag-deny",
    )
    source = gateway_call_export_source_from_rag_call(
        record,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="rag.retrieve",
    )
    assert source.record_kind is ExportRecordKind.RAG_CALL
    assert source.status is GatewayCallStatus.DENIED
    assert source.collection_id == "docs"
    assert source.latency_ms == 4
    assert source.hit_count == 2
    assert source.policy_rule_id == "rag-deny"


def test_runtime_event_projection_preserves_semantics() -> None:
    event = RuntimeEvent(
        tenant_id="tenant-a",
        agent_id="agent-1",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": "workspace.write_file",
            "latency_ms": 12,
            "status": ExportStatus.SUCCEEDED.value,
            "prompt": "secret prompt",
        },
        **runtime_event_test_identity(),
    )
    source = runtime_event_export_source_from_event(event)
    assert source.event_id == event.event_id
    assert source.run_id == event.run_id
    assert source.task_id == event.task_id
    assert source.event_type == event.event_type.value
    assert source.safe_payload["tool_id"] == "workspace.write_file"
    assert source.safe_payload["latency_ms"] == 12
    assert "prompt" not in source.safe_payload


def test_journal_ref_projection_preserves_semantics() -> None:
    ref = JournalRef(
        schema_version="journal.v1",
        run_id="run-1",
        tenant_id="tenant-a",
        event_count=5,
        parser_trace_count=2,
    )
    envelope = envelope_from_journal_ref(ref)
    assert envelope.record_kind is ExportRecordKind.JOURNAL_REF
    assert envelope.run_id == "run-1"
    assert envelope.tenant_id == "tenant-a"
    assert envelope.status is ExportStatus.SUCCEEDED
    assert envelope.counts["event_count"] == 5
    assert envelope.counts["parser_trace_count"] == 2

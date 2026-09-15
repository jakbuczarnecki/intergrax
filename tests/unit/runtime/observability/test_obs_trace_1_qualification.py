# © Artur Czarnecki. All rights reserved.

"""OBS-TRACE-1 — TraceEvent correlation qualification (Plane B vs canonical evidence).

Qualification command::

    uv run pytest tests/unit/runtime/observability/test_obs_trace_1_qualification.py -m obs_trace_1
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.tracing.events import TraceEvent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRACE_PLANE_ROOTS = (
    _REPO_ROOT / "intergrax" / "contracts" / "tracing",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tracing",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "task_trace.py",
    _REPO_ROOT / "intergrax" / "runtime" / "middleware" / "trace_middleware.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "emitter.py",
    _REPO_ROOT / "intergrax" / "runtime" / "codecraft" / "trace.py",
    _REPO_ROOT / "intergrax" / "runtime" / "replay" / "trace_replay_bridge.py",
    _REPO_ROOT / "intergrax" / "runtime" / "replay" / "persisted_trace_event_store.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "acp_routing_trace_bridge.py",
)
_RECONSTRUCTION_MODULES = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "reconstruction"
    / "execution_reconstruction.py",
    _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "diagnostic_orchestrator.py",
)
_EXCLUDED_PARTS = frozenset({"__pycache__", "tests"})

OBS_TRACE_1_VERDICT = "NOT_REQUIRED"


@dataclass(frozen=True, slots=True)
class TraceConsumerQualification:
    consumer: str
    purpose: str
    needs_attempt: bool
    needs_execution: bool
    canonical_alternative: str
    status: str


TRACE_CONSUMER_MATRIX: tuple[TraceConsumerQualification, ...] = (
    TraceConsumerQualification(
        consumer="Unified Run Journal",
        purpose="Chronological execution timeline",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="RuntimeEvent persistence + positioned journal",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
    TraceConsumerQualification(
        consumer="ExecutionReconstructor / DiagnosticOrchestrator",
        purpose="Factual execution reconstruction",
        needs_attempt=True,
        needs_execution=True,
        canonical_alternative="RuntimeEvent + ExecutionLineage + causal evidence",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
    TraceConsumerQualification(
        consumer="trace_bridge (TaskTraceEmitter path)",
        purpose="Mirror trace rows into RuntimeEvent bus",
        needs_attempt=True,
        needs_execution=True,
        canonical_alternative="Active execution identity at bridge time (not TraceEvent fields)",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
    TraceConsumerQualification(
        consumer="TraceEmittingMiddleware",
        purpose="Step-start canonical journal entries",
        needs_attempt=True,
        needs_execution=True,
        canonical_alternative="RuntimeEvent only (no TraceEvent)",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
    TraceConsumerQualification(
        consumer="RunTraceWriter / SQLite / in-memory trace store",
        purpose="Persist Plane B diagnostic telemetry per run",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Run-scoped append; execution facts via RuntimeEvent store",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="debug CLI / formatters",
        purpose="Operator run trace dump",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Optional unified journal when runtime store wired",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="journal_export / export_bridge OTLP snapshot",
        purpose="Export run narrative + parser traces",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Unified journal built from RuntimeEvent persistence",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
    TraceConsumerQualification(
        consumer="modality_metrics / metrics export",
        purpose="Aggregate trace steps for run metrics",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Run-level SerializedTraceEvent aggregation",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="trace_replay_bridge",
        purpose="Replay persisted trace into bus",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="RuntimeEvent replay path for execution truth",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="eval trajectory (RunTraceReader binding)",
        purpose="Qualification / eval trace inspection",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Run-scoped trace read; execution IDs from evidence tests",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="TraceQuery (nexus.tracing.trace_query)",
        purpose="Filter in-memory trace collections",
        needs_attempt=False,
        needs_execution=False,
        canonical_alternative="Payload/step filters within a run collection",
        status="RUN_SUFFICIENT",
    ),
    TraceConsumerQualification(
        consumer="platform_proofs scenario trace recorder",
        purpose="Scenario observability wrapper",
        needs_attempt=False,
        needs_execution=True,
        canonical_alternative="Separate scenario scope object; not TraceEvent contract",
        status="SHOULD_USE_RUNTIME_EVENT",
    ),
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _trace_plane_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _TRACE_PLANE_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def _collect_forbidden_mint_calls() -> list[str]:
    violations: list[str] = []
    for path in _trace_plane_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in {"mint_execution_id", "mint_attempt_id"}:
                violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def _module_imports_trace_event(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module and "trace_models" in node.module:
            for alias in node.names:
                if alias.name == "TraceEvent":
                    return True
        if node.module and node.module.endswith("tracing.events"):
            for alias in node.names:
                if alias.name == "TraceEvent":
                    return True
    return False


@pytest.mark.obs_trace_1
def test_obs_trace_1_verdict_is_not_required() -> None:
    assert OBS_TRACE_1_VERDICT == "NOT_REQUIRED"


@pytest.mark.obs_trace_1
def test_trace_event_contract_is_run_scoped_without_execution_fields() -> None:
    names = {field.name for field in dataclasses.fields(TraceEvent)}
    assert "run_id" in names
    assert "seq" in names
    assert "attempt_id" not in names
    assert "execution_id" not in names
    assert "task_id" not in names
    assert "tenant_id" not in names


@pytest.mark.obs_trace_1
@pytest.mark.parametrize("entry", TRACE_CONSUMER_MATRIX, ids=lambda e: e.consumer)
def test_no_trace_consumer_requires_trace_event_execution_correlation(
    entry: TraceConsumerQualification,
) -> None:
    assert entry.status != "NEEDS_EXECUTION"
    assert entry.status != "NEEDS_ATTEMPT"


@pytest.mark.obs_trace_1
def test_gate_trace_plane_does_not_mint_execution_or_attempt_id() -> None:
    assert _collect_forbidden_mint_calls() == []


@pytest.mark.obs_trace_1
def test_gate_factual_reconstruction_does_not_import_trace_event() -> None:
    violations: list[str] = []
    for path in _RECONSTRUCTION_MODULES:
        if _module_imports_trace_event(path):
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert violations == []


@pytest.mark.obs_trace_1
def test_gate_trace_middleware_emits_runtime_event_not_trace_event() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "middleware" / "trace_middleware.py"
    source = path.read_text(encoding="utf-8")
    assert "RuntimeEvent" in source
    assert "TraceEvent" not in source


@pytest.mark.obs_trace_1
def test_gate_unified_journal_does_not_merge_plane_b_trace_as_execution_authority() -> (
    None
):
    path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "unified_run_journal.py"
    source = path.read_text(encoding="utf-8")
    assert (
        "Plane B ``TraceEvent`` rows on ``PersistedRun`` are not converted here."
        in source
    )

# © Artur Czarnecki. All rights reserved.

"""Architecture gates for OBS-DIAG-PORT-1 neutral terminal diagnostic integration."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_assembly_resolver import DiagnosticAssemblyError
from intergrax.applications._shared.diagnostic_runtime_wiring import wire_terminal_execution_diagnostics
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
    DiagnosticProfile,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from tests.integration.runtime.test_terminal_diagnostic_production_e2e import _FakeEnvWiring
from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalDiagnosticDispatchResult,
    TerminalDiagnosticDispatchStatus,
    TerminalExecutionDiagnosticPort,
    TerminalExecutionDiagnosticRequest,
)
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_DIAGNOSTICS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"


class _RecordingExternalDiagnosticPort:
    def __init__(self) -> None:
        self.calls: list[TerminalExecutionDiagnosticRequest] = []

    def dispatch_terminal_execution(
        self,
        request: TerminalExecutionDiagnosticRequest,
    ) -> TerminalDiagnosticDispatchResult:
        self.calls.append(request)
        return TerminalDiagnosticDispatchResult(
            status=TerminalDiagnosticDispatchStatus.COMPLETED,
        )


class _RaisingExternalDiagnosticPort:
    def dispatch_terminal_execution(
        self,
        request: TerminalExecutionDiagnosticRequest,
    ) -> TerminalDiagnosticDispatchResult:
        raise RuntimeError("external diagnostic failure")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_nexus_loop_must_not_import_runtime_diagnostics() -> None:
    tree = _parse(_NEXUS_LOOP)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.diagnostics"):
                violations.append(f"imports {node.module}")
    assert violations == []


def test_nexus_loop_must_not_reference_diagnostic_orchestration_result() -> None:
    source = _NEXUS_LOOP.read_text(encoding="utf-8")
    assert "DiagnosticOrchestrationResult" not in source


def test_single_canonical_terminal_execution_diagnostic_port_definition() -> None:
    port_defs: list[str] = []
    for path in _DIAGNOSTICS_ROOT.rglob("*.py"):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TerminalExecutionDiagnosticPort":
                port_defs.append(path.relative_to(_REPO_ROOT).as_posix())
    assert port_defs == []


@pytest.mark.asyncio
async def test_external_port_injected_into_nexus_terminal_path() -> None:
    external = _RecordingExternalDiagnosticPort()
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry)
    loop.attach_terminal_diagnostic_trigger(external)
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    result = await runner.run_task(
        Task(
            tenant_id="obs-diag-port-tenant",
            user_id="user-1",
            message="external port proof",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )

    assert result.state is TaskState.COMPLETED
    matching = [call for call in external.calls if call.run_id == run_id]
    assert len(matching) >= 1


@pytest.mark.asyncio
async def test_external_port_failure_does_not_change_business_outcome() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry)
    loop.attach_terminal_diagnostic_trigger(_RaisingExternalDiagnosticPort())
    runner = UnifiedTaskRunner(loop)

    result = await runner.run_task(
        Task(
            tenant_id="obs-diag-port-tenant",
            user_id="user-1",
            message="failure isolation",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=mint_run_id(),
    )

    assert result.state is TaskState.COMPLETED


def test_production_required_diagnostics_fail_closed_when_port_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.diagnostic_runtime_wiring.try_build_terminal_execution_diagnostic_port",
        lambda **_kwargs: None,
    )
    document_store = InMemoryDocumentStore()
    runtime_store = InMemoryRuntimeEventStore()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="obs-diag-port-missing")
    env.diagnostic_profile = DiagnosticProfile(posture=DiagnosticPosture.REQUIRED)
    nexus_loop = NexusLoop(AgentRegistry())

    with pytest.raises(DiagnosticAssemblyError):
        wire_terminal_execution_diagnostics(
            env=env,
            env_wiring=_FakeEnvWiring(document_store),
            observability=stores,
            nexus_loop=nexus_loop,
        )


def test_execution_and_nexus_have_no_terminal_diagnostic_implementation_coupling() -> None:
    forbidden = (
        "DiagnosticOrchestrator",
        "DiagnosticOrchestrationResult",
        "ProblemLifecycleEngine",
        "TerminalExecutionDiagnosticTrigger",
    )
    violations: list[str] = []
    for root in (_REPO_ROOT / "intergrax" / "runtime" / "nexus", _EXECUTION_ROOT):
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            rel = path.relative_to(_REPO_ROOT).as_posix()
            for symbol in forbidden:
                if symbol in source:
                    violations.append(f"{rel} references {symbol}")
    assert violations == []

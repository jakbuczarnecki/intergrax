# © Artur Czarnecki. All rights reserved.

"""U4 — delegated subtask child execution closure static and contract gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.agent_distribution.delegated_subtasks import DelegatedSubtaskDelegate
from intergrax.applications._shared.production_delegated_subtask_child_execution_wiring import (
    ProductionDelegatedSubtaskChildExecutionPort,
    build_production_delegated_subtask_child_execution_port,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.execution_work_port import (
    DelegatedSpecialistChildWorkEnvelope,
    DelegatedSubtaskChildExecutionWorkPort,
    delegated_subtask_child_execution_work_port,
)
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FACTORY_RUNTIME = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "production_agent_capability_runtime.py"
)
_CHILD_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "production_delegated_subtask_child_execution_wiring.py"
)
_AC4_E2E = (
    _REPO_ROOT / "tests" / "unit" / "applications" / "test_ac4_phase9_production_composition_e2e.py"
)
_U4_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md"
)
_WORK_PORT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "execution_work_port.py"


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _factory_create_child_execution_optional(factory_source: str) -> bool:
    tree = ast.parse(factory_source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "DelegatedSubtaskServiceFactory":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "create":
                continue
            for arg in item.args.kwonlyargs:
                if arg.arg == "child_execution" and arg.annotation is None:
                    return True
                if arg.arg != "child_execution":
                    continue
                ann = arg.annotation
                if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
                    if isinstance(ann.right, ast.Constant) and ann.right.value is None:
                        return True
    return False


def _constructs_child_execution_runner(source: str, *, filename: str) -> bool:
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "ChildExecutionRunner":
            return True
        if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
            if func.value.id == "ChildExecutionRunner":
                return True
    return False


def test_u4_qualification_artifact_present() -> None:
    assert _U4_QUALIFICATION.is_file()


def test_u4_factory_requires_child_execution_and_does_not_construct_runner() -> None:
    source = _FACTORY_RUNTIME.read_text(encoding="utf-8")
    assert "ChildExecutionRunner" not in source
    assert "as_child_execution_port" not in source
    assert not _factory_create_child_execution_optional(source)
    assert not _constructs_child_execution_runner(source, filename=_rel(_FACTORY_RUNTIME))


def test_u4_composition_root_wires_canonical_child_execution_adapter() -> None:
    wiring = _CHILD_WIRING.read_text(encoding="utf-8")
    assert "build_production_delegated_subtask_child_execution_port" in wiring
    assert "delegated_subtask_child_execution_work_port" in wiring
    assert "child_execution_port_from_work_port" in wiring
    assert not _constructs_child_execution_runner(wiring, filename=_rel(_CHILD_WIRING))
    work_port = _WORK_PORT.read_text(encoding="utf-8")
    assert "class DelegatedSubtaskChildExecutionWorkPort" in work_port
    assert "ChildExecutionRunner" in work_port


def test_u4_production_e2e_passes_composition_root_child_port() -> None:
    source = _AC4_E2E.read_text(encoding="utf-8")
    assert "delegated_subtask_child_execution.port()" in source
    assert "child_execution=capability_runtime.delegated_subtask_child_execution.port()" in source


def test_u4_capability_runtime_exposes_delegated_subtask_child_execution_binding() -> None:
    source = _FACTORY_RUNTIME.read_text(encoding="utf-8")
    assert "delegated_subtask_child_execution: ProductionDelegatedSubtaskChildExecutionPort" in source
    assert "build_production_delegated_subtask_child_execution_port" in source


def test_u4_composition_root_does_not_import_child_execution_runner() -> None:
    wiring = _CHILD_WIRING.read_text(encoding="utf-8")
    tree = ast.parse(wiring, filename=_rel(_CHILD_WIRING))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "intergrax.runtime.execution.child":
            continue
        for alias in node.names:
            assert alias.name != "ChildExecutionRunner"


_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


@dataclass(frozen=True)
class _Ping:
    value: str


@dataclass(frozen=True)
class _Pong:
    value: str


class _EchoSpecialist(DelegatedSubtaskDelegate[_Ping, _Pong]):
    async def execute(self, request: _Ping) -> _Pong:
        return _Pong(value=request.value)


@dataclass
class _RecordingDelegatedSubtaskChildWorkPort:
    inner: DelegatedSubtaskChildExecutionWorkPort[_Ping, _Pong]
    execute_calls: list[
        ExecutionRequest[DelegatedSpecialistChildWorkEnvelope[_Ping, _Pong], _Pong]
    ] = field(default_factory=list)

    async def execute(
        self,
        request: ExecutionRequest[
            DelegatedSpecialistChildWorkEnvelope[_Ping, _Pong],
            _Pong,
        ],
    ) -> _Pong:
        self.execute_calls.append(request)
        return await self.inner.execute(request)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _root_authority() -> ParentExecutionAuthority:
    return ParentExecutionAuthority.unrestricted_root()


@pytest.mark.asyncio
async def test_u4_production_child_port_invokes_canonical_work_port() -> None:
    inner = delegated_subtask_child_execution_work_port(ledger=_UNLIMITED_LEDGER)
    recording = _RecordingDelegatedSubtaskChildWorkPort(inner=inner)
    binding = ProductionDelegatedSubtaskChildExecutionPort(_work_port=recording)
    child_port = binding.port()
    child_execution_id: ExecutionId | None = None
    parent_at_child: ExecutionId | None = None

    class _LineageSpecialist(DelegatedSubtaskDelegate[_Ping, _Pong]):
        async def execute(self, request: _Ping) -> _Pong:
            nonlocal child_execution_id, parent_at_child
            child_execution_id = require_active_execution_id()
            parent_at_child = peek_active_parent_execution_id()
            return _Pong(value=request.value)

    root = _root_identity()

    class _RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            return await child_port.execute_child(
                request=request,
                delegate=_LineageSpecialist(),
            )

    boundary = ExecutionBoundary[_Ping, _Pong](
        _RootDelegate(),
        identity=root,
        authority=_root_authority(),
    )
    result = await boundary.execute(_Ping(value="u4"))
    assert result.value == "u4"
    assert len(recording.execute_calls) == 1
    envelope = recording.execute_calls[0].input
    assert isinstance(envelope, DelegatedSpecialistChildWorkEnvelope)
    assert envelope.domain_request.value == "u4"
    assert child_execution_id is not None
    assert child_execution_id != root.execution_id
    assert parent_at_child == root.execution_id
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_u4_build_production_binding_uses_work_port_factory() -> None:
    binding = build_production_delegated_subtask_child_execution_port(
        ledger=_UNLIMITED_LEDGER,
    )
    child_port = binding.port()
    root = _root_identity()

    class _RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            return await child_port.execute_child(
                request=request,
                delegate=_EchoSpecialist(),
            )

    boundary = ExecutionBoundary[_Ping, _Pong](
        _RootDelegate(),
        identity=root,
        authority=_root_authority(),
    )
    pong = await boundary.execute(_Ping(value="factory"))
    assert pong.value == "factory"

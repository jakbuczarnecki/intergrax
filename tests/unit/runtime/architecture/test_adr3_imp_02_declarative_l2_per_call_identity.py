# © Artur Czarnecki. All rights reserved.

"""ADR3-IMP-02 — declarative L2 per-call immutable execution identity (M3 gates)."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvokeResult
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.knowledge.contracts.validation import JsonObject
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_L2_CONTRACT = _REPO / "intergrax/contracts/execution_bound_declarative_tool_invocation.py"
_NEXUS_ADAPTER = _REPO / "intergrax/runtime/nexus/agents/catalog_declarative_invoker.py"
_COMP_SESSION = _REPO / "intergrax/agents/persistence/compensation_tool_invoke_session.py"

_IDENTITY_MINT_FORBIDDEN = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "mint_task_id",
    }
)


def _protocol_method_names(source_path: Path, class_name: str) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _collect_call_names(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_adr3_imp_02_l2_protocol_has_no_bind_execution_identity() -> None:
    methods = _protocol_method_names(
        _L2_CONTRACT,
        "ExecutionBoundDeclarativeToolInvoker",
    )
    assert "bind_execution_identity" not in methods
    assert "invoke" in methods


def test_adr3_imp_02_l2_contract_is_nexus_free() -> None:
    source = _L2_CONTRACT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "runtime.nexus" not in node.module
            assert "nexus" not in node.module.split(".")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "nexus" not in alias.name


def test_adr3_imp_02_compensation_session_does_not_bind_before_invoke() -> None:
    source = _COMP_SESSION.read_text(encoding="utf-8")
    assert "bind_execution_identity" not in source


def test_adr3_imp_02_nexus_adapter_does_not_mint_identity() -> None:
    calls = _collect_call_names(_NEXUS_ADAPTER)
    violations = sorted(_IDENTITY_MINT_FORBIDDEN & calls)
    assert violations == []


@dataclass
class _NexusFreeDeclarativeInvoker:
    """Custom L2 implementation with no Nexus dependency or mutable bind."""

    observed_run_ids: list[str] = field(default_factory=list)

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        _ = tenant_id, task_id, agent_id, tool_id, args, idempotency_key
        self.observed_run_ids.append(run_id)
        return DeclarativeToolInvokeResult(status="success")


def test_adr3_imp_02_custom_impl_satisfies_protocol_without_nexus() -> None:
    invoker = _NexusFreeDeclarativeInvoker()
    assert isinstance(invoker, ExecutionBoundDeclarativeToolInvoker)
    source = inspect.getsource(_NexusFreeDeclarativeInvoker)
    assert "intergrax.runtime.nexus" not in source
    assert "bind_execution_identity" not in source


@pytest.mark.asyncio
async def test_adr3_imp_02_per_call_identity_projected() -> None:
    invoker = _NexusFreeDeclarativeInvoker()
    run_id = str(mint_run_id())
    await invoker.invoke(
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=str(_TASK_ID),
        agent_id="agent-adr3-m3",
        tool_id="noop",
        args={},
        idempotency_key="k1",
    )
    assert invoker.observed_run_ids == [run_id]


@pytest.mark.asyncio
async def test_adr3_imp_02_sequential_invocations_do_not_leak_identity() -> None:
    invoker = _NexusFreeDeclarativeInvoker()
    run_a = str(mint_run_id())
    run_b = str(mint_run_id())
    common = dict(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        agent_id="agent-adr3-m3",
        tool_id="noop",
        args={},
        idempotency_key=None,
    )
    await invoker.invoke(run_id=run_a, **common)
    await invoker.invoke(run_id=run_b, **common)
    assert invoker.observed_run_ids == [run_a, run_b]


@pytest.mark.asyncio
async def test_adr3_imp_02_compensation_session_forwards_identity_to_invoke() -> None:
    invoker = _NexusFreeDeclarativeInvoker()
    session = bound_compensation_tool_invoke_session(cast(ExecutionBoundDeclarativeToolInvoker, invoker))
    run_id = str(mint_run_id())
    task_id = str(mint_task_id())
    await session.invoke(
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
        agent_id="agent-comp",
        tool_id="noop",
        args={},
        idempotency_key="idem-1",
    )
    assert invoker.observed_run_ids == [run_id]


def test_adr3_imp_02_catalog_adapter_has_no_bind_execution_identity() -> None:
    methods = _protocol_method_names(_NEXUS_ADAPTER, "CatalogDeclarativeToolInvoker")
    assert "bind_execution_identity" not in methods


def test_adr3_imp_02_catalog_invoke_requires_explicit_per_call_identity() -> None:
    invoker = CatalogDeclarativeToolInvoker(tool_invoker=cast(object, lambda *a, **k: None))  # type: ignore[arg-type]
    run_id = str(mint_run_id())
    state = invoker._runtime_state(  # noqa: SLF001
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=str(_TASK_ID),
        agent_id="agent-catalog",
        user_id="",
    )
    assert str(state.run_id) == run_id
    assert state.request.tenant_id == _TENANT
    assert state.request.agent_id == "agent-catalog"

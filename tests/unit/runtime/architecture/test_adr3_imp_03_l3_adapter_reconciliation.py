# © Artur Czarnecki. All rights reserved.

"""ADR3-IMP-03 — L3 declarative adapter reconciliation (M4 gates)."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.tool_request import ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeRunBinding,
    CatalogDeclarativeToolInvoker,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_NEXUS_ADAPTER = _REPO / "intergrax/runtime/nexus/agents/catalog_declarative_invoker.py"
_L2_CONTRACT = _REPO / "intergrax/contracts/execution_bound_declarative_tool_invocation.py"

_IDENTITY_MINT_FORBIDDEN = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "mint_task_id",
    }
)

_BINDING_IDENTITY_ATTRS = frozenset({"run_id", "task_id", "agent_id", "tenant_id"})


def _adapter_source() -> str:
    return _NEXUS_ADAPTER.read_text(encoding="utf-8")


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


def _collect_binding_identity_reads(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Attribute):
            continue
        if not isinstance(node.value.value, ast.Name):
            continue
        if node.value.value.id != "self" or node.value.attr != "binding":
            continue
        if node.attr in _BINDING_IDENTITY_ATTRS:
            violations.append(node.attr)
    return sorted(set(violations))


def test_adr3_imp_03_no_resolve_invoke_identity_helper() -> None:
    assert "_resolve_invoke_identity" not in _adapter_source()


def test_adr3_imp_03_binding_has_no_execution_identity_fields() -> None:
    fields = {f.name for f in CatalogDeclarativeRunBinding.__dataclass_fields__.values()}
    assert _BINDING_IDENTITY_ATTRS.isdisjoint(fields)


def test_adr3_imp_03_invoke_identity_does_not_read_binding_execution_fields() -> None:
    violations = _collect_binding_identity_reads(_adapter_source())
    assert violations == []


def test_adr3_imp_03_invoke_requires_explicit_identity_parameters() -> None:
    sig = inspect.signature(CatalogDeclarativeToolInvoker.invoke)
    for name in ("tenant_id", "run_id", "task_id", "agent_id"):
        param = sig.parameters[name]
        assert param.default is inspect.Parameter.empty
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


def test_adr3_imp_03_nexus_adapter_does_not_mint_identity() -> None:
    violations = sorted(_IDENTITY_MINT_FORBIDDEN & _collect_call_names(_NEXUS_ADAPTER))
    assert violations == []


def test_adr3_imp_03_l2_contract_still_nexus_free() -> None:
    source = _L2_CONTRACT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "nexus" not in node.module


def test_adr3_imp_03_catalog_not_reexported_from_contracts() -> None:
    contracts_root = _REPO / "intergrax" / "contracts"
    for path in contracts_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "CatalogDeclarativeToolInvoker" not in text
        assert "catalog_declarative_invoker" not in text


@dataclass
class _RecordingRuntimeToolInvoker:
    observed_run_ids: list[str] = field(default_factory=list)

    def invoke(
        self,
        state: object,
        agent_id: str,
        request: object,
    ) -> object:
        runtime_state = cast(RuntimeState, state)
        self.observed_run_ids.append(str(runtime_state.run_id))
        _ = agent_id, request
        from intergrax.tools.execution_models import ToolExecutionResult

        return ToolExecutionResult.fail("tool_error", "stub")


def _catalog_invoker(recorder: _RecordingRuntimeToolInvoker) -> CatalogDeclarativeToolInvoker:
    return CatalogDeclarativeToolInvoker(
        tool_invoker=cast(RuntimeToolInvoker, recorder),
        production_mode=False,
    )


@pytest.mark.asyncio
async def test_adr3_imp_03_interleaved_invocations_project_distinct_runtime_state() -> None:
    recorder = _RecordingRuntimeToolInvoker()
    invoker = _catalog_invoker(recorder)
    run_a = str(mint_run_id())
    run_b = str(mint_run_id())
    common = dict(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        agent_id="agent-m4",
        tool_id="noop.tool",
        args={},
        idempotency_key=None,
    )

    def _ok_dispatch(state: object, request: object, trace_step: str) -> ToolResponse:
        runtime_state = cast(RuntimeState, state)
        recorder.observed_run_ids.append(str(runtime_state.run_id))
        _ = request, trace_step
        return ToolResponse(
            request_id="req-1",
            status=ToolResponseStatus.SUCCESS,
            output={},
        )

    with patch(
        "intergrax.runtime.nexus.agents.catalog_declarative_invoker.invoke_catalog_tool_request",
        side_effect=_ok_dispatch,
    ):
        await invoker.invoke(run_id=run_a, **common)
        await invoker.invoke(run_id=run_b, **common)
    assert recorder.observed_run_ids == [run_a, run_b]


def test_adr3_imp_03_runtime_state_projection_from_explicit_invoke_args() -> None:
    invoker = _catalog_invoker(_RecordingRuntimeToolInvoker())
    run_id = str(mint_run_id())
    state = invoker._runtime_state(  # noqa: SLF001
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=str(_TASK_ID),
        agent_id="agent-m4",
        user_id="user-1",
    )
    assert str(state.run_id) == run_id
    assert state.request.tenant_id == _TENANT
    assert state.request.agent_id == "agent-m4"
    assert state.request.user_id == "user-1"


def test_adr3_imp_03_bind_run_does_not_mutate_execution_identity_on_binding() -> None:
    invoker = _catalog_invoker(_RecordingRuntimeToolInvoker())
    before = CatalogDeclarativeRunBinding(
        user_id=invoker.binding.user_id,
        declarative_hitl_grant=invoker.binding.declarative_hitl_grant,
    )
    invoker.bind_run(
        run_id=str(mint_run_id()),
        task_id=str(mint_task_id()),
        agent_id="ignored",
        tenant_id="ignored",
        user_id="session-user",
    )
    assert invoker.binding.user_id == "session-user"
    assert not hasattr(invoker.binding, "run_id")
    assert before.declarative_hitl_grant == invoker.binding.declarative_hitl_grant


@pytest.mark.asyncio
async def test_adr3_imp_03_missing_identity_fails_closed() -> None:
    invoker = _catalog_invoker(_RecordingRuntimeToolInvoker())
    with pytest.raises(ValueError, match="explicit tenant_id"):
        await invoker.invoke(
            tenant_id="",
            run_id=str(mint_run_id()),
            task_id=str(_TASK_ID),
            agent_id="agent-m4",
            tool_id="noop",
            args={},
            idempotency_key=None,
        )

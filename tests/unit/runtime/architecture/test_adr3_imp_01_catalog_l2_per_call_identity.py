# © Artur Czarnecki. All rights reserved.

"""ADR3-IMP-01 — catalog L2 per-call immutable execution identity (M2 gates)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_L2_CONTRACT = _REPO / "intergrax/contracts/execution_bound_catalog_tool_invocation.py"
_NEXUS_ADAPTER = (
    _REPO / "intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py"
)
_CODECRAFT_WIRING = (
    _REPO / "intergrax/runtime/codecraft/wiring_bound_capability_execution.py"
)

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


def test_adr3_imp_01_l2_protocol_has_no_bind_execution_identity() -> None:
    methods = _protocol_method_names(
        _L2_CONTRACT,
        "ExecutionBoundCatalogToolInvoker",
    )
    assert "bind_execution_identity" not in methods
    assert "invoke" in methods


def test_adr3_imp_01_l2_contract_is_nexus_free() -> None:
    source = _L2_CONTRACT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "runtime.nexus" not in node.module
            assert "nexus" not in node.module.split(".")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "nexus" not in alias.name


def test_adr3_imp_01_codecraft_wiring_does_not_bind_catalog_identity() -> None:
    source = _CODECRAFT_WIRING.read_text(encoding="utf-8")
    assert "bind_execution_identity" not in source


def test_adr3_imp_01_nexus_adapter_does_not_mint_identity() -> None:
    calls = _collect_call_names(_NEXUS_ADAPTER)
    violations = sorted(_IDENTITY_MINT_FORBIDDEN & calls)
    assert violations == []


@dataclass
class _RecordingRuntimeToolInvoker:
    observed_run_ids: list[str] = field(default_factory=list)

    def invoke(
        self,
        state: object,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:
        from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

        runtime_state = cast(RuntimeState, state)
        self.observed_run_ids.append(str(runtime_state.run_id))
        _ = agent_id, request
        return ToolExecutionResult.fail("tool_error", "stub")


def _catalog_invoker(
    recorder: _RecordingRuntimeToolInvoker,
) -> NexusExecutionBoundCatalogToolInvoker:
    return NexusExecutionBoundCatalogToolInvoker(
        tool_invoker=cast(object, recorder),  # type: ignore[arg-type]
        policy_bundle=RuntimePolicyBundle(),
        caller_agent_id="worker-adr3-test",
        production_mode=False,
    )


def _invoke_request(run_id: str, step_id: str) -> ExecutionBoundCatalogToolInvokeRequest:
    return ExecutionBoundCatalogToolInvokeRequest(
        tool_id="sandbox.code_exec",
        input=CodeExecInput(code="1", language="python", timeout_s=5),
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        agent_id="worker-adr3-test",
        step_id=step_id,
    )


def test_adr3_imp_01_invoke_uses_request_identity_without_prior_bind() -> None:
    recorder = _RecordingRuntimeToolInvoker()
    catalog = _catalog_invoker(recorder)
    run_id = str(mint_run_id())
    catalog.invoke(_invoke_request(run_id, "adr3.imp01:a"))
    assert recorder.observed_run_ids == [run_id]
    assert catalog.binding.user_id == ""


def test_adr3_imp_01_sequential_invocations_do_not_leak_identity() -> None:
    recorder = _RecordingRuntimeToolInvoker()
    catalog = _catalog_invoker(recorder)
    run_a = str(mint_run_id())
    run_b = str(mint_run_id())
    catalog.invoke(_invoke_request(run_a, "adr3.imp01:seq-a"))
    catalog.invoke(_invoke_request(run_b, "adr3.imp01:seq-b"))
    assert recorder.observed_run_ids == [run_a, run_b]


def test_adr3_imp_01_runtime_state_projection_matches_request() -> None:
    catalog = _catalog_invoker(_RecordingRuntimeToolInvoker())
    run_id = str(mint_run_id())
    request = _invoke_request(run_id, "adr3.imp01:projection")
    state = catalog._runtime_state(request)
    assert str(state.run_id) == run_id
    assert state.request.run_id == run_id
    assert state.request.tenant_id == _TENANT
    assert state.request.agent_id == "worker-adr3-test"


def test_adr3_imp_01_invalid_run_id_fails_closed() -> None:
    catalog = _catalog_invoker(_RecordingRuntimeToolInvoker())
    bad = _invoke_request("not-a-valid-run-id", "adr3.imp01:bad")
    with pytest.raises(ValueError):
        catalog.invoke(bad)

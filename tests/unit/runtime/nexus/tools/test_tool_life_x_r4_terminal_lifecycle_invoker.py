# © Artur Czarnecki. All rights reserved.

"""R4 E1/E3 — RuntimeToolInvoker durable terminal recording on physical success."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import canonical_governed_execution_scope, canonical_run_id_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = canonical_run_id_for_tests("tool-life-r4-terminal")


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


class _Handler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(value=request.input.value)


class _InvokeState:
    tenant_id = "tenant-r4-terminal"
    run_id = _RUN_ID
    task_id = None
    verified_agent_governance_human_approval = None
    agent_governance_approval_consumption = None
    declarative_hitl_grant = None

    def __init__(self) -> None:
        self.request = type("Req", (), {"metadata": {}})()
        self._context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": None, "production_mode": False})()},
        )()

    @property
    def context(self):
        return self._context

    def trace_event(self, *args: object, **kwargs: object) -> None:
        return


def test_no_dependency_boundary_success_records_succeeded_before_return() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="terminal.success",
        name="terminal",
        description="",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )
    registry.register(contract, _Handler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        external_operation_store=store,
        external_operation_owner=owner,
        dependency_attempt_boundary=None,
    )
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="1",
        tool_id="terminal.success",
        input=_In(value=7),
    )
    with canonical_governed_execution_scope(_RUN_ID):
        out = invoker._execute_once(  # noqa: SLF001 — E1 durable terminal path under test
            _InvokeState(),
            "agent",
            contract,
            request,
        )
    assert out.value == 7
    records = list(store._records.values())  # noqa: SLF001 — qualification inspects durable store
    assert len(records) == 1
    record = next(iter(records))
    assert record.physical_state is ExternalOperationPhysicalState.SUCCEEDED

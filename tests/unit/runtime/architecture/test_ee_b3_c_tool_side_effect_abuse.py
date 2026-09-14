# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-05 tool side-effect and registry abuse."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import SideEffectRetrySafety, ToolContract
from intergrax.tools.execution_models import (
    ToolEffectCertainty,
    ToolExecutionRequest,
)
from intergrax.tools.registry import ToolRegistry
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        self.calls += 1
        return ValueOutput(result=request.input.value)


class _MinimalConfig:
    policy_bundle = None
    production_mode = False


class _MinimalContext:
    config = _MinimalConfig()


class MinimalState:
    run_id = "run_01234567890123456789012345678901"
    task_id = "task_01234567890123456789012345678901"
    tenant_id = "tenant-b3c"
    declarative_hitl_grant = None
    context = _MinimalContext()

    def trace_event(self, **kwargs: object) -> None:
        del kwargs


def _mutating_contract(tool_id: str = "abuse.mutate") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name="Mutate",
        description="mutating",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        side_effect_retry_safety=SideEffectRetrySafety.IDEMPOTENT,
        category="mutate",
    )


def test_ee_b3_c_unknown_tool_not_executed() -> None:
    executor = CountingExecutor()
    registry = ToolRegistry()
    invoker = RuntimeToolInvoker(registry=registry, executor=executor)  # type: ignore[arg-type]
    result = invoker.invoke(
        state=MinimalState(),  # type: ignore[arg-type]
        agent_id="agent",
        request=ToolExecutionRequest(
            run_id=MinimalState.run_id,
            step_id="s1",
            tool_id="forged.tool.path",
            input=ValueInput(value=1),
        ),
    )
    assert result.success is False
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED
    assert executor.calls == 0


def test_ee_b3_c_mutating_tool_without_side_effect_authorization_zero_calls() -> None:
    executor = CountingExecutor()
    contract = _mutating_contract()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=executor,  # type: ignore[arg-type]
        pre_effect_coordinator=IdempotencyPreEffectCoordinator(
            idempotency_store=InMemoryIdempotencyStore(),
        ),
    )
    with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError):
        invoker.invoke(
            state=MinimalState(),  # type: ignore[arg-type]
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=MinimalState.run_id,
                step_id="s1",
                tool_id=contract.tool_id,
                input=ValueInput(value=3),
                idempotency_key="b3c-no-auth",
            ),
        )
    assert executor.calls == 0


def test_ee_b3_c_arbitrary_import_path_not_used_in_invoker() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    invoker_source = (
        repo / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
    ).read_text(encoding="utf-8")
    assert "importlib" not in invoker_source

# © Artur Czarnecki. All rights reserved.

"""U2 contract tests — admitted compensation side-effect execution."""

from __future__ import annotations

import pytest

from intergrax.agents.persistence.compensation_side_effect_input import (
    compensation_side_effect_input_from_job,
)
from intergrax.agents.persistence.compensation_queue_store import CompensationJob
from intergrax.agents.persistence.compensation_tool_invoke_session import (
    bound_compensation_tool_invoke_session,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.contracts.compensation_side_effect_execution import CompensationSideEffectInput
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_run_id, require_active_execution_identity
from intergrax.contracts.side_effect import CompensationRequest
from intergrax.runtime.execution.compensation_side_effect import (
    build_runtime_compensation_side_effect_execution,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sample_input(*, run_id: str | None = None) -> CompensationSideEffectInput:
    resolved_run_id = run_id or str(mint_run_id())
    job = CompensationJob(
        run_id=resolved_run_id,
        tenant_id="tenant-u2",
        agent_id="agent-u2",
        step_index=2,
        request=CompensationRequest(
            original_side_effect_id="se-u2",
            compensation_tool_id="tool.compensate",
            args={"k": "v"},
            idempotency_key="comp:acp:u2",
        ),
    )
    return compensation_side_effect_input_from_job(job)


@pytest.mark.asyncio
async def test_u2_compensation_tool_runs_under_active_execution_identity() -> None:
    observed_run_id: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        active_run_id, _ = require_active_execution_identity()
        observed_run_id.append(str(active_run_id))
        assert kwargs["idempotency_key"] == "comp:acp:u2"
        return DeclarativeToolInvokeResult(status="success")

    execution = build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(
            CallableDeclarativeToolInvoker(_invoke),
        ),
        authority=ParentExecutionAuthority.unrestricted_root(),
    )
    work = _sample_input()
    result = await execution.execute(work)
    assert result.status == "success"
    assert observed_run_id == [work.run_id]


@pytest.mark.asyncio
async def test_u2_authority_unknown_denies_before_tool_invoke() -> None:
    invoked = False

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoked
        invoked = True
        return DeclarativeToolInvokeResult(status="success")

    execution = build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(
            CallableDeclarativeToolInvoker(_invoke),
        ),
        authority=ParentExecutionAuthority.unknown(),
    )
    result = await execution.execute(_sample_input())
    assert invoked is False
    assert result.status == "denied"


@pytest.mark.asyncio
async def test_u2_governance_denial_surfaces_as_failed_claim_semantics() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="denied", error="policy.denied")

    execution = build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(
            CallableDeclarativeToolInvoker(_invoke),
        ),
        authority=ParentExecutionAuthority.unrestricted_root(),
    )
    result = await execution.execute(_sample_input())
    assert result.status == "denied"
    assert result.error == "policy.denied"


@pytest.mark.asyncio
async def test_u2_production_wiring_builds_admitted_execution_port() -> None:
    from dataclasses import dataclass

    from intergrax.applications._shared.compensation_side_effect_wiring import (
        build_compensation_side_effect_execution,
    )
    from intergrax.contracts.compensation_side_effect_execution import (
        CompensationSideEffectExecutionPort,
    )

    @dataclass
    class _NexusStub:
        execution_budget_ledger_factory: object | None = None
        run_budget: object | None = None
        execution_lineage_persistence: object | None = None

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    port = build_compensation_side_effect_execution(
        _NexusStub(),  # type: ignore[arg-type]
        CallableDeclarativeToolInvoker(_invoke),
    )
    assert isinstance(port, CompensationSideEffectExecutionPort)
    work = _sample_input()
    assert (await port.execute(work)).status == "success"


@pytest.mark.asyncio
async def test_u2_lineage_preserves_parent_run_id() -> None:
    parent_run = str(mint_run_id())
    work = _sample_input(run_id=parent_run)

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        run_id, _ = require_active_execution_identity()
        assert str(run_id) == parent_run
        return DeclarativeToolInvokeResult(status="success")

    execution = build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(
            CallableDeclarativeToolInvoker(_invoke),
        ),
        authority=ParentExecutionAuthority.unrestricted_root(),
    )
    assert (await execution.execute(work)).status == "success"

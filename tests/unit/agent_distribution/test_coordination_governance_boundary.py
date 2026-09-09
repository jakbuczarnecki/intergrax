# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1 — coordination intent executor governance boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutOrchestrationPort,
    FanOutRequest,
)
from intergrax.agent_distribution.coordination_governance_adapter import (
    build_multi_agent_coordination_governance_request,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContributionId,
    CoordinationExecutionMode,
)
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationGovernanceRequiresHuman,
    CoordinationIntentExecutor,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationId,
    CoordinationRequest,
    CoordinationResult,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationGovernanceRequest,
)
from intergrax.contracts.execution_identity import mint_task_id
from testing_support.agent_distribution.coordination_governance import (
    allowing_coordination_governance,
    denying_coordination_governance,
    require_human_coordination_governance,
    unavailable_coordination_governance,
)
from tests.unit.agent_distribution.test_coordination_intent import (
    _fan_out_intent,
    _single_intent,
)
from tests.unit.agent_distribution.test_coordination_intent_executor import (
    _StaticOrchestrationPort,
    _TrackingCoordinationService,
    _TrackingFanOutService,
    _binding,
    _success_outcome,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    admin_test_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _RecordingGovernance:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0
        self.last_request: MultiAgentCoordinationGovernanceRequest | None = None

    def evaluate(self, request: MultiAgentCoordinationGovernanceRequest):
        self.calls += 1
        self.last_request = request
        return self._inner.evaluate(request)


def _executor(
    *,
    coordination: _TrackingCoordinationService | None = None,
    fan_out: _TrackingFanOutService | None = None,
    governance,
) -> tuple[
    CoordinationIntentExecutor[OcrRequest, OcrResult],
    _TrackingCoordinationService,
    _TrackingFanOutService,
    _RecordingGovernance,
]:
    coordination = coordination or _TrackingCoordinationService()
    fan_out = fan_out or _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    recording = _RecordingGovernance(governance)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=recording,
    )
    return executor, coordination, fan_out, recording


@pytest.mark.asyncio
async def test_single_allow_executes_coordination_path() -> None:
    executor, coordination, fan_out, governance = _executor(
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert governance.calls == 1
    assert governance.last_request is not None
    assert governance.last_request.execution_mode.value == "single"
    assert coordination.calls == 1
    assert fan_out.calls == 0
    assert result.mode is CoordinationExecutionMode.SINGLE


@pytest.mark.asyncio
async def test_single_deny_blocks_coordination_and_fan_out() -> None:
    executor, coordination, fan_out, governance = _executor(
        governance=denying_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    with pytest.raises(CoordinationGovernanceDenied):
        await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )

    assert governance.calls == 1
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_fan_out_allow_executes_bounded_fan_out_service() -> None:
    orchestration = _StaticOrchestrationPort(
        (
            _success_outcome("contrib-a", "a"),
            _success_outcome("contrib-b", "b"),
        ),
    )
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=orchestration),
    )
    executor, coordination, fan_out, governance = _executor(
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    binding = _binding(
        task_scope,
        pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
        ),
    )

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert governance.calls == 1
    assert fan_out.calls == 1
    assert orchestration.calls == 1
    assert coordination.calls == 0
    assert result.mode is CoordinationExecutionMode.FAN_OUT


@pytest.mark.asyncio
async def test_fan_out_deny_blocks_fan_out_and_orchestration() -> None:
    orchestration = _StaticOrchestrationPort(
        (
            _success_outcome("contrib-a", "a"),
            _success_outcome("contrib-b", "b"),
        ),
    )
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=orchestration),
    )
    executor, coordination, fan_out, governance = _executor(
        fan_out=fan_out,
        governance=denying_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b"))
    binding = _binding(
        task_scope,
        pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
        ),
    )

    with pytest.raises(CoordinationGovernanceDenied):
        await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )

    assert governance.calls == 1
    assert fan_out.calls == 0
    assert orchestration.calls == 0
    assert coordination.calls == 0


@pytest.mark.asyncio
async def test_require_human_blocks_execution_before_approval() -> None:
    executor, coordination, fan_out, governance = _executor(
        governance=require_human_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    with pytest.raises(CoordinationGovernanceRequiresHuman):
        await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )

    assert governance.calls == 1
    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_unavailable_governance_fail_closed() -> None:
    executor, coordination, fan_out, governance = _executor(
        governance=unavailable_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    with pytest.raises(CoordinationGovernanceDenied):
        await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )

    assert governance.calls == 1
    assert coordination.calls == 0


def test_governance_request_has_no_agent_or_lease_identity() -> None:
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    request = build_multi_agent_coordination_governance_request(
        intent,
        task_scope_id=binding.task_scope_id,
        application_id=binding.application_id,
        application_environment_id=binding.application_environment_id,
        principal=admin_test_principal(),
    )
    payload = request.model_dump(mode="json")
    serialized = str(payload)
    assert "lease-a" not in serialized
    assert "agent_id" not in MultiAgentCoordinationGovernanceRequest.model_fields
    assert "lease_id" not in MultiAgentCoordinationGovernanceRequest.model_fields


class _LeaseProbingCoordinationService:
    def __init__(self) -> None:
        self.seen_lease_ids: list[str] = []

    async def coordinate(self, request: CoordinationRequest, *, delegation, principal):
        del delegation, principal
        self.seen_lease_ids.append(str(request.lease_id))
        return CoordinationResult(
            coordination_id=request.coordination_id,
            delegated=SimpleNamespace(result=OcrResult(text="ok")),
        )


@pytest.mark.asyncio
async def test_deny_prevents_agent_lease_use_in_coordination_request() -> None:
    coordination = _LeaseProbingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=denying_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    with pytest.raises(CoordinationGovernanceDenied):
        await executor.execute(
            intent,
            binding=binding,
            principal=admin_test_principal(),
        )

    assert coordination.seen_lease_ids == []

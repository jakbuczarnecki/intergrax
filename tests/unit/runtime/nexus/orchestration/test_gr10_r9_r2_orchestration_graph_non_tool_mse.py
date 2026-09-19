# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R2 — orchestration graph / non-tool consequential MSE proofs."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotExecutor,
    OrchestrationConsequentialEffectBlockedError,
    authorize_orchestration_consequential_effect,
    execute_governed_orchestration_consequential_effect,
)
from intergrax.runtime.nexus.orchestration.orchestration_graph_meaningful_side_effect import (
    build_orchestration_graph_slot_enforcement_request,
    build_orchestration_graph_slot_meaningful_side_effect_request,
)
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
)
from intergrax.runtime.task.task import Task, TaskContext
pytestmark = pytest.mark.unit


class _RecordingPort:
    def __init__(self, *, action: PolicyAction) -> None:
        self.calls = 0
        self._action = action

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str,
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        permitted = self._action is PolicyAction.ALLOW
        decision = PolicyDecision(
            action=self._action,
            reason="test",
            policy_rule_id="test.rule",
        )
        from intergrax.contracts.collaborative_work import (
            CollaborativeWorkEnforcementResult,
            PolicyCompositionResult,
        )

        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=permitted,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=self._action
            in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE),
            governed_continuation_request=None,
        )


def _enforcement_request() -> CollaborativeWorkEnforcementRequest:
    side_effect = build_orchestration_graph_slot_meaningful_side_effect_request(
        slot_id=OrchestrationSlotId("slot-1"),
        operation_id="orchestration.custom_slot",
        resource_scope="resource/custom",
        side_effect_scope_id="scope/custom:1",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
    )
    return build_orchestration_graph_slot_enforcement_request(
        slot_id=OrchestrationSlotId("slot-1"),
        side_effect=side_effect,
        operation_id="orchestration.custom_slot",
        resource_scope="resource/custom",
    )


@pytest.fixture
def _identity_and_task():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            principal_id="principal-1",
        ),
    )
    task = Task(
        task_id=str(mint_task_id()),
        tenant_id="tenant-1",
        user_id="user-1",
        agent_id="agent-1",
        context=TaskContext(capability="orchestration"),
    )
    governed = ActiveGovernedExecutionTask()
    task_token = governed.bind(task)
    try:
        yield
    finally:
        governed.reset(task_token)
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)


def test_gr10_r9_r2_mse_deny_zero_effect(_identity_and_task: None) -> None:
    port = _RecordingPort(action=PolicyAction.DENY)
    with pytest.raises(OrchestrationConsequentialEffectBlockedError):
        authorize_orchestration_consequential_effect(
            port,
            enforcement_request=_enforcement_request(),
            production_mode=True,
            source_agent_id="agent-1",
            source_step_id="slot-1",
        )
    assert port.calls == 1


def test_gr10_r9_r2_mse_require_human_zero_effect(_identity_and_task: None) -> None:
    port = _RecordingPort(action=PolicyAction.REQUIRE_HUMAN)
    with pytest.raises(OrchestrationConsequentialEffectBlockedError):
        authorize_orchestration_consequential_effect(
            port,
            enforcement_request=_enforcement_request(),
            production_mode=True,
            source_agent_id="agent-1",
            source_step_id=None,
        )
    assert port.calls == 1


def test_gr10_r9_r2_mse_modify_zero_effect(_identity_and_task: None) -> None:
    port = _RecordingPort(action=PolicyAction.MODIFY)
    with pytest.raises(OrchestrationConsequentialEffectBlockedError):
        authorize_orchestration_consequential_effect(
            port,
            enforcement_request=_enforcement_request(),
            production_mode=True,
            source_agent_id="agent-1",
            source_step_id=None,
        )


@pytest.mark.asyncio
async def test_gr10_r9_r2_mse_allow_single_effect(_identity_and_task: None) -> None:
    port = _RecordingPort(action=PolicyAction.ALLOW)
    calls = 0

    async def effect() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await execute_governed_orchestration_consequential_effect(
        port,
        enforcement_request=_enforcement_request(),
        production_mode=True,
        source_agent_id="agent-1",
        source_step_id="slot-1",
        effect=effect,
    )
    assert result == "ok"
    assert calls == 1
    assert port.calls == 1


def test_gr10_r9_r2_missing_port_fail_closed(_identity_and_task: None) -> None:
    with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError):
        authorize_orchestration_consequential_effect(
            None,
            enforcement_request=_enforcement_request(),
            production_mode=True,
            source_agent_id="agent-1",
            source_step_id=None,
        )


@pytest.mark.asyncio
async def test_gr10_r9_r2_governed_slot_executor_deny_skips_inner(
    _identity_and_task: None,
) -> None:
    inner_calls = 0

    class _Inner:
        async def execute_slot(self, *, slot_id, payload):
            nonlocal inner_calls
            inner_calls += 1
            return payload

    port = _RecordingPort(action=PolicyAction.DENY)
    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=port,
        production_mode=True,
        build_enforcement_request=lambda _slot, _payload: _enforcement_request(),
    )
    from intergrax.contracts.orchestration_topology import OrchestrationSlotExecutionError

    with pytest.raises(OrchestrationSlotExecutionError):
        await executor.execute_slot(slot_id=OrchestrationSlotId("slot-1"), payload="x")
    assert inner_calls == 0
    assert port.calls == 1


@pytest.mark.asyncio
async def test_gr10_r9_r2_governed_slot_executor_allow_invokes_inner(
    _identity_and_task: None,
) -> None:
    port = _RecordingPort(action=PolicyAction.ALLOW)

    class _Inner:
        async def execute_slot(self, *, slot_id, payload):
            return f"{slot_id}:{payload}"

    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=port,
        production_mode=True,
        build_enforcement_request=lambda _slot, _payload: _enforcement_request(),
    )
    result = await executor.execute_slot(
        slot_id=OrchestrationSlotId("slot-1"),
        payload="payload",
    )
    assert result == "slot-1:payload"
    assert port.calls == 1


def test_gr10_r9_r2_custom_port_injected(_identity_and_task: None) -> None:
    port = _RecordingPort(action=PolicyAction.ALLOW)
    authorize_orchestration_consequential_effect(
        port,
        enforcement_request=_enforcement_request(),
        production_mode=True,
        source_agent_id="agent-1",
        source_step_id=None,
    )
    assert port.calls == 1

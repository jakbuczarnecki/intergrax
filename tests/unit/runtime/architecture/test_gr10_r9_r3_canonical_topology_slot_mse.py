# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R3 — mandatory MSE on canonical topology submit / recovery / continuation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    OrchestrationSlot,
    OrchestrationSlotContinuationRequest,
    OrchestrationSlotId,
    OrchestrationSlotRecoveryRequest,
    OrchestrationSlotStatus,
    OrchestrationTopology,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_host_task,
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.execution.orchestration_topology_slot_mse_enforcement import (
    build_orchestration_topology_slot_mse_policy,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUBMISSION = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "orchestration_topology_submission.py"
)


class _RecordingMsePort:
    def __init__(self, *, action: PolicyAction) -> None:
        self.calls = 0
        self._action = action
        self.scopes: list[str] = []

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str,
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        self.scopes.append(request.resource_scope)
        permitted = self._action is PolicyAction.ALLOW
        decision = PolicyDecision(
            action=self._action,
            reason="test",
            policy_rule_id="test.rule",
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


@dataclass(frozen=True, slots=True)
class _MutatingWork:
    value: int


class _MutatingSlotExecutor:
    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: _MutatingWork,
    ) -> int:
        del slot_id
        self.calls += 1
        return payload.value + 1


class _ContinuationExecutor:
    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0

    async def continue_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: _MutatingWork,
    ) -> int:
        del slot_id
        self.calls += 1
        return payload.value + 10


@pytest.fixture
def _identity_ctx():
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
    yield
    reset_active_execution_governance_identity(governance_token)
    reset_active_execution_identity(identity_token)


def _topology() -> OrchestrationTopology[_MutatingWork]:
    return OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=_MutatingWork(1),
            ),
        ),
    )


async def _run_submission(delegate: object, root_execution_id: str) -> object:
    budget_token = None
    if peek_active_execution_budget() is None:
        budget_token = bind_root_execution_budget(
            execution_id=root_execution_id,
            ledger=create_execution_budget_ledger(None),
        )
    try:
        return await ExecutionBoundary(
            delegate,
            identity=ExecutionIdentityBinding(
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=root_execution_id,
            ),
            authority=ParentExecutionAuthority.unknown(),
        ).execute(None)
    finally:
        if budget_token is not None:
            reset_active_execution_budget(budget_token)


@pytest.mark.asyncio
async def test_r9_r3_submit_deny_blocks_inner_executor(_identity_ctx) -> None:
    mse = _RecordingMsePort(action=PolicyAction.DENY)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
    )
    port = build_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        slot_mse_policy=policy,
    )
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-1",
        user_id="principal-1",
        task_id=mint_task_id(),
    )

    class _Delegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await port.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    result = await _run_submission(_Delegate(), mint_execution_id())
    assert mse.calls == 1
    assert inner.calls == 0
    assert result.outcomes[0].status is OrchestrationSlotStatus.FAILURE


@pytest.mark.asyncio
async def test_r9_r3_submit_allow_runs_inner_once(_identity_ctx) -> None:
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
    )
    port = build_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        slot_mse_policy=policy,
    )
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-1",
        user_id="principal-1",
        task_id=mint_task_id(),
    )

    class _Delegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await port.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    result = await _run_submission(_Delegate(), mint_execution_id())
    assert mse.calls == 1
    assert inner.calls == 1
    assert result.outcomes[0].status is OrchestrationSlotStatus.SUCCESS


@pytest.mark.asyncio
async def test_r9_r3_submit_missing_mse_port_fail_closed(_identity_ctx) -> None:
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=None,
        production_mode=True,
    )
    port = build_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        slot_mse_policy=policy,
    )
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-1",
        user_id="principal-1",
        task_id=mint_task_id(),
    )

    class _Delegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await port.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    await _run_submission(_Delegate(), mint_execution_id())
    assert inner.calls == 0


class _FlipMsePort(MeaningfulSideEffectAuthorizationPort):
    def __init__(self) -> None:
        self.calls = 0

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str,
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        if self.calls == 2:
            action = PolicyAction.DENY
        else:
            action = PolicyAction.ALLOW
        decision = PolicyDecision(
            action=action,
            reason="test",
            policy_rule_id="test.rule",
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
            permitted=action is PolicyAction.ALLOW,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


@pytest.mark.asyncio
async def test_r9_r3_recovery_fresh_mse_deny_after_prior_allow(_identity_ctx) -> None:
    flip_port = _FlipMsePort()
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=flip_port,
        production_mode=True,
    )
    port = build_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        slot_mse_policy=policy,
    )
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-1",
        user_id="principal-1",
        task_id=mint_task_id(),
    )
    topology = _topology()
    from intergrax.contracts.orchestration_topology import OrchestrationSlotExecutionError
    class _FailExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            raise OrchestrationSlotExecutionError(code="fail", message="boom")

    from intergrax.runtime.execution.orchestration_topology_submission import (
        resolve_orchestration_topology_execution_id,
    )

    class _SubmitAndRecoverDelegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                submit_result = await port.submit(
                    topology,
                    OrchestrationSchedulingPolicy(),
                    _FailExecutor(),
                )
                assert submit_result.outcomes[0].status is OrchestrationSlotStatus.FAILURE
                execution_id = resolve_orchestration_topology_execution_id(
                    host_task=host_task,
                    topology=topology,
                )
                await port.recover_failed_slot(
                    OrchestrationSlotRecoveryRequest(
                        execution_id=execution_id,
                        slot_id=OrchestrationSlotId("slot-a"),
                        correlation_id="recovery-1",
                        source_checkpoint_revision=1,
                    ),
                    slot_executor=inner,
                )
                await port.recover_failed_slot(
                    OrchestrationSlotRecoveryRequest(
                        execution_id=execution_id,
                        slot_id=OrchestrationSlotId("slot-a"),
                        correlation_id="recovery-2",
                        source_checkpoint_revision=1,
                    ),
                    slot_executor=inner,
                )
            finally:
                governed.reset(token)

    await _run_submission(_SubmitAndRecoverDelegate(), mint_execution_id())
    assert flip_port.calls == 3
    assert inner.calls == 1  # second recovery ALLOW; first recovery DENY did not execute effect


def test_r9_r3_submission_ast_mandatory_prepare_on_three_paths() -> None:
    source = _SUBMISSION.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_SUBMISSION))
    methods: dict[str, ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CanonicalOrchestrationTopologySubmissionPort":
            for child in node.body:
                if isinstance(child, ast.AsyncFunctionDef):
                    methods[child.name] = child
    for name in ("submit", "recover_failed_slot", "continue_slot"):
        func = methods.get(name)
        assert func is not None
        names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
        if name == "continue_slot":
            assert "prepare_orchestration_topology_slot_continuation_executor" in names
        else:
            assert "prepare_orchestration_topology_slot_executor" in names

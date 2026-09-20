# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R4 — mandatory production MSE composition and contract-based authority delegation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    PassthroughOrchestrationConsequentialEffectReliability,
)
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
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    OrchestrationSlot,
    OrchestrationSlotId,
    OrchestrationSlotStatus,
    OrchestrationTopology,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.orchestration_topology_submission import (
    OrchestrationTopologyMseCompositionError,
    build_lab_orchestration_topology_submission_port,
    build_production_orchestration_topology_submission_port,
    build_orchestration_topology_host_task,
)
from intergrax.runtime.execution.orchestration_topology_slot_mse_enforcement import (
    OrchestrationTopologySlotEffectAuthorityOwner,
    build_orchestration_topology_slot_mse_policy,
    prepare_orchestration_topology_slot_executor,
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
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    FanOutCoordinationSlotExecutor as CanonicalFanOutSlotExecutor,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENFORCEMENT = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "orchestration_topology_slot_mse_enforcement.py"
)


class _RecordingMsePort:
    def __init__(self, *, action: PolicyAction) -> None:
        self.calls = 0
        self.scopes: list[str] = []
        self._action = action

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str,
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        self.scopes.append(request.resource_scope)
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
            permitted=self._action is PolicyAction.ALLOW,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
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


class _DelegatedPhysicalSlotExecutor:
    """Alternate delegated implementation — different class name, same authority contract."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        self.calls = 0

    @property
    def orchestration_slot_effect_authority_owner(
        self,
    ) -> OrchestrationTopologySlotEffectAuthorityOwner:
        return OrchestrationTopologySlotEffectAuthorityOwner.PHYSICAL_DELEGATION

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: _MutatingWork,
    ) -> int:
        del slot_id, payload
        self.calls += 1
        return 99


class FanOutCoordinationSlotExecutor:
    """Spoofed unrelated class name — must not confer delegated authority."""

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: _MutatingWork,
    ) -> int:
        del slot_id, payload
        return 0


@pytest.fixture
def _identity_ctx():
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
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


def test_r9_r4_production_builder_rejects_missing_policy() -> None:
    with pytest.raises(OrchestrationTopologyMseCompositionError):
        build_production_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))


def test_r9_r4_lab_builder_allows_missing_policy() -> None:
    port = build_lab_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))
    assert port is not None


def test_r9_r4_prepare_without_policy_returns_raw_executor() -> None:
    inner = _MutatingSlotExecutor()
    prepared = prepare_orchestration_topology_slot_executor(inner, policy=None)
    assert prepared is inner


def test_r9_r4_canonical_fanout_executor_skips_mse_wrap() -> None:
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
        effect_reliability=PassthroughOrchestrationConsequentialEffectReliability(),
    )
    inner = CanonicalFanOutSlotExecutor(
        coordination=object(),  # type: ignore[arg-type]
        principal=object(),  # type: ignore[arg-type]
    )
    prepared = prepare_orchestration_topology_slot_executor(inner, policy=policy)
    assert prepared is inner


def test_r9_r4_alternate_delegated_implementation_skips_mse_wrap() -> None:
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
        effect_reliability=PassthroughOrchestrationConsequentialEffectReliability(),
    )
    inner = _DelegatedPhysicalSlotExecutor()
    prepared = prepare_orchestration_topology_slot_executor(inner, policy=policy)
    assert prepared is inner
    assert not isinstance(prepared, GovernedOrchestrationSlotExecutor)


def test_r9_r4_spoofed_class_name_does_not_skip_mse_wrap() -> None:
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
        effect_reliability=PassthroughOrchestrationConsequentialEffectReliability(),
    )
    inner = FanOutCoordinationSlotExecutor()
    prepared = prepare_orchestration_topology_slot_executor(inner, policy=policy)
    assert isinstance(prepared, GovernedOrchestrationSlotExecutor)


@pytest.mark.asyncio
async def test_r9_r4_parallel_slots_distinct_mse_scopes(_identity_ctx) -> None:
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    policy = build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=mse,
        production_mode=True,
        effect_reliability=PassthroughOrchestrationConsequentialEffectReliability(),
    )
    port = build_production_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        slot_mse_policy=policy,
    )
    inner = _MutatingSlotExecutor()
    topology = OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=_MutatingWork(1),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("slot-b"),
                payload=_MutatingWork(2),
            ),
        ),
    )
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
                    topology,
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    result = await _run_submission(_Delegate(), mint_execution_id())
    assert mse.calls == 2
    assert len(set(mse.scopes)) == 2
    assert inner.calls == 2
    assert all(
        outcome.status is OrchestrationSlotStatus.SUCCESS for outcome in result.outcomes
    )


def test_r9_r4_authority_module_no_type_name_coupling() -> None:
    source = _ENFORCEMENT.read_text(encoding="utf-8-sig")
    assert "_DELEGATED_MSE_EXECUTOR_TYPE_NAMES" not in source
    assert "FanOutCoordinationSlotExecutor" not in source
    tree = ast.parse(source, filename=str(_ENFORCEMENT))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                node.func.attr == "__name__"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "type"
            ):
                pytest.fail("type(...).__name__ must not drive authority ownership")


def test_r9_r4_prepare_no_policy_none_early_return_in_production_path() -> None:
    source = _ENFORCEMENT.read_text(encoding="utf-8-sig")
    assert "if policy is None:\n        return slot_executor" in source.replace("\r\n", "\n")
    submission = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "execution"
        / "orchestration_topology_submission.py"
    )
    sub_source = submission.read_text(encoding="utf-8-sig")
    assert "build_production_orchestration_topology_submission_port" in sub_source
    assert "OrchestrationTopologyMseCompositionError" in sub_source

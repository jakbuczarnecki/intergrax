# © Artur Czarnecki. All rights reserved.

"""GR-10-R13-R2 — production Reliability composition adoption through real host factories."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.orchestration_topology_production_composition import (
    build_governed_contractor_production_orchestration_topology_submission_port,
)
from applications.governed_contractor_application.host.stores import (
    InMemoryProviderInvocationStore,
)
from intergrax.runtime.execution.orchestration_topology_production_composition import (
    build_strict_production_orchestration_topology_submission_port,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCategory,
    ExternalEffectCapabilitySupport,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityRequest,
    ExternalEffectRepeatEligibilityVerdict,
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
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
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.orchestration_topology_production_composition import (
    build_orchestration_reliability_composition,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_host_task,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_boundary import (
    OrchestrationConsequentialEffectDefinitiveFailureError,
    OrchestrationConsequentialEffectUncertaintyError,
    orchestration_slot_invocation_id,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_composition import (
    OrchestrationConsequentialEffectReliabilityCompositionError,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_RUNTIME_COMPOSITION = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "orchestration_topology_production_composition.py"
)
_GOVERNED_COMPOSITION = (
    _REPO_ROOT
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "orchestration_topology_production_composition.py"
)
_SLOT_IDEMPOTENCY_KEY = "slot:slot-a"


@pytest.fixture
def _identity_ctx():
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="tenant-strict",
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


class _RecordingMsePort:
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
    value: int = 0


class _MutatingSlotExecutor:
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


def _topology() -> OrchestrationTopology[_MutatingWork]:
    return OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("slot-a"),
                payload=_MutatingWork(1),
            ),
        ),
    )


def _harness_port(
    store: DurableTestProviderInvocationStore,
    *,
    tenant_id: str = "tenant-strict",
    mse: _RecordingMsePort | None = None,
):
    return build_strict_production_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        provider_invocation_store=store,
        tenant_id=tenant_id,
        clock=lambda: _T0,
        meaningful_side_effect_authorization=mse,
    )


def test_strict_composition_missing_store_fails() -> None:
    with pytest.raises(OrchestrationConsequentialEffectReliabilityCompositionError):
        build_orchestration_reliability_composition(
            provider_invocation_store=None,
            clock=lambda: _T0,
            tenant_id="t1",
        )


def test_strict_composition_non_durable_store_fails() -> None:
    with pytest.raises(OrchestrationConsequentialEffectReliabilityCompositionError):
        build_orchestration_reliability_composition(
            provider_invocation_store=InMemoryProviderInvocationStore(),
            clock=lambda: _T0,
            tenant_id="t1",
        )


def test_strict_composition_custom_durable_store_passes() -> None:
    store = DurableTestProviderInvocationStore()
    port = build_orchestration_reliability_composition(
        provider_invocation_store=store,
        clock=lambda: _T0,
        tenant_id="t1",
    )
    assert port is not None


def test_static_gate_runtime_composition_wires_store_to_boundary() -> None:
    source = _RUNTIME_COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_strict_production_orchestration_topology_slot_mse_policy" in source
    assert "provider_invocation_store" in source
    assert "build_production_orchestration_topology_submission_port" in source


def test_static_gate_governed_contractor_composition_wires_store() -> None:
    source = _GOVERNED_COMPOSITION.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_GOVERNED_COMPOSITION))
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "build_governed_contractor_production_orchestration_topology_submission_port"
        for node in tree.body
    )


@pytest.mark.asyncio
async def test_real_host_success_persists_succeeded(_identity_ctx) -> None:
    store = DurableTestProviderInvocationStore()
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)
    port = _harness_port(store, mse=mse)
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-strict",
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
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="tenant-strict",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_real_host_generic_error_unknown_no_second_physical_call(_identity_ctx) -> None:
    store = DurableTestProviderInvocationStore()
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)

    class _FailingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            self.calls += 1
            raise RuntimeError("post-dispatch generic")

    port = _harness_port(store, mse=mse)
    inner = _FailingExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-strict",
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

    with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
        await _run_submission(_Delegate(), mint_execution_id())

    assert inner.calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="tenant-strict",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_real_host_definitive_failure_records_failed(_identity_ctx) -> None:
    store = DurableTestProviderInvocationStore()
    mse = _RecordingMsePort(action=PolicyAction.ALLOW)

    class _DefinitiveExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            self.calls += 1
            raise OrchestrationConsequentialEffectDefinitiveFailureError("typed not executed")

    port = _harness_port(store, mse=mse)
    inner = _DefinitiveExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-strict",
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

    with pytest.raises(OrchestrationConsequentialEffectDefinitiveFailureError):
        await _run_submission(_Delegate(), mint_execution_id())

    assert inner.calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="tenant-strict",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.FAILED


@pytest.mark.asyncio
async def test_restart_visibility_across_reconstructed_store() -> None:
    store_a = DurableTestProviderInvocationStore()
    boundary = build_orchestration_reliability_composition(
        provider_invocation_store=store_a,
        clock=lambda: _T0,
        tenant_id="tenant-restart",
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    async def _ok() -> str:
        return "ok"

    try:
        await boundary.execute_admitted_effect(
            slot_id="slot-a",
            operation_id="op-restart",
            idempotency_key="op-restart",
            execute=_ok,
        )
    finally:
        reset_active_execution_identity(token)

    invocation_id = orchestration_slot_invocation_id(
        tenant_id="tenant-restart",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-restart",
    )
    snapshot_invocation = store_a.get_invocation(invocation_id)
    snapshot_outcome = store_a.get_outcome(invocation_id)
    assert snapshot_invocation is not None
    assert snapshot_outcome is not None
    assert snapshot_outcome.status is ProviderInvocationStatus.SUCCEEDED

    store_b = DurableTestProviderInvocationStore()
    store_b.put_invocation(snapshot_invocation)
    store_b.put_outcome(snapshot_outcome)
    assert store_b.get_invocation(invocation_id) == snapshot_invocation
    assert store_b.get_outcome(invocation_id) == snapshot_outcome


def test_governed_contractor_host_factory_accepts_durable_store() -> None:
    store = DurableTestProviderInvocationStore()
    port = build_governed_contractor_production_orchestration_topology_submission_port(
        NexusLoop(AgentRegistry()),
        provider_invocation_store=store,
        tenant_id="governed-contractor-tenant",
        clock=lambda: _T0,
        meaningful_side_effect_authorization=None,
    )
    assert port is not None


@pytest.mark.asyncio
async def test_topology_unknown_gr7_repeat_eligibility_requires_reconciliation() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = build_orchestration_reliability_composition(
        provider_invocation_store=store,
        clock=lambda: _T0,
        tenant_id="tenant-gr7",
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    async def _timeout() -> None:
        raise TimeoutError("t")

    try:
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-unknown",
                idempotency_key="op-unknown",
                execute=_timeout,
            )
    finally:
        reset_active_execution_identity(token)

    invocation_id = orchestration_slot_invocation_id(
        tenant_id="tenant-gr7",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-unknown",
    )
    invocation = store.get_invocation(invocation_id)
    outcome = store.get_outcome(invocation_id)
    assert invocation is not None
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.UNKNOWN
    contract = ExternalEffectContract(
        contract_id="c-orch",
        operation_key="topology_slot:slot-a",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("probe",),
    )
    decision = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=invocation,
            outcome=outcome,
            effect_contract=contract,
        ),
    )
    assert decision.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert decision.reason in {
        ExternalEffectRepeatEligibilityReason.DENIED_RECONCILIATION_REQUIRED,
        ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED,
    }

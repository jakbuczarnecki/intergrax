# © Artur Czarnecki. All rights reserved.

"""GR-10-R13-R4 — resolved production port Governance→Reliability E2E (no rebuilt port, no fake MSE)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.harness_host_composition import (
    resolve_harness_host_orchestration_topology_submission_port,
)
from intergrax.contracts.execution_identity import mint_execution_id, mint_task_id
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    OrchestrationSlotId,
    OrchestrationSlotStatus,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.orchestration_topology_submission import (
    CanonicalOrchestrationTopologySubmissionPort,
    build_orchestration_topology_host_task,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_boundary import (
    OrchestrationConsequentialEffectDefinitiveFailureError,
    OrchestrationConsequentialEffectUncertaintyError,
    orchestration_slot_invocation_id,
)
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

from tests.unit.runtime.architecture.test_gr10_r13_r3_production_host_reliability_adoption import (
    _MutatingSlotExecutor,
    _MutatingWork,
    _SharedProviderInvocationBacking,
    _SLOT_IDEMPOTENCY_KEY,
    _TENANT_MANIFEST,
    _run_submission,
    _strict_host_app,
    _topology,
    production_host_task_id,
    ReconstructableDurableProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

pytest_plugins = (
    "tests.unit.runtime.architecture.test_gr10_r13_r3_production_host_reliability_adoption",
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R4_FILE = Path(__file__)


def _resolved_port(runtime: object):
    port = resolve_harness_host_orchestration_topology_submission_port(runtime)
    assert port is runtime.orchestration_topology.submission_port
    return port


async def _submit_via_resolved_port(
    runtime: object,
    *,
    inner: object,
) -> object:
    port = _resolved_port(runtime)
    host_task = build_orchestration_topology_host_task(
        tenant_id=runtime.tenant_id,
        user_id="principal-1",
        task_id=production_host_task_id(),
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

    return await _run_submission(_Delegate(), mint_execution_id())


def _invocation_id(runtime: object) -> str:
    return orchestration_slot_invocation_id(
        tenant_id=runtime.tenant_id,
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )


def test_canonical_production_mse_gate_allows_topology_slot(_identity_ctx) -> None:
    from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
    from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
    from intergrax.collaborative_work.persistence import collaborative_work_core_repositories
    from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
    from intergrax.contracts.orchestration_topology import OrchestrationSlotId
    from intergrax.runtime.execution.orchestration_topology_slot_mse_enforcement import (
        default_topology_slot_enforcement_request,
    )
    from governed_contractor_application.host.production_external_work_composition import (
        resolve_production_runtime_policy_bundle_evaluator,
    )
    from tests.unit.runtime.architecture.test_gr10_r13_r3_production_host_reliability_adoption import (
        _default_settings,
    )

    settings = _default_settings()
    core = collaborative_work_core_repositories(settings.collaborative_work_repositories)
    evaluator = resolve_production_runtime_policy_bundle_evaluator(settings)
    assert evaluator is not None
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=core.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=core.membership,
            principal_authority_repository=core.principal_authority,
            delegation_repository=core.delegation,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(core.policy),
        runtime_policy_evaluator=evaluator,
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(
        build_orchestration_topology_host_task(
            tenant_id=_TENANT_MANIFEST,
            user_id="principal-1",
            task_id=production_host_task_id(),
        ),
    )
    try:
        request = default_topology_slot_enforcement_request(
            OrchestrationSlotId("slot-a"),
            _MutatingWork(1),
        )
        result = gate.evaluate(request)
    finally:
        governed.reset(token)
    assert result.composition.decision.action is PolicyAction.ALLOW, result.composition.decision


def test_static_gate_r4_file_no_rebuilt_topology_port_builder() -> None:
    tree = ast.parse(_R4_FILE.read_text(encoding="utf-8-sig"), filename=str(_R4_FILE))
    banned_names = {
        "_RecordingMsePort",
        "_runtime_topology_port_with_mse",
        "build_governed_contractor_production_orchestration_topology_submission_port",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise AssertionError(f"forbidden identifier in R13-R4 E2E: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_names


@pytest.mark.asyncio
async def test_resolved_production_port_success_with_canonical_mse_allow(
    _identity_ctx,
    tmp_path: Path,
) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    port = _resolved_port(runtime)
    assert isinstance(port, CanonicalOrchestrationTopologySubmissionPort)
    inner = _MutatingSlotExecutor()
    result = await _submit_via_resolved_port(runtime, inner=inner)
    assert result.outcomes[0].status is OrchestrationSlotStatus.SUCCESS, (
        result.outcomes[0].failure
    )
    assert inner.calls == 1
    outcome = store.get_outcome(_invocation_id(runtime))
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.SUCCEEDED
    invocation = store.get_invocation(_invocation_id(runtime))
    assert invocation is not None
    assert runtime.tenant_id == _TENANT_MANIFEST
    assert invocation.idempotency_key == _SLOT_IDEMPOTENCY_KEY


def test_process_composition_store_matches_runtime_topology_store(
    tmp_path: Path,
) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    wiring = runtime.orchestration_topology
    assert wiring is not None
    assert wiring.provider_invocation_store is store


@pytest.mark.asyncio
async def test_resolved_production_port_unknown_no_replay(
    _identity_ctx,
    tmp_path: Path,
) -> None:
    store = DurableTestProviderInvocationStore()
    runtime = _strict_host_app(store, tmp_path).state.harness_runtime

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

    inner = _FailingExecutor()
    with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
        await _submit_via_resolved_port(runtime, inner=inner)
    assert inner.calls == 1
    assert store.get_outcome(_invocation_id(runtime)).status is ProviderInvocationStatus.UNKNOWN  # type: ignore[union-attr]
    with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
        await _submit_via_resolved_port(runtime, inner=inner)
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_resolved_production_port_definitive_failure(
    _identity_ctx,
    tmp_path: Path,
) -> None:
    store = DurableTestProviderInvocationStore()
    runtime = _strict_host_app(store, tmp_path).state.harness_runtime

    class _DefinitiveFailureExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            raise OrchestrationConsequentialEffectDefinitiveFailureError("typed definitive")

    with pytest.raises(OrchestrationConsequentialEffectDefinitiveFailureError):
        await _submit_via_resolved_port(runtime, inner=_DefinitiveFailureExecutor())
    assert store.get_outcome(_invocation_id(runtime)).status is ProviderInvocationStatus.FAILED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_resolved_production_port_mse_deny_blocks_reliability(
    _identity_ctx,
    tmp_path: Path,
) -> None:
    from tests.unit.runtime.architecture.test_gr10_r13_r3_production_host_reliability_adoption import (
        _settings_topology_runtime_deny,
    )

    store = DurableTestProviderInvocationStore()
    runtime = _strict_host_app(
        store,
        tmp_path,
        settings=_settings_topology_runtime_deny(),
    ).state.harness_runtime
    inner = _MutatingSlotExecutor()
    result = await _submit_via_resolved_port(runtime, inner=inner)
    assert result.outcomes[0].status is OrchestrationSlotStatus.FAILURE
    assert inner.calls == 0
    assert store.get_invocation(_invocation_id(runtime)) is None


@pytest.mark.asyncio
async def test_restart_uses_resolved_production_port_with_real_allow(
    tmp_path: Path,
) -> None:
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        reset_active_execution_identity,
    )
    from intergrax.runtime.governance.active_execution_governance_identity import (
        ActiveExecutionGovernanceIdentity,
        bind_active_execution_governance_identity,
        reset_active_execution_governance_identity,
    )

    backing = _SharedProviderInvocationBacking()
    store_a = ReconstructableDurableProviderInvocationStore(backing)
    runtime_a = _strict_host_app(store_a, tmp_path / "host-a").state.harness_runtime
    inner = _MutatingSlotExecutor()
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=runtime_a.tenant_id,
            workspace_id="workspace-1",
            principal_id="principal-1",
        ),
    )
    try:
        await _submit_via_resolved_port(runtime_a, inner=inner)
    finally:
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)
    invocation_id = _invocation_id(runtime_a)
    assert backing.outcomes[invocation_id].status is ProviderInvocationStatus.SUCCEEDED

    store_b = ReconstructableDurableProviderInvocationStore(backing)
    runtime_b = _strict_host_app(store_b, tmp_path / "host-b").state.harness_runtime
    assert runtime_b.orchestration_topology.provider_invocation_store is store_b
    assert store_b.get_outcome(invocation_id) is not None


@pytest.mark.asyncio
async def test_unknown_gr7_repeat_eligibility_via_resolved_port(
    _identity_ctx,
    tmp_path: Path,
) -> None:
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

    store = DurableTestProviderInvocationStore()
    runtime = _strict_host_app(store, tmp_path).state.harness_runtime

    class _FailingExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            raise RuntimeError("post-dispatch generic")

    with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
        await _submit_via_resolved_port(runtime, inner=_FailingExecutor())
    invocation = store.get_invocation(_invocation_id(runtime))
    outcome = store.get_outcome(_invocation_id(runtime))
    assert invocation is not None and outcome is not None
    contract = ExternalEffectContract(
        contract_id="c-orch-r4",
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

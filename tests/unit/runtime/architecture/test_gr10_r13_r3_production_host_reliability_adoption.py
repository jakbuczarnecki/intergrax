# © Artur Czarnecki. All rights reserved.

"""GR-10-R13-R3 — actual production host Reliability adoption through real factories."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.stores import InMemoryProviderInvocationStore
from governed_contractor_application.host.agent_builders import GOVERNED_CONTRACTOR_AGENT_BUILDERS
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.applications._shared.harness_host_composition import (
    resolve_harness_host_orchestration_topology_submission_port,
)
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
    resolve_harness_host_orchestration_topology_wiring,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
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
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    OrchestrationSlot,
    OrchestrationSlotId,
    OrchestrationSlotStatus,
    OrchestrationTopology,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
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
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    export_durable_continuation_state,
    execution_continuation_state_store_from_durable_export,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


_REPO_ROOT = Path(__file__).resolve().parents[4]
_FACTORY = (
    _REPO_ROOT
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "factory.py"
)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SLOT_IDEMPOTENCY_KEY = "orchestration.graph_slot:slot:slot-a"
_TOPOLOGY_CANONICAL_OPERATION_ID = _SLOT_IDEMPOTENCY_KEY
_TENANT_MANIFEST = "governed_contractor"
_PRODUCTION_HOST_TASK_ID = mint_task_id()


@dataclass
class _SharedProviderInvocationBacking:
    invocations: dict[str, ProviderInvocation] = field(default_factory=dict)
    outcomes: dict[str, ProviderInvocationOutcome] = field(default_factory=dict)


class ReconstructableDurableProviderInvocationStore:
    """Durable store sharing backing across reconstructed provider instances."""

    def __init__(self, backing: _SharedProviderInvocationBacking) -> None:
        self._backing = backing

    @property
    def is_durable(self) -> bool:
        return True

    def put_invocation(self, invocation: ProviderInvocation) -> None:
        self._backing.invocations[invocation.invocation_id] = invocation

    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        return self._backing.invocations.get(invocation_id)

    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        self._backing.outcomes[outcome.invocation_id] = outcome

    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        return self._backing.outcomes.get(invocation_id)


def _seed_active_registry_projection(
    composition: ProductionProcessComposition,
    *,
    application_id: str,
    application_environment_id: str,
    projection: object,
) -> None:
    stores = composition.agent_platform_runtime.stores
    stores.registry_projection_store.put(projection)
    stores.serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=projection.evidence.runtime_revision_id,  # type: ignore[attr-defined]
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )


def _topology_collaborative_work_repositories(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
) -> object:
    from governed_contractor_application.host.collaborative_work_local_fixture import (
        build_in_memory_collaborative_work_repositories,
    )
    from external_contractor_adapter.side_effect_actions import (
        ACTION_ACCEPT_QUOTE,
        ACTION_CREATE_EXTERNAL_WORK,
    )
    from intergrax.collaborative_work.repository import (
        CreateCollaborativeOperationPolicyProfileCommand,
        CreateCollaborativePolicyRuleCommand,
        CreatePrincipalAuthorityGrantCommand,
        CreateWorkspaceMembershipCommand,
    )
    from intergrax.contracts.collaborative_work import (
        AuthorityGrantStatus,
        CollaborativeOperationPolicyProfileStatus,
        CollaborativePolicyRuleStatus,
        MembershipStatus,
        OperationPolicyRequirement,
        PolicyCompositionLayer,
        PolicyLayerApplicability,
        WorkspaceMembershipRole,
    )
    from intergrax.contracts.runtime_policy import PolicyAction

    external_scope = "external_work.mutate"
    topology_scope = "orchestration/topology/slot/slot-a"
    repositories = build_in_memory_collaborative_work_repositories()
    repositories.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    repositories.principal_authority.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=(external_scope, topology_scope),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    for scope in (external_scope, topology_scope):
        repositories.policy.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                policy_rule_id=f"workspace-allow-{scope.replace('/', '-')}",
                layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                authority_scope=scope,
                action=PolicyAction.ALLOW,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )
    for operation_id in (
        ACTION_CREATE_EXTERNAL_WORK,
        ACTION_ACCEPT_QUOTE,
        "orchestration.graph_slot:slot:slot-a",
    ):
        scope = (
            topology_scope
            if operation_id == "orchestration.graph_slot:slot:slot-a"
            else external_scope
        )
        repositories.operation_profile.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
                authority_scope=scope,
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )
    return repositories


def _r13r3_policy_bundle() -> object:
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )
    from intergrax.contracts.runtime_policy_bundle import (
        PolicyBundleRule,
        build_immutable_runtime_policy_bundle,
    )

    base = _test_policy_bundle()
    return build_immutable_runtime_policy_bundle(
        bundle_id="r13r3-production-policy",
        version="1.0.0",
        rules=(
            *base.rules,
            PolicyBundleRule(
                rule_id="r13r3.ORCH_TOPOLOGY_SLOT",
                description="allow orchestration topology slot-a",
                effect="allow",
                match_action="orchestration.graph_slot:slot:slot-a",
            ),
        ),
        issued_at=base.issued_at,
    )


def _default_settings() -> GovernedContractorBackendSettings:
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from governed_contractor_application.host.production_external_work_composition import (
        wire_governed_contractor_production_external_work_settings,
    )
    from intergrax.contracts.execution_identity import mint_task_id
    from tests.unit.runtime.governance.gr3_test_support import StaticActiveTaskScope

    fake = DeterministicExternalWorkFake()
    cw = _topology_collaborative_work_repositories(
        tenant_id=_TENANT_MANIFEST,
        workspace_id="workspace-1",
        principal_id="principal-1",
    )
    base = replace(
        GovernedContractorBackendSettings(
            include_mcp=False,
            include_scheduler=False,
            include_interaction_routes=False,
        ),
        external_work_integration=fake,
        runtime_policy_bundle=_r13r3_policy_bundle(),  # type: ignore[arg-type]
        collaborative_work_repositories=cw,
    )
    return wire_governed_contractor_production_external_work_settings(
        base,
        task_scope=StaticActiveTaskScope(_PRODUCTION_HOST_TASK_ID),
    )


def production_host_task_id() -> str:
    return _PRODUCTION_HOST_TASK_ID


def _continuation_store():
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def _settings_topology_runtime_deny() -> GovernedContractorBackendSettings:
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )

    return replace(
        _default_settings(),
        runtime_policy_bundle=_test_policy_bundle(),  # type: ignore[arg-type]
    )


def _strict_host_app(
    store: DurableTestProviderInvocationStore | ReconstructableDurableProviderInvocationStore,
    tmp_path: Path,
    *,
    settings: GovernedContractorBackendSettings | None = None,
):
    platform_persistence = build_reference_production_platform_persistence(
        db_path=tmp_path / "platform-kv.db",
    )
    composition = ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(
            platform_persistence=platform_persistence,
        ),
        provider_invocation_store=store,
    )
    resolved_settings = settings or _default_settings()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(
        resolved_settings,
    )
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=GOVERNED_CONTRACTOR_AGENT_BUILDERS,
        revision_id="rev-r13r3-strict-host",
        settings=resolved_settings,
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    platform = composition.agent_platform_runtime.platform_persistence
    return create_governed_contractor_backend_app(
        registry_projection=projection,
        process_composition=composition,
        settings=resolved_settings,
        document_store=platform.document_store,
        key_value_cache=platform.kv_store,
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        checkpoints_db_path=tmp_path / "checkpoints.db",
        execution_continuation_state_store=_continuation_store(),
    )


@pytest.fixture
def _identity_ctx():
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT_MANIFEST,
            workspace_id="workspace-1",
            principal_id="principal-1",
        ),
    )
    yield
    reset_active_execution_governance_identity(governance_token)
    reset_active_execution_identity(identity_token)


@dataclass
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


def _resolved_topology_port(runtime: object):
    port = resolve_harness_host_orchestration_topology_submission_port(runtime)
    assert port is runtime.orchestration_topology.submission_port
    return port


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
        ).execute(object())
    finally:
        if budget_token is not None:
            reset_active_execution_budget(budget_token)


def test_static_gate_factory_wires_governed_topology_builder_and_store() -> None:
    source = _FACTORY.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_FACTORY))
    assert "build_governed_contractor_production_orchestration_topology_submission_port" in source
    assert "require_strict_orchestration_topology_reliability" in source
    assert "provider_invocation_store" in source
    assert "InMemoryProviderInvocationStore" not in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_harness_host_runtime"
        for node in ast.walk(tree)
    )


def test_strict_process_app_missing_store_fails_at_factory(tmp_path: Path) -> None:
    platform_persistence = build_reference_production_platform_persistence(
        db_path=tmp_path / "platform-kv-missing.db",
    )
    composition = ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(
            platform_persistence=platform_persistence,
        ),
    )
    settings = _default_settings()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=GOVERNED_CONTRACTOR_AGENT_BUILDERS,
        revision_id="rev-r13r3-missing-store",
        settings=settings,
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    with pytest.raises(HarnessHostOrchestrationTopologyReliabilityCompositionError):
        create_governed_contractor_backend_app(
            registry_projection=projection,
            process_composition=composition,
            settings=settings,
            document_store=composition.agent_platform_runtime.platform_persistence.document_store,
            key_value_cache=composition.agent_platform_runtime.platform_persistence.kv_store,
            trace_db_path=tmp_path / "trace.db",
            execution_continuation_state_store=_continuation_store(),
        )


def test_strict_process_app_non_durable_store_fails(tmp_path: Path) -> None:
    platform_persistence = build_reference_production_platform_persistence(
        db_path=tmp_path / "platform-kv-nondurable.db",
    )
    composition = ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(
            platform_persistence=platform_persistence,
        ),
        provider_invocation_store=InMemoryProviderInvocationStore(),
    )
    settings = _default_settings()
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=GOVERNED_CONTRACTOR_AGENT_BUILDERS,
        revision_id="rev-r13r3-nondurable",
        settings=settings,
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    with pytest.raises(HarnessHostOrchestrationTopologyReliabilityCompositionError):
        create_governed_contractor_backend_app(
            registry_projection=projection,
            process_composition=composition,
            settings=settings,
            document_store=composition.agent_platform_runtime.platform_persistence.document_store,
            key_value_cache=composition.agent_platform_runtime.platform_persistence.kv_store,
            trace_db_path=tmp_path / "trace.db",
            execution_continuation_state_store=_continuation_store(),
        )


def test_runtime_exposes_topology_submission_from_factory(tmp_path: Path) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    wiring = resolve_harness_host_orchestration_topology_wiring(runtime)
    assert wiring.provider_invocation_store is store
    port = resolve_harness_host_orchestration_topology_submission_port(runtime)
    assert port is wiring.submission_port
    assert runtime.tenant_id == _TENANT_MANIFEST


@pytest.mark.asyncio
async def test_real_host_success_e2e_via_runtime_port(_identity_ctx, tmp_path: Path) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    port = resolve_harness_host_orchestration_topology_submission_port(runtime)
    from intergrax.runtime.execution.orchestration_topology_submission import (
        CanonicalOrchestrationTopologySubmissionPort,
    )

    assert isinstance(port, CanonicalOrchestrationTopologySubmissionPort)
    exec_port = _resolved_topology_port(runtime)
    inner = _MutatingSlotExecutor()
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
                return await exec_port.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    result = await _run_submission(_Delegate(), mint_execution_id())
    assert result.outcomes[0].status is OrchestrationSlotStatus.SUCCESS, (
        result.outcomes[0].failure
    )
    assert inner.calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id=runtime.tenant_id,
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_real_host_unknown_e2e_via_runtime_port(_identity_ctx, tmp_path: Path) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    exec_port = _resolved_topology_port(runtime)

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
                return await exec_port.submit(
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
        tenant_id=runtime.tenant_id,
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    assert store.get_outcome(invocation_id).status is ProviderInvocationStatus.UNKNOWN  # type: ignore[union-attr]

    with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
        await _run_submission(_Delegate(), mint_execution_id())
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_real_host_definitive_failure_e2e_via_runtime_port(
    _identity_ctx,
    tmp_path: Path,
) -> None:
    store = DurableTestProviderInvocationStore()
    app = _strict_host_app(store, tmp_path)
    runtime = app.state.harness_runtime
    exec_port = _resolved_topology_port(runtime)

    class _DefinitiveFailureExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _MutatingWork,
        ) -> int:
            del slot_id, payload
            raise OrchestrationConsequentialEffectDefinitiveFailureError("typed definitive")

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
                return await exec_port.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    _DefinitiveFailureExecutor(),
                )
            finally:
                governed.reset(token)

    with pytest.raises(OrchestrationConsequentialEffectDefinitiveFailureError):
        await _run_submission(_Delegate(), mint_execution_id())
    invocation_id = orchestration_slot_invocation_id(
        tenant_id=runtime.tenant_id,
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    assert store.get_outcome(invocation_id).status is ProviderInvocationStatus.FAILED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_restart_visibility_via_shared_backing_and_rebuilt_host(
    tmp_path: Path,
) -> None:
    backing = _SharedProviderInvocationBacking()
    store_a = ReconstructableDurableProviderInvocationStore(backing)
    app_a = _strict_host_app(store_a, tmp_path / "host-a")
    runtime_a = app_a.state.harness_runtime
    port_a = _resolved_topology_port(runtime_a)
    inner = _MutatingSlotExecutor()
    host_task = build_orchestration_topology_host_task(
        tenant_id=runtime_a.tenant_id,
        user_id="principal-1",
        task_id=production_host_task_id(),
    )
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

    class _Delegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await port_a.submit(
                    _topology(),
                    OrchestrationSchedulingPolicy(),
                    inner,
                )
            finally:
                governed.reset(token)

    try:
        await _run_submission(_Delegate(), mint_execution_id())
    finally:
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)

    invocation_id = orchestration_slot_invocation_id(
        tenant_id=runtime_a.tenant_id,
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=_SLOT_IDEMPOTENCY_KEY,
    )
    assert backing.outcomes[invocation_id].status is ProviderInvocationStatus.SUCCEEDED

    store_b = ReconstructableDurableProviderInvocationStore(backing)
    app_b = _strict_host_app(store_b, tmp_path / "host-b")
    runtime_b = app_b.state.harness_runtime
    assert runtime_b.orchestration_topology is not None
    assert runtime_b.orchestration_topology.provider_invocation_store is store_b
    assert store_b.get_outcome(invocation_id) is not None


def test_lab_factory_without_process_composition_does_not_require_store() -> None:
    settings = _default_settings()
    manifest = build_governed_contractor_manifest()
    env = build_governed_contractor_environment_profile(settings)
    if env.execution_mode.value == "strict":
        pytest.skip("environment defaults to strict; lab path not applicable")
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=GOVERNED_CONTRACTOR_AGENT_BUILDERS,
        revision_id="rev-r13r3-lab",
        settings=settings,
    )
    app = create_governed_contractor_backend_app(
        registry_projection=projection,
        settings=settings,
    )
    assert app.state.harness_runtime.orchestration_topology is None

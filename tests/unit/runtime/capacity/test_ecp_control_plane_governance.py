# © Artur Czarnecki. All rights reserved.

"""ECP control-plane mutation governance proofs (ECP-CPM1–ECP-CPM32)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import pytest

from intergrax.applications._shared.production_capacity_governance_wiring import (
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import resolve_production_capacity_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.capacity.control_plane_governance import (
    CELERY_POOL_RESOURCE_TYPE,
    EcpGovernanceBlockedError,
    K8S_DEPLOYMENT_RESOURCE_TYPE,
    MUTATION_TYPE_SCALE_CELERY_WORKERS,
    MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
    StaticEcpResourceTenantResolver,
    build_scale_celery_workers_mutation_request,
    build_scale_k8s_deployment_mutation_request,
    celery_workers_revision_token,
    k8s_replicas_revision_token,
)
from intergrax.runtime.capacity.governed_capacity_mutation import GovernedCapacityMutationExecutor
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind, ScalingTarget
from intergrax.runtime.capacity.production_adapters import (
    CeleryProductionAdapter,
    ProductionCapacityAdapters,
    apply_production_scale_probe,
    build_production_capacity_adapters,
)
from intergrax.runtime.capacity.provisioner import (
    GovernedExecutionRequiredError,
    ProvisionerExecutionMode,
    ScalingProvisioner,
    StaleCapacityStateError,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-ecp"
_OTHER_TENANT = "tenant-other"
_DEPLOYMENT = "nexus-host"
_POOL = "default"


@dataclass
class _RecordingK8s:
    replicas: int = 2
    scale_calls: list[int] = field(default_factory=list)

    def get_replicas(self, *, deployment: str) -> int:
        del deployment
        return self.replicas

    def scale_workload(self, *, deployment: str, replicas: int) -> int:
        del deployment
        self.scale_calls.append(replicas)
        self.replicas = replicas
        return replicas


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


def _service_principal(tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="capacity-probe",
        principal_type=PrincipalType.SERVICE,
        auth_subject="capacity-probe",
    )


def _build_executor(
    *,
    k8s: _RecordingK8s | None = None,
    celery: CeleryProductionAdapter | None = None,
    evaluator: _RecordingEvaluator | None = None,
    tenant_id: str = _TENANT,
    execution_mode: ProvisionerExecutionMode = ProvisionerExecutionMode.UNRESTRICTED,
) -> tuple[GovernedCapacityMutationExecutor, _RecordingK8s, CeleryProductionAdapter, _RecordingEvaluator]:
    kubernetes = k8s or _RecordingK8s()
    celery_adapter = celery or CeleryProductionAdapter(worker_count=2)
    recording = evaluator or _RecordingEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=recording)
    provisioner = ScalingProvisioner(
        kubernetes=kubernetes,
        celery=celery_adapter,
        execution_mode=execution_mode,
    )
    executor = GovernedCapacityMutationExecutor(
        provisioner=provisioner,
        mutation_boundary=boundary,
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=tenant_id),
    )
    return executor, kubernetes, celery_adapter, recording


def test_ecp_cpm1_allow_k8s_exact_target_and_evidence() -> None:
    executor, k8s, _celery, recording = _build_executor()
    result = executor.scale_k8s_deployment(
        principal=_service_principal(),
        tenant_id=_TENANT,
        mutation_id="mut-k8s-allow",
        deployment=_DEPLOYMENT,
        delta=1,
    )
    assert len(recording.calls) == 1
    request = recording.calls[0]
    assert request.mutation_type == MUTATION_TYPE_SCALE_K8S_DEPLOYMENT
    assert request.resource_type == K8S_DEPLOYMENT_RESOURCE_TYPE
    assert request.resource_id == _DEPLOYMENT
    assert request.current_revision == k8s_replicas_revision_token(
        deployment=_DEPLOYMENT,
        replicas=2,
    )
    assert request.target_revision == k8s_replicas_revision_token(
        deployment=_DEPLOYMENT,
        replicas=3,
    )
    assert result.authorization_evidence.mutation_id == "mut-k8s-allow"
    assert result.authorization_evidence.policy_action is PolicyAction.ALLOW
    assert k8s.scale_calls == [3]
    assert k8s.replicas == 3


def test_ecp_cpm2_deny_k8s_zero_scale_calls() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    executor, k8s, _celery, _recording = _build_executor(evaluator=evaluator)
    with pytest.raises(EcpGovernanceBlockedError) as exc_info:
        executor.scale_k8s_deployment(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-deny",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    assert exc_info.value.policy_action == PolicyAction.DENY.value
    assert k8s.scale_calls == []
    assert k8s.replicas == 2


def test_ecp_cpm3_require_human_k8s_zero_scale_calls() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    executor, k8s, _celery, _recording = _build_executor(evaluator=evaluator)
    with pytest.raises(EcpGovernanceBlockedError) as exc_info:
        executor.scale_k8s_deployment(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-human",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    assert exc_info.value.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert exc_info.value.authorization_scope is not None
    assert exc_info.value.authorization_scope.mutation_id == "mut-k8s-human"
    assert k8s.scale_calls == []


def test_ecp_cpm4_wrong_tenant_blocked_before_provider_mutation() -> None:
    executor, k8s, _celery, recording = _build_executor()
    with pytest.raises(EcpGovernanceBlockedError) as exc_info:
        executor.scale_k8s_deployment(
            principal=_service_principal(tenant_id=_OTHER_TENANT),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-tenant",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    assert exc_info.value.tenant_scope_denial is not None
    assert exc_info.value.authorization_evidence is None
    assert recording.calls == []
    assert k8s.scale_calls == []


def test_ecp_cpm5_missing_boundary_fail_closed() -> None:
    k8s = _RecordingK8s()
    celery = CeleryProductionAdapter(worker_count=2)
    provisioner = ScalingProvisioner(kubernetes=k8s, celery=celery)
    executor = GovernedCapacityMutationExecutor(
        provisioner=provisioner,
        mutation_boundary=None,
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    with pytest.raises(EcpGovernanceBlockedError) as exc_info:
        executor.scale_k8s_deployment(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-missing",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    assert exc_info.value.blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"
    assert k8s.scale_calls == []


def test_ecp_cpm6_caller_mutation_id_in_evidence() -> None:
    executor, _k8s, _celery, _recording = _build_executor()
    mutation_id = "caller-stable-mutation-id-42"
    result = executor.scale_k8s_deployment(
        principal=_service_principal(),
        tenant_id=_TENANT,
        mutation_id=mutation_id,
        deployment=_DEPLOYMENT,
        delta=1,
    )
    assert result.authorization_evidence.mutation_id == mutation_id
    assert _recording.calls[0].mutation_id == mutation_id


def test_ecp_cpm7_current_target_binding_changes_digest() -> None:
    principal = _service_principal()
    request_a = build_scale_k8s_deployment_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        mutation_id="mut-digest",
        deployment=_DEPLOYMENT,
        current_replicas=2,
        target_replicas=3,
    )
    request_b = build_scale_k8s_deployment_mutation_request(
        principal=principal,
        tenant_id=_TENANT,
        mutation_id="mut-digest",
        deployment=_DEPLOYMENT,
        current_replicas=5,
        target_replicas=6,
    )
    assert control_plane_mutation_request_digest(request_a) != control_plane_mutation_request_digest(
        request_b
    )


@dataclass
class _FlakyK8s:
    replicas: int = 2
    scale_calls: list[int] = field(default_factory=list)
    _read_count: int = 0

    def get_replicas(self, *, deployment: str) -> int:
        del deployment
        self._read_count += 1
        if self._read_count == 1:
            return 2
        if self._read_count == 2:
            return 5
        return self.replicas

    def scale_workload(self, *, deployment: str, replicas: int) -> int:
        del deployment
        self.scale_calls.append(replicas)
        self.replicas = replicas
        return replicas


def test_ecp_cpm8_stale_k8s_state_blocks_apply() -> None:
    k8s = _FlakyK8s()
    executor, _k8s_ref, _celery, _recording = _build_executor(k8s=k8s)
    with pytest.raises(StaleCapacityStateError) as exc_info:
        executor.scale_k8s_deployment(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-stale",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    assert exc_info.value.authorized_current == 2
    assert exc_info.value.observed_current == 5
    assert k8s.scale_calls == []


def test_ecp_cpm9_allow_celery_exact_target_and_evidence() -> None:
    executor, _k8s, celery, recording = _build_executor()
    result = executor.scale_celery_workers(
        principal=_service_principal(),
        tenant_id=_TENANT,
        mutation_id="mut-celery-allow",
        pool_id=_POOL,
        delta=1,
    )
    assert len(recording.calls) == 1
    request = recording.calls[0]
    assert request.mutation_type == MUTATION_TYPE_SCALE_CELERY_WORKERS
    assert request.resource_type == CELERY_POOL_RESOURCE_TYPE
    assert request.current_revision == celery_workers_revision_token(
        pool_id=_POOL,
        worker_count=2,
    )
    assert request.target_revision == celery_workers_revision_token(
        pool_id=_POOL,
        worker_count=3,
    )
    assert result.authorization_evidence.policy_action is PolicyAction.ALLOW
    assert celery.worker_count == 3


def test_ecp_cpm10_deny_celery_zero_worker_mutation() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    executor, _k8s, celery, _recording = _build_executor(evaluator=evaluator)
    with pytest.raises(EcpGovernanceBlockedError):
        executor.scale_celery_workers(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-celery-deny",
            pool_id=_POOL,
            delta=1,
        )
    assert celery.worker_count == 2


def test_ecp_cpm11_production_probe_uses_governed_facade() -> None:
    recording = _RecordingEvaluator()
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=recording),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert apply_production_scale_probe(
        adapters,
        principal=_service_principal(),
        tenant_id=_TENANT,
        k8s_mutation_id="probe-k8s",
        celery_mutation_id="probe-celery",
    )
    assert len(recording.calls) == 2
    assert {adapter_field.name for adapter_field in fields(ProductionCapacityAdapters)} == {
        "governed_executor",
        "kubernetes_backend",
        "kubernetes_observation",
        "celery_observation",
    }


def test_ecp_cpm12_production_adapters_expose_governed_executor_only() -> None:
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert isinstance(adapters, ProductionCapacityAdapters)
    assert adapters.governed_executor is not None
    assert {adapter_field.name for adapter_field in fields(ProductionCapacityAdapters)} == {
        "governed_executor",
        "kubernetes_backend",
        "kubernetes_observation",
        "celery_observation",
    }


def test_ecp_cpm13_provider_side_effect_only_after_allow() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    executor, k8s, celery, _recording = _build_executor(evaluator=evaluator)
    with pytest.raises(EcpGovernanceBlockedError):
        executor.scale_k8s_deployment(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-k8s-blocked",
            deployment=_DEPLOYMENT,
            delta=1,
        )
    with pytest.raises(EcpGovernanceBlockedError):
        executor.scale_celery_workers(
            principal=_service_principal(),
            tenant_id=_TENANT,
            mutation_id="mut-celery-blocked",
            pool_id=_POOL,
            delta=1,
        )
    assert k8s.replicas == 2
    assert celery.worker_count == 2


def test_ecp_cpm14_production_missing_policy_fails_closed() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_production_capacity_wiring(env)
    assert wiring.enabled is True
    assert wiring.adapters is None
    assert wiring.probe_passed is False


def test_ecp_cpm15_production_supplied_deny_policy_zero_provider_effect() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    deny_evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    governance = build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=deny_evaluator,
        ),
    )
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    assert wiring.enabled is True
    assert wiring.adapters is not None
    assert wiring.probe_passed is False
    assert wiring.adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 2
    assert wiring.adapters.celery_observation.get_worker_count() == 2


def test_ecp_cpm16_no_permissive_local_production_evaluator() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(env)
    assert governance.mutation_authorization_boundary is None
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    assert wiring.enabled is True
    assert wiring.adapters is None
    assert wiring.probe_passed is False


def test_ecp_cpm17_production_adapters_block_raw_exact_target_mutation() -> None:
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert isinstance(adapters.governed_executor, GovernedCapacityMutationExecutor)
    provisioner = ScalingProvisioner(
        kubernetes=_RecordingK8s(),
        celery=CeleryProductionAdapter(worker_count=2),
        execution_mode=ProvisionerExecutionMode.GOVERNED_ONLY,
    )
    with pytest.raises(GovernedExecutionRequiredError):
        provisioner.apply(
            ScalingAction(
                kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
                target=ScalingTarget.NEXUS_HOST,
                delta=1,
            ),
        )
    with pytest.raises(GovernedExecutionRequiredError):
        provisioner.apply(
            ScalingAction(
                kind=ScalingActionKind.SCALE_CELERY_WORKERS,
                target=ScalingTarget.CELERY_POOL,
                delta=1,
            ),
        )


def test_ecp_cpm18_maintenance_path_cannot_manufacture_allow_policy() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(env)
    assert governance.mutation_authorization_boundary is None
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    assert wiring.enabled is True
    assert wiring.adapters is None
    assert wiring.probe_passed is False


def _scheduler_service_principal(tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="capacity-scheduler",
        principal_type=PrincipalType.SERVICE,
        auth_subject="capacity-scheduler",
    )


def _build_governed_scheduler(
    *,
    k8s: _RecordingK8s | None = None,
    celery: CeleryProductionAdapter | None = None,
    evaluator: _RecordingEvaluator | None = None,
    tenant_id: str = _TENANT,
    execution_identity: RequestIdentity | None = None,
    governed_executor: GovernedCapacityMutationExecutor | None = None,
    requires_governed: bool = True,
) -> tuple[
    "CapacityScheduler",
    _RecordingK8s,
    CeleryProductionAdapter,
    GovernedCapacityMutationExecutor,
    _RecordingEvaluator,
]:
    import asyncio

    from intergrax.runtime.capacity.collector import CapacitySignalCollector
    from intergrax.runtime.capacity.contracts import ScalingPolicy, ScalingRule
    from intergrax.runtime.capacity.evaluator import ScalingEvaluator
    from intergrax.runtime.capacity.scheduler import CapacityScheduler

    kubernetes = k8s or _RecordingK8s()
    celery_adapter = celery or CeleryProductionAdapter(worker_count=2)
    recording = evaluator or _RecordingEvaluator()
    if governed_executor is None:
        governed_executor, _, _, recording = _build_executor(
            k8s=kubernetes,
            celery=celery_adapter,
            evaluator=recording,
            tenant_id=tenant_id,
        )
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="k8s",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            ),
            ScalingRule(
                rule_id="celery",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            ),
        ],
    )
    collector = CapacitySignalCollector()
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=ScalingEvaluator(policy),
        provisioner=None,
        execution_identity=execution_identity or _scheduler_service_principal(tenant_id),
        governed_capacity_executor=governed_executor,
        tenant_id=tenant_id,
        requires_governed_execution=requires_governed,
    )
    return scheduler, kubernetes, celery_adapter, governed_executor, recording


def test_ecp_cpm19_automatic_scheduler_k8s_allow() -> None:
    import asyncio

    scheduler, k8s, _celery, executor, recording = _build_governed_scheduler()
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert len(recording.calls) == 1
    request = recording.calls[0]
    assert request.mutation_type == MUTATION_TYPE_SCALE_K8S_DEPLOYMENT
    assert request.principal.principal_type is PrincipalType.SERVICE
    assert request.principal.user_id == "capacity-scheduler"
    assert k8s.scale_calls == [3]
    assert len(executor.scheduler_applied_actions) == 1
    assert executor.scheduler_applied_actions[0].action_id == request.mutation_id


def test_ecp_cpm20_automatic_scheduler_k8s_deny() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler(
        evaluator=evaluator,
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert executor.scheduler_recorded_failures


def test_ecp_cpm21_automatic_scheduler_celery_allow() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    scheduler, _k8s, celery, executor, recording = _build_governed_scheduler()
    action = ScalingAction(
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    celery_calls = [
        call for call in recording.calls if call.mutation_type == MUTATION_TYPE_SCALE_CELERY_WORKERS
    ]
    assert len(celery_calls) == 1
    assert celery_calls[0].principal.principal_type is PrincipalType.SERVICE
    assert celery.worker_count == 3
    assert len(executor.scheduler_applied_actions) == 1


def test_ecp_cpm22_automatic_scheduler_celery_deny() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    scheduler, _k8s, celery, executor, _recording = _build_governed_scheduler(
        evaluator=evaluator,
    )
    action = ScalingAction(
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert celery.worker_count == 2
    assert executor.scheduler_applied_actions == ()
    assert executor.scheduler_recorded_failures


def test_ecp_cpm23_missing_governed_executor_fail_closed() -> None:
    import asyncio

    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler(
        governed_executor=None,
        requires_governed=True,
    )
    scheduler._governed_capacity_executor = None
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert len(scheduler.blocked_outcomes) == 1
    assert scheduler.blocked_outcomes[0].blocker_code == "ECP_SCHEDULER_MISSING_EXECUTOR"


def test_ecp_cpm24_wrong_tenant_automatic_scheduler_blocked() -> None:
    import asyncio

    scheduler, k8s, _celery, executor, recording = _build_governed_scheduler(
        execution_identity=_scheduler_service_principal(tenant_id=_OTHER_TENANT),
        tenant_id=_TENANT,
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert recording.calls == []
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm25_scheduler_mutation_id_uses_action_id() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    scheduler, _k8s, _celery, executor, recording = _build_governed_scheduler()
    action = ScalingAction(
        action_id="stable-scheduler-action-id",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert recording.calls[0].mutation_id == "stable-scheduler-action-id"
    assert executor.scheduler_applied_actions[0].action_id == "stable-scheduler-action-id"


def test_ecp_cpm26_require_human_zero_provider_effect() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler(
        evaluator=evaluator,
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert any("ECP_BLOCKED_BY_REQUIRE_HUMAN" in failure for failure in executor.scheduler_recorded_failures)


def test_ecp_cpm27_scheduler_production_path_no_raw_provisioner_apply() -> None:
    import asyncio
    from unittest.mock import patch

    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler()
    with patch.object(ScalingProvisioner, "apply", side_effect=AssertionError("raw apply bypass")):
        scheduler._collector.record_backpressure()
        asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]


def test_ecp_cpm28_ceiling_automatic_mutation_fail_closed() -> None:
    import asyncio

    from intergrax.runtime.capacity.ceiling_patcher import BoundedOrchestrationCeilingPatcher
    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    patcher = BoundedOrchestrationCeilingPatcher(max_inflight_nodes=10)
    scheduler, _k8s, _celery, executor, _recording = _build_governed_scheduler()
    executor.attach_scheduler_dependencies(ceiling_patcher=patcher)
    action = ScalingAction(
        kind=ScalingActionKind.RAISE_ORCHESTRATION_CEILING,
        target=ScalingTarget.ORCHESTRATION_CEILING,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert patcher.max_inflight_nodes == 10
    assert executor.scheduler_applied_actions == ()
    assert any(
        "ECP_SCHEDULER_CEILING_UNSUPPORTED" in failure
        for failure in executor.scheduler_recorded_failures
    )


def test_ecp_cpm29_require_human_scheduler_preserves_governance_outcome() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler(
        evaluator=evaluator,
    )
    action = ScalingAction(
        action_id="scheduler-require-human-action",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert len(scheduler.blocked_outcomes) == 1
    blocked = scheduler.blocked_outcomes[0]
    assert blocked.action_id == "scheduler-require-human-action"
    assert blocked.blocker_code == "ECP_BLOCKED_BY_REQUIRE_HUMAN"
    assert blocked.policy_action == PolicyAction.REQUIRE_HUMAN.value
    assert blocked.authorization_scope is not None
    assert blocked.authorization_scope.mutation_id == "scheduler-require-human-action"
    assert blocked.authorization_evidence is not None
    assert blocked.authorization_evidence.mutation_id == "scheduler-require-human-action"
    assert blocked.authorization_evidence.policy_action is PolicyAction.REQUIRE_HUMAN


def test_ecp_cpm30_deny_scheduler_preserves_canonical_evidence() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.DENY, reason="blocked"),
    )
    scheduler, k8s, celery, executor, _recording = _build_governed_scheduler(
        evaluator=evaluator,
    )
    k8s_action = ScalingAction(
        action_id="scheduler-deny-k8s",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    celery_action = ScalingAction(
        action_id="scheduler-deny-celery",
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(
        actions=(k8s_action, celery_action),
        evaluation_status="planned",
    )
    asyncio.run(scheduler._apply_plan(plan))
    assert k8s.scale_calls == []
    assert celery.worker_count == 2
    assert executor.scheduler_applied_actions == ()
    assert len(scheduler.blocked_outcomes) == 2
    for blocked in scheduler.blocked_outcomes:
        assert blocked.blocker_code == "ECP_BLOCKED_BY_POLICY"
        assert blocked.policy_action == PolicyAction.DENY.value
        assert blocked.authorization_evidence is not None
        assert blocked.authorization_evidence.policy_action is PolicyAction.DENY
        assert blocked.authorization_evidence.mutation_id == blocked.action_id


def test_ecp_cpm31_wrong_tenant_scheduler_preserves_tenant_denial_without_evidence() -> None:
    import asyncio

    scheduler, k8s, _celery, executor, recording = _build_governed_scheduler(
        execution_identity=_scheduler_service_principal(tenant_id=_OTHER_TENANT),
        tenant_id=_TENANT,
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert recording.calls == []
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert len(scheduler.blocked_outcomes) == 1
    blocked = scheduler.blocked_outcomes[0]
    assert blocked.blocker_code == "ECP_BLOCKED_BY_TENANT_AUTHORITY"
    assert blocked.policy_action == PolicyAction.DENY.value
    assert blocked.tenant_scope_denial is not None
    assert blocked.tenant_scope_denial.reason == "principal_tenant_mismatch"
    assert blocked.authorization_evidence is None
    assert blocked.authorization_scope is None


def test_ecp_cpm32_stale_state_scheduler_blocked_without_cpm_evidence() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    scheduler, k8s, _celery, executor, _recording = _build_governed_scheduler(
        k8s=_FlakyK8s(),
    )
    action = ScalingAction(
        action_id="scheduler-stale-k8s",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert len(scheduler.blocked_outcomes) == 1
    blocked = scheduler.blocked_outcomes[0]
    assert blocked.action_id == "scheduler-stale-k8s"
    assert blocked.blocker_code == "ECP_SCHEDULER_STALE_STATE"
    assert blocked.policy_action is None
    assert blocked.authorization_evidence is None
    assert blocked.authorization_scope is None
    assert blocked.tenant_scope_denial is None


def _human_principal(tenant_id: str = _TENANT) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="capacity-operator",
        principal_type=PrincipalType.USER,
        auth_subject="capacity-operator",
    )


def _build_hitl_governed_scheduler(
    *,
    k8s: _RecordingK8s | None = None,
    celery: CeleryProductionAdapter | None = None,
    evaluator: _RecordingEvaluator | None = None,
    tenant_id: str = _TENANT,
    require_hitl_policy: bool = False,
) -> tuple[
    "CapacityScheduler",
    _RecordingK8s,
    CeleryProductionAdapter,
    GovernedCapacityMutationExecutor,
    _RecordingEvaluator,
    "CapacityApprovalQueue",
    "ControlPlaneMutationApprovalCoordinator",
]:
    import asyncio

    from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue
    from intergrax.runtime.capacity.collector import CapacitySignalCollector
    from intergrax.runtime.capacity.contracts import ScalingPolicy, ScalingRule
    from intergrax.runtime.capacity.evaluator import ScalingEvaluator
    from intergrax.runtime.capacity.scheduler import CapacityScheduler
    from intergrax.runtime.governance.control_plane_mutation_approval import (
        ApprovalConsumingControlPlaneMutationEvaluator,
        ControlPlaneMutationApprovalCoordinator,
    )

    kubernetes = k8s or _RecordingK8s()
    celery_adapter = celery or CeleryProductionAdapter(worker_count=2)
    recording = evaluator or _RecordingEvaluator()
    coordinator = ControlPlaneMutationApprovalCoordinator()
    queue = CapacityApprovalQueue(coordinator=coordinator)
    wrapped = ApprovalConsumingControlPlaneMutationEvaluator(
        inner=recording,
        coordinator=coordinator,
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=wrapped)
    provisioner = ScalingProvisioner(
        kubernetes=kubernetes,
        celery=celery_adapter,
        execution_mode=ProvisionerExecutionMode.GOVERNED_ONLY,
    )
    governed_executor = GovernedCapacityMutationExecutor(
        provisioner=provisioner,
        mutation_boundary=boundary,
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=tenant_id),
        approval_coordinator=coordinator,
    )
    policy = ScalingPolicy(
        enabled=True,
        require_hitl_for_scale_up=require_hitl_policy,
        rules=[
            ScalingRule(
                rule_id="k8s",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            ),
            ScalingRule(
                rule_id="celery",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            ),
        ],
    )
    collector = CapacitySignalCollector()
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=ScalingEvaluator(policy),
        provisioner=None,
        approval_queue=queue,
        execution_identity=_scheduler_service_principal(tenant_id),
        governed_capacity_executor=governed_executor,
        tenant_id=tenant_id,
        requires_governed_execution=True,
    )
    return scheduler, kubernetes, celery_adapter, governed_executor, recording, queue, coordinator


def test_ecp_cpm33_require_human_k8s_pending_preserves_scope() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    pending = queue.list_pending()
    assert len(pending) == 1
    record = pending[0]
    assert record.mutation_id == record.action.action_id
    assert record.authorization_scope.mutation_id == record.mutation_id
    assert record.authorization_evidence.policy_action is PolicyAction.REQUIRE_HUMAN


def test_ecp_cpm34_human_approve_k8s_resume_exact_target_once() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    mutation_id = queue.list_pending()[0].mutation_id
    grant = queue.approve_mutation(mutation_id, _human_principal())
    assert grant is not None
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]
    assert len(executor.scheduler_applied_actions) == 1
    assert coordinator.is_consumed(grant.grant_id)


def test_ecp_cpm35_human_deny_zero_provider_effect() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    mutation_id = queue.list_pending()[0].mutation_id
    denial = queue.deny_mutation(mutation_id, _human_principal())
    assert denial is not None
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert coordinator.get_denial(mutation_id) is not None
    assert queue.approve_mutation(mutation_id, _human_principal()) is None


def test_ecp_cpm36_wrong_approver_tenant_fail_closed() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    mutation_id = queue.list_pending()[0].mutation_id
    assert queue.approve_mutation(mutation_id, _human_principal(tenant_id=_OTHER_TENANT)) is None
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm37_wrong_mutation_approval_cannot_resume_other_action() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    action_a = ScalingAction(
        action_id="hitl-action-a",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    action_b = ScalingAction(
        action_id="hitl-action-b",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    pending_a = scheduler._governed_capacity_executor.prepare_k8s_pending_authorization(
        principal=_scheduler_service_principal(),
        tenant_id=_TENANT,
        mutation_id=action_a.action_id,
        deployment=_DEPLOYMENT,
        delta=1,
    )
    pending_b = scheduler._governed_capacity_executor.prepare_k8s_pending_authorization(
        principal=_scheduler_service_principal(),
        tenant_id=_TENANT,
        mutation_id=action_b.action_id,
        deployment=_DEPLOYMENT,
        delta=1,
    )
    queue.submit_pending(
        plan_id="plan-a",
        action=action_a,
        authorization_scope=pending_a.authorization_scope,
        authorization_evidence=pending_a.authorization_evidence,
        service_principal=_scheduler_service_principal(),
    )
    queue.submit_pending(
        plan_id="plan-b",
        action=action_b,
        authorization_scope=pending_b.authorization_scope,
        authorization_evidence=pending_b.authorization_evidence,
        service_principal=_scheduler_service_principal(),
    )
    grant_a = queue.approve_mutation(action_a.action_id, _human_principal())
    assert grant_a is not None
    resumable_a = queue.drain_resumable()[0]
    assert resumable_a.mutation_id == action_a.action_id
    from intergrax.runtime.capacity.approval_queue import CapacityResumableMutation

    resumable_b = CapacityResumableMutation(
        mutation_id=action_b.action_id,
        plan_id="plan-b",
        action=action_b,
        authorization_scope=pending_b.authorization_scope,
        approval_evidence_ref=grant_a.grant_id,
        service_principal=_scheduler_service_principal(),
    )
    scheduler._resume_governed_mutation(resumable_b)
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm38_resource_mismatch_k8s_approval_not_celery() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    k8s_action = ScalingAction(
        action_id="hitl-k8s-only",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    celery_action = ScalingAction(
        action_id="hitl-celery-only",
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(
        actions=(k8s_action, celery_action),
        evaluation_status="planned",
    )
    asyncio.run(scheduler._apply_plan(plan))
    assert queue.list_pending()
    k8s_pending = next(r for r in queue.list_pending() if r.mutation_id == k8s_action.action_id)
    queue.approve_mutation(k8s_pending.mutation_id, _human_principal())
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]
    assert celery.worker_count == 2
    assert len([a for a in executor.scheduler_applied_actions if a.action_id == celery_action.action_id]) == 0


def test_ecp_cpm39_stale_current_after_approval_no_provider_write() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    flaky = _FlakyK8s()
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator, k8s=flaky)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    mutation_id = queue.list_pending()[0].mutation_id
    queue.approve_mutation(mutation_id, _human_principal())
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    assert any("ECP_SCHEDULER_STALE_STATE" in failure for failure in executor.scheduler_recorded_failures)


def test_ecp_cpm40_target_mismatch_cannot_execute() -> None:
    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    action = ScalingAction(
        action_id="target-mismatch",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    pending = scheduler._governed_capacity_executor.prepare_k8s_pending_authorization(
        principal=_scheduler_service_principal(),
        tenant_id=_TENANT,
        mutation_id=action.action_id,
        deployment=_DEPLOYMENT,
        delta=1,
    )
    queue.submit_pending(
        plan_id="plan-target",
        action=action,
        authorization_scope=pending.authorization_scope,
        authorization_evidence=pending.authorization_evidence,
        service_principal=_scheduler_service_principal(),
    )
    grant = queue.approve_mutation(action.action_id, _human_principal())
    assert grant is not None
    mismatched_scope = pending.authorization_scope.model_copy(
        update={
            "target_revision": k8s_replicas_revision_token(
                deployment=_DEPLOYMENT,
                replicas=4,
            ),
        },
    )
    from intergrax.runtime.capacity.approval_queue import CapacityResumableMutation

    resumable = CapacityResumableMutation(
        mutation_id=action.action_id,
        plan_id="plan-target",
        action=action,
        authorization_scope=mismatched_scope,
        approval_evidence_ref=grant.grant_id,
        service_principal=_scheduler_service_principal(),
    )
    scheduler._resume_governed_mutation(resumable)
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm41_double_drain_cannot_execute_twice() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    queue.approve_mutation(queue.list_pending()[0].mutation_id, _human_principal())
    asyncio.run(scheduler.tick())
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]
    assert len(executor.scheduler_applied_actions) == 1


def test_ecp_cpm42_missing_approval_dependency_fail_closed() -> None:
    import asyncio

    from intergrax.runtime.capacity.approval_queue import CapacityResumableMutation

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    action = ScalingAction(
        action_id="missing-approval",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    pending = scheduler._governed_capacity_executor.prepare_k8s_pending_authorization(
        principal=_scheduler_service_principal(),
        tenant_id=_TENANT,
        mutation_id=action.action_id,
        deployment=_DEPLOYMENT,
        delta=1,
    )
    resumable = CapacityResumableMutation(
        mutation_id=action.action_id,
        plan_id="plan-missing",
        action=action,
        authorization_scope=pending.authorization_scope,
        approval_evidence_ref="cpm-grant:missing",
        service_principal=_scheduler_service_principal(),
    )
    scheduler._resume_governed_mutation(resumable)
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm43_multi_action_per_action_approval_scope() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    k8s_action = ScalingAction(
        action_id="multi-k8s",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    celery_action = ScalingAction(
        action_id="multi-celery",
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(
        actions=(k8s_action, celery_action),
        evaluation_status="planned",
    )
    asyncio.run(scheduler._apply_plan(plan))
    assert len(queue.list_pending()) == 2
    queue.approve_mutation(k8s_action.action_id, _human_principal())
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]
    assert celery.worker_count == 2
    assert len(executor.scheduler_applied_actions) == 1


def test_ecp_cpm44_local_hitl_flag_does_not_bypass_canonical_governance() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok"),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator, require_hitl_policy=True)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()
    pending = queue.list_pending()[0]
    assert pending.authorization_evidence.policy_action is PolicyAction.REQUIRE_HUMAN
    assert queue.approve_mutation(pending.mutation_id, _human_principal()) is not None
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == [3]


def test_ecp_cpm45_scheduler_require_human_routable_to_pending_flow() -> None:
    import asyncio

    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    action = ScalingAction(
        action_id="route-hitl-action",
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0].mutation_id == action.action_id
    assert pending[0].authorization_evidence.mutation_id == action.action_id
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm46_plan_approve_without_approver_cannot_authorize() -> None:
    import asyncio

    evaluator = _RecordingEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
    )
    scheduler, k8s, _celery, executor, _recording, queue, _coordinator = (
        _build_hitl_governed_scheduler(evaluator=evaluator)
    )
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    plan_id = queue.list_pending()[0].plan_id
    assert queue.approve(plan_id) is None
    asyncio.run(scheduler.tick())
    assert k8s.scale_calls == []
    assert executor.scheduler_applied_actions == ()


def test_ecp_cpm47_production_adapters_do_not_expose_k8s_mutation_surface() -> None:
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    observation = adapters.kubernetes_observation
    assert observation.get_replicas(deployment=_DEPLOYMENT) == 2
    with pytest.raises(AttributeError):
        observation.scale_workload(deployment=_DEPLOYMENT, replicas=3)


def test_ecp_cpm48_production_adapters_do_not_expose_celery_mutation_surface() -> None:
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    observation = adapters.celery_observation
    assert observation.get_worker_count() == 2
    with pytest.raises(AttributeError):
        observation.scale_workers(delta=1)


def test_ecp_cpm49_production_probe_verifies_without_raw_provider_access() -> None:
    recording = _RecordingEvaluator()
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=recording),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert apply_production_scale_probe(
        adapters,
        principal=_service_principal(),
        tenant_id=_TENANT,
        k8s_mutation_id="probe-k8s-readonly",
        celery_mutation_id="probe-celery-readonly",
    )
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3
    assert adapters.celery_observation.get_worker_count() == 3
    assert len(recording.calls) == 2


def _build_production_scaling_wiring(
    *,
    evaluator: _RecordingEvaluator | None = None,
) -> tuple:
    import asyncio

    from intergrax.applications._shared.scaling_wiring import wire_application_scaling
    from intergrax.applications.contracts.environment_profile.sub_profiles import ScalingProfile
    from intergrax.runtime.capacity.contracts import ScalingPolicy, ScalingRule

    env = ApplicationEnvironmentProfile.product_defaults()
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="k8s",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            ),
            ScalingRule(
                rule_id="celery",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            ),
        ],
    )
    env = env.model_copy(
        update={
            "governance": env.governance.model_copy(
                update={
                    "scaling": ScalingProfile(policy=policy, production_adapters_enabled=True),
                },
            ),
        },
    )
    recording = evaluator or _RecordingEvaluator()
    governance = build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=recording,
        ),
    )
    adapters = build_production_capacity_adapters(
        mutation_boundary=governance.mutation_authorization_boundary,
        tenant_resolver=governance.tenant_resolver,
    )
    wiring = wire_application_scaling(
        env,
        production_capacity_adapters=adapters,
        production_capacity_governance=governance,
    )
    return wiring, adapters, recording, asyncio


def test_ecp_cpm50_production_scheduler_k8s_through_governed_executor() -> None:
    wiring, adapters, recording, asyncio = _build_production_scaling_wiring()
    assert wiring.scheduler is not None
    assert wiring.provisioner is None
    scheduler = wiring.scheduler
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_SCALE_K8S_DEPLOYMENT
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3


def test_ecp_cpm51_production_scheduler_celery_through_governed_executor() -> None:
    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    wiring, adapters, recording, asyncio = _build_production_scaling_wiring()
    assert wiring.scheduler is not None
    assert wiring.provisioner is None
    scheduler = wiring.scheduler
    action = ScalingAction(
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_SCALE_CELERY_WORKERS
    assert adapters.celery_observation.get_worker_count() == 3


def test_ecp_cpm52_missing_governance_fail_closed_after_composition_refactor() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(env)
    production_wiring = resolve_production_capacity_wiring(env, governance=governance)
    assert production_wiring.enabled is True
    assert production_wiring.adapters is None
    assert production_wiring.probe_passed is False


def test_ecp_cpm53_live_kubernetes_backend_only_behind_governed_write_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LiveLikeKubernetes:
        replicas: int = 2

        def get_replicas(self, deployment: str, *, namespace: str | None = None) -> int:
            del deployment, namespace
            return self.replicas

        def scale_workload(self, deployment: str, *, replicas: int, namespace: str | None = None) -> int:
            del deployment, namespace
            self.replicas = replicas
            return replicas

    monkeypatch.setattr(
        "intergrax.runtime.capacity.production_adapters.resolve_kubernetes_backend",
        lambda: (_LiveLikeKubernetes(), "live"),
    )
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert adapters.kubernetes_backend == "live"
    with pytest.raises(AttributeError):
        adapters.kubernetes_observation.scale_workload(deployment=_DEPLOYMENT, replicas=3)
    adapters.governed_executor.scale_k8s_deployment(
        principal=_service_principal(),
        tenant_id=_TENANT,
        mutation_id="live-k8s-governed",
        deployment=_DEPLOYMENT,
        delta=1,
    )
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3


def test_ecp_cpm54_governed_executor_public_api_does_not_expose_scaling_provisioner() -> None:
    executor, _, _, _ = _build_executor()
    with pytest.raises(AttributeError):
        executor.provisioner  # type: ignore[attr-defined]

    recording = _RecordingEvaluator()
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=recording,
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert {adapter_field.name for adapter_field in fields(ProductionCapacityAdapters)} == {
        "governed_executor",
        "kubernetes_backend",
        "kubernetes_observation",
        "celery_observation",
    }

    wiring, _, _, _asyncio = _build_production_scaling_wiring()
    assert wiring.provisioner is None
    assert wiring.scheduler is not None

    adapters.governed_executor.scale_k8s_deployment(
        principal=_service_principal(),
        tenant_id=_TENANT,
        mutation_id="cpm54-governed",
        deployment=_DEPLOYMENT,
        delta=1,
    )
    assert len(recording.calls) == 1
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3


def test_ecp_cpm55_production_wiring_attaches_scheduler_deps_without_provisioner_escape() -> None:
    wiring, adapters, _, _ = _build_production_scaling_wiring()
    assert wiring.provisioner is None
    assert wiring.scheduler is not None
    assert wiring.scheduler._provisioner is None
    with pytest.raises(AttributeError):
        adapters.governed_executor.provisioner  # type: ignore[attr-defined]


def test_ecp_cpm56_production_scheduler_k8s_through_governed_executor() -> None:
    wiring, adapters, recording, asyncio = _build_production_scaling_wiring()
    scheduler = wiring.scheduler
    assert scheduler is not None
    scheduler._collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_SCALE_K8S_DEPLOYMENT
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3


def test_ecp_cpm57_production_scheduler_celery_through_governed_executor() -> None:
    from intergrax.runtime.capacity.contracts import ScalingActionPlan

    wiring, adapters, recording, asyncio = _build_production_scaling_wiring()
    scheduler = wiring.scheduler
    assert scheduler is not None
    action = ScalingAction(
        kind=ScalingActionKind.SCALE_CELERY_WORKERS,
        target=ScalingTarget.CELERY_POOL,
        delta=1,
    )
    plan = ScalingActionPlan(actions=(action,), evaluation_status="planned")
    asyncio.run(scheduler._apply_plan(plan))
    assert len(recording.calls) == 1
    assert recording.calls[0].mutation_type == MUTATION_TYPE_SCALE_CELERY_WORKERS
    assert adapters.celery_observation.get_worker_count() == 3


def test_ecp_cpm58_read_only_observation_still_works_after_provisioner_surface_closure() -> None:
    recording = _RecordingEvaluator()
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=recording),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 2
    assert adapters.celery_observation.get_worker_count() == 2
    assert apply_production_scale_probe(
        adapters,
        principal=_service_principal(),
        tenant_id=_TENANT,
        k8s_mutation_id="probe-k8s-readonly-r2",
        celery_mutation_id="probe-celery-readonly-r2",
    )
    assert adapters.kubernetes_observation.get_replicas(deployment=_DEPLOYMENT) == 3
    assert adapters.celery_observation.get_worker_count() == 3


def test_ecp_cpm59_governed_only_remains_fail_closed_for_legacy_apply_path() -> None:
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_RecordingEvaluator(),
        ),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
    )
    legacy_provisioner = ScalingProvisioner(
        kubernetes=_RecordingK8s(),
        celery=CeleryProductionAdapter(worker_count=2),
        execution_mode=ProvisionerExecutionMode.GOVERNED_ONLY,
    )
    with pytest.raises(GovernedExecutionRequiredError):
        legacy_provisioner.apply(
            ScalingAction(
                kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
                target=ScalingTarget.NEXUS_HOST,
                delta=1,
            ),
        )
    with pytest.raises(AttributeError):
        adapters.governed_executor.provisioner  # type: ignore[attr-defined]

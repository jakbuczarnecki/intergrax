# © Artur Czarnecki. All rights reserved.

"""ECP control-plane mutation governance proofs (ECP-CPM1–ECP-CPM18)."""

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
        "kubernetes",
        "celery",
        "governed_executor",
        "kubernetes_backend",
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
        "kubernetes",
        "celery",
        "governed_executor",
        "kubernetes_backend",
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
    assert wiring.adapters.kubernetes.get_replicas(deployment=_DEPLOYMENT) == 2
    assert wiring.adapters.celery.worker_count == 2


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

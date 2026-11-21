# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) CAP-1..CAP-7: production capacity wiring fails closed without a cluster."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from intergrax.applications._shared.production_capacity_governance_wiring import (
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import (
    CapacityQualification,
    resolve_production_capacity_wiring,
)
from intergrax.applications._shared.scaling_wiring import wire_application_scaling
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import ScalingProfile
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.capacity.contracts import (
    ScalingAction,
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
    ScalingTarget,
)
from intergrax.runtime.capacity.control_plane_governance import StaticEcpResourceTenantResolver
from intergrax.runtime.capacity.production_adapters import (
    CELERY_BACKEND_KIND,
    CapacityBackendNotConfiguredError,
    CapacityBackendPosture,
    apply_production_scale_probe,
    build_production_capacity_adapters,
    resolve_kubernetes_backend,
)
from intergrax.runtime.capacity.provisioner import GovernedExecutionRequiredError
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-fc2"
_K8S_URL = "INTERGRAX_KUBERNETES_URL"


@dataclass
class _RecordingEvaluator:
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return PolicyDecision(action=PolicyAction.ALLOW, reason="ok")


def _scaling_env() -> ApplicationEnvironmentProfile:
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
        ],
    )
    return env.model_copy(
        update={
            "governance": env.governance.model_copy(
                update={"scaling": ScalingProfile(policy=policy, production_adapters_enabled=True)},
            ),
        },
    )


def _governance(env: ApplicationEnvironmentProfile, recording: _RecordingEvaluator):
    return build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=recording),
    )


@pytest.fixture(autouse=True)
def _no_cluster_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_K8S_URL, raising=False)


def test_product_wiring_without_cluster_is_unqualified_and_builds_no_adapters() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    recording = _RecordingEvaluator()
    wiring = resolve_production_capacity_wiring(env, governance=_governance(env, recording))
    assert wiring.enabled is True
    assert wiring.adapters is None
    assert wiring.probe_passed is False
    assert wiring.qualification is CapacityQualification.UNQUALIFIED_NO_CLUSTER
    assert "kubernetes backend not configured" in wiring.reason
    assert recording.calls == []


def test_unqualified_capacity_refuses_scale_mutation_through_application_scaling() -> None:
    env = _scaling_env()
    recording = _RecordingEvaluator()
    governance = _governance(env, recording)
    capacity = resolve_production_capacity_wiring(env, governance=governance)
    assert capacity.adapters is None
    scaling = wire_application_scaling(
        env,
        production_capacity_adapters=capacity.adapters,
        production_capacity_governance=governance,
    )
    assert scaling.provisioner is not None
    action = ScalingAction(
        kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
        target=ScalingTarget.NEXUS_HOST,
        delta=1,
    )
    with pytest.raises(GovernedExecutionRequiredError):
        scaling.provisioner.apply(action)
    assert scaling.provisioner._kubernetes is None
    assert scaling.provisioner._celery is None
    assert recording.calls == []


def test_build_adapters_default_posture_raises_without_url() -> None:
    with pytest.raises(CapacityBackendNotConfiguredError, match="kubernetes backend not configured"):
        build_production_capacity_adapters(
            mutation_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator()),
            tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
        )


def test_resolve_kubernetes_backend_default_posture_never_returns_in_memory() -> None:
    with pytest.raises(CapacityBackendNotConfiguredError):
        resolve_kubernetes_backend()
    with pytest.raises(CapacityBackendNotConfiguredError):
        resolve_kubernetes_backend(posture=CapacityBackendPosture.PRODUCTION)
    _backend, kind = resolve_kubernetes_backend(posture=CapacityBackendPosture.CI_DETERMINISTIC)
    assert kind == "in_memory"


def test_in_memory_backend_wiring_never_reports_probe_passed() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    recording = _RecordingEvaluator()
    wiring = resolve_production_capacity_wiring(
        env,
        governance=_governance(env, recording),
        backend_posture=CapacityBackendPosture.CI_DETERMINISTIC,
    )
    assert wiring.adapters is not None
    assert wiring.adapters.kubernetes_backend == "in_memory"
    assert wiring.probe_passed is False
    assert wiring.qualification is CapacityQualification.CI_DETERMINISTIC_ONLY
    assert len(recording.calls) == 2


def test_celery_intent_only_backend_is_reported_not_counted_as_live() -> None:
    assert CELERY_BACKEND_KIND == "intent_only"
    assert {member.name for member in CapacityQualification} == {
        "NOT_ENABLED",
        "UNQUALIFIED_NO_CLUSTER",
        "CI_DETERMINISTIC_ONLY",
        "LIVE_KUBERNETES_CELERY_INTENT_ONLY",
    }
    adapters = build_production_capacity_adapters(
        mutation_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator()),
        tenant_resolver=StaticEcpResourceTenantResolver(tenant_id=_TENANT),
        backend_posture=CapacityBackendPosture.CI_DETERMINISTIC,
    )
    assert adapters.celery_observation.get_worker_count() == 2
    assert not any("live_celery" in member.value for member in CapacityQualification)


def test_live_kubernetes_with_intent_only_celery_is_not_fully_qualified(
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
        lambda **_kwargs: (_LiveLikeKubernetes(), "live"),
    )
    env = ApplicationEnvironmentProfile.product_defaults()
    recording = _RecordingEvaluator()
    governance = _governance(env, recording)
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    assert wiring.adapters is not None
    assert wiring.adapters.kubernetes_backend == "live"
    assert wiring.qualification is CapacityQualification.LIVE_KUBERNETES_CELERY_INTENT_ONLY
    probe = apply_production_scale_probe(
        wiring.adapters,
        principal=governance.principal,
        tenant_id=governance.tenant_id,
        k8s_mutation_id="live-k8s",
        celery_mutation_id="live-celery",
    )
    assert wiring.probe_passed is True
    assert probe is True

# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.scaling_wiring import wire_application_scaling
from intergrax.runtime.capacity import ScalingTarget
from intergrax.runtime.capacity.ahi_bridge import scaling_action_from_ahi_proposal
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import (
    CapacitySignal,
    ScalingAction,
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
)
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.metrics import export_capacity_metrics, record_scale_action
from intergrax.runtime.capacity.provisioner import ScalingProvisioner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ecp_package_imports() -> None:
    from intergrax.runtime import capacity  # noqa: F401

    assert ScalingTarget.NEXUS_HOST.value == "nexus_host"


def test_scaling_profile_on_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.scaling_profile.policy.enabled is False


def test_scaling_evaluator_hysteresis() -> None:
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="q",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            )
        ],
    )
    evaluator = ScalingEvaluator(policy)
    signals = list(CapacitySignalCollector().collect(backpressure_rate=0.0))
    signals.append(
        CapacitySignal(
            target=ScalingTarget.CELERY_POOL,
            metric_name="queue_depth",
            value=12.0,
        )
    )
    plan = evaluator.evaluate(signals)
    assert plan.evaluation_status == "planned"
    assert plan.actions


def test_kubernetes_provisioner_scale() -> None:
    class _Client:
        replicas = 2

        def get_replicas(self, deployment: str, *, namespace: str) -> int:
            return self.replicas

        def scale_workload(self, deployment: str, *, replicas: int, namespace: str) -> int:
            self.replicas = replicas
            return replicas

        def health(self) -> bool:
            return True

    from intergrax.integrations._shared.p5.clients import KubernetesCloudPlatform

    k8s = KubernetesCloudPlatform(_Client(), namespace="lab")
    provisioner = ScalingProvisioner(kubernetes=k8s)
    from intergrax.runtime.capacity.contracts import ScalingAction

    ok = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            target=ScalingTarget.NEXUS_HOST,
            delta=1,
        ),
        deployment="nexus-host",
    )
    assert ok is True
    assert k8s.get_replicas(deployment="nexus-host") == 3


def test_scaling_wiring_noop_when_disabled() -> None:
    wiring = wire_application_scaling(ApplicationEnvironmentProfile.lab_defaults())
    assert wiring.scheduler is None


def test_ahi_bridge_action() -> None:
    action = scaling_action_from_ahi_proposal(ceiling_delta=2, reason="approved")
    assert action.delta == 2


def test_capacity_metrics_export() -> None:
    record_scale_action(target="nexus_host")
    metrics = export_capacity_metrics()
    assert metrics["harness_scale_actions_total"] >= 1.0

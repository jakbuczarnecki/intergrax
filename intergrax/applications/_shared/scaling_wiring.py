# © Artur Czarnecki. All rights reserved.

"""Elastic capacity host wiring (ECP-1.4 / ECP-5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.provisioner import ScalingProvisioner
from intergrax.runtime.capacity.scheduler import CapacityScheduler


@dataclass(frozen=True, slots=True)
class ApplicationScalingWiring:
    """No-op when scaling policy disabled."""

    collector: CapacitySignalCollector | None
    evaluator: ScalingEvaluator | None
    provisioner: ScalingProvisioner | None
    scheduler: CapacityScheduler | None


def wire_application_scaling(
    env: ApplicationEnvironmentProfile,
) -> ApplicationScalingWiring:
    policy = env.scaling_profile.policy
    if not policy.enabled:
        return ApplicationScalingWiring(None, None, None, None)
    collector = CapacitySignalCollector()
    evaluator = ScalingEvaluator(policy)
    provisioner = ScalingProvisioner()
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=evaluator,
        provisioner=provisioner,
    )
    return ApplicationScalingWiring(collector, evaluator, provisioner, scheduler)

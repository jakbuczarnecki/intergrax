# © Artur Czarnecki. All rights reserved.

"""Elastic capacity host wiring (ECP-1.4 / ECP-5.2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.capacity.action_gate import CapacityActionGate
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.event_bridge import CapacityEventBridge
from intergrax.runtime.capacity.events import PublishFn
from intergrax.runtime.capacity.provisioner import ScalingProvisioner
from intergrax.runtime.capacity.scheduler import CapacityScheduler
from intergrax.runtime.events.event_bus import RuntimeEventBus


@dataclass(frozen=True, slots=True)
class ApplicationScalingWiring:
    """No-op when scaling policy disabled."""

    collector: CapacitySignalCollector | None
    evaluator: ScalingEvaluator | None
    provisioner: ScalingProvisioner | None
    scheduler: CapacityScheduler | None
    event_bridge: CapacityEventBridge | None


def wire_application_scaling(
    env: ApplicationEnvironmentProfile,
    *,
    event_bus: RuntimeEventBus | None = None,
    publish: PublishFn | None = None,
    queue_depth_provider: Callable[[], float] | None = None,
) -> ApplicationScalingWiring:
    policy = env.scaling_profile.policy
    if not policy.enabled:
        return ApplicationScalingWiring(None, None, None, None, None)
    collector = CapacitySignalCollector(
        publish=publish,
        queue_depth_provider=queue_depth_provider,
    )
    evaluator = ScalingEvaluator(policy, publish=publish)
    provisioner = ScalingProvisioner(action_gate=CapacityActionGate())
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=evaluator,
        provisioner=provisioner,
    )
    event_bridge: CapacityEventBridge | None = None
    if event_bus is not None:
        event_bridge = CapacityEventBridge(collector, event_bus)
        event_bridge.attach()
    return ApplicationScalingWiring(collector, evaluator, provisioner, scheduler, event_bridge)

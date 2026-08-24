# © Artur Czarnecki. All rights reserved.

"""Elastic capacity host wiring (ECP-1.4 / ECP-5.2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.orchestration_wiring import resolve_max_inflight_nodes
from intergrax.applications._shared.production_capacity_governance_wiring import (
    ProductionCapacityGovernance,
)
from intergrax.runtime.capacity.production_adapters import ProductionCapacityAdapters
from intergrax.runtime.capacity.provisioner import ProvisionerExecutionMode
from intergrax.runtime.capacity.ceiling_patcher import BoundedOrchestrationCeilingPatcher
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.capacity.action_gate import CapacityActionGate
from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.event_bridge import CapacityEventBridge
from intergrax.runtime.capacity.events import PublishFn
from intergrax.runtime.capacity.provisioner import ScalingProvisioner
from intergrax.runtime.capacity.queue_depth import make_queue_depth_provider
from intergrax.runtime.capacity.governed_capacity_mutation import GovernedCapacityMutationExecutor
from intergrax.runtime.capacity.scheduler import CapacityScheduler
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ControlPlaneMutationApprovalCoordinator,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus


@dataclass(frozen=True, slots=True)
class ApplicationScalingWiring:
    """No-op when scaling policy disabled."""

    collector: CapacitySignalCollector | None
    evaluator: ScalingEvaluator | None
    provisioner: ScalingProvisioner | None
    scheduler: CapacityScheduler | None
    event_bridge: CapacityEventBridge | None
    approval_queue: CapacityApprovalQueue | None


def wire_application_scaling(
    env: ApplicationEnvironmentProfile,
    *,
    event_bus: RuntimeEventBus | None = None,
    publish: PublishFn | None = None,
    queue_depth_provider: Callable[[], float] | None = None,
    kv_store: DistributedKVStore | None = None,
    tenant_id: str = "harness",
    production_capacity_adapters: ProductionCapacityAdapters | None = None,
    production_capacity_governance: ProductionCapacityGovernance | None = None,
) -> ApplicationScalingWiring:
    policy = env.scaling_profile.policy
    if not policy.enabled:
        return ApplicationScalingWiring(None, None, None, None, None, None)
    requires_governed_execution = (
        env.application_profile is ApplicationProfile.PRODUCT
        and env.scaling_profile.production_adapters_enabled
    )
    resolved_tenant_id = tenant_id
    execution_identity = None
    governed_executor = None
    if requires_governed_execution and production_capacity_governance is not None:
        resolved_tenant_id = production_capacity_governance.tenant_id
        execution_identity = production_capacity_governance.principal
        if production_capacity_adapters is not None:
            governed_executor = production_capacity_adapters.governed_executor
    resolved_queue_depth = queue_depth_provider
    if resolved_queue_depth is None and kv_store is not None:
        resolved_queue_depth = make_queue_depth_provider(kv_store, resolved_tenant_id)
    approval_coordinator: ControlPlaneMutationApprovalCoordinator | None = None
    approval_queue: CapacityApprovalQueue | None = None
    if policy.require_hitl_for_scale_up or requires_governed_execution:
        approval_coordinator = ControlPlaneMutationApprovalCoordinator()
        approval_queue = CapacityApprovalQueue(coordinator=approval_coordinator)
    collector = CapacitySignalCollector(
        publish=publish,
        queue_depth_provider=resolved_queue_depth,
    )
    evaluator = ScalingEvaluator(policy, publish=publish)
    ceiling_patcher = BoundedOrchestrationCeilingPatcher(
        max_inflight_nodes=resolve_max_inflight_nodes(env) or 8,
    )
    kubernetes_backend = None
    celery_backend = None
    if production_capacity_adapters is not None:
        kubernetes_backend = production_capacity_adapters.kubernetes
        celery_backend = production_capacity_adapters.celery
    provisioner = ScalingProvisioner(
        kubernetes=kubernetes_backend,
        celery=celery_backend,
        action_gate=CapacityActionGate(),
        ceiling_patcher=ceiling_patcher,
        publish=publish,
        execution_mode=(
            ProvisionerExecutionMode.GOVERNED_ONLY if requires_governed_execution
            else ProvisionerExecutionMode.UNRESTRICTED
        ),
    )
    if (
        requires_governed_execution
        and production_capacity_governance is not None
        and approval_coordinator is not None
    ):
        governed_executor = GovernedCapacityMutationExecutor(
            provisioner=provisioner,
            mutation_boundary=production_capacity_governance.mutation_authorization_boundary,
            tenant_resolver=production_capacity_governance.tenant_resolver,
            approval_coordinator=approval_coordinator,
        )
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=evaluator,
        provisioner=provisioner,
        approval_queue=approval_queue,
        publish=publish,
        execution_identity=execution_identity,
        governed_capacity_executor=governed_executor,
        tenant_id=resolved_tenant_id if requires_governed_execution else None,
        requires_governed_execution=requires_governed_execution,
    )
    event_bridge: CapacityEventBridge | None = None
    if event_bus is not None:
        event_bridge = CapacityEventBridge(collector, event_bus)
        event_bridge.attach()
    return ApplicationScalingWiring(
        collector,
        evaluator,
        provisioner,
        scheduler,
        event_bridge,
        approval_queue,
    )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production worker recovery governed fulfillment wiring (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.autonomous_work.capability_acquisition_ports import (
    WorkerCapabilityProfileResolver,
)
from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionPort,
)
from intergrax.autonomous_work.repository import WorkerPrincipalBindingRepository
from intergrax.autonomous_work.worker_capability_fulfillment_composition import (
    build_worker_capability_fulfillment_coordinator,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_capability_fulfillment_ports import (
    CapabilityRealizationCoordinatorPort,
    WorkerCapabilityDirectReuseFulfillmentPort,
    WorkerCapabilityRecoveryPort,
)
from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
    WorkerConfiguredCapabilityFulfillmentService,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingResolver,
)
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationRealizationPort,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ExistingCapabilityConfigurationOpportunityReadPort,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningStore,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationResolution,
)
from intergrax.runtime.execution.execution_bound_integration_pinning_adapter import (
    ExecutionBoundIntegrationConfigurationExecutionPinningAdapter,
)
from intergrax.runtime.execution.execution_integration_configuration_pinning_ports import (
    ExecutionIntegrationConfigurationExecutionPinningPort,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_composition import (
    build_worker_qualified_capability_resume_coordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_ports import (
    QualifiedCapabilityBindingPort,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_async_service import (
    WorkerRecoveryCapabilityFulfillmentAsyncService,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_ports import (
    WorkerRecoveryFulfillmentTaskContextReadPort,
    WorkerRecoveryObstacleCapabilityNeedReadPort,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_provider import (
    DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_request_builder import (
    WorkerRecoveryCapabilityFulfillmentRequestBuilder,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.runtime.execution.governed_task_scoped_qualified_capability_execution_dispatch import (
    ActiveTaskRegistryGovernedExecutionTaskLookup,
    GovernedTaskScopedQualifiedCapabilityExecutionDispatchService,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionAsyncDispatchPort,
    QualifiedCapabilityExecutionDispatchPort,
)


class GovernedFulfillmentInnerDispatchPort(
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionAsyncDispatchPort,
    Protocol,
):
    """Sync + async dispatch surface required by governed task-scoped wrapper."""
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_async_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAsyncAdapter,
)


@dataclass(frozen=True, slots=True)
class WorkerRecoveryGovernedFulfillmentWiring:
    """Production composition artifact — not a semantic owner."""

    request_builder: WorkerRecoveryCapabilityFulfillmentRequestBuilder
    fulfillment: WorkerCapabilityFulfillmentCoordinator
    fulfillment_sync: WorkerRecoveryCapabilityFulfillmentService
    fulfillment_async: WorkerRecoveryCapabilityFulfillmentAsyncService
    governed_dispatch: GovernedTaskScopedQualifiedCapabilityExecutionDispatchService
    resume: WorkerQualifiedCapabilityResumeCoordinator
    integration_configuration_pinning: (
        ExecutionIntegrationConfigurationExecutionPinningPort | None
    ) = None


def build_worker_recovery_governed_fulfillment_wiring(
    *,
    recovery: WorkerCapabilityRecoveryPort,
    direct_reuse: WorkerCapabilityDirectReuseFulfillmentPort,
    inner_dispatch: GovernedFulfillmentInnerDispatchPort,
    binding: QualifiedCapabilityBindingPort,
    obstacle_capability_need_reader: WorkerRecoveryObstacleCapabilityNeedReadPort,
    task_context_reader: WorkerRecoveryFulfillmentTaskContextReadPort,
    principal_binding_repository: WorkerPrincipalBindingRepository,
    capability_profile_resolver: WorkerCapabilityProfileResolver,
    realization: CapabilityRealizationCoordinatorPort | None = None,
    opportunity_read: ExistingCapabilityConfigurationOpportunityReadPort | None = None,
    configuration_realization: ExistingCapabilityConfigurationRealizationPort | None = None,
    configuration_pinning_store: ExecutionIntegrationConfigurationPinningStore | None = None,
    authority_admission: WorkerExecutionAdmissionPort | None = None,
) -> WorkerRecoveryGovernedFulfillmentWiring:
    """Wire production builder, sync/async fulfillment, governed dispatch, and resume coordinator."""
    episode_context = DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider(
        obstacle_capability_need_reader=obstacle_capability_need_reader,
        task_context_reader=task_context_reader,
        principal_binding_repository=principal_binding_repository,
        capability_profile_resolver=capability_profile_resolver,
    )
    governed_dispatch = GovernedTaskScopedQualifiedCapabilityExecutionDispatchService(
        inner=inner_dispatch,
        task_lookup=ActiveTaskRegistryGovernedExecutionTaskLookup(),
    )
    sync_execution = WorkerQualifiedCapabilityExecutionEngineAdapter(
        dispatch=governed_dispatch,
    )
    async_execution = WorkerQualifiedCapabilityExecutionEngineAsyncAdapter(
        dispatch=governed_dispatch,
    )
    resume_coordinator = build_worker_qualified_capability_resume_coordinator(
        binding=binding,
        execution=sync_execution,
        async_execution=async_execution,
        authority_admission=authority_admission,
    )
    configured_fulfillment = None
    if opportunity_read is not None and configuration_realization is not None:
        configured_fulfillment = WorkerConfiguredCapabilityFulfillmentService(
            opportunity_read=opportunity_read,
            realization=configuration_realization,
            principal_binding_resolver=WorkerPrincipalBindingResolver(
                principal_binding_repository,
            ),
        )
    integration_configuration_pinning = None
    if configuration_pinning_store is not None:
        integration_configuration_pinning = (
            ExecutionBoundIntegrationConfigurationExecutionPinningAdapter(
                resolution=ExecutionBoundIntegrationResolution(
                    pinning_store=configuration_pinning_store,
                ),
            )
        )
    fulfillment_coordinator = build_worker_capability_fulfillment_coordinator(
        recovery=recovery,
        resume=resume_coordinator,
        direct_reuse=direct_reuse,
        realization=realization,
        configured_fulfillment=configured_fulfillment,
    )
    request_builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(
        episode_context=episode_context,
    )
    recovery_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=fulfillment_coordinator,
    )
    recovery_fulfillment_async = WorkerRecoveryCapabilityFulfillmentAsyncService(
        fulfillment=fulfillment_coordinator,
    )
    return WorkerRecoveryGovernedFulfillmentWiring(
        request_builder=request_builder,
        fulfillment=fulfillment_coordinator,
        fulfillment_sync=recovery_fulfillment,
        fulfillment_async=recovery_fulfillment_async,
        governed_dispatch=governed_dispatch,
        resume=resume_coordinator,
        integration_configuration_pinning=integration_configuration_pinning,
    )


__all__ = [
    "WorkerRecoveryGovernedFulfillmentWiring",
    "build_worker_recovery_governed_fulfillment_wiring",
]

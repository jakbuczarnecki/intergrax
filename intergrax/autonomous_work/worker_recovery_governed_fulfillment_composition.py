# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production worker recovery governed fulfillment wiring (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_async_service import (
    WorkerRecoveryCapabilityFulfillmentAsyncService,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context import (
    WorkerRecoveryCapabilityFulfillmentEpisodeContextPort,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_request_builder import (
    WorkerRecoveryCapabilityFulfillmentRequestBuilder,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.runtime.execution.governed_task_scoped_qualified_capability_execution_dispatch import (
    ActiveTaskRegistryGovernedExecutionTaskLookup,
    GovernedTaskScopedQualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_async_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAsyncAdapter,
)


def build_worker_recovery_governed_fulfillment_wiring(
    *,
    fulfillment_coordinator: WorkerCapabilityFulfillmentCoordinator,
    episode_context: WorkerRecoveryCapabilityFulfillmentEpisodeContextPort,
    inner_dispatch: QualifiedCapabilityExecutionDispatchService,
) -> tuple[
    WorkerRecoveryCapabilityFulfillmentRequestBuilder,
    WorkerRecoveryCapabilityFulfillmentService,
    WorkerRecoveryCapabilityFulfillmentAsyncService,
    GovernedTaskScopedQualifiedCapabilityExecutionDispatchService,
]:
    """Wire production builder, sync/async fulfillment, and governed-task scoped dispatch."""
    governed_dispatch = GovernedTaskScopedQualifiedCapabilityExecutionDispatchService(
        inner=inner_dispatch,
        task_lookup=ActiveTaskRegistryGovernedExecutionTaskLookup(),
    )
    _ = governed_dispatch
    request_builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(
        episode_context=episode_context,
    )
    recovery_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=fulfillment_coordinator,
    )
    recovery_fulfillment_async = WorkerRecoveryCapabilityFulfillmentAsyncService(
        fulfillment=fulfillment_coordinator,
    )
    return (
        request_builder,
        recovery_fulfillment,
        recovery_fulfillment_async,
        governed_dispatch,
    )


def wire_governed_execution_into_resume_coordinator(
    *,
    resume: WorkerQualifiedCapabilityResumeCoordinator,
    governed_dispatch: GovernedTaskScopedQualifiedCapabilityExecutionDispatchService,
) -> tuple[
    WorkerQualifiedCapabilityExecutionEngineAdapter,
    WorkerQualifiedCapabilityExecutionEngineAsyncAdapter,
]:
    """Attach governed scoped dispatch to sync/async worker execution adapters."""
    sync_execution = WorkerQualifiedCapabilityExecutionEngineAdapter(
        dispatch=governed_dispatch,
    )
    async_execution = WorkerQualifiedCapabilityExecutionEngineAsyncAdapter(
        dispatch=governed_dispatch,
    )
    resume._execution = sync_execution  # noqa: SLF001 — composition root explicit wiring
    resume._async_execution = async_execution  # noqa: SLF001
    return sync_execution, async_execution


__all__ = [
    "build_worker_recovery_governed_fulfillment_wiring",
    "wire_governed_execution_into_resume_coordinator",
]

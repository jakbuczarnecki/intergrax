# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable recovery episode context provider — read/project only (UCA-6C-R6-R5.8-R2-H1-R1)."""

from __future__ import annotations

from intergrax.autonomous_work.capability_acquisition_ports import (
    WorkerCapabilityProfileResolver,
)
from intergrax.autonomous_work.repository import WorkerPrincipalBindingRepository
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context import (
    WorkerRecoveryCapabilityFulfillmentEpisodeContext,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_ports import (
    WorkerRecoveryFulfillmentTaskContextReadPort,
    WorkerRecoveryObstacleCapabilityNeedReadPort,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryEpisode,
    WorkerRecoveryOrchestrationRequest,
)


class DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider:
    """Project fulfillment facts from durable recovery owners — fail closed on mismatch."""

    def __init__(
        self,
        *,
        obstacle_capability_need_reader: WorkerRecoveryObstacleCapabilityNeedReadPort,
        task_context_reader: WorkerRecoveryFulfillmentTaskContextReadPort,
        principal_binding_repository: WorkerPrincipalBindingRepository,
        capability_profile_resolver: WorkerCapabilityProfileResolver,
    ) -> None:
        self._obstacle_need_reader = obstacle_capability_need_reader
        self._task_context_reader = task_context_reader
        self._principal_binding_repository = principal_binding_repository
        self._capability_profile_resolver = capability_profile_resolver

    def resolve_episode_context(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentEpisodeContext | None:
        if episode.recovery_decision_id != request.decision.decision_id:
            return None
        if episode.worker_instance_id != request.original_source.worker_instance_id:
            return None
        if episode.obstacle_id != request.decision.obstacle_id:
            return None
        worker_need = self._obstacle_need_reader.get_obstacle_capability_need(
            worker_instance_id=episode.worker_instance_id,
            obstacle_id=episode.obstacle_id,
        )
        if worker_need is None:
            return None
        if worker_need.worker_instance_id != episode.worker_instance_id:
            return None
        if worker_need.obstacle_id != episode.obstacle_id:
            return None
        if worker_need.recovery_decision_id != request.decision.decision_id:
            return None
        run_id = request.resume_target.run_id or episode.resume_target.run_id
        task_context = self._task_context_reader.resolve_task_context(run_id=run_id)
        if task_context is None:
            return None
        principal_binding = self._principal_binding_repository.get(
            worker_instance_id=episode.worker_instance_id,
        )
        if principal_binding is None:
            return None
        if principal_binding.tenant_id != task_context.tenant_id:
            return None
        policy = self._capability_profile_resolver.resolve(
            worker_need.capability_profile_ref,
        )
        requested_scopes = request.resume_target.requested_scopes
        if not requested_scopes:
            requested_scopes = episode.resume_target.requested_scopes
        return WorkerRecoveryCapabilityFulfillmentEpisodeContext(
            worker_need=worker_need,
            recovery_decision=request.decision,
            capability_profile_ref=worker_need.capability_profile_ref,
            tenant_id=task_context.tenant_id,
            task_id=task_context.task_id,
            requested_authority_scopes=requested_scopes,
            allow_generic_acquisition=policy.generated_capability_allowed,
            run_id=task_context.run_id,
            attempt_id=task_context.attempt_id,
            codecraft_profile_ref=worker_need.codecraft_profile_ref,
        )


__all__ = ["DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider"]

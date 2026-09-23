# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production recovery episode → WorkerCapabilityFulfillmentRequest projection (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context import (
    WorkerRecoveryCapabilityFulfillmentEpisodeContext,
    WorkerRecoveryCapabilityFulfillmentEpisodeContextPort,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityAcquisitionRequest,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryEpisode,
    WorkerRecoveryOrchestrationRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)


class WorkerRecoveryCapabilityFulfillmentRequestBuilder:
    """Stateless projection adapter — fail closed when episode context is incomplete."""

    def __init__(
        self,
        *,
        episode_context: WorkerRecoveryCapabilityFulfillmentEpisodeContextPort,
    ) -> None:
        self._episode_context = episode_context

    def build_fulfillment_request(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
    ) -> WorkerCapabilityFulfillmentRequest | None:
        context = self._episode_context.resolve_episode_context(
            episode=episode,
            request=request,
        )
        if context is None:
            return None
        if not _episode_context_correlates(episode, request, context):
            return None
        need = context.worker_need
        if need.recovery_episode_id is None:
            need = replace(need, recovery_episode_id=episode.recovery_episode_id)
        elif need.recovery_episode_id != episode.recovery_episode_id:
            return None
        acquisition_request = WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=context.recovery_decision,
            capability_profile_ref=context.capability_profile_ref,
            codecraft_profile_ref=context.codecraft_profile_ref,
        )
        return WorkerCapabilityFulfillmentRequest(
            acquisition_request=acquisition_request,
            worker_instance_id=episode.worker_instance_id,
            tenant_id=context.tenant_id,
            task_id=context.task_id,
            requested_at=need.requested_at,
            requested_authority_scopes=context.requested_authority_scopes,
            allow_generic_acquisition=context.allow_generic_acquisition,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
        )


def _episode_context_correlates(
    episode: WorkerRecoveryEpisode,
    request: WorkerRecoveryOrchestrationRequest,
    context: WorkerRecoveryCapabilityFulfillmentEpisodeContext,
) -> bool:
    if episode.worker_instance_id != context.worker_need.worker_instance_id:
        return False
    if episode.obstacle_id != context.worker_need.obstacle_id:
        return False
    if request.decision.decision_id != context.recovery_decision.decision_id:
        return False
    if request.decision.decision_id != context.worker_need.recovery_decision_id:
        return False
    return True


__all__ = ["WorkerRecoveryCapabilityFulfillmentRequestBuilder"]

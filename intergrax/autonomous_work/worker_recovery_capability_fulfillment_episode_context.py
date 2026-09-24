# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Episode-scoped inputs for worker recovery fulfillment projection (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import WorkerRecoveryDecision
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryEpisode,
    WorkerRecoveryOrchestrationRequest,
)
from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId


@dataclass(frozen=True, slots=True)
class WorkerRecoveryCapabilityFulfillmentEpisodeContext:
    """Truthful recovery episode facts — no discovery/acquisition authority."""

    worker_need: WorkerCapabilityNeed
    recovery_decision: WorkerRecoveryDecision
    capability_profile_ref: CapabilityProfileRef
    tenant_id: str
    task_id: TaskId
    requested_authority_scopes: tuple[str, ...]
    allow_generic_acquisition: bool = True
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    codecraft_profile_ref: CodecraftProfileRef | None = None


class WorkerRecoveryCapabilityFulfillmentEpisodeContextPort(Protocol):
    """Resolve durable recovery facts required to project a fulfillment request."""

    def resolve_episode_context(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentEpisodeContext | None: ...


__all__ = [
    "WorkerRecoveryCapabilityFulfillmentEpisodeContext",
    "WorkerRecoveryCapabilityFulfillmentEpisodeContextPort",
]

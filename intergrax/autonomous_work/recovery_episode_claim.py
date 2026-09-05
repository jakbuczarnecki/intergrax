# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared recovery episode claim resolution for repository adapters (AW-6B)."""

from __future__ import annotations

from intergrax.autonomous_work.repository import (
    WorkerRecoveryEpisodeClaim,
    WorkerRecoveryEpisodeClaimStatus,
    WorkerRecoveryEpisodeCreateStatus,
    WorkerRecoveryEpisodeCreateResult,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryEpisode,
    recovery_episodes_logically_equivalent,
)


def resolve_recovery_episode_create(
    incoming: WorkerRecoveryEpisode,
    stored: WorkerRecoveryEpisode | None,
) -> WorkerRecoveryEpisodeCreateResult:
    """Resolve create-or-get against optional stored canonical episode."""
    if stored is None:
        return WorkerRecoveryEpisodeCreateResult(
            status=WorkerRecoveryEpisodeCreateStatus.CREATED,
            episode=incoming,
        )
    if recovery_episodes_logically_equivalent(incoming, stored):
        return WorkerRecoveryEpisodeCreateResult(
            status=WorkerRecoveryEpisodeCreateStatus.EXISTING,
            episode=stored,
        )
    return WorkerRecoveryEpisodeCreateResult(
        status=WorkerRecoveryEpisodeCreateStatus.CONFLICT,
        episode=stored,
    )


def resolve_recovery_attempt_claim(
    *,
    stored: WorkerRecoveryEpisode,
    attempt_number: int,
    claimed_episode: WorkerRecoveryEpisode,
) -> WorkerRecoveryEpisodeClaim:
    """Resolve attempt claim against stored episode state."""
    if (
        stored.claimed_attempt_number is not None
        and stored.last_execution_id is None
    ):
        return WorkerRecoveryEpisodeClaim(
            status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
            episode=stored,
        )
    if stored.claimed_attempt_number == attempt_number:
        if stored.last_execution_id is not None:
            return WorkerRecoveryEpisodeClaim(
                status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
                episode=stored,
            )
        return WorkerRecoveryEpisodeClaim(
            status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
            episode=stored,
        )
    if stored.claimed_attempt_number is not None:
        return WorkerRecoveryEpisodeClaim(
            status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
            episode=stored,
        )
    return WorkerRecoveryEpisodeClaim(
        status=WorkerRecoveryEpisodeClaimStatus.CLAIMED,
        episode=claimed_episode,
    )

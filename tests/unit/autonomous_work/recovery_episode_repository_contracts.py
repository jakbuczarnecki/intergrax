# © Artur Czarnecki. All rights reserved.

"""Shared worker recovery episode repository contract suite (AW-6B)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime

from intergrax.autonomous_work.repository import (
    WorkerRecoveryEpisodeClaimStatus,
    WorkerRecoveryEpisodeCreateStatus,
    WorkerRecoveryEpisodeRepository,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    DECISION_POLICY_VERSION,
    RecoveryStrategy,
    WorkerObstacleSourceKind,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    RecoveryEpisodeStatus,
    WorkerOriginalWorkSource,
    WorkerRecoveryEpisode,
    WorkerRecoveryResumeTarget,
    WorkerRecoveryResumeTargetKind,
    derive_recovery_episode_id,
    recovery_episodes_logically_equivalent,
)
from intergrax.contracts.autonomous_work.revision import Revision, initial_revision
from intergrax.contracts.autonomous_work.ids import mint_wake_up_id
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id
from tests.unit.autonomous_work import repository_contracts as contract_suite

_UTC = UTC
_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_GOAL_ID = contract_suite.mint_worker_goal_id()
_RESP_ID = contract_suite.mint_responsibility_id()


def recovery_episode(**overrides: object) -> WorkerRecoveryEpisode:
    obstacle_id = f"{_WORKER_ID}:execution_failure:execution/terminal/failed-1:occ-1"
    decision_id = f"{obstacle_id}:aw-6a.v1"
    episode_id = derive_recovery_episode_id(
        worker_instance_id=_WORKER_ID,
        obstacle_id=obstacle_id,
        recovery_decision_id=decision_id,
    )
    base = WorkerRecoveryEpisode(
        recovery_episode_id=episode_id,
        worker_instance_id=_WORKER_ID,
        obstacle_id=obstacle_id,
        recovery_decision_id=decision_id,
        decision_policy_version=DECISION_POLICY_VERSION,
        strategy=RecoveryStrategy.RETRY,
        original_source=WorkerOriginalWorkSource(
            worker_instance_id=_WORKER_ID,
            source_kind=WorkerObstacleSourceKind.EXECUTION_FAILURE,
            source_ref="execution/terminal/failed-1",
        ),
        resume_target=WorkerRecoveryResumeTarget(
            kind=WorkerRecoveryResumeTargetKind.GOAL_DECISION,
            source_ref=str(_GOAL_ID),
            goal_id=_GOAL_ID,
            goal_revision=Revision(0),
            responsibility_id=_RESP_ID,
            wake_up_id=mint_wake_up_id(),
        ),
        started_at=_NOW,
        status=RecoveryEpisodeStatus.PENDING,
        attempt_count=0,
        revision=initial_revision(),
        max_attempts=2,
    )
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)


def run_recovery_episode_repository_contract_suite(
    factory: Callable[[], WorkerRecoveryEpisodeRepository],
) -> None:
    repo = factory()
    seed = recovery_episode()
    created = repo.create_or_get(seed)
    assert created.status is WorkerRecoveryEpisodeCreateStatus.CREATED
    assert recovery_episodes_logically_equivalent(created.episode, seed)

    existing = repo.create_or_get(seed)
    assert existing.status is WorkerRecoveryEpisodeCreateStatus.EXISTING
    assert existing.episode == created.episode

    conflicting = repo.create_or_get(
        recovery_episode(recovery_decision_id=f"{seed.recovery_decision_id}:other"),
    )
    assert conflicting.status is WorkerRecoveryEpisodeCreateStatus.CONFLICT

    claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=created.episode.revision,
        claimed_at=_NOW,
    )
    assert claim.status is WorkerRecoveryEpisodeClaimStatus.CLAIMED
    assert claim.episode.attempt_count == 1
    assert claim.episode.status is RecoveryEpisodeStatus.IN_PROGRESS

    duplicate_claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=2,
        expected_revision=claim.episode.revision,
        claimed_at=_NOW,
    )
    assert duplicate_claim.status is WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED

    execution_id = mint_execution_id()
    bound = repo.record_execution(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=claim.episode.revision,
        execution_id=execution_id,
        recorded_at=_NOW,
    )
    assert bound.last_execution_id == execution_id

    loaded = repo.get(recovery_episode_id=seed.recovery_episode_id)
    assert loaded == bound


def test_recovery_episode_create_claim_and_bind(
    factory: Callable[[], WorkerRecoveryEpisodeRepository],
) -> None:
    run_recovery_episode_repository_contract_suite(factory)


def test_recovery_episode_orphaned_claim_blocks_next_attempt(
    factory: Callable[[], WorkerRecoveryEpisodeRepository],
) -> None:
    repo = factory()
    seed = recovery_episode()
    created = repo.create_or_get(seed)
    claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=created.episode.revision,
        claimed_at=_NOW,
    )
    assert claim.status is WorkerRecoveryEpisodeClaimStatus.CLAIMED
    next_claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=2,
        expected_revision=claim.episode.revision,
        claimed_at=_NOW,
    )
    assert next_claim.status is WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED
    assert next_claim.episode.last_execution_id is None


def test_recovery_episode_concurrent_create_same_payload(
    factory: Callable[[], WorkerRecoveryEpisodeRepository],
) -> None:
    seed = recovery_episode()
    repo = factory()
    results: list[WorkerRecoveryEpisodeCreateStatus] = []

    def worker() -> None:
        results.append(repo.create_or_get(seed).status)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert WorkerRecoveryEpisodeCreateStatus.CREATED in results
    assert WorkerRecoveryEpisodeCreateStatus.EXISTING in results


def test_recovery_episode_terminal_restart(
    factory: Callable[[], WorkerRecoveryEpisodeRepository],
) -> None:
    repo = factory()
    seed = recovery_episode()
    created = repo.create_or_get(seed)
    claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=created.episode.revision,
        claimed_at=_NOW,
    )
    execution_id = mint_execution_id()
    bound = repo.record_execution(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=claim.episode.revision,
        execution_id=ExecutionId(str(execution_id)),
        recorded_at=_NOW,
    )
    terminal = repo.mark_succeeded(
        recovery_episode_id=seed.recovery_episode_id,
        expected_revision=bound.revision,
        completed_at=_NOW,
    )
    assert terminal.status is RecoveryEpisodeStatus.SUCCEEDED

    loaded = repo.get(recovery_episode_id=seed.recovery_episode_id)
    assert loaded == terminal

    reentry = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=2,
        expected_revision=terminal.revision,
        claimed_at=_NOW,
    )
    assert reentry.status is WorkerRecoveryEpisodeClaimStatus.TERMINAL

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local reference repository for worker recovery episodes (AW-6B)."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime

from intergrax.autonomous_work.recovery_episode_claim import (
    resolve_recovery_attempt_claim,
    resolve_recovery_episode_create,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkRevisionConflict,
    AutonomousWorkRepositoryCapabilities,
    WorkerRecoveryEpisodeClaim,
    WorkerRecoveryEpisodeClaimStatus,
    WorkerRecoveryEpisodeCreateResult,
    WorkerRecoveryEpisodeCreateStatus,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    RecoveryEpisodeStatus,
    WorkerRecoveryEpisode,
    is_terminal_recovery_episode_status,
)
from intergrax.contracts.autonomous_work.references import ExternalDependencyReference
from intergrax.contracts.autonomous_work.revision import Revision, initial_revision
from intergrax.contracts.execution_identity import ExecutionId


class InMemoryWorkerRecoveryEpisodeRepository:
    """Thread-safe process-local reference repository for recovery episodes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, WorkerRecoveryEpisode] = {}

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.worker_recovery_episode.in_memory",
            durable=False,
            reference_only=True,
        )

    def get(self, *, recovery_episode_id: str) -> WorkerRecoveryEpisode | None:
        with self._lock:
            return self._records.get(recovery_episode_id)

    def create_or_get(
        self,
        episode: WorkerRecoveryEpisode,
    ) -> WorkerRecoveryEpisodeCreateResult:
        with self._lock:
            stored = self._records.get(episode.recovery_episode_id)
            result = resolve_recovery_episode_create(episode, stored)
            if result.status is WorkerRecoveryEpisodeCreateStatus.CREATED:
                self._records[episode.recovery_episode_id] = episode
            return result

    def claim_attempt(
        self,
        *,
        recovery_episode_id: str,
        attempt_number: int,
        expected_revision: Revision,
        claimed_at: datetime,
    ) -> WorkerRecoveryEpisodeClaim:
        with self._lock:
            stored = self._records.get(recovery_episode_id)
            if stored is None:
                raise KeyError(f"recovery episode not found: {recovery_episode_id}")
            if is_terminal_recovery_episode_status(stored.status):
                return WorkerRecoveryEpisodeClaim(
                    status=WorkerRecoveryEpisodeClaimStatus.TERMINAL,
                    episode=stored,
                )
            if stored.revision != expected_revision:
                return WorkerRecoveryEpisodeClaim(
                    status=WorkerRecoveryEpisodeClaimStatus.REVISION_CONFLICT,
                    episode=stored,
                )
            if attempt_number != stored.attempt_count + 1:
                return WorkerRecoveryEpisodeClaim(
                    status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
                    episode=stored,
                )
            if (
                stored.claimed_attempt_number is not None
                and stored.last_execution_id is None
            ):
                return WorkerRecoveryEpisodeClaim(
                    status=WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED,
                    episode=stored,
                )
            claimed = replace(
                stored,
                status=RecoveryEpisodeStatus.IN_PROGRESS,
                attempt_count=attempt_number,
                claimed_attempt_number=attempt_number,
                last_attempt_at=claimed_at,
                revision=Revision(stored.revision.value + 1),
            )
            claim = resolve_recovery_attempt_claim(
                stored=stored,
                attempt_number=attempt_number,
                claimed_episode=claimed,
            )
            if claim.status is WorkerRecoveryEpisodeClaimStatus.CLAIMED:
                self._records[recovery_episode_id] = claimed
            return claim

    def record_execution(
        self,
        *,
        recovery_episode_id: str,
        attempt_number: int,
        expected_revision: Revision,
        execution_id: ExecutionId,
        recorded_at: datetime,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            self._require_revision(stored, expected_revision)
            if stored.claimed_attempt_number != attempt_number:
                raise ValueError("execution binding requires matching claimed attempt")
            updated = replace(
                stored,
                last_execution_id=execution_id,
                last_attempt_at=recorded_at,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def record_continuity_resume(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        continuity_resume_revision: Revision,
        recorded_at: datetime,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            if stored.continuity_resume_completed:
                if stored.continuity_resume_revision != continuity_resume_revision:
                    raise ValueError("continuity resume revision mismatch on idempotent replay")
                return stored
            self._require_revision(stored, expected_revision)
            updated = replace(
                stored,
                continuity_resume_completed=True,
                continuity_resume_revision=continuity_resume_revision,
                last_attempt_at=recorded_at,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def mark_waiting(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        next_retry_at: datetime | None,
        dependency_ref: ExternalDependencyReference | None,
        updated_at: datetime,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            self._require_revision(stored, expected_revision)
            updated = replace(
                stored,
                status=RecoveryEpisodeStatus.WAITING,
                next_retry_at=next_retry_at,
                dependency_ref=dependency_ref,
                claimed_attempt_number=None,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def mark_waiting_for_human(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        human_decision_ref: str,
        updated_at: datetime,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            self._require_revision(stored, expected_revision)
            updated = replace(
                stored,
                status=RecoveryEpisodeStatus.WAITING_FOR_HUMAN,
                human_decision_ref=human_decision_ref,
                claimed_attempt_number=None,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def mark_succeeded(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        terminal_reason: str | None = None,
    ) -> WorkerRecoveryEpisode:
        return self._mark_terminal(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            completed_at=completed_at,
            status=RecoveryEpisodeStatus.SUCCEEDED,
            terminal_reason=terminal_reason,
        )

    def mark_failed(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        terminal_reason: str,
        last_failure_ref: str | None = None,
    ) -> WorkerRecoveryEpisode:
        return self._mark_terminal(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            completed_at=completed_at,
            status=RecoveryEpisodeStatus.FAILED,
            terminal_reason=terminal_reason,
            last_failure_ref=last_failure_ref,
        )

    def mark_escalated(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        terminal_reason: str,
    ) -> WorkerRecoveryEpisode:
        return self._mark_terminal(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            completed_at=completed_at,
            status=RecoveryEpisodeStatus.ESCALATED,
            terminal_reason=terminal_reason,
        )

    def mark_quarantined(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        terminal_reason: str,
    ) -> WorkerRecoveryEpisode:
        return self._mark_terminal(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            completed_at=completed_at,
            status=RecoveryEpisodeStatus.QUARANTINED,
            terminal_reason=terminal_reason,
        )

    def mark_stopped(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        terminal_reason: str,
    ) -> WorkerRecoveryEpisode:
        return self._mark_terminal(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            completed_at=completed_at,
            status=RecoveryEpisodeStatus.STOPPED,
            terminal_reason=terminal_reason,
        )

    def record_attempt_outcome(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        attempt_number: int,
        finished_at: datetime,
        last_failure_ref: str | None,
        next_retry_at: datetime | None,
        status: RecoveryEpisodeStatus,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            self._require_revision(stored, expected_revision)
            updated = replace(
                stored,
                status=status,
                last_failure_ref=last_failure_ref,
                next_retry_at=next_retry_at,
                claimed_attempt_number=None,
                last_attempt_at=finished_at,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def _mark_terminal(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        completed_at: datetime,
        status: RecoveryEpisodeStatus,
        terminal_reason: str | None,
        last_failure_ref: str | None = None,
    ) -> WorkerRecoveryEpisode:
        with self._lock:
            stored = self._require_episode(recovery_episode_id)
            self._require_revision(stored, expected_revision)
            updated = replace(
                stored,
                status=status,
                completed_at=completed_at,
                terminal_reason=terminal_reason,
                last_failure_ref=last_failure_ref,
                claimed_attempt_number=None,
                next_retry_at=None,
                revision=Revision(stored.revision.value + 1),
            )
            self._records[recovery_episode_id] = updated
            return updated

    def _require_episode(self, recovery_episode_id: str) -> WorkerRecoveryEpisode:
        stored = self._records.get(recovery_episode_id)
        if stored is None:
            raise KeyError(f"recovery episode not found: {recovery_episode_id}")
        return stored

    def _require_revision(
        self,
        stored: WorkerRecoveryEpisode,
        expected_revision: Revision,
    ) -> None:
        if stored.revision != expected_revision:
            raise AutonomousWorkRevisionConflict(
                (
                    f"WorkerRecoveryEpisode revision conflict for "
                    f"{stored.recovery_episode_id}: expected "
                    f"{expected_revision.value}, actual {stored.revision.value}"
                ),
                entity_kind="WorkerRecoveryEpisode",
                entity_id=stored.recovery_episode_id,
                expected_revision=expected_revision,
                actual_revision=stored.revision,
            )

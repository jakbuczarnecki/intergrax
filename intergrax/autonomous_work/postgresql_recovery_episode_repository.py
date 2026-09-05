# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL repository for durable worker recovery episodes (AW-6B)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from intergrax.autonomous_work.postgresql_repository import PostgreSQLAutonomousWorkStore
from intergrax.autonomous_work.recovery_episode_claim import (
    resolve_recovery_attempt_claim,
    resolve_recovery_episode_create,
)
from intergrax.autonomous_work.recovery_episode_serialization import (
    worker_recovery_episode_from_json,
    worker_recovery_episode_to_json,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
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
from intergrax.contracts.autonomous_work.revision import Revision
from intergrax.contracts.execution_identity import ExecutionId

_CAPABILITIES = AutonomousWorkRepositoryCapabilities(
    backend_id="autonomous_work.worker_recovery_episode.postgresql",
    durable=True,
    reference_only=False,
)


class PostgreSQLWorkerRecoveryEpisodeRepository:
    """Production repository for durable worker recovery episodes."""

    _TABLE = "aw_worker_recovery_episodes"

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def get(self, *, recovery_episode_id: str) -> WorkerRecoveryEpisode | None:
        with self._store.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT record_json FROM {self._TABLE}
                WHERE recovery_episode_id = %s
                """,
                (recovery_episode_id.strip(),),
            ).fetchone()
            if row is None:
                return None
            return worker_recovery_episode_from_json(row["record_json"])

    def create_or_get(
        self,
        episode: WorkerRecoveryEpisode,
    ) -> WorkerRecoveryEpisodeCreateResult:
        record_json = worker_recovery_episode_to_json(episode)
        with self._store.transaction() as conn:
            inserted = conn.execute(
                f"""
                INSERT INTO {self._TABLE} (
                    recovery_episode_id, worker_instance_id, record_json, revision
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (recovery_episode_id) DO NOTHING
                RETURNING record_json
                """,
                (
                    episode.recovery_episode_id.strip(),
                    episode.worker_instance_id.strip(),
                    record_json,
                    episode.revision.value,
                ),
            ).fetchone()
            if inserted is not None:
                return WorkerRecoveryEpisodeCreateResult(
                    status=WorkerRecoveryEpisodeCreateStatus.CREATED,
                    episode=worker_recovery_episode_from_json(inserted["record_json"]),
                )
            row = conn.execute(
                f"""
                SELECT record_json FROM {self._TABLE}
                WHERE recovery_episode_id = %s
                """,
                (episode.recovery_episode_id.strip(),),
            ).fetchone()
            if row is None:
                raise RuntimeError("recovery episode conflict without stored canonical episode")
            stored = worker_recovery_episode_from_json(row["record_json"])
            return resolve_recovery_episode_create(episode, stored)

    def claim_attempt(
        self,
        *,
        recovery_episode_id: str,
        attempt_number: int,
        expected_revision: Revision,
        claimed_at: datetime,
    ) -> WorkerRecoveryEpisodeClaim:
        with self._store.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT record_json, revision FROM {self._TABLE}
                WHERE recovery_episode_id = %s
                FOR UPDATE
                """,
                (recovery_episode_id.strip(),),
            ).fetchone()
            if row is None:
                raise KeyError(f"recovery episode not found: {recovery_episode_id}")
            stored = worker_recovery_episode_from_json(row["record_json"])
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
                conn.execute(
                    f"""
                    UPDATE {self._TABLE}
                    SET record_json = %s, revision = %s
                    WHERE recovery_episode_id = %s AND revision = %s
                    """,
                    (
                        worker_recovery_episode_to_json(claimed),
                        claimed.revision.value,
                        recovery_episode_id.strip(),
                        expected_revision.value,
                    ),
                )
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
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: replace(
                stored,
                last_execution_id=execution_id,
                last_attempt_at=recorded_at,
                revision=Revision(stored.revision.value + 1),
            ),
            validate=lambda stored: stored.claimed_attempt_number == attempt_number,
        )

    def record_continuity_resume(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        continuity_resume_revision: Revision,
        recorded_at: datetime,
    ) -> WorkerRecoveryEpisode:
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: (
                stored
                if stored.continuity_resume_completed
                else replace(
                    stored,
                    continuity_resume_completed=True,
                    continuity_resume_revision=continuity_resume_revision,
                    last_attempt_at=recorded_at,
                    revision=Revision(stored.revision.value + 1),
                )
            ),
            validate=lambda stored: (
                not stored.continuity_resume_completed
                or stored.continuity_resume_revision == continuity_resume_revision
            ),
        )

    def mark_waiting(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        next_retry_at: datetime | None,
        dependency_ref: ExternalDependencyReference | None,
        updated_at: datetime,
    ) -> WorkerRecoveryEpisode:
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: replace(
                stored,
                status=RecoveryEpisodeStatus.WAITING,
                next_retry_at=next_retry_at,
                dependency_ref=dependency_ref,
                claimed_attempt_number=None,
                revision=Revision(stored.revision.value + 1),
            ),
        )

    def mark_waiting_for_human(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        human_decision_ref: str,
        updated_at: datetime,
    ) -> WorkerRecoveryEpisode:
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: replace(
                stored,
                status=RecoveryEpisodeStatus.WAITING_FOR_HUMAN,
                human_decision_ref=human_decision_ref,
                claimed_attempt_number=None,
                revision=Revision(stored.revision.value + 1),
            ),
        )

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
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: replace(
                stored,
                status=status,
                last_failure_ref=last_failure_ref,
                next_retry_at=next_retry_at,
                claimed_attempt_number=None,
                last_attempt_at=finished_at,
                revision=Revision(stored.revision.value + 1),
            ),
        )

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
        return self._mutate(
            recovery_episode_id=recovery_episode_id,
            expected_revision=expected_revision,
            mutator=lambda stored: replace(
                stored,
                status=status,
                completed_at=completed_at,
                terminal_reason=terminal_reason,
                last_failure_ref=last_failure_ref,
                claimed_attempt_number=None,
                next_retry_at=None,
                revision=Revision(stored.revision.value + 1),
            ),
        )

    def _mutate(
        self,
        *,
        recovery_episode_id: str,
        expected_revision: Revision,
        mutator,
        validate=None,
    ) -> WorkerRecoveryEpisode:
        with self._store.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT record_json, revision FROM {self._TABLE}
                WHERE recovery_episode_id = %s
                FOR UPDATE
                """,
                (recovery_episode_id.strip(),),
            ).fetchone()
            if row is None:
                raise KeyError(f"recovery episode not found: {recovery_episode_id}")
            stored = worker_recovery_episode_from_json(row["record_json"])
            if stored.revision != expected_revision:
                raise AutonomousWorkRevisionConflict(
                    (
                        f"WorkerRecoveryEpisode revision conflict for "
                        f"{recovery_episode_id}"
                    ),
                    entity_kind="WorkerRecoveryEpisode",
                    entity_id=recovery_episode_id,
                    expected_revision=expected_revision,
                    actual_revision=stored.revision,
                )
            if validate is not None and not validate(stored):
                raise ValueError("recovery episode mutation validation failed")
            updated = mutator(stored)
            conn.execute(
                f"""
                UPDATE {self._TABLE}
                SET record_json = %s, revision = %s
                WHERE recovery_episode_id = %s AND revision = %s
                """,
                (
                    worker_recovery_episode_to_json(updated),
                    updated.revision.value,
                    recovery_episode_id.strip(),
                    expected_revision.value,
                ),
            )
            return updated

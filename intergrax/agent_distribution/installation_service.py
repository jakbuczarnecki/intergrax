# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Installation lifecycle domain service (AGENT_DISTRIBUTION §11, AP-4)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    InstallationLifecycleError,
    InstallationSlotConflict,
)
from intergrax.agent_distribution.events import TransitionResult, distribution_event
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import AgentInstallationRecord, InstallationState
from intergrax.agent_distribution.package_trust import assert_installation_trust_record_acceptable
from intergrax.agent_distribution.stores import AgentInstallationStore
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord


_ALLOWED_TRANSITIONS: dict[InstallationState, frozenset[InstallationState]] = {
    InstallationState.CANDIDATE: frozenset(
        {InstallationState.VERIFIED, InstallationState.FAILED_CANDIDATE}
    ),
    InstallationState.VERIFIED: frozenset({InstallationState.INSTALLED_ACTIVE}),
    InstallationState.INSTALLED_ACTIVE: frozenset(
        {InstallationState.INSTALLED_PREVIOUS, InstallationState.REVOKED}
    ),
    InstallationState.INSTALLED_PREVIOUS: frozenset({InstallationState.INSTALLED_ACTIVE, InstallationState.REVOKED}),
    InstallationState.REVOKED: frozenset({InstallationState.REMOVED_TOMBSTONE}),
}


class InstallationService:
    """Transactional installation lifecycle operations — fail closed."""

    def __init__(self, store: AgentInstallationStore) -> None:
        self._store = store

    def create_candidate_installation(
        self,
        *,
        installation_id: str,
        installation_slot_id: str,
        environment_id: str,
        package_identity: AgentPackageIdentity,
    ) -> TransitionResult[AgentInstallationRecord]:
        record = AgentInstallationRecord(
            installation_id=installation_id,
            installation_slot_id=installation_slot_id,
            environment_id=environment_id,
            package_identity=package_identity,
            installation_state=InstallationState.CANDIDATE,
            created_at=datetime.now(UTC),
        )
        persisted = self._store.persist_installation(record)
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "installation.created",
                    installation_id,
                    installation_slot_id=installation_slot_id,
                ),
            ),
        )

    def mark_verified(
        self,
        installation_id: str,
        *,
        artifact_store_ref: str,
        trust_record: AgentInstallationTrustRecord,
    ) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.VERIFIED)
        assert_installation_trust_record_acceptable(
            trust_record,
            package_identity=record.package_identity,
        )
        updated = record.model_copy(
            update={
                "installation_state": InstallationState.VERIFIED,
                "artifact_store_ref": artifact_store_ref,
                "trust_record": trust_record,
            }
        )
        persisted = self._store.persist_installation(updated)
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "installation.verified",
                    installation_id,
                    installation_slot_id=record.installation_slot_id,
                ),
            ),
        )

    def mark_failed_candidate(self, installation_id: str) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.FAILED_CANDIDATE)
        updated = record.model_copy(
            update={"installation_state": InstallationState.FAILED_CANDIDATE}
        )
        persisted = self._store.persist_installation(updated)
        return TransitionResult(value=persisted)

    def promote_verified_to_active(
        self,
        installation_id: str,
        *,
        expected_active_installation_id: str | None = None,
    ) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.INSTALLED_ACTIVE)
        if record.artifact_store_ref is None:
            raise InstallationLifecycleError("verified installation requires artifact_store_ref")

        prior_active = self._store.get_active_installation_for_slot(record.installation_slot_id)
        if prior_active is not None and prior_active.installation_id == record.installation_id:
            raise InstallationLifecycleError("installation is already active for slot")

        now = datetime.now(UTC)
        demoted_prior: AgentInstallationRecord | None = None
        if prior_active is not None:
            demoted_prior = prior_active.model_copy(
                update={
                    "installation_state": InstallationState.INSTALLED_PREVIOUS,
                    "active_for_slot": False,
                    "superseded_at": now,
                }
            )

        promoted = record.model_copy(
            update={
                "installation_state": InstallationState.INSTALLED_ACTIVE,
                "active_for_slot": True,
                "previous_installation_ref": prior_active.installation_id if prior_active else None,
            }
        )

        persisted, _ = self._store.atomic_promote_active_installation(
            demoted_prior=demoted_prior,
            promoted=promoted,
            expected_active_installation_id=expected_active_installation_id,
        )

        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "installation.activated",
                    installation_id,
                    installation_slot_id=record.installation_slot_id,
                    prior_active_installation_id=(
                        prior_active.installation_id if prior_active is not None else ""
                    ),
                ),
            ),
        )

    def revoke_installation(self, installation_id: str) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.REVOKED)
        if record.active_for_slot:
            raise InstallationLifecycleError("cannot revoke active installation without demotion")
        updated = record.model_copy(update={"installation_state": InstallationState.REVOKED})
        persisted = self._store.persist_installation(updated)
        return TransitionResult(
            value=persisted,
            events=(distribution_event("installation.revoked", installation_id),),
        )

    def tombstone_installation(self, installation_id: str) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.REMOVED_TOMBSTONE)
        if record.active_for_slot:
            raise InstallationLifecycleError("cannot tombstone active installation")
        updated = record.model_copy(
            update={
                "installation_state": InstallationState.REMOVED_TOMBSTONE,
                "tombstoned_at": datetime.now(UTC),
            }
        )
        persisted = self._store.persist_installation(updated)
        return TransitionResult(value=persisted)

    def resolve_active_for_slot(
        self,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        return self._store.get_active_installation_for_slot(installation_slot_id)

    def rollback_slot_to_previous(
        self,
        installation_slot_id: str,
        *,
        expected_active_installation_id: str,
    ) -> TransitionResult[AgentInstallationRecord]:
        active = self._store.get_active_installation_for_slot(installation_slot_id)
        if active is None:
            raise AgentDistributionNotFoundError("no active installation for slot")
        if active.installation_id != expected_active_installation_id:
            raise InstallationSlotConflict("active installation does not match expected value")
        if active.previous_installation_ref is None:
            raise InstallationLifecycleError("active installation has no rollback target")

        previous = self._require_installation(active.previous_installation_ref)
        if previous.installation_state not in {
            InstallationState.INSTALLED_PREVIOUS,
            InstallationState.INSTALLED_ACTIVE,
        }:
            raise InstallationLifecycleError("rollback target is not a restorable installation")

        now = datetime.now(UTC)
        demoted_active = active.model_copy(
            update={
                "installation_state": InstallationState.INSTALLED_PREVIOUS,
                "active_for_slot": False,
                "superseded_at": now,
            }
        )
        restored = previous.model_copy(
            update={
                "installation_state": InstallationState.INSTALLED_ACTIVE,
                "active_for_slot": True,
                "previous_installation_ref": active.installation_id,
            }
        )

        persisted, _ = self._store.atomic_promote_active_installation(
            demoted_prior=demoted_active,
            promoted=restored,
            expected_active_installation_id=expected_active_installation_id,
        )

        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    "installation.activated",
                    restored.installation_id,
                    installation_slot_id=installation_slot_id,
                    rollback_from=active.installation_id,
                ),
            ),
        )

    def _require_installation(self, installation_id: str) -> AgentInstallationRecord:
        record = self._store.get_installation(installation_id)
        if record is None:
            raise AgentDistributionNotFoundError(f"installation {installation_id} was not found")
        return record

    @staticmethod
    def _require_transition(current: InstallationState, target: InstallationState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise InstallationLifecycleError(
                f"illegal installation transition {current.value} -> {target.value}"
            )

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
from intergrax.agent_distribution.events import (
    EVENT_INSTALLATION_ACTIVATED,
    EVENT_INSTALLATION_CREATED,
    EVENT_INSTALLATION_FAILED_CANDIDATE,
    EVENT_INSTALLATION_RETIRED,
    EVENT_INSTALLATION_REVOKED,
    EVENT_INSTALLATION_TOMBSTONED,
    EVENT_INSTALLATION_VERIFIED,
    TransitionResult,
    distribution_event,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import AgentInstallationRecord, InstallationState
from intergrax.agent_distribution.package_trust import assert_installation_trust_record_acceptable
from intergrax.agent_distribution.stores import AgentInstallationStore
from intergrax.agent_distribution.trust import AgentInstallationTrustRecord

_ALLOWED_TRANSITIONS: dict[InstallationState, frozenset[InstallationState]] = {
    InstallationState.CANDIDATE: frozenset({InstallationState.VERIFIED, InstallationState.FAILED_CANDIDATE}),
    InstallationState.VERIFIED: frozenset({InstallationState.INSTALLED_ACTIVE, InstallationState.FAILED_CANDIDATE}),
    InstallationState.INSTALLED_ACTIVE: frozenset(
        {
            InstallationState.INSTALLED_PREVIOUS,
            InstallationState.REVOKED,
            InstallationState.RETIRED,
        }
    ),
    InstallationState.INSTALLED_PREVIOUS: frozenset({InstallationState.INSTALLED_ACTIVE, InstallationState.REVOKED}),
    InstallationState.REVOKED: frozenset({InstallationState.REMOVED_TOMBSTONE}),
}

# Evidence contract (PDC-D1-R017): the canonical event that records entering each
# installation state. A state without an entry here has no lifecycle evidence.
# "installed_previous" is recorded by the successor's "installation.activated"
# event (prior_active_installation_id attribute) because demotion and promotion
# are one atomic store transition.
INSTALLATION_STATE_EVIDENCE_EVENT: dict[InstallationState, str] = {
    InstallationState.CANDIDATE: EVENT_INSTALLATION_CREATED,
    InstallationState.VERIFIED: EVENT_INSTALLATION_VERIFIED,
    InstallationState.INSTALLED_ACTIVE: EVENT_INSTALLATION_ACTIVATED,
    InstallationState.INSTALLED_PREVIOUS: EVENT_INSTALLATION_ACTIVATED,
    InstallationState.FAILED_CANDIDATE: EVENT_INSTALLATION_FAILED_CANDIDATE,
    InstallationState.REVOKED: EVENT_INSTALLATION_REVOKED,
    InstallationState.REMOVED_TOMBSTONE: EVENT_INSTALLATION_TOMBSTONED,
    InstallationState.RETIRED: EVENT_INSTALLATION_RETIRED,
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
        self._require_slot_not_retired(environment_id, installation_slot_id)
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
                    EVENT_INSTALLATION_CREATED,
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
                    EVENT_INSTALLATION_VERIFIED,
                    installation_id,
                    installation_slot_id=record.installation_slot_id,
                ),
            ),
        )

    def mark_failed_candidate(
        self,
        installation_id: str,
        *,
        reason_code: str | None = None,
    ) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.FAILED_CANDIDATE)
        updated = record.model_copy(update={"installation_state": InstallationState.FAILED_CANDIDATE})
        persisted = self._store.persist_installation(updated)
        attributes = {"installation_slot_id": record.installation_slot_id}
        if reason_code is not None:
            attributes["reason_code"] = reason_code
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    EVENT_INSTALLATION_FAILED_CANDIDATE,
                    installation_id,
                    **attributes,
                ),
            ),
        )

    def promote_verified_to_active(
        self,
        installation_id: str,
        *,
        expected_active_installation_id: str | None = None,
    ) -> TransitionResult[AgentInstallationRecord]:
        record = self._require_installation(installation_id)
        self._require_transition(record.installation_state, InstallationState.INSTALLED_ACTIVE)
        self._require_slot_not_retired(record.environment_id, record.installation_slot_id)
        if record.artifact_store_ref is None:
            raise InstallationLifecycleError("verified installation requires artifact_store_ref")

        prior_active = self._store.get_active_installation_for_slot(
            record.environment_id,
            record.installation_slot_id,
        )
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
                    EVENT_INSTALLATION_ACTIVATED,
                    installation_id,
                    installation_slot_id=record.installation_slot_id,
                    prior_active_installation_id=(prior_active.installation_id if prior_active is not None else ""),
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
            events=(distribution_event(EVENT_INSTALLATION_REVOKED, installation_id),),
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
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    EVENT_INSTALLATION_TOMBSTONED,
                    installation_id,
                    installation_slot_id=record.installation_slot_id,
                ),
            ),
        )

    def retire_installation(
        self,
        installation_id: str,
        *,
        expected_active_installation_id: str,
        reason: str | None = None,
    ) -> TransitionResult[AgentInstallationRecord]:
        """Retire the active installation of a slot (ADR-PDC-01, PDC-D1-R014).

        The record and its trust evidence stay in the store (no tombstone). The slot
        loses its active installation, so it is excluded from every new effective
        roster, and create_candidate_installation / promote_verified_to_active
        refuse the slot afterwards. Runtime revisions that already pin the package
        digest are not touched here: executions finish under their pinned revision.
        """
        record = self._require_installation(installation_id)
        if installation_id != expected_active_installation_id:
            raise InstallationSlotConflict("retire target does not match expected active installation")
        self._require_transition(record.installation_state, InstallationState.RETIRED)
        active = self._store.get_active_installation_for_slot(
            record.environment_id,
            record.installation_slot_id,
        )
        if active is None or active.installation_id != installation_id:
            raise InstallationSlotConflict("installation is not the active installation for its slot")
        retired = record.model_copy(
            update={
                "installation_state": InstallationState.RETIRED,
                "active_for_slot": False,
                "retired_at": datetime.now(UTC),
            }
        )
        persisted = self._store.persist_installation(
            retired,
            expected_active_installation_id=installation_id,
        )
        attributes = {"installation_slot_id": record.installation_slot_id}
        if reason is not None:
            attributes["reason"] = reason
        return TransitionResult(
            value=persisted,
            events=(
                distribution_event(
                    EVENT_INSTALLATION_RETIRED,
                    installation_id,
                    **attributes,
                ),
            ),
        )

    def find_completed_retirement(
        self,
        environment_id: str,
        installation_slot_id: str,
        installation_id: str,
    ) -> AgentInstallationRecord | None:
        """Return the slot installation if it already completed this retirement (idempotent replay)."""
        for item in self._store.list_installations_for_slot(environment_id, installation_slot_id):
            if item.installation_state is InstallationState.RETIRED and item.installation_id == installation_id:
                return item
        return None

    def is_slot_retired(self, environment_id: str, installation_slot_id: str) -> bool:
        return any(
            item.installation_state is InstallationState.RETIRED
            for item in self._store.list_installations_for_slot(
                environment_id,
                installation_slot_id,
            )
        )

    def resolve_active_for_slot(
        self,
        environment_id: str,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        return self._store.get_active_installation_for_slot(
            environment_id,
            installation_slot_id,
        )

    def rollback_slot_to_previous(
        self,
        environment_id: str,
        installation_slot_id: str,
        *,
        expected_active_installation_id: str,
    ) -> TransitionResult[AgentInstallationRecord]:
        active = self._store.get_active_installation_for_slot(
            environment_id,
            installation_slot_id,
        )
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
                    EVENT_INSTALLATION_ACTIVATED,
                    restored.installation_id,
                    installation_slot_id=installation_slot_id,
                    rollback_from=active.installation_id,
                ),
            ),
        )

    def _require_slot_not_retired(self, environment_id: str, installation_slot_id: str) -> None:
        if self.is_slot_retired(environment_id, installation_slot_id):
            raise InstallationLifecycleError(
                "installation slot is retired; new installations and activations are refused"
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
            raise InstallationLifecycleError(f"illegal installation transition {current.value} -> {target.value}")

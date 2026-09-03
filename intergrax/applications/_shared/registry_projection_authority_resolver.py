# © Artur Czarnecki. All rights reserved.

"""Canonical registry projection authority resolution (ADR-AGENT-007 Phase 4D).

Resolves immutable historical lifecycle authority for one ``runtime_revision_id``
from store ports and ``EffectiveRosterAuthorityService``. Caller-supplied roster,
lock, or materialization overrides are forbidden at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.errors import (
    EffectiveRosterAuthorityConflict,
    EffectiveRosterAuthorityError,
    EffectiveRosterAuthorityNotFound,
    RuntimeMaterializationConflict,
)
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
    validate_runtime_materialization_record_matches_revision,
)
from intergrax.agent_distribution.runtime_revision import (
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import (
    MaterializedRuntimeLockStore,
    RuntimeMaterializationStore,
    RuntimeRevisionStore,
)

_PROJECTION_ELIGIBLE_REVISION_STATES = frozenset(
    {
        RuntimeRevisionState.VALIDATED,
        RuntimeRevisionState.ACTIVE,
        RuntimeRevisionState.SUPERSEDED,
    }
)


class RegistryProjectionAuthorityError(Exception):
    """Canonical registry projection authority resolution failed."""


class RegistryProjectionAuthorityNotFound(RegistryProjectionAuthorityError):
    """Required canonical authority record is missing."""


class RegistryProjectionAuthorityConflict(RegistryProjectionAuthorityError):
    """Canonical authority records disagree across lifecycle stores."""


@dataclass(frozen=True, slots=True)
class ResolvedRegistryProjectionAuthority:
    """Frozen canonical lifecycle authority for one runtime revision projection."""

    runtime_revision: RuntimeRevision
    effective_roster: EffectiveRoster
    materialized_runtime_lock: MaterializedRuntimeLock
    runtime_materialization: RuntimeMaterializationRecord


class RegistryProjectionAuthorityResolver:
    """Resolve and cross-validate canonical projection authority for one revision."""

    def __init__(
        self,
        *,
        revision_store: RuntimeRevisionStore,
        effective_roster_authority: EffectiveRosterAuthorityService,
        lock_store: MaterializedRuntimeLockStore,
        materialization_store: RuntimeMaterializationStore,
    ) -> None:
        self._revision_store = revision_store
        self._effective_roster_authority = effective_roster_authority
        self._lock_store = lock_store
        self._materialization_store = materialization_store

    def require_for_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> ResolvedRegistryProjectionAuthority:
        normalized_revision_id = runtime_revision_id.strip()
        if not normalized_revision_id:
            raise RegistryProjectionAuthorityNotFound(
                "runtime_revision_id must be non-empty"
            )

        revision = self._revision_store.get_revision(normalized_revision_id)
        if revision is None:
            raise RegistryProjectionAuthorityNotFound(
                f"runtime revision not found: {normalized_revision_id!r}"
            )
        if revision.runtime_revision_id != normalized_revision_id:
            raise RegistryProjectionAuthorityConflict(
                "runtime revision id mismatch with canonical store record"
            )
        if revision.application_id != application_id:
            raise RegistryProjectionAuthorityConflict(
                "runtime revision application_id mismatch with projection scope"
            )
        if revision.application_environment_id != application_environment_id:
            raise RegistryProjectionAuthorityConflict(
                "runtime revision application_environment_id mismatch with projection scope"
            )
        if revision.revision_state not in _PROJECTION_ELIGIBLE_REVISION_STATES:
            raise RegistryProjectionAuthorityConflict(
                f"runtime revision state {revision.revision_state.value!r} is not projection-eligible"
            )

        try:
            effective_roster = self._effective_roster_authority.require_for_revision(
                revision
            )
        except EffectiveRosterAuthorityNotFound as exc:
            raise RegistryProjectionAuthorityNotFound(str(exc)) from exc
        except EffectiveRosterAuthorityConflict as exc:
            raise RegistryProjectionAuthorityConflict(str(exc)) from exc
        except EffectiveRosterAuthorityError as exc:
            raise RegistryProjectionAuthorityError(str(exc)) from exc

        materialization = self._materialization_store.get_by_revision(
            normalized_revision_id
        )
        if materialization is None:
            raise RegistryProjectionAuthorityNotFound(
                f"missing canonical materialization record for {normalized_revision_id!r}"
            )
        try:
            validate_runtime_materialization_record_matches_revision(
                revision, materialization
            )
        except RuntimeMaterializationConflict as exc:
            raise RegistryProjectionAuthorityConflict(str(exc)) from exc

        artifact_digest = revision.materialization_artifact_digest
        if artifact_digest is None:
            raise RegistryProjectionAuthorityConflict(
                "runtime revision requires materialization_artifact_digest"
            )
        if artifact_digest != materialization.materialization_artifact_digest:
            raise RegistryProjectionAuthorityConflict(
                "materialization_artifact_digest mismatch with canonical materialization record"
            )

        lock_id = revision.materialized_runtime_lock_id
        if lock_id is None:
            raise RegistryProjectionAuthorityConflict(
                "runtime revision requires materialized_runtime_lock_id"
            )
        lock = self._lock_store.get_lock(lock_id)
        if lock is None:
            raise RegistryProjectionAuthorityNotFound(
                f"canonical materialized runtime lock not found: {lock_id!r}"
            )
        lock = _validate_canonical_lock_authority(
            runtime_revision=revision,
            materialization=materialization,
            materialized_runtime_lock=lock,
        )

        _validate_cross_authority_invariants(
            runtime_revision=revision,
            effective_roster=effective_roster,
            materialized_runtime_lock=lock,
            runtime_materialization=materialization,
        )

        return ResolvedRegistryProjectionAuthority(
            runtime_revision=revision,
            effective_roster=effective_roster,
            materialized_runtime_lock=lock,
            runtime_materialization=materialization,
        )


def _validated_lock_identity(lock: MaterializedRuntimeLock) -> MaterializedRuntimeLock:
    if lock.lock_id is None or lock.lock_digest is None:
        return lock.with_content_identity()
    return lock


def _validate_canonical_lock_authority(
    *,
    runtime_revision: RuntimeRevision,
    materialization: RuntimeMaterializationRecord,
    materialized_runtime_lock: MaterializedRuntimeLock,
) -> MaterializedRuntimeLock:
    lock = _validated_lock_identity(materialized_runtime_lock)
    lock_id = runtime_revision.materialized_runtime_lock_id
    lock_digest = runtime_revision.materialized_runtime_lock_digest
    if lock_id is None or lock_digest is None:
        raise RegistryProjectionAuthorityConflict(
            "runtime revision requires materialized runtime lock identity"
        )
    if lock.lock_id != lock_id:
        raise RegistryProjectionAuthorityConflict(
            "canonical materialized runtime lock id mismatch with runtime revision"
        )
    if lock.lock_digest != lock_digest:
        raise RegistryProjectionAuthorityConflict(
            "canonical materialized runtime lock digest mismatch with runtime revision"
        )
    if materialization.materialized_runtime_lock_id != lock_id:
        raise RegistryProjectionAuthorityConflict(
            "canonical materialized runtime lock id mismatch with materialization record"
        )
    if materialization.materialized_runtime_lock_digest != lock_digest:
        raise RegistryProjectionAuthorityConflict(
            "canonical materialized runtime lock digest mismatch with materialization record"
        )
    return lock


def _validate_cross_authority_invariants(
    *,
    runtime_revision: RuntimeRevision,
    effective_roster: EffectiveRoster,
    materialized_runtime_lock: MaterializedRuntimeLock,
    runtime_materialization: RuntimeMaterializationRecord,
) -> None:
    roster_revision_id = runtime_revision.effective_roster_revision_id
    if roster_revision_id is None:
        raise RegistryProjectionAuthorityConflict(
            "runtime revision requires effective_roster_revision_id"
        )
    if effective_roster.effective_roster_revision_id != roster_revision_id:
        raise RegistryProjectionAuthorityConflict(
            "runtime revision effective_roster_revision_id mismatch with effective roster"
        )

    lock_id = runtime_revision.materialized_runtime_lock_id
    lock_digest = runtime_revision.materialized_runtime_lock_digest
    if lock_id is None or lock_digest is None:
        raise RegistryProjectionAuthorityConflict(
            "runtime revision requires materialized runtime lock identity"
        )
    if materialized_runtime_lock.lock_id != lock_id:
        raise RegistryProjectionAuthorityConflict(
            "materialized runtime lock id mismatch with runtime revision"
        )
    if materialized_runtime_lock.lock_digest != lock_digest:
        raise RegistryProjectionAuthorityConflict(
            "materialized runtime lock digest mismatch with runtime revision"
        )

    if (
        runtime_materialization.runtime_revision_id
        != runtime_revision.runtime_revision_id
    ):
        raise RegistryProjectionAuthorityConflict(
            "runtime materialization runtime_revision_id mismatch with runtime revision"
        )
    if runtime_materialization.materialized_runtime_lock_id != lock_id:
        raise RegistryProjectionAuthorityConflict(
            "runtime materialization lock id mismatch with runtime revision"
        )
    if runtime_materialization.materialized_runtime_lock_digest != lock_digest:
        raise RegistryProjectionAuthorityConflict(
            "runtime materialization lock digest mismatch with runtime revision"
        )

    artifact_digest = runtime_revision.materialization_artifact_digest
    if artifact_digest is None:
        raise RegistryProjectionAuthorityConflict(
            "runtime revision requires materialization_artifact_digest"
        )
    if runtime_materialization.materialization_artifact_digest != artifact_digest:
        raise RegistryProjectionAuthorityConflict(
            "materialization_artifact_digest mismatch with runtime revision"
        )


__all__ = [
    "RegistryProjectionAuthorityConflict",
    "RegistryProjectionAuthorityError",
    "RegistryProjectionAuthorityNotFound",
    "RegistryProjectionAuthorityResolver",
    "ResolvedRegistryProjectionAuthority",
]

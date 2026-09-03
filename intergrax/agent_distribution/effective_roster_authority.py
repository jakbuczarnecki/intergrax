# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Historical EffectiveRoster authority resolution for RuntimeRevision (ADR-AGENT-007)."""

from __future__ import annotations

from intergrax.agent_distribution.errors import (
    EffectiveRosterAuthorityConflict,
    EffectiveRosterAuthorityNotFound,
)
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import EffectiveRosterSnapshotStore


class EffectiveRosterAuthorityService:
    """Resolves immutable historical EffectiveRoster authority for RuntimeRevision.

    Fail-closed: missing or invalid snapshot authority raises typed domain errors.
    """

    def __init__(
        self,
        *,
        snapshot_store: EffectiveRosterSnapshotStore,
    ) -> None:
        self._snapshot_store = snapshot_store

    def require_for_revision(
        self,
        revision: RuntimeRevision,
    ) -> EffectiveRoster:
        roster_revision_id = revision.effective_roster_revision_id
        snapshot = self._snapshot_store.get_by_revision(roster_revision_id)
        if snapshot is None:
            raise EffectiveRosterAuthorityNotFound(
                "runtime revision lacks canonical effective roster snapshot authority"
            )
        self._validate_snapshot_matches_revision(snapshot, revision)
        return snapshot

    @staticmethod
    def _validate_snapshot_matches_revision(
        snapshot: EffectiveRoster,
        revision: RuntimeRevision,
    ) -> None:
        roster_revision_id = revision.effective_roster_revision_id
        if snapshot.effective_roster_revision_id != roster_revision_id:
            raise EffectiveRosterAuthorityConflict(
                "effective roster snapshot revision id mismatch"
            )
        if snapshot.compute_revision_id() != roster_revision_id:
            raise EffectiveRosterAuthorityConflict(
                "effective roster snapshot content identity mismatch"
            )
        if snapshot.application_id != revision.application_id:
            raise EffectiveRosterAuthorityConflict(
                "effective roster snapshot application id mismatch"
            )
        if snapshot.application_environment_id != revision.application_environment_id:
            raise EffectiveRosterAuthorityConflict(
                "effective roster snapshot application environment id mismatch"
            )
        if snapshot.manifest_release_id != revision.application_release_id:
            raise EffectiveRosterAuthorityConflict(
                "effective roster snapshot manifest release id mismatch"
            )


__all__ = ["EffectiveRosterAuthorityService"]

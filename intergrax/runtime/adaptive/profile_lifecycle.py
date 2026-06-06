# © Artur Czarnecki. All rights reserved.

"""Profile version lifecycle state machine (Phase W-ADAPT-3.2)."""

from __future__ import annotations

from intergrax.runtime.adaptive.contracts import ProfileVersionRecord, ProfileVersionStatus
from intergrax.runtime.adaptive.profile_version_store import ProfileVersionStore

_ALLOWED_TRANSITIONS: dict[ProfileVersionStatus, frozenset[ProfileVersionStatus]] = {
    ProfileVersionStatus.DRAFT: frozenset({ProfileVersionStatus.SHADOW}),
    ProfileVersionStatus.SHADOW: frozenset(
        {ProfileVersionStatus.CANARY, ProfileVersionStatus.DRAFT}
    ),
    ProfileVersionStatus.CANARY: frozenset(
        {ProfileVersionStatus.ACTIVE, ProfileVersionStatus.DRAFT}
    ),
    ProfileVersionStatus.ACTIVE: frozenset(
        {ProfileVersionStatus.RETIRED, ProfileVersionStatus.DRAFT}
    ),
    ProfileVersionStatus.RETIRED: frozenset({ProfileVersionStatus.ACTIVE}),
}


class ProfileLifecycleTransitionError(ValueError):
    """Raised when a profile version status transition is not allowed."""


def validate_profile_transition(
    *,
    current: ProfileVersionStatus,
    target: ProfileVersionStatus,
) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ProfileLifecycleTransitionError(
            f"Unsupported profile transition: {current.value} -> {target.value}"
        )


class ProfileVersionLifecycleManager:
    """Applies validated status transitions through a profile version store."""

    def __init__(self, store: ProfileVersionStore) -> None:
        self._store = store

    def transition(
        self,
        version_id: str,
        *,
        target: ProfileVersionStatus,
    ) -> ProfileVersionRecord:
        current = self._store.get(version_id)
        if current is None:
            raise ValueError(f"Unknown profile version: {version_id}")
        validate_profile_transition(current=current.status, target=target)
        updated = current.model_copy(update={"status": target})
        return self._store.save_status(updated)

# © Artur Czarnecki. All rights reserved.

"""Append-only in-memory effective profile revision store (P1.2)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionConflictError,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)


def _scope_key(scope: EffectiveProfileRevisionScope) -> tuple[str, str | None]:
    return (scope.application_id, scope.tenant_id)


class InMemoryEffectiveProfileRevisionStore:
    """Test and local adapter for append-only revision persistence."""

    def __init__(self) -> None:
        self._revisions: dict[tuple[str, str | None, str], EffectiveProfileRevision] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def save(self, revision: EffectiveProfileRevision) -> None:
        app_id, tenant_id = _scope_key(revision.scope)
        key = (app_id, tenant_id, revision.revision_id.value)
        if key in self._revisions:
            raise EffectiveProfileRevisionConflictError(
                f"revision already exists: {revision.revision_id.value}"
            )
        self._revisions[key] = revision

    def get(
        self,
        revision_id: EffectiveProfileRevisionId,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision | None:
        app_id, tenant_id = _scope_key(scope)
        return self._revisions.get((app_id, tenant_id, revision_id.value))

    def latest(
        self,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision | None:
        app_id, tenant_id = _scope_key(scope)
        prefix = (app_id, tenant_id)
        matches = [
            revision
            for key, revision in self._revisions.items()
            if key[0] == prefix[0] and key[1] == prefix[1]
        ]
        if not matches:
            return None
        return matches[-1]

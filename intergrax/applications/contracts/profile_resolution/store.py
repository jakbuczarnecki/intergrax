# © Artur Czarnecki. All rights reserved.

"""Effective profile revision durable storage contract (P1.2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)


class EffectiveProfileRevisionStore(Protocol):
    """Append-only immutable effective profile revision store."""

    def save(self, revision: EffectiveProfileRevision) -> None:
        """Persist one immutable revision; must not overwrite an existing id."""

    def get(
        self,
        revision_id: EffectiveProfileRevisionId,
        *,
        scope: EffectiveProfileRevisionScope,
    ) -> EffectiveProfileRevision | None:
        """Load one revision by identity within the declared scope."""

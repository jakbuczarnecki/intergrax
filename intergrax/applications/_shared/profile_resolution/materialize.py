# © Artur Czarnecki. All rights reserved.

"""Canonical effective profile revision materialization (P1.2)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionError,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    mint_effective_profile_revision_id,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)


def materialize_effective_profile_revision(
    resolution: ProfileResolution,
    *,
    scope: EffectiveProfileRevisionScope,
    predecessor_revision_id: EffectiveProfileRevisionId | None = None,
    revision_id: EffectiveProfileRevisionId | None = None,
    store: EffectiveProfileRevisionStore | None = None,
) -> EffectiveProfileRevision:
    """
    Admit one immutable effective profile revision from resolved evidence.

    Consumes ``ProfileResolution`` — does not resolve configuration.
    """
    resolved_id = revision_id or mint_effective_profile_revision_id()
    revision = EffectiveProfileRevision(
        revision_id=resolved_id,
        fingerprint=resolution.fingerprint,
        effective_profile=resolution.effective_profile.model_copy(deep=True),
        resolution=resolution,
        scope=scope,
        predecessor_revision_id=predecessor_revision_id,
    )
    if revision.fingerprint != resolution.fingerprint:
        raise EffectiveProfileRevisionError("revision fingerprint must match resolution authority")
    if store is not None:
        store.save(revision)
    return revision

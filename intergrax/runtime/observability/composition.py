# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host composition for revision ordering authority (TRACE-BITEMP-2)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.contracts.bitemporal_knowledge import RevisionOrderingAuthority
from intergrax.integrations.providers.relational_store.sqlite.config import DEFAULT_DATA_DIR
from intergrax.runtime.observability.canonical_revision_ordering_provider import (
    CanonicalRevisionOrderingProvider,
)
from intergrax.runtime.observability.revision_ordering_store import (
    RevisionOrderingStoreTestHooks,
)

ENV_REVISION_ORDERING_DB = "INTERGRAX_REVISION_ORDERING_DB"
REVISION_ORDERING_DB_NAME = "intergrax_revision_ordering.db"


def resolve_revision_ordering_db_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_REVISION_ORDERING_DB)
    if env:
        return Path(env)
    return DEFAULT_DATA_DIR / REVISION_ORDERING_DB_NAME


def open_revision_ordering_authority(
    *,
    db_path: Path | str | None = None,
    test_hooks: RevisionOrderingStoreTestHooks | None = None,
) -> RevisionOrderingAuthority:
    resolved = resolve_revision_ordering_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return CanonicalRevisionOrderingProvider.open(
        str(resolved),
        test_hooks=test_hooks,
    )

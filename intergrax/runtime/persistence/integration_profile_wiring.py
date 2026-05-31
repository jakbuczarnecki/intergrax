# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Resolve runtime persistence from ``IntegrationProfile`` (Phase M.8).

Prefer ``profile.resolve(IntegrationCategory.RELATIONAL_STORE)`` or
``create_sqlite_integration()`` over direct imports of runtime SQLite classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug


def sqlite_bundle_for_profile(
    profile: IntegrationProfile,
    *,
    data_dir: Path | str | None = None,
    **config_overrides: object,
) -> Optional[SQLiteIntegrationBundle]:
    slug = profile.slug_for_category(IntegrationCategory.RELATIONAL_STORE)
    if slug != IntegrationSlug.SQLITE.value:
        return None
    opts = dict(profile.options_for_slug(IntegrationSlug.SQLITE))
    opts.update(config_overrides)
    return create_sqlite_integration(data_dir=data_dir, **opts)


def open_trace_store_from_profile(
    profile: IntegrationProfile,
    *,
    db_path: Path | None = None,
) -> Any:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_trace_store

        return create_sqlite_trace_store(db_path=db_path)
    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return bundle.trace_store
    raise RuntimeError("relational_store is not sqlite — configure trace store explicitly")


def open_runtime_event_store_from_profile(
    profile: IntegrationProfile,
    *,
    db_path: Path | None = None,
) -> Any:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_runtime_event_store

        return create_sqlite_runtime_event_store(db_path=db_path)
    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return bundle.runtime_event_store
    raise RuntimeError("relational_store is not sqlite — configure runtime event store explicitly")

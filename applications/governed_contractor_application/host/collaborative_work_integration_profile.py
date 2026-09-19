# © Artur Czarnecki. All rights reserved.

"""Governed Contractor — Collaborative Work integration profile for harness host assembly."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile


def resolve_governed_contractor_collaborative_work_integration_profile(
    manifest: ApplicationManifest,
    *,
    trace_db_path: Path | None,
) -> IntegrationProfile:
    """
    Product manifest integration selection for Collaborative Work persistence.

    When diagnostics attach local trace storage, relational SQLite options are scoped
    to the same storage directory as the harness trace database.
    """
    profile = manifest.integration_profile
    if trace_db_path is None:
        return profile
    if profile.slug_for_category(IntegrationCategory.RELATIONAL_STORE) != SQLITE.slug:
        return profile
    storage_dir = trace_db_path.parent
    sqlite_options = dict(profile.options.get(SQLITE.slug, {}))
    sqlite_options.setdefault("data_dir", str(storage_dir))
    sqlite_options.setdefault(
        "relational_db",
        str(storage_dir / "collaborative_work.db"),
    )
    return profile.model_copy(
        update={
            "options": {
                **profile.options,
                SQLITE.slug: sqlite_options,
            },
        },
    )


__all__ = ["resolve_governed_contractor_collaborative_work_integration_profile"]

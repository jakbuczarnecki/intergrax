# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite integration plugin — type-based registration alternative to manifest-only."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store
from intergrax.integrations.registry.catalog_manifests import SQLITE


class SqliteIntegrationPlugin:
    """Register via :func:`register_integration_plugin` or ``IntegrationProfile(relational_store=SqliteIntegrationPlugin)``."""

    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return SQLITE

    @classmethod
    def create_integration(cls, **kwargs: Any) -> Any:
        return create_sqlite_relational_store(**kwargs)

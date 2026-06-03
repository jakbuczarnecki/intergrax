# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_sql in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.azure_sql.bundle import create_azure_sql_relational_store
from intergrax.integrations.providers.relational_store.azure_sql.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_azure_sql_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_azure_sql_relational_store, override=override)

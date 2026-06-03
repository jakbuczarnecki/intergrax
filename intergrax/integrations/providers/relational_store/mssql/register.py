# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mssql in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store
from intergrax.integrations.providers.relational_store.mssql.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_mssql_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_mssql_relational_store, override=override)

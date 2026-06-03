# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register databricks in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.databricks.bundle import create_databricks_relational_store
from intergrax.integrations.providers.relational_store.databricks.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_databricks_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_databricks_relational_store, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register snowflake in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.snowflake.bundle import create_snowflake_relational_store
from intergrax.integrations.providers.relational_store.snowflake.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_snowflake_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_snowflake_relational_store, override=override)

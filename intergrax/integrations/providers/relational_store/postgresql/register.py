# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register postgresql in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.postgresql.bundle import create_postgresql_relational_store
from intergrax.integrations.providers.relational_store.postgresql.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_postgresql_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_postgresql_relational_store, override=override)

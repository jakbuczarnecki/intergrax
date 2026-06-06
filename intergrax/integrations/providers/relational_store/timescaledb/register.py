# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register timescaledb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.timescaledb.bundle import create_timescaledb_relational_store
from intergrax.integrations.providers.relational_store.timescaledb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_timescaledb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_timescaledb_relational_store, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register falkordb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.graph_store.falkordb.bundle import create_falkordb_graph_store
from intergrax.integrations.providers.graph_store.falkordb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_falkordb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_falkordb_graph_store, override=override)

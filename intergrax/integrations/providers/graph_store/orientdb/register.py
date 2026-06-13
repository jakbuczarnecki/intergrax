# © Artur Czarnecki. All rights reserved.

from intergrax.integrations.providers.graph_store.orientdb.bundle import create_orientdb_graph_store
from intergrax.integrations.providers.graph_store.orientdb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_orientdb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_orientdb_graph_store, override=override)

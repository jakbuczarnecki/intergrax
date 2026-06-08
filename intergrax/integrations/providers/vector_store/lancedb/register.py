# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register lancedb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.lancedb.bundle import create_lancedb_vector_store
from intergrax.integrations.providers.vector_store.lancedb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_lancedb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_lancedb_vector_store, override=override)

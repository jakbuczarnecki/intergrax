# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register qdrant in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.integrations.providers.vector_store.qdrant.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_qdrant_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_qdrant_vector_store, override=override)

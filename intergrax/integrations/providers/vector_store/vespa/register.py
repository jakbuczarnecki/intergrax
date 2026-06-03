# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register vespa in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store
from intergrax.integrations.providers.vector_store.vespa.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_vespa_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_vespa_vector_store, override=override)

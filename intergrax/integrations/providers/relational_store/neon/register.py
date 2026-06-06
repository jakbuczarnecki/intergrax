# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register neon in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.neon.bundle import create_neon_relational_store
from intergrax.integrations.providers.relational_store.neon.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_neon_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_neon_relational_store, override=override)

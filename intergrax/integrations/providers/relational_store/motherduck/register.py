# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register motherduck in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.motherduck.bundle import create_motherduck_relational_store
from intergrax.integrations.providers.relational_store.motherduck.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_motherduck_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_motherduck_relational_store, override=override)

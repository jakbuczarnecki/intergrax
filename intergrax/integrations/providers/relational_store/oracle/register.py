# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register oracle in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.oracle.bundle import create_oracle_relational_store
from intergrax.integrations.providers.relational_store.oracle.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_oracle_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_oracle_relational_store, override=override)

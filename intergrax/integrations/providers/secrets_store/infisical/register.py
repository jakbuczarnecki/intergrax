# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register infisical in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.infisical.bundle import create_infisical_secrets_store
from intergrax.integrations.providers.secrets_store.infisical.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_infisical_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_infisical_secrets_store, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register keycloak in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.keycloak.bundle import create_keycloak_identity_provider
from intergrax.integrations.providers.identity_provider.keycloak.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_keycloak_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_keycloak_identity_provider, override=override)

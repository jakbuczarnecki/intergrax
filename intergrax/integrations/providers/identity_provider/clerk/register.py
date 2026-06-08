# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register clerk in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.clerk.bundle import create_clerk_identity_provider
from intergrax.integrations.providers.identity_provider.clerk.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_clerk_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_clerk_identity_provider, override=override)

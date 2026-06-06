# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register workos in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.workos.bundle import create_workos_identity_provider
from intergrax.integrations.providers.identity_provider.workos.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_workos_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_workos_identity_provider, override=override)

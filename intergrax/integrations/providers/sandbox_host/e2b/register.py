# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register e2b in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
from intergrax.integrations.providers.sandbox_host.e2b.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_e2b_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_e2b_sandbox_host, override=override)

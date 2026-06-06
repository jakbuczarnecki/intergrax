# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register trivy in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner
from intergrax.integrations.providers.security_scanner.trivy.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_trivy_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_trivy_security_scanner, override=override)

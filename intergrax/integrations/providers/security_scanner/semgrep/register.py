# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register semgrep in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.semgrep.bundle import create_semgrep_security_scanner
from intergrax.integrations.providers.security_scanner.semgrep.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_semgrep_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_semgrep_security_scanner, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register snyk in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.security_scanner.snyk.bundle import create_snyk_security_scanner
from intergrax.integrations.providers.security_scanner.snyk.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.security_scanner.snyk.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_snyk_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_snyk_security_scanner,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

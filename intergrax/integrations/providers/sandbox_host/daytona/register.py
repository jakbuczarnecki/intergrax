# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register daytona in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.sandbox_host.daytona.bundle import create_daytona_sandbox_host
from intergrax.integrations.providers.sandbox_host.daytona.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.sandbox_host.daytona.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_daytona_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_daytona_sandbox_host,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

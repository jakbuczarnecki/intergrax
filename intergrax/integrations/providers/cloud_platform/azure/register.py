# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.cloud_platform.azure.bundle import create_azure_cloud_platform
from intergrax.integrations.providers.cloud_platform.azure.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.cloud_platform.azure.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_azure_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_azure_cloud_platform,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

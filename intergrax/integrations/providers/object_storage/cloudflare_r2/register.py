# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register cloudflare_r2 in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.cloudflare_r2.bundle import create_cloudflare_r2_object_storage
from intergrax.integrations.providers.object_storage.cloudflare_r2.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.object_storage.cloudflare_r2.contract_spec import CONTRACT_SPECS


def register_cloudflare_r2_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_cloudflare_r2_object_storage, override=override, contract_specs=CONTRACT_SPECS)

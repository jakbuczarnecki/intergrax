# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register exa in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.exa.bundle import create_exa_search_provider
from intergrax.integrations.providers.search_provider.exa.manifest import MANIFEST
from intergrax.integrations.providers.search_provider.exa.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_exa_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_exa_search_provider, override=override, contract_specs=CONTRACT_SPECS)

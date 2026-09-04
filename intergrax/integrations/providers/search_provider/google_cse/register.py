# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_cse in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.google_cse.bundle import create_google_cse_search_provider
from intergrax.integrations.providers.search_provider.google_cse.manifest import MANIFEST
from intergrax.integrations.providers.search_provider.google_cse.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_google_cse_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_google_cse_search_provider, override=override, contract_specs=CONTRACT_SPECS)

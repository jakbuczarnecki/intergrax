# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register algolia in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.algolia.bundle import create_algolia_search_provider
from intergrax.integrations.providers.search_provider.algolia.manifest import MANIFEST
from intergrax.integrations.providers.search_provider.algolia.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_algolia_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_algolia_search_provider, override=override, contract_specs=CONTRACT_SPECS)

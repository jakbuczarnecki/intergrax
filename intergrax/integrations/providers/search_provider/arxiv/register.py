# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register arxiv in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.search_provider.arxiv.bundle import create_arxiv_search_provider
from intergrax.integrations.providers.search_provider.arxiv.manifest import MANIFEST
from intergrax.integrations.providers.search_provider.arxiv.contract_spec import CONTRACT_SPECS
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_arxiv_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_arxiv_search_provider, override=override, contract_specs=CONTRACT_SPECS)

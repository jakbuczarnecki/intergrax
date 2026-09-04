# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sharepoint in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge
from intergrax.integrations.providers.wiki_knowledge.sharepoint.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.wiki_knowledge.sharepoint.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_sharepoint_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_sharepoint_wiki_knowledge,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
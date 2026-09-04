# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register notion in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge
from intergrax.integrations.providers.wiki_knowledge.notion.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.wiki_knowledge.notion.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_notion_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_notion_wiki_knowledge,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register wikipedia in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.wikipedia.bundle import create_wikipedia_wiki_knowledge
from intergrax.integrations.providers.wiki_knowledge.wikipedia.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.wiki_knowledge.wikipedia.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_wikipedia_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_wikipedia_wiki_knowledge,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
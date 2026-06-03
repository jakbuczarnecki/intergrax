# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register confluence in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_confluence_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_confluence_wiki_knowledge, override=override)

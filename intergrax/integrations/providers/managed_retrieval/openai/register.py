# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register OpenAI managed retrieval in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.managed_retrieval.openai.bundle import create_openai_managed_retrieval
from intergrax.integrations.providers.managed_retrieval.openai.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_openai_managed_retrieval_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_openai_managed_retrieval, override=override)

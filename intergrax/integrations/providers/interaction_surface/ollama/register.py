# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register ollama in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.interaction_surface.ollama.bundle import create_ollama_interaction_surface
from intergrax.integrations.providers.interaction_surface.ollama.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_ollama_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_ollama_interaction_surface, override=override)

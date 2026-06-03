# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register slash_command in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.interaction_surface.slash_command.bundle import create_slash_command_interaction_surface
from intergrax.integrations.providers.interaction_surface.slash_command.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_slash_command_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_slash_command_interaction_surface, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register lab_json in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.interaction_surface.lab_json.bundle import create_lab_json_interaction_surface
from intergrax.integrations.providers.interaction_surface.lab_json.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_lab_json_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_lab_json_interaction_surface, override=override)

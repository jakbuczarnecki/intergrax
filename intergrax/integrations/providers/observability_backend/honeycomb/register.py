# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register honeycomb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.honeycomb.bundle import create_honeycomb_observability_backend
from intergrax.integrations.providers.observability_backend.honeycomb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_honeycomb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_honeycomb_observability_backend, override=override)

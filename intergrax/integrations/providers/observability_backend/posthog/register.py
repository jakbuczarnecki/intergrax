# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register posthog in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.posthog.bundle import create_posthog_observability_backend
from intergrax.integrations.providers.observability_backend.posthog.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_posthog_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_posthog_observability_backend, override=override)

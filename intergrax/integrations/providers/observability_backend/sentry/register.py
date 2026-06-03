# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sentry in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend
from intergrax.integrations.providers.observability_backend.sentry.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_sentry_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_sentry_observability_backend, override=override)

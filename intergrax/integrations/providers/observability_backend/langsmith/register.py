# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register langsmith in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.langsmith.bundle import create_langsmith_observability_backend
from intergrax.integrations.providers.observability_backend.langsmith.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_langsmith_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_langsmith_observability_backend, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register langfuse in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend
from intergrax.integrations.providers.observability_backend.langfuse.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_langfuse_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_langfuse_observability_backend, override=override)

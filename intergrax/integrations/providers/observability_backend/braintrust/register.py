# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register braintrust in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.braintrust.bundle import create_braintrust_observability_backend
from intergrax.integrations.providers.observability_backend.braintrust.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_braintrust_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_braintrust_observability_backend, override=override)

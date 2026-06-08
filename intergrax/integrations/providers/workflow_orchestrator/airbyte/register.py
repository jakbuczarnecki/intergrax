# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register airbyte in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.airbyte.bundle import create_airbyte_workflow_orchestrator
from intergrax.integrations.providers.workflow_orchestrator.airbyte.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_airbyte_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_airbyte_workflow_orchestrator, override=override)

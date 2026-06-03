# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_devops in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import create_azure_devops_issue_tracker
from intergrax.integrations.providers.issue_tracker.azure_devops.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_azure_devops_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_azure_devops_issue_tracker, override=override)

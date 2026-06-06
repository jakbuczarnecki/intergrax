# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register servicenow in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.servicenow.bundle import create_servicenow_issue_tracker
from intergrax.integrations.providers.issue_tracker.servicenow.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_servicenow_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_servicenow_issue_tracker, override=override)

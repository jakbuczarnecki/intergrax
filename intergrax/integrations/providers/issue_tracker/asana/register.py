# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register asana in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.asana.bundle import create_asana_issue_tracker
from intergrax.integrations.providers.issue_tracker.asana.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_asana_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_asana_issue_tracker, override=override)

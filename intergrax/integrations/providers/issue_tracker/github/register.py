# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register github in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
from intergrax.integrations.providers.issue_tracker.github.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_github_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_github_issue_tracker, override=override)

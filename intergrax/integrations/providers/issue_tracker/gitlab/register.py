# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gitlab in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker
from intergrax.integrations.providers.issue_tracker.gitlab.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.issue_tracker.gitlab.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_gitlab_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_gitlab_issue_tracker,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
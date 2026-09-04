# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register jira in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.jira.bundle import create_jira_issue_tracker
from intergrax.integrations.providers.issue_tracker.jira.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.issue_tracker.jira.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_jira_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_jira_issue_tracker,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
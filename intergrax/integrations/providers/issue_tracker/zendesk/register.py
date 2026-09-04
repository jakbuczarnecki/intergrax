# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register zendesk in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.zendesk.bundle import create_zendesk_issue_tracker
from intergrax.integrations.providers.issue_tracker.zendesk.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.issue_tracker.zendesk.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_zendesk_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_zendesk_issue_tracker,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
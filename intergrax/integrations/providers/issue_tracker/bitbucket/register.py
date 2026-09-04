# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register bitbucket in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.issue_tracker.bitbucket.bundle import create_bitbucket_issue_tracker
from intergrax.integrations.providers.issue_tracker.bitbucket.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.issue_tracker.bitbucket.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_bitbucket_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_bitbucket_issue_tracker,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register google_workspace in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite
from intergrax.integrations.providers.collaboration_suite.google_workspace.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.collaboration_suite.google_workspace.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_google_workspace_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_google_workspace_collaboration_suite,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
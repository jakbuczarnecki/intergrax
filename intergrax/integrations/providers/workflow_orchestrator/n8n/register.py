# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register n8n in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.n8n.bundle import create_n8n_workflow_orchestrator
from intergrax.integrations.providers.workflow_orchestrator.n8n.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.workflow_orchestrator.n8n.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_n8n_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_n8n_workflow_orchestrator,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

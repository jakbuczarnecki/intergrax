# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register prefect in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.prefect.bundle import create_prefect_workflow_orchestrator
from intergrax.integrations.providers.workflow_orchestrator.prefect.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.workflow_orchestrator.prefect.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_prefect_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_prefect_workflow_orchestrator,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

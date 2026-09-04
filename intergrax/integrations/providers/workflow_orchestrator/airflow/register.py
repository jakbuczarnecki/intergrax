# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register airflow in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.workflow_orchestrator.airflow.bundle import create_airflow_workflow_orchestrator
from intergrax.integrations.providers.workflow_orchestrator.airflow.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.workflow_orchestrator.airflow.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_airflow_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_airflow_workflow_orchestrator,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

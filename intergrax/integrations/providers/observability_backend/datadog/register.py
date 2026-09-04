# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register datadog in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend
from intergrax.integrations.providers.observability_backend.datadog.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.datadog.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_datadog_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_datadog_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

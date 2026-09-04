# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register splunk in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.splunk.bundle import create_splunk_observability_backend
from intergrax.integrations.providers.observability_backend.splunk.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.splunk.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_splunk_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_splunk_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

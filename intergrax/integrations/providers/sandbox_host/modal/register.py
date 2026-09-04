# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register modal in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.sandbox_host.modal.bundle import create_modal_sandbox_host
from intergrax.integrations.providers.sandbox_host.modal.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.sandbox_host.modal.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_modal_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_modal_sandbox_host,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

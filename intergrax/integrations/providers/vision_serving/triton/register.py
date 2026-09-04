# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register triton in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vision_serving.triton.bundle import create_triton_vision_serving
from intergrax.integrations.providers.vision_serving.triton.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.vision_serving.triton.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_triton_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_triton_vision_serving,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

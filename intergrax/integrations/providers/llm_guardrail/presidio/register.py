# © Artur Czarnecki. All rights reserved.

"""Register presidio in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.presidio.bundle import create_presidio_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.presidio.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.presidio.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_presidio_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_presidio_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

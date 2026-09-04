# © Artur Czarnecki. All rights reserved.

"""Register azure_content_safety in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.azure_content_safety.bundle import create_azure_content_safety_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.azure_content_safety.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.azure_content_safety.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_azure_content_safety_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_azure_content_safety_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

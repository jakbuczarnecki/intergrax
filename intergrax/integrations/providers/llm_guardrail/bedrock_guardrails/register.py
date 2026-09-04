# © Artur Czarnecki. All rights reserved.

"""Register bedrock_guardrails in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.bundle import create_bedrock_guardrails_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_bedrock_guardrails_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_bedrock_guardrails_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

# © Artur Czarnecki. All rights reserved.

"""Register guardrails_ai in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.guardrails_ai.bundle import create_guardrails_ai_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.guardrails_ai.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.guardrails_ai.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_guardrails_ai_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_guardrails_ai_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )

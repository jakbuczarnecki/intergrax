# © Artur Czarnecki. All rights reserved.

"""Register all LLM guardrail catalog slugs (M-P12.*)."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.azure_content_safety.register import (
    register_azure_content_safety_integration,
)
from intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.register import (
    register_bedrock_guardrails_integration,
)
from intergrax.integrations.providers.llm_guardrail.guardrails_ai.register import (
    register_guardrails_ai_integration,
)
from intergrax.integrations.providers.llm_guardrail.lakera.register import register_lakera_integration
from intergrax.integrations.providers.llm_guardrail.llama_guard.register import register_llama_guard_integration
from intergrax.integrations.providers.llm_guardrail.llm_guard.register import register_llm_guard_integration
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.register import (
    register_nemo_guardrails_integration,
)
from intergrax.integrations.providers.llm_guardrail.openguardrails.register import (
    register_openguardrails_integration,
)
from intergrax.integrations.providers.llm_guardrail.presidio.register import register_presidio_integration

GUARD_SLUGS: tuple[str, ...] = (
    "llm_guard",
    "guardrails_ai",
    "nemo_guardrails",
    "openguardrails",
    "presidio",
    "llama_guard",
    "lakera",
    "azure_content_safety",
    "bedrock_guardrails",
)


def register_llm_guardrail_integrations(*, override: bool = False) -> None:
    register_llm_guard_integration(override=override)
    register_guardrails_ai_integration(override=override)
    register_nemo_guardrails_integration(override=override)
    register_openguardrails_integration(override=override)
    register_presidio_integration(override=override)
    register_llama_guard_integration(override=override)
    register_lakera_integration(override=override)
    register_azure_content_safety_integration(override=override)
    register_bedrock_guardrails_integration(override=override)

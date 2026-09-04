# © Artur Czarnecki. All rights reserved.

"""Factory helpers for OpenGuardrails guardrail adapter."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bundles.http_guardrail import create_openguardrails_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.openguardrails.integration import (
    OPENGUARDRAILS_PROVIDER_ID,
    OpenguardrailsLlmGuardrailIntegration,
    OpenguardrailsLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_openguardrails_llm_guardrail",
    "create_openguardrails_llm_guardrail_integration",
]


def create_openguardrails_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> OpenguardrailsLlmGuardrailIntegration:
    """Build a contract-based OpenGuardrails guardrail integration."""
    if backend is not None:
        return OpenguardrailsLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=OPENGUARDRAILS_PROVIDER_ID,
            display_name="OpenGuardrails",
            enabled=enabled,
            config=OpenguardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return OpenguardrailsLlmGuardrailIntegration.for_provider(
        provider_id=OPENGUARDRAILS_PROVIDER_ID,
        display_name="OpenGuardrails",
        config=OpenguardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_openguardrails_llm_guardrail(**_kwargs: object) -> OpenguardrailsLlmGuardrailIntegration:
    """Catalog factory for ``openguardrails`` / ``llm_guardrail``."""
    backend = create_openguardrails_backend()
    return OpenguardrailsLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=OPENGUARDRAILS_PROVIDER_ID,
        display_name="OpenGuardrails",
        enabled=True,
        config=OpenguardrailsLlmGuardrailIntegrationConfig(enabled=True),
    )

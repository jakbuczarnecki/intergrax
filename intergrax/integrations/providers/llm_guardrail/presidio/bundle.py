# © Artur Czarnecki. All rights reserved.

"""Factory helpers for Presidio guardrail adapter."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.bundles.presidio import create_presidio_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.presidio.integration import (
    PRESIDIO_PROVIDER_ID,
    PresidioLlmGuardrailIntegration,
    PresidioLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_presidio_llm_guardrail",
    "create_presidio_llm_guardrail_integration",
]


def create_presidio_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> PresidioLlmGuardrailIntegration:
    """Build a contract-based Presidio guardrail integration."""
    if backend is not None:
        return PresidioLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=PRESIDIO_PROVIDER_ID,
            display_name="Presidio",
            enabled=enabled,
            config=PresidioLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return PresidioLlmGuardrailIntegration.for_provider(
        provider_id=PRESIDIO_PROVIDER_ID,
        display_name="Presidio",
        config=PresidioLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_presidio_llm_guardrail(**_kwargs: object) -> PresidioLlmGuardrailIntegration:
    """Catalog factory for ``presidio`` / ``llm_guardrail``."""
    backend = create_presidio_backend()
    return PresidioLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=PRESIDIO_PROVIDER_ID,
        display_name="Presidio",
        enabled=True,
        config=PresidioLlmGuardrailIntegrationConfig(enabled=True),
    )

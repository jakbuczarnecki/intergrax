# © Artur Czarnecki. All rights reserved.

"""Guardrail profile → runtime bridge (M-P12-WIRE.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend


@dataclass(frozen=True, slots=True)
class GuardrailWiringOptions:
    enabled: bool
    backend_slug: str | None


def resolve_guardrail_wiring_options(env: ApplicationEnvironmentProfile) -> GuardrailWiringOptions:
    binding = env.integration_profile.llm_guardrail
    slug = binding.resolved_slug() if binding is not None else None
    return GuardrailWiringOptions(enabled=slug is not None, backend_slug=slug)


def resolve_guardrail_backend(env: ApplicationEnvironmentProfile) -> LlmGuardrailBackend | None:
    options = resolve_guardrail_wiring_options(env)
    if not options.enabled or options.backend_slug is None:
        return None
    resolved = env.integration_profile.resolve(IntegrationCategory.LLM_GUARDRAIL)
    return resolved  # type: ignore[return-value]

# © Artur Czarnecki. All rights reserved.

"""Guardrail profile → runtime bridge (M-P12-WIRE.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.llm_guardrail import GuardrailBackendOptions, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._factory import (
    create_chained_guardrail_backend,
    create_guardrail_backend,
)
from intergrax.integrations.providers.llm_guardrail.register_all import GUARD_SLUGS


@dataclass(frozen=True, slots=True)
class GuardrailWiringOptions:
    enabled: bool
    backend_slug: str | None


def guardrail_backend_options(env: ApplicationEnvironmentProfile) -> GuardrailBackendOptions:
    profile = env.guardrail_profile
    return GuardrailBackendOptions(
        secondary_slug=profile.secondary_slug,
        colang_config_path=profile.colang_config_path,
        bedrock_guardrail_policy_id=profile.bedrock_guardrail_policy_id,
        inference_slug=profile.inference_slug,
    )


def resolve_guardrail_wiring_options(env: ApplicationEnvironmentProfile) -> GuardrailWiringOptions:
    binding = env.integration_profile.llm_guardrail
    slug = binding.resolved_slug() if binding is not None else None
    return GuardrailWiringOptions(enabled=slug is not None, backend_slug=slug)


def _resolve_secondary_slug(
    env: ApplicationEnvironmentProfile,
    primary_slug: str,
) -> str | None:
    if env.guardrail_profile.secondary_slug:
        return env.guardrail_profile.secondary_slug
    options = env.integration_profile.options or {}
    for key in options:
        if key != primary_slug and key in GUARD_SLUGS:
            return key
    return None


def resolve_guardrail_backend(env: ApplicationEnvironmentProfile) -> LlmGuardrailBackend | None:
    options = resolve_guardrail_wiring_options(env)
    if not options.enabled or options.backend_slug is None:
        return None
    backend_options = guardrail_backend_options(env)
    primary_slug = options.backend_slug
    secondary_slug = _resolve_secondary_slug(env, primary_slug)
    if secondary_slug:
        return create_chained_guardrail_backend(
            primary_slug,
            secondary_slug,
            options=backend_options,
        )
    return create_guardrail_backend(primary_slug, options=backend_options)


def apply_guardrail_profiles_to_runtime_config(
    config: object,
    env: ApplicationEnvironmentProfile,
) -> object:
    """Attach guardrail middleware for runtime pipeline LLM steps when enabled."""
    from intergrax.applications._shared.application_guardrail_middleware import LlmGuardrailMiddleware
    from intergrax.runtime.nexus.config import RuntimeConfig

    if not isinstance(config, RuntimeConfig):
        return config
    if not env.guardrail_profile.enabled:
        return config
    backend = resolve_guardrail_backend(env)
    if backend is None:
        return config
    config.metadata["guardrail_middleware"] = LlmGuardrailMiddleware(
        backend,
        env.guardrail_profile,
    )
    wiring = resolve_guardrail_wiring_options(env)
    if wiring.backend_slug:
        config.metadata["llm_guardrail_slug"] = wiring.backend_slug
    return config

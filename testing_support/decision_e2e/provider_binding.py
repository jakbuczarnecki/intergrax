# © Artur Czarnecki. All rights reserved.

"""Canonical INTERGRAX_LLM_* binding for DS-E2E scenario qualification."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.catalog_capabilities import unwrap_catalog_capability_adapter
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env

_BINDING_SOURCE = "application_environment_profile.llm_profile"


@dataclass(frozen=True, slots=True)
class QualificationProviderBinding:
    """Resolved provider/model identity for qualification evidence."""

    requested_provider: str | None
    requested_model: str | None
    resolved_provider: str
    resolved_model: str | None
    binding_source: str
    profile: LLMProfile


def _explicit_env_value(prefix: str, suffix: str) -> str | None:
    raw = os.environ.get(f"{prefix}_{suffix}")
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider).strip().lower()


def _adapter_identity(adapter: LLMAdapter) -> tuple[str, str]:
    inner = unwrap_catalog_capability_adapter(adapter)
    return _provider_slug(inner.provider), inner.model


def bind_qualification_llm_profile(
    environment: ApplicationEnvironmentProfile,
    *,
    prefix: str = "INTERGRAX_LLM",
    adapter_resolver: Callable[[ApplicationEnvironmentProfile], LLMAdapter] | None = None,
) -> tuple[QualificationProviderBinding | None, str | None]:
    """
    Bind ``llm_profile_from_env`` into ``environment`` and verify adapter resolution.

    Returns ``(binding, block_reason)``; block_reason is set on fail-closed paths.
    """
    requested_provider = _explicit_env_value(prefix, "PROVIDER")
    requested_model = _explicit_env_value(prefix, "MODEL")

    try:
        profile = llm_profile_from_env(prefix=prefix)
    except ValueError as exc:
        if requested_provider is not None:
            return None, f"invalid explicit qualification provider/model: {exc}"
        return None, f"qualification llm profile resolution failed: {exc}"

    environment.llm_profile = profile

    try:
        profile.create_adapter()
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        if requested_provider is not None or requested_model is not None:
            return None, (
                "explicit qualification provider/model could not be resolved: "
                f"{type(exc).__name__}"
            )
        return None, f"qualification adapter unavailable: {type(exc).__name__}"

    if adapter_resolver is None:
        from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
            resolve_scenario_llm_adapter,
        )

        adapter_resolver = resolve_scenario_llm_adapter

    try:
        adapter = adapter_resolver(environment)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        if requested_provider is not None or requested_model is not None:
            return None, (
                "explicit qualification provider/model could not be resolved: "
                f"{type(exc).__name__}"
            )
        return None, f"qualification adapter resolution failed: {type(exc).__name__}"

    resolved_provider, resolved_model = _adapter_identity(adapter)
    profile_provider = _provider_slug(profile.provider)
    profile_model = profile.model

    if profile_provider != resolved_provider or profile_model != resolved_model:
        return None, (
            "qualification provider binding mismatch between profile and resolved adapter: "
            f"profile={profile_provider}/{profile_model} "
            f"resolved={resolved_provider}/{resolved_model}"
        )

    if requested_provider is not None and requested_provider.lower() != resolved_provider:
        return None, (
            "requested provider does not match resolved adapter: "
            f"requested={requested_provider} resolved={resolved_provider}"
        )
    if requested_model is not None and requested_model != resolved_model:
        return None, (
            "requested model does not match resolved adapter: "
            f"requested={requested_model} resolved={resolved_model}"
        )

    return (
        QualificationProviderBinding(
            requested_provider=requested_provider,
            requested_model=requested_model,
            resolved_provider=resolved_provider,
            resolved_model=resolved_model,
            binding_source=_BINDING_SOURCE,
            profile=profile,
        ),
        None,
    )

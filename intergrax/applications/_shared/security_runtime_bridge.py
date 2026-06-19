# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile security fields to wiring options (Phase SEC-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.security.defense_registry import resolve_security_defense_plugins


@dataclass(frozen=True, slots=True)
class SecurityWiringOptions:
    """Resolved security wiring flags for Tier-3 hosts."""

    prompt_defense_enabled: bool
    tool_injection_defense_enabled: bool
    retrieval_poisoning_defense_enabled: bool
    tenant_security_verify_enabled: bool
    defense_plugin_ids: tuple[str, ...] = ()
    defense_bundle_ids: tuple[str, ...] = ()
    defense_middleware_names: tuple[str, ...] = ()
    encryption_enforcement_enabled: bool = False
    secrets_store_configured: bool = False
    require_secrets_store_for_encryption: bool = False


def _secrets_store_configured(env: ApplicationEnvironmentProfile) -> bool:
    profile = env.integration_profile
    if profile is None:
        return False
    slug = profile.slug_for_category(IntegrationCategory.SECRETS_STORE)
    return bool(slug and slug.strip())


def resolve_security_wiring_options(
    profile: ApplicationSecurityProfile,
    *,
    env: ApplicationEnvironmentProfile | None = None,
) -> SecurityWiringOptions:
    """Translate ``ApplicationSecurityProfile`` into host wiring flags."""
    plugins = resolve_security_defense_plugins(
        tuple(profile.defense_plugin_ids),
        tuple(profile.defense_bundle_ids),
    )
    defense_names = tuple(f"SecurityDefense:{plugin.plugin_id}" for plugin in plugins)
    secrets_configured = _secrets_store_configured(env) if env is not None else False
    return SecurityWiringOptions(
        prompt_defense_enabled=profile.prompt_defense_enabled,
        tool_injection_defense_enabled=profile.tool_injection_defense_enabled,
        retrieval_poisoning_defense_enabled=profile.retrieval_poisoning_defense_enabled,
        tenant_security_verify_enabled=profile.tenant_security_verify_enabled,
        defense_plugin_ids=tuple(profile.defense_plugin_ids),
        defense_bundle_ids=tuple(profile.defense_bundle_ids),
        defense_middleware_names=defense_names,
        encryption_enforcement_enabled=profile.encryption_enforcement_enabled,
        secrets_store_configured=secrets_configured,
        require_secrets_store_for_encryption=profile.require_secrets_store_for_encryption,
    )


def apply_security_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: ApplicationSecurityProfile,
) -> RuntimeConfig:
    """Record security posture on runtime config for downstream Nexus steps."""
    config.security_profile = profile
    return config


def apply_security_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared security profile and optional vendor guardrail (M-P12-WIRE.2)."""
    config = apply_security_profile_to_runtime_config(config, env.security_profile)
    from intergrax.applications._shared.guardrail_runtime_bridge import (
        resolve_guardrail_wiring_options,
    )

    guardrail = resolve_guardrail_wiring_options(env)
    if guardrail.enabled:
        config.metadata["llm_guardrail_slug"] = guardrail.backend_slug or ""
    return config

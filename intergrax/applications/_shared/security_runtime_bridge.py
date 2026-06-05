# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile security fields to wiring options (Phase SEC-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class SecurityWiringOptions:
    """Resolved security wiring flags for Tier-3 hosts."""

    prompt_defense_enabled: bool
    tool_injection_defense_enabled: bool
    retrieval_poisoning_defense_enabled: bool
    tenant_security_verify_enabled: bool


def resolve_security_wiring_options(
    profile: ApplicationSecurityProfile,
) -> SecurityWiringOptions:
    """Translate ``ApplicationSecurityProfile`` into host wiring flags."""
    return SecurityWiringOptions(
        prompt_defense_enabled=profile.prompt_defense_enabled,
        tool_injection_defense_enabled=profile.tool_injection_defense_enabled,
        retrieval_poisoning_defense_enabled=profile.retrieval_poisoning_defense_enabled,
        tenant_security_verify_enabled=profile.tenant_security_verify_enabled,
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
    """Apply environment-declared security profile."""
    return apply_security_profile_to_runtime_config(config, env.security_profile)

# © Artur Czarnecki. All rights reserved.

"""Tier-3 security wiring (Phase SEC-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.application_security_wiring import (
    register_application_security_hooks,
)
from intergrax.applications._shared.security_runtime_bridge import (
    SecurityWiringOptions,
    resolve_security_wiring_options,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def _enabled_middleware_names(options: SecurityWiringOptions) -> tuple[str, ...]:
    names: list[str] = []
    if options.prompt_defense_enabled:
        names.append("PromptDefenseMiddleware")
    if options.tool_injection_defense_enabled:
        names.append("ToolInjectionDefenseMiddleware")
    if options.tenant_security_verify_enabled:
        names.append("TenantSecurityMiddleware")
    return tuple(names)


@dataclass(frozen=True, slots=True)
class ApplicationSecurityWiring:
    """Resolved security artifacts for a Tier-3 host."""

    profile: ApplicationSecurityProfile
    options: SecurityWiringOptions
    enabled_middleware: tuple[str, ...]


def wire_application_security(
    env: ApplicationEnvironmentProfile,
) -> ApplicationSecurityWiring:
    """Resolve security profile and expected middleware from environment."""
    profile = env.security_profile
    options = resolve_security_wiring_options(profile)
    return ApplicationSecurityWiring(
        profile=profile,
        options=options,
        enabled_middleware=_enabled_middleware_names(options),
    )


def apply_application_security_wiring(
    nexus: NexusLoop,
    wiring: ApplicationSecurityWiring,
) -> None:
    """Attach V-SEC middleware to ``NexusLoop`` from resolved wiring."""
    register_application_security_hooks(nexus, wiring.profile)

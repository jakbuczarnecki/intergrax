# © Artur Czarnecki. All rights reserved.

"""Security assembly validation for Tier-3 hosts (Phase SEC-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.security_wiring import ApplicationSecurityWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.applications._shared.security_runtime_bridge import SecurityWiringOptions
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop


@dataclass(frozen=True, slots=True)
class SecurityAssemblyValidationResult:
    """Outcome of security assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class SecurityAssemblyError(ValueError):
    """Raised when security assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def _middleware_names_on_nexus(nexus: NexusLoop) -> frozenset[str]:
    pipeline = nexus._middleware  # noqa: SLF001 — assembly verification
    if not isinstance(pipeline, MiddlewarePipeline):
        return frozenset()
    return frozenset(middleware.name for middleware in pipeline._middleware)  # noqa: SLF001


def validate_security_wiring(
    wiring: ApplicationSecurityWiring,
    env: ApplicationEnvironmentProfile,
) -> SecurityAssemblyValidationResult:
    """Validate security wiring matches environment profile requirements."""
    errors: list[str] = []
    profile = env.security_profile
    options = wiring.options

    if options.prompt_defense_enabled != profile.prompt_defense_enabled:
        errors.append("prompt_defense_enabled mismatch between wiring and security_profile")
    if options.tool_injection_defense_enabled != profile.tool_injection_defense_enabled:
        errors.append("tool_injection_defense_enabled mismatch between wiring and security_profile")
    if options.retrieval_poisoning_defense_enabled != profile.retrieval_poisoning_defense_enabled:
        errors.append("retrieval_poisoning_defense_enabled mismatch between wiring and security_profile")
    if options.tenant_security_verify_enabled != profile.tenant_security_verify_enabled:
        errors.append("tenant_security_verify_enabled mismatch between wiring and security_profile")

    if env.identity_profile.tenant_required and not profile.tenant_security_verify_enabled:
        errors.append("identity_profile.tenant_required requires tenant_security_verify_enabled")

    expected = frozenset(wiring.enabled_middleware)
    if expected != frozenset(_expected_middleware_from_profile(profile, wiring.options)):
        errors.append("enabled_middleware must match ApplicationSecurityProfile toggles")

    if profile.require_secrets_store_for_encryption and not _secrets_store_configured(env):
        errors.append(
            "require_secrets_store_for_encryption requires integration_profile.secrets_store",
        )

    if (
        profile.encryption_enforcement_enabled
        and profile.require_secrets_store_for_encryption
        and not _secrets_store_configured(env)
        and env.execution_mode == ExecutionMode.STRICT
    ):
        errors.append("encryption enforcement requires secrets_store on strict hosts")

    for plugin_id in profile.defense_plugin_ids:
        if _resolve_defense_plugin(plugin_id) is None and env.execution_mode == ExecutionMode.STRICT:
            errors.append(f"unknown security defense plugin id: {plugin_id}")

    for bundle_id in profile.defense_bundle_ids:
        if _resolve_defense_plugin(bundle_id) is None:
            errors.append(f"unknown security defense bundle id: {bundle_id}")

    return SecurityAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def _secrets_store_configured(env: ApplicationEnvironmentProfile) -> bool:
    from intergrax.applications._shared.security_runtime_bridge import (
        _secrets_store_configured as _configured,
    )

    return _configured(env)


def _resolve_defense_plugin(plugin_id: str):
    from intergrax.runtime.security.defense_registry import get_security_defense_plugin

    return get_security_defense_plugin(plugin_id)


def _expected_middleware_from_profile(
    profile: ApplicationSecurityProfile,
    options: SecurityWiringOptions | None = None,
) -> tuple[str, ...]:
    if options is not None:
        from intergrax.applications._shared.security_wiring import _enabled_middleware_names

        return _enabled_middleware_names(options)
    names: list[str] = []
    if profile.encryption_enforcement_enabled:
        names.append("EncryptionEnforcementMiddleware")
    if profile.prompt_defense_enabled:
        names.append("PromptDefenseMiddleware")
    if profile.tool_injection_defense_enabled:
        names.append("ToolInjectionDefenseMiddleware")
    if profile.tenant_security_verify_enabled:
        names.append("TenantSecurityMiddleware")
    return tuple(names)


def validate_security_nexus_wiring(
    wiring: ApplicationSecurityWiring,
    nexus: NexusLoop,
) -> SecurityAssemblyValidationResult:
    """Validate Nexus middleware matches resolved security wiring."""
    errors: list[str] = []
    attached = _middleware_names_on_nexus(nexus)
    for name in wiring.enabled_middleware:
        if name not in attached:
            errors.append(f"missing security middleware on NexusLoop: {name}")
    return SecurityAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_security_assembly_valid(
    wiring: ApplicationSecurityWiring,
    env: ApplicationEnvironmentProfile,
    *,
    nexus: NexusLoop | None = None,
) -> None:
    """Raise :class:`SecurityAssemblyError` when security validation fails."""
    profile_result = validate_security_wiring(wiring, env)
    errors = list(profile_result.errors)
    if nexus is not None:
        nexus_result = validate_security_nexus_wiring(wiring, nexus)
        errors.extend(nexus_result.errors)
    if errors:
        raise SecurityAssemblyError(errors)

# © Artur Czarnecki. All rights reserved.

"""Security assembly validation for Tier-3 hosts (Phase SEC-2)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from intergrax.applications._shared.security_runtime_bridge import SecurityWiringOptions
from intergrax.applications._shared.security_wiring import ApplicationSecurityWiring
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationAssemblyInspectionTarget,
)
from intergrax.core.security.security_configuration import (
    AUDIT_REGIONS_INVALID,
    AUDIT_SINK_MISSING,
    DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
    PLUGIN_TRUST_POLICY_MISSING,
    RETRIEVAL_POISONING_PROFILE_MALFORMED,
    SIGNING_SECRET_ENV_INVALID,
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
    retrieval_poisoning_defense_enabled_or_default,
)

# Environment variable read by ``resolve_host_attestation_sealer_from_env``.
HOST_ATTESTATION_SIGNING_KEY_ENV = "INTERGRAX_EBE_HOST_SIGNING_KEY"


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


def _middleware_names_on_host(
    host: HostOrchestrationAssemblyInspectionTarget,
) -> frozenset[str]:
    return host.middleware.registered_middleware_names()


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
    # PDC-D2-R008 / PDC-D6-R014: the poisoning defense is ON unless the profile explicitly says False.
    if options.retrieval_poisoning_defense_enabled != retrieval_poisoning_defense_enabled_or_default(profile):
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

    from intergrax.runtime.security.defense_plugin import SecurityFailMode
    from intergrax.runtime.security.defense_registry import resolve_security_defense_plugins

    for plugin in resolve_security_defense_plugins(
        tuple(profile.defense_plugin_ids),
        tuple(profile.defense_bundle_ids),
    ):
        if plugin.fail_mode is not SecurityFailMode.FAIL_CLOSED:
            errors.append(
                f"security defense plugin {plugin.plugin_id!r} must use fail_mode=FAIL_CLOSED for host composition",
            )

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


def validate_security_host_wiring(
    wiring: ApplicationSecurityWiring,
    host: HostOrchestrationAssemblyInspectionTarget,
) -> SecurityAssemblyValidationResult:
    """Validate orchestration host middleware matches resolved security wiring."""
    errors: list[str] = []
    attached = _middleware_names_on_host(host)
    for name in wiring.enabled_middleware:
        if name not in attached:
            errors.append(f"missing security middleware on orchestration host: {name}")
    return SecurityAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_security_assembly_valid(
    wiring: ApplicationSecurityWiring,
    env: ApplicationEnvironmentProfile,
    *,
    orchestration_host: HostOrchestrationAssemblyInspectionTarget | None = None,
    nexus: HostOrchestrationAssemblyInspectionTarget | None = None,
) -> None:
    """Raise :class:`SecurityAssemblyError` when security validation fails."""
    profile_result = validate_security_wiring(wiring, env)
    errors = list(profile_result.errors)
    resolved_host = orchestration_host if orchestration_host is not None else nexus
    if resolved_host is not None:
        host_result = validate_security_host_wiring(wiring, resolved_host)
        errors.extend(host_result.errors)
    if errors:
        raise SecurityAssemblyError(errors)


def validate_strict_security_configuration(
    env: ApplicationEnvironmentProfile,
    *,
    environ: Mapping[str, str] | None = None,
    audit_evidence_persistence: EvidencePersistencePort | None = None,
    plugin_entry_points_enabled: bool = False,
    plugin_trust_policy: object | None = None,
) -> tuple[SecurityConfigurationError, ...]:
    """Collect missing or malformed security configuration for a product (strict) profile.

    Returns typed issues without raising so that callers can report all of them; use
    :func:`assert_strict_security_configuration` to fail closed. Lab profiles have no
    mandatory items here. Nothing is defaulted: each issue is a rejected item.
    """
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ()
    issues: list[SecurityConfigurationError] = []
    source = environ if environ is not None else os.environ

    identity = env.identity_profile
    if identity.critical_action_signing_enabled:
        secret_env = (identity.critical_action_signing_secret_env or "").strip()
        if not secret_env:
            issues.append(
                SecurityConfigurationError(
                    item="critical_action_signing",
                    reason_code=SIGNING_SECRET_ENV_INVALID,
                    message="critical_action_signing_secret_env is blank",
                )
            )
        else:
            value = source.get(secret_env)
            if value is None or not value.strip():
                issues.append(
                    SecurityConfigurationError(
                        item="critical_action_signing",
                        reason_code=SIGNING_SECRET_MISSING,
                        message=f"signing secret is not set in {secret_env!r}",
                    )
                )

    security = env.security_profile
    if not isinstance(security.retrieval_poisoning_defense_enabled, bool):
        issues.append(
            SecurityConfigurationError(
                item="retrieval_poisoning_defense",
                reason_code=RETRIEVAL_POISONING_PROFILE_MALFORMED,
                message="retrieval_poisoning_defense_enabled must be an explicit boolean",
            )
        )

    if security.immutable_audit_trail_enabled:
        regions = [r.strip() for r in security.audit_trail_regions if isinstance(r, str) and r.strip()]
        if len(set(regions)) < 2 or len(regions) != len(security.audit_trail_regions):
            issues.append(
                SecurityConfigurationError(
                    item="security_audit_trail",
                    reason_code=AUDIT_REGIONS_INVALID,
                    message="audit_trail_regions must list at least two distinct non-blank regions",
                )
            )
        if audit_evidence_persistence is None:
            issues.append(
                SecurityConfigurationError(
                    item="security_audit_trail",
                    reason_code=AUDIT_SINK_MISSING,
                    message="no EvidencePersistencePort supplied for the immutable audit trail",
                )
            )

    # PDC-D6-R027: host attestation signing falls back to the public PoC demo seed when this
    # variable is unset (runtime/attestation/host_attestation.py). A strict profile must
    # therefore supply the key explicitly; the assembly refuses the configuration otherwise.
    export_profile = env.execution_boundary_export_profile
    if export_profile is not None and export_profile.enabled and export_profile.host_signing_enabled:
        host_key = source.get(HOST_ATTESTATION_SIGNING_KEY_ENV)
        if host_key is None or not host_key.strip():
            issues.append(
                SecurityConfigurationError(
                    item="host_attestation",
                    reason_code=DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
                    message=(
                        f"host attestation signing requires {HOST_ATTESTATION_SIGNING_KEY_ENV}; "
                        "the demo signing seed is not available in a strict profile"
                    ),
                )
            )

    if plugin_entry_points_enabled and plugin_trust_policy is None:
        issues.append(
            SecurityConfigurationError(
                item="plugin_provenance",
                reason_code=PLUGIN_TRUST_POLICY_MISSING,
                message="plugin entry-point discovery requires an AgentPackageTrustPolicy",
            )
        )
    return tuple(issues)


def assert_strict_security_configuration(
    env: ApplicationEnvironmentProfile,
    *,
    environ: Mapping[str, str] | None = None,
    audit_evidence_persistence: EvidencePersistencePort | None = None,
    plugin_entry_points_enabled: bool = False,
    plugin_trust_policy: object | None = None,
) -> None:
    """Raise the first :class:`SecurityConfigurationError` for a strict profile (fail closed)."""
    issues = validate_strict_security_configuration(
        env,
        environ=environ,
        audit_evidence_persistence=audit_evidence_persistence,
        plugin_entry_points_enabled=plugin_entry_points_enabled,
        plugin_trust_policy=plugin_trust_policy,
    )
    if issues:
        raise issues[0]

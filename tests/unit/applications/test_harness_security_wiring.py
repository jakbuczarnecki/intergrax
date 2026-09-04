# © Artur Czarnecki. All rights reserved.

"""SEC-1/2: Security runtime bridge and assembly validation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications._shared.security_assembly_resolver import (
    SecurityAssemblyError,
    assert_security_assembly_valid,
    validate_security_wiring,
)
from intergrax.applications._shared.security_runtime_bridge import (
    resolve_security_wiring_options,
)
from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
    IdentityProfile,
)
from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_resolve_security_wiring_options_maps_profile_fields() -> None:
    profile = ApplicationSecurityProfile(
        prompt_defense_enabled=False,
        tool_injection_defense_enabled=True,
        retrieval_poisoning_defense_enabled=False,
        tenant_security_verify_enabled=True,
    )
    options = resolve_security_wiring_options(profile)
    assert options.prompt_defense_enabled is False
    assert options.tool_injection_defense_enabled is True
    assert options.retrieval_poisoning_defense_enabled is False
    assert options.tenant_security_verify_enabled is True


def test_wire_application_security_enabled_middleware_names() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.wire")
    wiring = wire_application_security(env)
    assert "PromptDefenseMiddleware" in wiring.enabled_middleware
    assert "ToolInjectionDefenseMiddleware" in wiring.enabled_middleware
    assert "TenantSecurityMiddleware" in wiring.enabled_middleware


def test_assert_security_assembly_valid_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.valid")
    wiring = wire_application_security(env)
    assert_security_assembly_valid(wiring, env)


def test_validate_security_wiring_requires_tenant_verify_when_tenant_required() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.tenant")
    env.identity_profile = IdentityProfile(tenant_required=True)
    env.security_profile = ApplicationSecurityProfile(tenant_security_verify_enabled=False)
    wiring = wire_application_security(env)
    result = validate_security_wiring(wiring, env)
    assert not result.valid
    assert any("tenant_security_verify_enabled" in error for error in result.errors)


def test_validate_security_wiring_rejects_mismatched_options() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.reject")
    wiring = wire_application_security(env)
    wiring = type(wiring)(
        profile=wiring.profile,
        options=type(wiring.options)(
            prompt_defense_enabled=not wiring.options.prompt_defense_enabled,
            tool_injection_defense_enabled=wiring.options.tool_injection_defense_enabled,
            retrieval_poisoning_defense_enabled=wiring.options.retrieval_poisoning_defense_enabled,
            tenant_security_verify_enabled=wiring.options.tenant_security_verify_enabled,
        ),
        enabled_middleware=wiring.enabled_middleware,
    )
    with pytest.raises(SecurityAssemblyError):
        assert_security_assembly_valid(wiring, env)


def test_materialize_runtime_config_applies_security_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.runtime")
    request = RuntimeRequest(
        message="hello",
        tenant_id="t1",
        agent_id="echo",
        user_id="user-1",
        session_id="session-1",
    )
    manifest = build_lab_manifest(LabApplicationSettings.from_env())
    build_ctx = ApplicationBuildContext.for_manifest(manifest, environment=env)
    config = materialize_runtime_config(request, build_ctx, env)
    assert config.security_profile is not None
    assert config.security_profile.prompt_defense_enabled == env.security_profile.prompt_defense_enabled


def test_build_harness_host_runtime_wires_security_middleware() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    pipeline = resolve_harness_host_nexus_loop_legacy(runtime)._middleware  # noqa: SLF001
    assert isinstance(pipeline, MiddlewarePipeline)
    names = {middleware.name for middleware in pipeline._middleware}  # noqa: SLF001
    for middleware_name in runtime.security.enabled_middleware:
        assert middleware_name in names

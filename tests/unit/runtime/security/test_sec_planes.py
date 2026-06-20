# © Artur Czarnecki. All rights reserved.

"""SEC-PLANES: security defense plugins, encryption enforcement, assembly."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.security_assembly_resolver import (
    SecurityAssemblyError,
    assert_security_assembly_valid,
    validate_security_wiring,
)
from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ApplicationSecurityProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.data_classification import DataClassification
from intergrax.integrations.registry.presets import harness_defense_stack
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.security.defense_registry import (
    get_security_defense_plugin,
    register_security_defense_plugin,
    reset_security_defense_registry_for_tests,
    resolve_security_defense_plugins,
)
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
from intergrax.runtime.security.encryption_policy import evaluate_encryption_enforcement
from intergrax.applications._shared.application_security_wiring import (
    register_application_security_hooks,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _BlockToolPlugin:
    plugin_id = "lab.block_tool"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 57
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(
            allowed=False,
            reasons=["lab block"],
            plugin_id=self.plugin_id,
            hook_point=point.value,
        )


@pytest.fixture(autouse=True)
def _reset_defense_registry() -> None:
    reset_security_defense_registry_for_tests()


def test_shipped_strict_injection_bundle_registered() -> None:
    plugin = get_security_defense_plugin("harness.strict_injection")
    assert plugin is not None
    assert HookPoint.BEFORE_TOOL_CALL in plugin.hook_points


def test_resolve_defense_plugins_from_bundle_ids() -> None:
    plugins = resolve_security_defense_plugins((), ("harness.strict_injection",))
    assert len(plugins) == 1
    assert plugins[0].plugin_id == "harness.strict_injection"


@pytest.mark.asyncio
async def test_plugin_middleware_blocks_before_tool_call() -> None:
    register_security_defense_plugin(_BlockToolPlugin())
    middleware = PluginSecurityDefenseMiddleware(_BlockToolPlugin())
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"tool_id": "echo", "arguments": {"q": "x"}},
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action.value == "block"


def test_wire_application_security_includes_defense_middleware_name() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.defense")
    env.security_profile = ApplicationSecurityProfile(
        defense_bundle_ids=["harness.strict_injection"],
    )
    wiring = wire_application_security(env)
    assert "SecurityDefense:harness.strict_injection" in wiring.enabled_middleware


def test_encryption_policy_blocks_restricted_without_secrets_backend() -> None:
    decision = evaluate_encryption_enforcement(
        payload={"value": {"data_classification": DataClassification.RESTRICTED.value}},
        secrets_store_configured=False,
        enforcement_enabled=True,
    )
    assert not decision.allowed


@pytest.mark.asyncio
async def test_encryption_middleware_blocks_memory_write() -> None:
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=False,
    )
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={
            "value": {"data_classification": "restricted", "secret": "x"},
        },
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "block"


def test_strict_host_rejects_unknown_defense_plugin_id() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.unknown")
    env.meta = env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env.security_profile = ApplicationSecurityProfile(defense_plugin_ids=["missing.plugin"])
    wiring = wire_application_security(env)
    result = validate_security_wiring(wiring, env)
    assert not result.valid


def test_register_application_security_hooks_attaches_defense_plugin() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.hooks")
    env.security_profile = ApplicationSecurityProfile(
        prompt_defense_enabled=False,
        tool_injection_defense_enabled=False,
        tenant_security_verify_enabled=False,
        defense_bundle_ids=["harness.strict_injection"],
    )
    wiring = wire_application_security(env)
    nexus = NexusLoop(AgentRegistry())
    register_application_security_hooks(nexus, wiring.profile, options=wiring.options)
    pipeline = nexus._middleware  # noqa: SLF001
    assert isinstance(pipeline, MiddlewarePipeline)
    names = {item.name for item in pipeline._middleware}  # noqa: SLF001
    assert "SecurityDefense:harness.strict_injection" in names


def test_harness_defense_stack_includes_secrets_store() -> None:
    register_default_integrations()
    profile = harness_defense_stack()
    assert profile.secrets_store is not None


def test_production_security_envelope_requires_secrets_when_configured() -> None:
    from intergrax.applications.contracts.environment_profile.bundles import SecurityEnvelope

    envelope = SecurityEnvelope.production()
    assert envelope.application_security.encryption_enforcement_enabled is True
    assert envelope.application_security.defense_bundle_ids == ["harness.strict_injection"]


def test_assert_security_assembly_valid_with_defense_bundle() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.bundle.valid")
    env.security_profile = ApplicationSecurityProfile(
        defense_bundle_ids=["harness.strict_injection"],
        prompt_defense_enabled=False,
        tool_injection_defense_enabled=False,
        tenant_security_verify_enabled=False,
    )
    wiring = wire_application_security(env)
    assert_security_assembly_valid(wiring, env)


def test_strict_encryption_requires_secrets_store_integration() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.enc.strict")
    env.meta = env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    env.security_profile = ApplicationSecurityProfile(
        encryption_enforcement_enabled=True,
        require_secrets_store_for_encryption=True,
        prompt_defense_enabled=False,
        tool_injection_defense_enabled=False,
        tenant_security_verify_enabled=False,
    )
    wiring = wire_application_security(env)
    with pytest.raises(SecurityAssemblyError):
        assert_security_assembly_valid(wiring, env)

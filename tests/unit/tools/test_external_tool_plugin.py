# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.examples.custom_echo import CustomEchoToolPlugin
from intergrax.tools.examples.custom_echo.plugin import CUSTOM_ECHO_TOOL_ID, CustomEchoInput
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_external_tool_plugin_invokes_via_runtime_invoker() -> None:
    register_tool_plugin(CustomEchoToolPlugin)
    registry = build_registry_from_profile(
        ToolProfile(enabled_bundles=["custom_echo"]),
        ctx=ToolWiringContext(),
    )
    assert registry.has(CUSTOM_ECHO_TOOL_ID)

    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    run_id = canonical_run_id_for_tests("custom_echo_run")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                step_id="step/1",
                tool_id=CUSTOM_ECHO_TOOL_ID,
                input=CustomEchoInput(message="hello"),
            ),
        )
    assert result.success is True
    assert result.output is not None
    assert result.output.message == "hello"


def test_external_side_effect_plugin_requires_enforcing_authorization() -> None:
    from pydantic import BaseModel

    from intergrax.applications._shared.policy_wiring import wire_policy_bundle
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        PolicyRulesProfile,
    )
    from intergrax.runtime.nexus.errors.meaningful_side_effect_authorization_error import (
        MeaningfulSideEffectAuthorizationRequiredError,
    )
    from intergrax.tools.core.contracts import ToolContract

    class PluginInput(BaseModel):
        value: int

    class PluginOutput(BaseModel):
        value: int

    class PluginHandler:
        def execute(self, request: ToolExecutionRequest[PluginInput]) -> PluginOutput:
            return PluginOutput(value=request.input.value)

    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id="plugin.side_effect.echo",
            name="plugin.side_effect.echo",
            description="side effect plugin",
            input_schema=PluginInput,
            output_schema=PluginOutput,
            error_mapping={},
            side_effects=True,
        ),
        handler=PluginHandler(),
    )
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    run_id = canonical_run_id_for_tests("plugin_side_effect_run")
    state = build_runtime_state_for_tests(run_id=run_id)
    request = ToolExecutionRequest(
        run_id=run_id,
        step_id="step/1",
        tool_id="plugin.side_effect.echo",
        input=PluginInput(value=7),
    )

    with canonical_execution_identity_scope(run_id):
        with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError):
            invoker.invoke(state=state, agent_id="agent", request=request)

    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="plugin.side_effect.allow")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    allowed_state = build_runtime_state_for_tests(run_id=run_id)
    allowed_state.context.config.policy_bundle = wire_policy_bundle(allow_env)

    with canonical_execution_identity_scope(run_id):
        allowed = invoker.invoke(state=allowed_state, agent_id="agent", request=request)

    assert allowed.success is True
    assert allowed.output is not None
    assert allowed.output.value == 7

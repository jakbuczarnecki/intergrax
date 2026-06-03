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
from testing_support.builder import build_runtime_state_for_tests

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
    state = build_runtime_state_for_tests(run_id="custom_echo_run")
    result = invoker.invoke(
        state=state,
        agent_id="agent",
        request=ToolExecutionRequest(
            run_id="custom_echo_run",
            step_id="step/1",
            tool_id=CUSTOM_ECHO_TOOL_ID,
            input=CustomEchoInput(message="hello"),
        ),
    )
    assert result.success is True
    assert result.output is not None
    assert result.output.message == "hello"

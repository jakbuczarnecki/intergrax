# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-0 / TOOL-ENG-3: catalog planner bootstrap and scope policy wiring."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.catalog_runtime_bridge import (
    apply_tool_engine_settings_from_environment,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.planner_bootstrap import wire_catalog_tool_planner_if_enabled
from intergrax.runtime.tools.idempotent_invoker import IdempotentToolInvoker
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


class _Handler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value + 1)


class _TraceState:
    run_id = "run-bootstrap"
    tenant_id = "tenant-bootstrap"

    def trace_event(self, **kwargs: object) -> None:
        pass


def _register_dummy(registry: ToolRegistry, tool_id: str = "dummy.tool") -> None:
    contract = tools_agent_make_contract(tool_id, _In, _Out)
    registry.register(contract, _Handler())


def test_apply_tool_engine_settings_from_environment_sets_planner_prompt_id() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reasoning_profile": ReasoningProfile(tool_planner_prompt_id="custom_planner"),
        }
    )
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_tool_engine_settings_from_environment(config, env)

    assert config.tool_planner_prompt_id == "custom_planner"


def test_wire_catalog_tool_planner_when_tools_enabled() -> None:
    registry = ToolRegistry()
    _register_dummy(registry)
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tools_mode="auto",
        tool_profile=None,
    )

    wire_catalog_tool_planner_if_enabled(config, registry)

    assert isinstance(config.tool_planner, CatalogToolPlanner)


def test_wire_catalog_tool_planner_skips_when_tools_mode_off() -> None:
    registry = ToolRegistry()
    _register_dummy(registry)
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tools_mode="off",
    )

    wire_catalog_tool_planner_if_enabled(config, registry)

    assert config.tool_planner is None


def test_runtime_context_build_wires_planner_and_scope_policy() -> None:
    from intergrax.tools.registry.profile import ToolProfile

    scope = StaticToolScopePolicy(allowed_tools={"interaction.list_sessions"})
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_mode="auto",
        tool_scope_policy=scope,
        tool_profile=ToolProfile(enabled_bundles=("interaction",)),
    )

    ctx = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )

    assert isinstance(ctx.config.tool_planner, CatalogToolPlanner)
    invoker = ctx.config.tool_invoker
    assert invoker is not None
    base = invoker._base_invoker if isinstance(invoker, IdempotentToolInvoker) else invoker
    assert isinstance(base, RuntimeToolInvoker)
    assert base._scope_policy is scope


def test_runtime_context_scope_policy_denies_on_invoke_path() -> None:
    from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor

    registry = ToolRegistry()
    _register_dummy(registry, "forbidden.tool")
    scope = StaticToolScopePolicy(allowed_tools=set())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        scope_policy=scope,
    )

    request = ToolExecutionRequest(
        run_id="run-deny",
        step_id="tools",
        tool_id="forbidden.tool",
        input=_In(value=1),
    )

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=_TraceState(), agent_id="agent", request=request)

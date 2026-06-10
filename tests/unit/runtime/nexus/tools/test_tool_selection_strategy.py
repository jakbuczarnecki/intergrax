# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-5: ToolSelectionStrategy and planner subset resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.catalog_runtime_bridge import (
    apply_tool_engine_settings_from_environment,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.catalog_dispatch import resolve_tool_registry
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_selection import (
    RetrievalTopKSelectionStrategy,
    SkillPackSelectionStrategy,
    StaticAllowListSelectionStrategy,
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.skills.registry import SkillProfile
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry, build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _Handler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


def _registry_with_tools(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(tools_agent_make_contract(tool_id, _In, _Out), _Handler())
    return registry


def test_static_strategy_uses_plan_allow_list() -> None:
    ctx = ToolSelectionContext(
        registry=_registry_with_tools("alpha.tool", "beta.tool"),
        query="find alpha",
        plan_allowed_tool_ids=("alpha.tool",),
    )
    strategy = StaticAllowListSelectionStrategy()
    assert strategy.select_tool_ids(ctx) == ("alpha.tool",)


def test_resolve_planner_allowed_tool_ids_intersects_plan_and_skill_pack() -> None:
    bootstrap_catalogs(register_shipped=True, skill_bundle_ids=("harness",))
    tool_registry = ToolRegistry()
    from intergrax.tools.registry.bootstrap import register_default_tools

    register_default_tools()
    build_registry_from_profile(
        ToolProfile(enabled_bundles=("interaction", "harness", "catalog")),
        registry=tool_registry,
    )
    skill_profile = SkillProfile(enabled_bundles=["harness"])

    ctx = ToolSelectionContext(
        registry=tool_registry,
        query="list tools",
        skill_profile=skill_profile,
        plan_allowed_tool_ids=("catalog.list_tools", "missing.tool"),
    )
    resolved = resolve_planner_allowed_tool_ids(ToolSelectionMode.SKILL_PACK, ctx)

    assert resolved is not None
    assert "catalog.list_tools" in resolved
    assert "missing.tool" not in resolved


def test_retrieval_top_k_prefers_matching_metadata() -> None:
    from intergrax.tools.core.contracts import ToolContract

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="jira.search_tasks",
            name="jira.search_tasks",
            input_schema=_In,
            output_schema=_Out,
            description="Search Jira issues by project",
            tags=("jira", "search"),
            error_mapping={},
            side_effects=False,
        ),
        _Handler(),
    )
    registry.register(
        ToolContract(
            tool_id="workspace.read_file",
            name="workspace.read_file",
            input_schema=_In,
            output_schema=_Out,
            description="Read workspace file",
            error_mapping={},
            side_effects=False,
        ),
        _Handler(),
    )

    ctx = ToolSelectionContext(registry=registry, query="search jira project", top_k=1)
    selected = RetrievalTopKSelectionStrategy().select_tool_ids(ctx)

    assert selected == ("jira.search_tasks",)


def test_apply_tool_engine_settings_sets_lab_skill_pack_mode() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_tool_engine_settings_from_environment(config, env)

    assert config.tool_selection_mode == ToolSelectionMode.SKILL_PACK
    assert config.tool_selection_top_k == 20


@pytest.mark.asyncio
async def test_tools_step_uses_skill_pack_subset() -> None:
    bootstrap_catalogs(register_shipped=True, skill_bundle_ids=("harness",))
    tool_registry = ToolRegistry()
    from intergrax.tools.registry.bootstrap import register_default_tools

    register_default_tools()
    build_registry_from_profile(
        ToolProfile(enabled_bundles=("interaction", "harness", "catalog")),
        registry=tool_registry,
    )
    invoker = RuntimeToolInvoker(
        registry=tool_registry,
        executor=RegistryToolExecutor(tool_registry),
    )

    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_invoker=invoker,
        tool_selection_mode=ToolSelectionMode.SKILL_PACK,
        skill_profile=SkillProfile(enabled_bundles=["harness"]),
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    state = RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="list catalog tools",
        ),
        run_id="run-selection",
    )

    planner = MagicMock()
    planner.plan_tools = MagicMock(return_value=MagicMock(tool_plan=None))
    config.tool_planner = planner
    config.tools_mode = "auto"

    await ToolsStep().run(state)

    allowed = planner.plan_tools.call_args.kwargs["allowed_tool_ids"]
    assert allowed is not None
    assert "catalog.list_tools" in allowed
    assert "interaction.list_sessions" not in allowed

# © Artur Czarnecki. All rights reserved.

"""LKW.1.11: ApplicationToolWiring.registry parity with runtime gateway invoker."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.runtime_config_bridge import (
    build_runtime_context_from_environment,
    materialize_runtime_config,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.tool_request import ToolRequest
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.catalog_dispatch import (
    is_registered_catalog_tool,
    resolve_tool_registry,
)
from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import FakeLLMAdapter, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]

PARITY_TOOL_ID = "catalog.registry_parity_probe"


class _ParityIn(BaseModel):
    query: str = "probe"


class _ParityOut(BaseModel):
    ok: bool = True


class _ParityHandler(ToolHandler[_ParityIn, _ParityOut]):
    async def execute(self, input: _ParityIn) -> _ParityOut:
        return _ParityOut(ok=True)


def _wired_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract(PARITY_TOOL_ID, _ParityIn, _ParityOut),
        _ParityHandler(),
    )
    return registry


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-parity",
        agent_id="agent-parity",
        user_id="user-parity",
        session_id="session-parity",
        message="registry parity probe",
    )


def _build_context(registry: ToolRegistry) -> ApplicationBuildContext:
    tool_profile = ToolProfile(enabled=(PARITY_TOOL_ID,))
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="registry.parity")
    return ApplicationBuildContext.for_manifest(
        object(),
        tool_profile=tool_profile,
        tool_registry=registry,
        policy_bundle=RuntimePolicyBundle(),
        environment=env,
    )


@pytest.mark.asyncio
async def test_runtime_gateway_uses_wired_application_tool_registry() -> None:
    registry = _wired_registry()
    build_ctx = _build_context(registry)
    env = build_ctx.environment
    assert env is not None

    config = materialize_runtime_config(
        _request(),
        build_ctx,
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    assert config.tool_registry is registry

    runtime_ctx = build_runtime_context_from_environment(
        _request(),
        build_ctx,
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    runtime_registry = resolve_tool_registry(runtime_ctx.config.tool_invoker)
    assert runtime_registry is registry
    assert is_registered_catalog_tool(runtime_registry, PARITY_TOOL_ID)

    state = RuntimeState(
        context=runtime_ctx,
        request=_request(),
        run_id="run-registry-parity",
        tool_traces=[],
    )
    gateway = RuntimeToolGateway.for_state(state, allowed_tools=[PARITY_TOOL_ID])
    response = await gateway.invoke(
        ToolRequest(
            tool_name=PARITY_TOOL_ID,
            agent_id="agent-parity",
            step_id="parity",
            input={"query": "probe"},
        )
    )

    assert not (response.error or "").startswith("unknown_capability_tool:")

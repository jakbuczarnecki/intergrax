# © Artur Czarnecki. All rights reserved.
"""T-IGX-076-A: MCP composition inherits host wiring_fails_closed posture."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.applications._shared.mcp_harness_principal import McpIdentityCompositionError
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    nexus_legacy_server_definition,
)
from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = pytest.mark.unit

_LOCAL = McpIdentityMode.LOCAL_DEV_EXPLICIT


def _strict_fails_closed() -> bool:
    env = ApplicationEnvironmentProfile(
        application_profile=ApplicationProfile.LAB,
        execution_mode=ExecutionMode.STRICT,
    )
    return wiring_fails_closed(env)


def _host() -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    return port


def _compose_kwargs(*, fails_closed: bool) -> dict[str, object]:
    return {
        "display_name": "posture-mcp",
        "host_execution": _host(),
        "registry": AgentRegistry(),
        "default_capability": "demo.cap",
        "identity_mode": _LOCAL,
        "fails_closed": fails_closed,
    }


def test_wiring_fails_closed_strict_profile() -> None:
    assert _strict_fails_closed() is True


def test_compose_application_mcp_server_refuses_local_when_fails_closed() -> None:
    with pytest.raises(McpIdentityCompositionError):
        compose_application_mcp_server(**_compose_kwargs(fails_closed=True))


def test_build_nexus_mcp_server_refuses_local_when_fails_closed() -> None:
    with pytest.raises(McpIdentityCompositionError):
        build_nexus_mcp_server(
            name="posture-nexus",
            host_execution=_host(),
            registry=AgentRegistry(),
            default_capability="demo.cap",
            identity_mode=_LOCAL,
            fails_closed=True,
        )


def test_compose_mcp_server_from_definition_refuses_local_when_fails_closed() -> None:
    definition = nexus_legacy_server_definition(
        server_id="posture-engine",
        display_name="posture",
        default_capability="demo.cap",
        default_tenant_id="mcp",
        default_user_id="mcp-user",
        catalog_tool_ids=(),
        include_catalog_surface=False,
    )
    deps = McpServerCompositionDeps(
        host_execution=_host(),
        registry=AgentRegistry(),
        identity_mode=_LOCAL,
        fails_closed=True,
    )
    with pytest.raises(McpIdentityCompositionError):
        compose_mcp_server_from_definition(definition, deps)


def test_strict_posture_wiring_refuses_local_via_engine() -> None:
    definition = nexus_legacy_server_definition(
        server_id="posture-engine-strict",
        display_name="posture",
        default_capability="demo.cap",
        default_tenant_id="mcp",
        default_user_id="mcp-user",
        catalog_tool_ids=(),
        include_catalog_surface=False,
    )
    deps = McpServerCompositionDeps(
        host_execution=_host(),
        registry=AgentRegistry(),
        identity_mode=_LOCAL,
        fails_closed=_strict_fails_closed(),
    )
    with pytest.raises(McpIdentityCompositionError):
        compose_mcp_server_from_definition(definition, deps)

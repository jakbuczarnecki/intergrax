# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.declarative_tool_executor import CallableDeclarativeToolInvoker
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvokeResult
from intergrax.agents.persistence.tool_invoker_wiring import (
    attach_declarative_tool_invoker,
    inject_acp_tool_invoker_metadata,
    resolve_declarative_tool_invoker_from_metadata,
    wire_acp_run_request_with_tool_invoker,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_metadata_roundtrip_for_declarative_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    metadata = attach_declarative_tool_invoker({}, invoker)
    resolved = resolve_declarative_tool_invoker_from_metadata(metadata)
    assert resolved is invoker


def test_wire_acp_run_request_attaches_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="t1"),
    )
    wired = wire_acp_run_request_with_tool_invoker(request, invoker)
    assert wired.metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] is invoker


def test_inject_acp_tool_invoker_metadata_wires_host_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    metadata: dict[str, object] = {}
    inject_acp_tool_invoker_metadata(
        metadata,
        invoker,
        run_id="run-1",
        agent_id="agent-1",
        tenant_id="tenant-1",
    )
    assert metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] is invoker

# © Artur Czarnecki. All rights reserved.

"""MCP-RC MRC-2 — one bounded-result choke point and one canonical wire-error contract (R-P7-03, R-P7-05)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.mcp_catalog_invocation_bridge import _bound_error_payload
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_wire_errors import (
    bound_cross_loop_catalog_payload,
    mcp_code_for_legacy,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.mcp_result_bounds import (
    DEFAULT_MCP_MAX_SERIALIZED_BYTES,
    mcp_serialized_size_exceeds,
)
from intergrax.tools.registry.runtime import ToolRegistry
from pydantic import BaseModel
from tests.qualification.mcp_server_x.reference_app.build import (
    build_mcp_server_x_reference_qualification_server,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MAX = DEFAULT_MCP_MAX_SERIALIZED_BYTES
_CODE = McpServerDefinitionErrorCode


def _host(answer: str = "ok") -> AsyncMock:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer=answer,
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )
    )
    return port


def _server(*, host: AsyncMock | None = None, echo=None, tool_registry: ToolRegistry | None = None):
    return build_mcp_server_x_reference_qualification_server(
        host_execution=host or _host(),
        registry=AgentRegistry(),
        custom_echo=echo,
        tool_registry=tool_registry,
        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,
    )


# --- bounded size primitive -------------------------------------------------------------------------


def test_size_check_is_exact_at_the_boundary() -> None:
    # json for a bare string is the value plus two quotes.
    fits = "x" * (_MAX - 2)
    assert mcp_serialized_size_exceeds(fits, max_bytes=_MAX) is False
    assert mcp_serialized_size_exceeds(fits + "x", max_bytes=_MAX) is True


def test_size_check_counts_escaped_multibyte_characters() -> None:
    assert mcp_serialized_size_exceeds("é" * 200, max_bytes=800) is True
    assert mcp_serialized_size_exceeds("é" * 100, max_bytes=800) is False


def test_size_check_aborts_without_materializing_the_whole_value() -> None:
    produced = 0

    class _Endless:
        def __iter__(self):
            nonlocal produced
            while True:
                produced += 1
                yield "x" * 1024

    class _Wrapper(list):
        def __len__(self) -> int:
            return 1  # non-empty so the encoder iterates

        def __iter__(self):
            return iter(_Endless())

    assert mcp_serialized_size_exceeds(_Wrapper(), max_bytes=64 * 1024) is True
    assert produced < 200


# --- every result-producing path is bounded with the declared code --------------------------------


@pytest.mark.asyncio
async def test_custom_host_tool_result_over_the_bound_is_a_typed_error() -> None:
    async def _huge(message: str = "") -> dict[str, object]:
        return {"echo": "x" * (_MAX + 10)}

    async with Client(_server(echo=_huge)) as client:
        with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
            await client.call_tool("qual_echo", {"message": "m"})


@pytest.mark.asyncio
async def test_custom_host_tool_nested_payload_over_the_bound_is_a_typed_error() -> None:
    async def _nested(message: str = "") -> dict[str, object]:
        return {"a": {"b": [{"c": "y" * 1000} for _ in range(2000)]}}

    async with Client(_server(echo=_nested)) as client:
        with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
            await client.call_tool("qual_echo", {"message": "m"})


@pytest.mark.asyncio
async def test_custom_host_tool_result_within_the_bound_is_unchanged() -> None:
    async with Client(_server()) as client:
        result = await client.call_tool("qual_echo", {"message": "small"})
    assert result.is_error is False
    assert result.data["echo"] == "small"


@pytest.mark.asyncio
async def test_run_agent_answer_over_the_bound_is_is_error_true_not_a_normal_result() -> None:
    server = _server(host=_host("y" * (_MAX + 1024)))
    async with Client(server) as client:
        with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
            await client.call_tool("run_agent", {"message": "x"})


@pytest.mark.asyncio
async def test_concurrent_oversize_calls_all_fail_typed_and_the_server_stays_usable() -> None:
    async def _huge(message: str = "") -> dict[str, object]:
        return {"echo": "x" * (_MAX + 10)}

    async with Client(_server(echo=_huge)) as client:
        outcomes = await asyncio.gather(
            *(client.call_tool("qual_echo", {"message": str(i)}) for i in range(6)),
            return_exceptions=True,
        )
        assert all(isinstance(o, ToolError) and "MCP_RESULT_BOUND_EXCEEDED" in str(o) for o in outcomes)
        listed = await client.call_tool("list_agents", {})
        assert listed.is_error is False


@pytest.mark.asyncio
async def test_custom_tool_exception_text_is_not_echoed_unbounded_on_the_wire() -> None:
    async def _boom(message: str = "") -> dict[str, object]:
        raise RuntimeError("secret-path C:/internal " + "z" * 200_000)

    async with Client(_server(echo=_boom)) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("qual_echo", {"message": "m"})
    text = str(excinfo.value)
    assert "MCP_INTERNAL_ERROR" in text
    assert "secret-path" not in text
    assert len(text) < 2048


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _registry_with_big_description() -> ToolRegistry:
    async def _handler(_input: _In, _ctx: object) -> _Out:
        return _Out()

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="rc.big",
            name="rc.big",
            description="d" * (_MAX + 1024),
            input_schema=_In,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="rc",
        ),
        _handler,
    )
    return registry


@pytest.mark.asyncio
async def test_describe_catalog_tool_over_the_bound_is_a_typed_error() -> None:
    server = _server(tool_registry=_registry_with_big_description())
    async with Client(server) as client:
        with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
            await client.call_tool("describe_catalog_tool", {"tool_id": "rc.big"})


@pytest.mark.asyncio
async def test_list_catalog_tools_over_the_bound_is_a_typed_error_never_an_empty_list() -> None:
    server = _server(tool_registry=_registry_with_big_description())
    async with Client(server) as client:
        with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
            await client.call_tool("list_catalog_tools", {})


@pytest.mark.asyncio
async def test_unknown_tool_error_does_not_echo_unbounded_caller_input() -> None:
    server = _server(tool_registry=_registry_with_big_description())
    async with Client(server) as client:
        result_or_error = await asyncio.gather(
            client.call_tool("describe_catalog_tool", {"tool_id": "q" * 100_000}),
            return_exceptions=True,
        )
    outcome = result_or_error[0]
    text = str(outcome) if isinstance(outcome, BaseException) else str(outcome.data)
    assert len(text) < 4096


# --- canonical code map -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("unauthorized", _CODE.MCP_PRINCIPAL_REQUIRED),
        ("forbidden", _CODE.MCP_PRINCIPAL_REQUIRED),
        ("missing_caller_identity", _CODE.MCP_PRINCIPAL_REQUIRED),
        ("identity_conflict", _CODE.MCP_TENANT_ASSERTION_CONFLICT),
        ("wrong_tenant", _CODE.MCP_TENANT_ASSERTION_CONFLICT),
        ("mcp_payload_byte_limit_exceeded", _CODE.MCP_RESULT_BOUND_EXCEEDED),
        ("invalid_task_id", _CODE.MCP_EXECUTION_BINDING_INVALID),
        ("missing_execution_binding", _CODE.MCP_EXECUTION_BINDING_INVALID),
        ("stale_run_binding", _CODE.MCP_EXECUTION_BINDING_INVALID),
        ("stale_step_binding", _CODE.MCP_EXECUTION_BINDING_INVALID),
        ("unauthorized_execution_caller", _CODE.MCP_CALLER_NOT_AUTHORIZED),
        ("unknown_tool", _CODE.MCP_CATALOG_TOOL_REFUSED),
        ("tool_disabled", _CODE.MCP_CATALOG_TOOL_REFUSED),
        ("invalid_tool_input", _CODE.MCP_TOOL_INPUT_INVALID),
        ("invalid_tool_contract", _CODE.MCP_TOOL_INPUT_INVALID),
        ("execution_terminal", _CODE.MCP_EXECUTION_REFUSED),
        ("execution_cancelled", _CODE.MCP_EXECUTION_REFUSED),
        ("execution_admission_denied", _CODE.MCP_EXECUTION_REFUSED),
        ("mcp_concurrent_call_limit_exceeded", _CODE.MCP_CONCURRENT_CALL_LIMIT_EXCEEDED),
        ("internal_error", _CODE.MCP_INTERNAL_ERROR),
        ("something_new_and_unmapped", _CODE.MCP_INTERNAL_ERROR),
    ],
)
def test_legacy_codes_map_to_declared_codes(legacy: str, expected: McpServerDefinitionErrorCode) -> None:
    assert mcp_code_for_legacy(legacy) is expected


def test_bridge_error_payloads_carry_the_canonical_code_and_keep_the_legacy_string() -> None:
    payload = _bound_error_payload("execution_cancelled", "cooperative cancel requested")
    assert payload["error"] == "execution_cancelled"
    assert payload["mcp_code"] == "MCP_EXECUTION_REFUSED"


def test_cross_loop_payload_is_bounded_and_does_not_leak_exception_text() -> None:
    leaked = bound_cross_loop_catalog_payload({"error": "internal_error", "detail": "C:/secret/path boom"})
    assert leaked["error"] == "internal_error"
    assert "secret" not in str(leaked)
    assert leaked["mcp_code"] == "MCP_INTERNAL_ERROR"
    with pytest.raises(ToolError, match="MCP_RESULT_BOUND_EXCEEDED"):
        bound_cross_loop_catalog_payload({"success": True, "output": {"blob": "x" * (_MAX + 10)}})
    ok = {"success": True, "output": {"v": 1}}
    assert bound_cross_loop_catalog_payload(ok) == ok

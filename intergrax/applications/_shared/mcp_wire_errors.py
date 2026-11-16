# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single owner of the MCP wire-error contract and of the bounded-result choke point.

Wire shape: ``"<MCP_CODE>: <detail>"`` carried by ``ToolError`` (``isError=true`` on the protocol). Legacy
lowercase codes that earlier runtime paths emitted are mapped to declared ``MCP_*`` codes and kept in the
detail (contract v1.1, OD-MCP-RC-01 D-4).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastmcp.exceptions import ToolError

from intergrax.applications._shared.mcp_engine_concurrency import McpConcurrentToolCallLimitExceeded
from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode
from intergrax.tools.mcp_result_bounds import (
    DEFAULT_MCP_MAX_SERIALIZED_BYTES,
    mcp_serialized_size_exceeds,
)

_C = McpServerDefinitionErrorCode
_LOG = logging.getLogger(__name__)

LEGACY_WIRE_CODE_MAP: Mapping[str, McpServerDefinitionErrorCode] = {
    "unauthorized": _C.MCP_PRINCIPAL_REQUIRED,
    "forbidden": _C.MCP_PRINCIPAL_REQUIRED,
    "missing_caller_identity": _C.MCP_PRINCIPAL_REQUIRED,
    "identity_conflict": _C.MCP_TENANT_ASSERTION_CONFLICT,
    "wrong_tenant": _C.MCP_TENANT_ASSERTION_CONFLICT,
    "mcp_payload_byte_limit_exceeded": _C.MCP_RESULT_BOUND_EXCEEDED,
    "invalid_task_id": _C.MCP_EXECUTION_BINDING_INVALID,
    "invalid_run_id": _C.MCP_EXECUTION_BINDING_INVALID,
    "missing_execution_binding": _C.MCP_EXECUTION_BINDING_INVALID,
    "stale_run_binding": _C.MCP_EXECUTION_BINDING_INVALID,
    "stale_step_binding": _C.MCP_EXECUTION_BINDING_INVALID,
    "unauthorized_execution_caller": _C.MCP_CALLER_NOT_AUTHORIZED,
    "unknown_tool": _C.MCP_CATALOG_TOOL_REFUSED,
    "tool_disabled": _C.MCP_CATALOG_TOOL_REFUSED,
    "invalid_tool_input": _C.MCP_TOOL_INPUT_INVALID,
    "invalid_tool_contract": _C.MCP_TOOL_INPUT_INVALID,
    "execution_terminal": _C.MCP_EXECUTION_REFUSED,
    "execution_cancelled": _C.MCP_EXECUTION_REFUSED,
    "execution_admission_denied": _C.MCP_EXECUTION_REFUSED,
    "mcp_concurrent_call_limit_exceeded": _C.MCP_CONCURRENT_CALL_LIMIT_EXCEEDED,
    "internal_error": _C.MCP_INTERNAL_ERROR,
}


def mcp_code_for_legacy(legacy_code: str) -> McpServerDefinitionErrorCode:
    """Declared MCP code for a legacy runtime string; unknown strings map to ``MCP_INTERNAL_ERROR``."""
    return LEGACY_WIRE_CODE_MAP.get(str(legacy_code).strip(), _C.MCP_INTERNAL_ERROR)


def mcp_tool_error(code: McpServerDefinitionErrorCode | str, detail: str = "") -> ToolError:
    """Build the canonical MCP wire error."""
    code_text = code.value if isinstance(code, McpServerDefinitionErrorCode) else str(code)
    return ToolError(f"{code_text}: {detail}" if detail else code_text)


@asynccontextmanager
async def sanitized_tool_errors(tool_name: str) -> AsyncIterator[None]:
    """Run a tool body so that raw exception text never reaches the wire (MCP-RC AF-01, AF-10).

    Wire errors (``ToolError``), cancellation and the declared concurrency-limit error pass through unchanged;
    any other exception becomes ``MCP_INTERNAL_ERROR`` carrying only the exception class name, and the full
    exception is logged server-side.
    """
    try:
        yield
    except (ToolError, asyncio.CancelledError, McpConcurrentToolCallLimitExceeded):
        raise
    except Exception as exc:
        _LOG.exception("MCP tool %s failed", tool_name)
        raise mcp_tool_error(_C.MCP_INTERNAL_ERROR, f"tool execution failed ({type(exc).__name__})") from exc


def bound_mcp_result_or_raise(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MCP_MAX_SERIALIZED_BYTES,
) -> Any:
    """Return *value* unchanged, or raise ``MCP_RESULT_BOUND_EXCEEDED`` (``isError=true``) when too large."""
    if mcp_serialized_size_exceeds(value, max_bytes=max_bytes):
        raise mcp_tool_error(
            _C.MCP_RESULT_BOUND_EXCEEDED,
            "tool result exceeds the serialized byte ceiling",
        )
    return value


_GENERIC_INTERNAL_DETAIL = "catalog child execution failed"


def bound_cross_loop_catalog_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bound and sanitize a payload produced by the cross-loop catalog ingress (runtime-owned serializer)."""
    result = dict(payload)
    error = result.get("error")
    if isinstance(error, str):
        result.setdefault("mcp_code", mcp_code_for_legacy(error).value)
        if error == "internal_error":
            result["detail"] = _GENERIC_INTERNAL_DETAIL
    return bound_mcp_result_or_raise(result)


__all__ = [
    "LEGACY_WIRE_CODE_MAP",
    "bound_cross_loop_catalog_payload",
    "bound_mcp_result_or_raise",
    "mcp_code_for_legacy",
    "mcp_tool_error",
    "sanitized_tool_errors",
]

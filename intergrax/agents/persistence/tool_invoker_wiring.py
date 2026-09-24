# © Artur Czarnecki. All rights reserved.

"""Host wiring for declarative catalog tool invoker (ACP-PROD-2 depth)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest


def attach_declarative_tool_invoker(
    metadata: dict[str, Any],
    invoker: ExecutionBoundDeclarativeToolInvoker | None,
) -> dict[str, Any]:
    wired = dict(metadata)
    if invoker is not None:
        wired[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] = invoker
    return wired


def resolve_declarative_tool_invoker_from_metadata(
    metadata: dict[str, Any],
) -> ExecutionBoundDeclarativeToolInvoker | None:
    candidate = metadata.get(AcpMetadataKey.DECLARATIVE_TOOL_INVOKER)
    if candidate is None:
        return None
    if isinstance(candidate, ExecutionBoundDeclarativeToolInvoker):
        return candidate
    raise TypeError(
        "declarative tool invoker metadata must implement ExecutionBoundDeclarativeToolInvoker",
    )


def inject_acp_tool_invoker_metadata(
    metadata: dict[str, Any],
    invoker: ExecutionBoundDeclarativeToolInvoker | None,
    *,
    task_id: str,
    run_id: str,
    agent_id: str,
    tenant_id: str,
) -> None:
    """Mutate task/runtime metadata with the host catalog tool invoker when wired."""
    if invoker is None:
        return
    metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] = invoker


def wire_acp_run_request_with_tool_invoker(
    request: AgentRunRequest,
    invoker: ExecutionBoundDeclarativeToolInvoker | None,
) -> AgentRunRequest:
    if invoker is None:
        return request
    return request.model_copy(
        update={
            "metadata": attach_declarative_tool_invoker(dict(request.metadata), invoker),
        },
    )

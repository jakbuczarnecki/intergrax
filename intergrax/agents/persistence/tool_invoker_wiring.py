# © Artur Czarnecki. All rights reserved.

"""Host wiring for declarative catalog tool invoker (ACP-PROD-2 depth)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
    resolve_declarative_tool_invoker,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest


def attach_declarative_tool_invoker(
    metadata: dict[str, Any],
    invoker: DeclarativeToolInvoker | None,
) -> dict[str, Any]:
    wired = dict(metadata)
    if invoker is not None:
        wired[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] = invoker
    return wired


def resolve_declarative_tool_invoker_from_metadata(
    metadata: dict[str, Any],
) -> DeclarativeToolInvoker | None:
    return resolve_declarative_tool_invoker(
        metadata.get(AcpMetadataKey.DECLARATIVE_TOOL_INVOKER),
    )


def inject_acp_tool_invoker_metadata(
    metadata: dict[str, Any],
    invoker: DeclarativeToolInvoker | None,
    *,
    task_id: str,
    run_id: str,
    agent_id: str,
    tenant_id: str,
) -> None:
    """Mutate task/runtime metadata with the host catalog tool invoker when wired."""
    if invoker is None:
        return
    if isinstance(invoker, CatalogDeclarativeToolInvoker):
        invoker.bind_run(
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
    metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] = invoker


def wire_acp_run_request_with_tool_invoker(
    request: AgentRunRequest,
    invoker: DeclarativeToolInvoker | None,
) -> AgentRunRequest:
    if invoker is None:
        return request
    return request.model_copy(
        update={
            "metadata": attach_declarative_tool_invoker(dict(request.metadata), invoker),
        },
    )

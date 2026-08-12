# © Artur Czarnecki. All rights reserved.

"""Shared source-operation coordinator wiring for RAG tool paths."""

from __future__ import annotations

import socket

from intergrax.distributed.source_operation import (
    SourceOperationCoordinator,
    resolve_source_operation_coordinator,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.registry.wiring import ToolWiringContext

_COORDINATOR_CACHE_KEY = "source_operation_coordinator"


def _default_owner_id() -> str:
    host = socket.gethostname().strip() or "local"
    return f"rag-runtime:{host}"


def shared_source_operation_coordinator(
    ctx: ToolWiringContext,
) -> SourceOperationCoordinator:
    cached = ctx.extras.get(_COORDINATOR_CACHE_KEY)
    if isinstance(cached, SourceOperationCoordinator):
        return cached
    coordinator = resolve_source_operation_coordinator(
        ctx.document_store,
        owner_id=_default_owner_id(),
    )
    ctx.extras[_COORDINATOR_CACHE_KEY] = coordinator
    return coordinator


def bind_source_operation_coordinator(
    ctx: ToolWiringContext,
    manager: BaseVectorstoreManager | None,
) -> None:
    if not isinstance(manager, VectorstoreManager):
        return
    manager.set_source_operation_coordinator(
        shared_source_operation_coordinator(ctx),
    )

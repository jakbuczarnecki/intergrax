# © Artur Czarnecki. All rights reserved.

"""Host wiring for ``ReliabilityProfile.idempotency_store`` injection (ACP-CLOSE-PROD-6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.task.task import Task


def attach_idempotency_store_wiring(
    metadata: dict[str, Any],
    store: IdempotencyStore,
) -> dict[str, Any]:
    wired = dict(metadata)
    wired[AcpMetadataKey.IDEMPOTENCY_STORE] = store
    return wired


def wire_acp_run_request_with_idempotency_store(
    request: AgentRunRequest,
    store: IdempotencyStore,
) -> AgentRunRequest:
    return request.model_copy(
        update={
            "metadata": attach_idempotency_store_wiring(
                dict(request.metadata),
                store,
            ),
        },
    )


def resolve_idempotency_store_from_metadata(
    metadata: dict[str, Any],
) -> IdempotencyStore | None:
    store = metadata.get(AcpMetadataKey.IDEMPOTENCY_STORE)
    if isinstance(store, IdempotencyStore):
        return store
    return None


def inject_acp_idempotency_store_metadata(
    metadata: dict[str, Any],
    store: IdempotencyStore | None,
) -> None:
    if store is None:
        return
    if not metadata.get(AcpMetadataKey.SESSION_ENABLED):
        return
    metadata[AcpMetadataKey.IDEMPOTENCY_STORE] = store


def overlay_idempotency_pre_effect_on_runtime_context(
    runtime_context: object,
    metadata: dict[str, Any],
) -> None:
    """Attach pre-effect idempotency coordinator when host metadata carries a store."""
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.runtime.nexus.tools.catalog_dispatch import resolve_tool_registry
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
    from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
        build_production_runtime_tool_invoker,
    )

    if not isinstance(runtime_context, RuntimeContext):
        return
    store = resolve_idempotency_store_from_metadata(metadata)
    if store is None:
        return
    invoker = runtime_context.config.tool_invoker
    if not isinstance(invoker, RuntimeToolInvoker):
        return
    if invoker._pre_effect_coordinator is not None:
        return
    registry = resolve_tool_registry(invoker)
    if registry is None:
        return
    runtime_context.config.tool_invoker = build_production_runtime_tool_invoker(
        registry=registry,
        executor=invoker._executor,
        scope_policy=invoker._scope_policy,
        idempotency_store=store,
        production_mode=runtime_context.config.production_mode,
        meaningful_side_effect_authorization=invoker._meaningful_side_effect_authorization,
        agent_runtime_governance=invoker._agent_runtime_governance,
        inner_execution_guard=invoker._inner_execution_guard,
        sandbox_availability=invoker._sandbox_availability,
    )
    runtime_context.config.idempotency_store = store


def make_acp_idempotency_store_task_enricher(
    store: IdempotencyStore | None,
) -> Callable[[Task], Task] | None:
    """Build a task enricher that wires idempotency store into task metadata."""
    if store is None:
        return None

    def enricher(task: Task) -> Task:
        metadata = dict(task.metadata)
        inject_acp_idempotency_store_metadata(metadata, store)
        return task.model_copy(update={"metadata": metadata})

    return enricher

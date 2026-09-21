# © Artur Czarnecki. All rights reserved.

"""UAEP runtime overlay for host-injected idempotency store (X6 / ACP-CLOSE-PROD-6)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    recompose_runtime_tool_invoker_with_idempotency_store,
)


def apply_host_idempotency_pre_effect_to_runtime_context(
    runtime_context: RuntimeContext,
    request_metadata: dict[str, Any],
) -> None:
    """Attach pre-effect idempotency coordinator when host metadata carries a store."""
    from intergrax.agents.persistence.idempotency_store_wiring import (
        resolve_idempotency_store_from_metadata,
    )

    store = resolve_idempotency_store_from_metadata(request_metadata)
    if store is None:
        return
    invoker = runtime_context.config.tool_invoker
    if not isinstance(invoker, RuntimeToolInvoker):
        return
    replacement = recompose_runtime_tool_invoker_with_idempotency_store(
        invoker,
        idempotency_store=store,
        production_mode=runtime_context.config.production_mode,
    )
    if replacement is None:
        return
    runtime_context.config.tool_invoker = replacement
    runtime_context.config.idempotency_store = store


__all__ = ["apply_host_idempotency_pre_effect_to_runtime_context"]

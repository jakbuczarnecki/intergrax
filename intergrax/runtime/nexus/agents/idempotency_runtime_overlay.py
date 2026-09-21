# © Artur Czarnecki. All rights reserved.

"""UAEP runtime overlay for host-injected idempotency store (X6 / ACP-CLOSE-PROD-6)."""

from __future__ import annotations

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker


def apply_host_idempotency_pre_effect_to_runtime_context(
    runtime_context: RuntimeContext,
    idempotency_store: IdempotencyStore | None,
) -> None:
    """Attach pre-effect idempotency coordinator when host supplies a store."""
    if idempotency_store is None:
        return
    invoker = runtime_context.config.tool_invoker
    if not isinstance(invoker, RuntimeToolInvoker):
        return
    replacement = invoker.with_idempotency_store(
        idempotency_store,
        production_mode=runtime_context.config.production_mode,
    )
    runtime_context.config.tool_invoker = replacement
    runtime_context.config.idempotency_store = idempotency_store


__all__ = ["apply_host_idempotency_pre_effect_to_runtime_context"]

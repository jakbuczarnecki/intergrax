# © Artur Czarnecki. All rights reserved.

"""Typed runtime dependencies for canonical ContextEngine assembly (CE-01-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.ucl_orchestration import (
    NEXUS_UCL_RUNTIME_HANDLE,
    NexusUCLRuntimeDependencies,
)
from intergrax.runtime.wiring.context_runtime_bridge import CONTEXT_OPTIMIZATION_POLICY_HANDLE

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.nexus.config import RuntimeConfig


class ContextEngineRuntimeConfig(Protocol):
    """Minimal runtime config surface required by canonical ContextEngine assembly."""

    llm_adapter: LLMAdapter | None
    production_mode: bool
    metadata: dict[str, Any]

CANONICAL_SEMANTIC_HANDLE_KEYS: frozenset[str] = frozenset(
    {
        "runtime_config",
        "messages",
        "max_output_tokens",
        CONTEXT_OPTIMIZATION_POLICY_HANDLE,
        NEXUS_UCL_RUNTIME_HANDLE,
    }
)

CANONICAL_ASSEMBLY_OBSERVABILITY_HANDLE_KEYS: frozenset[str] = frozenset(
    {
        "event_bus",
        "node_id",
        "agent_id",
    }
)


@dataclass(frozen=True, slots=True)
class ContextAssemblyRuntimeDependencies:
    """Explicit runtime inputs for ``DefaultNexusContextEngine.assemble`` (not source payloads)."""

    runtime_config: ContextEngineRuntimeConfig
    base_messages: tuple[ChatMessage, ...] = ()
    max_output_tokens: int | None = None
    optimization_policy: ContextOptimizationPolicy | None = None
    ucl_runtime: NexusUCLRuntimeDependencies | None = None
    event_bus: RuntimeEventBus | None = None
    node_id: str | None = None
    agent_id: str | None = None


def _coerce_base_messages(raw: object) -> tuple[ChatMessage, ...]:
    if not isinstance(raw, list) or not raw:
        return ()
    typed: list[ChatMessage] = []
    for item in raw:
        if isinstance(item, ChatMessage):
            typed.append(item)
    return tuple(typed)


def _coerce_max_output_tokens(raw: object) -> int | None:
    if type(raw) is int and raw > 0:
        return raw
    return None


def build_context_assembly_runtime_dependencies(
    *,
    runtime_config: ContextEngineRuntimeConfig,
    messages: list[Any] | tuple[Any, ...] | None = None,
    max_output_tokens: int | None = None,
    optimization_policy: ContextOptimizationPolicy | None = None,
    ucl_runtime: NexusUCLRuntimeDependencies | None = None,
    event_bus: RuntimeEventBus | None = None,
    node_id: str | None = None,
    agent_id: str | None = None,
) -> ContextAssemblyRuntimeDependencies:
    base = _coerce_base_messages(list(messages or []))
    return ContextAssemblyRuntimeDependencies(
        runtime_config=runtime_config,
        base_messages=base,
        max_output_tokens=max_output_tokens,
        optimization_policy=optimization_policy,
        ucl_runtime=ucl_runtime,
        event_bus=event_bus,
        node_id=node_id,
        agent_id=agent_id,
    )


def try_build_runtime_from_legacy_handles(
    handles: dict[str, Any],
) -> ContextAssemblyRuntimeDependencies | None:
    """Writer-side compatibility: one-shot mapping from legacy handle keys."""
    runtime_config = handles.get("runtime_config")
    if runtime_config is None or not hasattr(runtime_config, "llm_adapter"):
        return None
    optimization_policy = handles.get(CONTEXT_OPTIMIZATION_POLICY_HANDLE)
    if optimization_policy is not None and not isinstance(optimization_policy, ContextOptimizationPolicy):
        optimization_policy = None
    ucl_runtime = handles.get(NEXUS_UCL_RUNTIME_HANDLE)
    if ucl_runtime is not None and not isinstance(ucl_runtime, NexusUCLRuntimeDependencies):
        ucl_runtime = None
    event_bus = handles.get("event_bus")
    if not isinstance(event_bus, RuntimeEventBus):
        event_bus = None
    node_id = handles.get("node_id")
    agent_id = handles.get("agent_id")
    return ContextAssemblyRuntimeDependencies(
        runtime_config=runtime_config,
        base_messages=_coerce_base_messages(handles.get("messages")),
        max_output_tokens=_coerce_max_output_tokens(handles.get("max_output_tokens")),
        optimization_policy=optimization_policy,
        ucl_runtime=ucl_runtime,
        event_bus=event_bus,
        node_id=node_id if isinstance(node_id, str) else None,
        agent_id=agent_id if isinstance(agent_id, str) else None,
    )


def ensure_context_assembly_runtime(ctx: Any) -> Any:
    """Populate ``ctx.runtime`` from legacy handles when callers still pass handles only."""
    if getattr(ctx, "runtime", None) is not None:
        return ctx
    hydrated = try_build_runtime_from_legacy_handles(getattr(ctx, "handles", {}) or {})
    if hydrated is not None:
        ctx.runtime = hydrated
    return ctx

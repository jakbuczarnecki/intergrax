# © Artur Czarnecki. All rights reserved.

"""Typed runtime dependencies for canonical ContextEngine assembly (CE-01-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.ucl_orchestration import NexusUCLRuntimeDependencies

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.nexus.config import RuntimeConfig


class ContextEngineRuntimeConfig(Protocol):
    """Minimal runtime config surface required by canonical ContextEngine assembly."""

    llm_adapter: LLMAdapter | None
    production_mode: bool
    metadata: dict[str, Any]

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

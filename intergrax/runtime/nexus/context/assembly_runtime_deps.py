# © Artur Czarnecki. All rights reserved.

"""Typed runtime dependencies for canonical ContextEngine assembly (CE-01-R1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from intergrax.context.assembly_runtime import (
    ContextAssemblyUCLRuntime,
    ContextEngineRuntimeConfig,
)
from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy

@dataclass(frozen=True, slots=True)
class ContextAssemblyRuntimeDependencies:
    """Explicit runtime inputs for ``DefaultNexusContextEngine.assemble`` (not source payloads)."""

    runtime_config: ContextEngineRuntimeConfig
    base_messages: tuple[ChatMessage, ...] = ()
    max_output_tokens: int | None = None
    optimization_policy: ContextOptimizationPolicy | None = None
    ucl_runtime: ContextAssemblyUCLRuntime | None = None
    event_bus: RuntimeEventRecorderPort | None = None
    node_id: str | None = None
    agent_id: str | None = None


def build_context_assembly_runtime_dependencies(
    *,
    runtime_config: ContextEngineRuntimeConfig,
    messages: Sequence[ChatMessage] | None = None,
    max_output_tokens: int | None = None,
    optimization_policy: ContextOptimizationPolicy | None = None,
    ucl_runtime: ContextAssemblyUCLRuntime | None = None,
    event_bus: RuntimeEventRecorderPort | None = None,
    node_id: str | None = None,
    agent_id: str | None = None,
) -> ContextAssemblyRuntimeDependencies:
    base = tuple(messages or ())
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

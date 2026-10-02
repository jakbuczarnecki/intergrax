# © Artur Czarnecki. All rights reserved.

"""Reusable typed assembly runtime fixtures for CE / Nexus context tests."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.context.contracts import ContextProviderContext
from intergrax.context.source_inputs import ContextProviderSourceInputs
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.context.assembly_runtime import (
    ContextAssemblyUCLRuntime,
    validate_context_assembly_ucl_runtime,
)
from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    ContextAssemblyRuntimeDependencies,
    build_context_assembly_runtime_dependencies,
)
from intergrax.runtime.nexus.context.ucl_orchestration import NEXUS_UCL_RUNTIME_HANDLE
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.wiring.context_runtime_bridge import CONTEXT_OPTIMIZATION_POLICY_HANDLE

_SEMANTIC_HANDLE_KEYS = frozenset(
    {
        "runtime_config",
        "messages",
        "max_output_tokens",
        "context_optimization_policy",
        CONTEXT_OPTIMIZATION_POLICY_HANDLE,
        NEXUS_UCL_RUNTIME_HANDLE,
        "event_bus",
        "node_id",
        "agent_id",
    }
)


def build_test_assembly_runtime(
    *,
    runtime_config: RuntimeConfig,
    messages: Sequence[ChatMessage] | None = None,
    max_output_tokens: int | None = None,
    optimization_policy: ContextOptimizationPolicy | None = None,
    ucl_runtime: ContextAssemblyUCLRuntime | None = None,
    event_bus: RuntimeEventRecorderPort | None = None,
    node_id: str | None = None,
    agent_id: str | None = None,
) -> ContextAssemblyRuntimeDependencies:
    return build_context_assembly_runtime_dependencies(
        runtime_config=runtime_config,
        messages=messages,
        max_output_tokens=max_output_tokens,
        optimization_policy=optimization_policy,
        ucl_runtime=ucl_runtime,
        event_bus=event_bus,
        node_id=node_id,
        agent_id=agent_id,
    )


def provider_context_for_engine_assembly(
    *,
    runtime_config: RuntimeConfig,
    messages: Sequence[ChatMessage] | None = None,
    max_output_tokens: int | None = None,
    optimization_policy: ContextOptimizationPolicy | None = None,
    ucl_runtime: ContextAssemblyUCLRuntime | None = None,
    event_bus: RuntimeEventRecorderPort | None = None,
    node_id: str | None = None,
    agent_id: str | None = None,
    engine_id: str = "default",
    sources: ContextProviderSourceInputs | None = None,
    handles: dict[str, object] | None = None,
) -> ContextProviderContext:
    runtime = build_test_assembly_runtime(
        runtime_config=runtime_config,
        messages=messages,
        max_output_tokens=max_output_tokens,
        optimization_policy=optimization_policy,
        ucl_runtime=ucl_runtime,
        event_bus=event_bus,
        node_id=node_id,
        agent_id=agent_id,
    )
    return ContextProviderContext(
        engine_id=engine_id,
        sources=sources or ContextProviderSourceInputs(),
        runtime=runtime,
        handles=dict(handles or {}),
    )


def provider_context_from_legacy_style_handles(
    handles: dict[str, object],
    *,
    engine_id: str = "default",
    sources: ContextProviderSourceInputs | None = None,
) -> ContextProviderContext:
    """Migrate legacy handle dicts to typed ``runtime``; auxiliary keys remain in ``handles``."""
    raw = dict(handles)
    runtime_config = raw.pop("runtime_config", None)
    if runtime_config is None:
        raise ValueError("runtime_config is required for assembly provider context")
    messages_raw = raw.pop("messages", [])
    messages: list[ChatMessage] = []
    if isinstance(messages_raw, list):
        messages = [m for m in messages_raw if isinstance(m, ChatMessage)]
    max_output_tokens = raw.pop("max_output_tokens", None)
    if max_output_tokens is not None and not (type(max_output_tokens) is int and max_output_tokens > 0):
        max_output_tokens = None
    optimization_policy = raw.pop("context_optimization_policy", None)
    if optimization_policy is None:
        optimization_policy = raw.pop(CONTEXT_OPTIMIZATION_POLICY_HANDLE, None)
    if optimization_policy is not None and not isinstance(optimization_policy, ContextOptimizationPolicy):
        optimization_policy = None
    ucl_runtime = raw.pop(NEXUS_UCL_RUNTIME_HANDLE, None)
    if ucl_runtime is not None:
        try:
            validate_context_assembly_ucl_runtime(ucl_runtime)
        except ValueError:
            ucl_runtime = None
    event_bus = raw.pop("event_bus", None)
    if event_bus is not None and not isinstance(event_bus, RuntimeEventRecorderPort):
        event_bus = None
    node_id = raw.pop("node_id", None)
    agent_id = raw.pop("agent_id", None)
    for key in _SEMANTIC_HANDLE_KEYS:
        raw.pop(key, None)
    return provider_context_for_engine_assembly(
        runtime_config=runtime_config,  # type: ignore[arg-type]
        messages=messages,
        max_output_tokens=max_output_tokens if type(max_output_tokens) is int else None,
        optimization_policy=optimization_policy,
        ucl_runtime=ucl_runtime,
        event_bus=event_bus,
        node_id=node_id if isinstance(node_id, str) else None,
        agent_id=agent_id if isinstance(agent_id, str) else None,
        engine_id=engine_id,
        sources=sources,
        handles=raw,
    )

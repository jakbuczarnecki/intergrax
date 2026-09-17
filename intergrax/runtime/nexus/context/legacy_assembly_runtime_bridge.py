# © Artur Czarnecki. All rights reserved.

"""LEGACY COMPATIBILITY ONLY — NOT CANONICAL.

Maps legacy ``ContextProviderContext.handles`` semantic keys to typed
``ContextAssemblyRuntimeDependencies``. Call explicitly before canonical assembly;
``DefaultNexusContextEngine`` and ``ContextProviderContext`` do not invoke this path.
"""

from __future__ import annotations

from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    ContextAssemblyRuntimeDependencies,
)
from intergrax.runtime.nexus.context.ucl_orchestration import (
    NEXUS_UCL_RUNTIME_HANDLE,
    NexusUCLRuntimeDependencies,
)
from intergrax.runtime.wiring.context_runtime_bridge import CONTEXT_OPTIMIZATION_POLICY_HANDLE


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


def build_context_assembly_runtime_from_legacy_handles(
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


def try_build_runtime_from_legacy_handles(
    handles: dict[str, Any],
) -> ContextAssemblyRuntimeDependencies | None:
    """Deprecated alias — prefer ``build_context_assembly_runtime_from_legacy_handles``."""
    return build_context_assembly_runtime_from_legacy_handles(handles)

# © Artur Czarnecki. All rights reserved.

"""Provider handle builders for graph context assembly (CE-PROV-CTX)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.task.task import Task

WORKSPACE_FILES_METADATA_KEY = "workspace_files"


def workspace_files_from_task(task: Task) -> dict[str, str]:
    """Read workspace file map from task metadata when present."""
    raw = task.metadata.get(WORKSPACE_FILES_METADATA_KEY)
    if not isinstance(raw, dict) or not raw:
        return {}
    return {str(path): str(content) for path, content in raw.items()}


def build_graph_provider_handles(
    task: Task,
    *,
    runtime_config: RuntimeConfig,
    messages: list[Any],
    event_bus: RuntimeEventBus | None,
    node_id: str,
    agent_id: str | None,
    engine_id: str,
) -> dict[str, Any]:
    """Assemble runtime handles for ``ContextProviderContext`` on graph nodes."""
    handles: dict[str, Any] = {
        "runtime_config": runtime_config,
        "messages": messages,
        "event_bus": event_bus,
        "node_id": node_id,
        "agent_id": agent_id,
        "engine_id": engine_id,
    }
    workspace_files = workspace_files_from_task(task)
    if workspace_files:
        handles["workspace_files"] = workspace_files
    memory_profile = task.metadata.get("memory_profile")
    if isinstance(memory_profile, dict) and memory_profile.get("enable_session_vector_index"):
        handles["enable_session_vector_index"] = True
    vector_hits = task.metadata.get("session_vector_hits")
    if isinstance(vector_hits, list):
        handles["session_vector_hits"] = vector_hits
    return handles

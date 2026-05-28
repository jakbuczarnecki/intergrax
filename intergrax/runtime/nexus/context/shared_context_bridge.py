# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge ``SharedTaskContext`` into task memory for UAEP read path (§42.14, §42.35)."""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.context.shared_task_context import SharedTaskContext
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator
from intergrax.runtime.task_memory.limits import TaskMemoryLimits
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence


def hydrate_shared_context_memory(
    store: TaskMemoryPersistence,
    *,
    tenant_id: str,
    task_id: str,
    shared: SharedTaskContext,
    limits: Optional[TaskMemoryLimits] = None,
) -> None:
    """
    Mirror ``SharedTaskContext`` into task memory for agent reads via ``MemoryView``.

    Runtime-owned hydration — bypasses agent write policy on the shared namespace.
    """
    namespace = shared.memory_namespace
    resolved_limits = limits or TaskMemoryLimits()
    provenance = {"source": "shared_task_context", "version": shared.version}
    for node_id, payload in shared.structured_outputs.items():
        TaskMemoryCoordinator.write(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=node_id,
            value=dict(payload),
            provenance=provenance,
            limits=resolved_limits,
        )
    for label, artifact in shared.artifacts.items():
        TaskMemoryCoordinator.write(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace,
            key=f"artifact:{label}",
            value=artifact.model_dump(mode="json"),
            provenance=provenance,
            limits=resolved_limits,
        )

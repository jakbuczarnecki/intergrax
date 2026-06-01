# © Artur Czarnecki. All rights reserved.

"""Delegation-scoped memory namespaces (Phase R-Delegate.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from intergrax.runtime.task_memory.policy import MemoryAccessPolicy


class TaskMemoryMetadataKey(StrEnum):
    DELEGATION_MEMORY_NAMESPACE = "delegation_memory_namespace"
    PARENT_RUN_ID = "parent_run_id"
    PARENT_NODE_ID = "parent_node_id"


def apply_delegation_memory_namespace(
    policy: MemoryAccessPolicy,
    metadata: Mapping[str, Any],
) -> MemoryAccessPolicy:
    """
    When a graph node runs as a delegation child, allow its isolated namespace.

    Other namespaces remain governed by the base policy.
    """
    raw = metadata.get(TaskMemoryMetadataKey.DELEGATION_MEMORY_NAMESPACE)
    if not isinstance(raw, str) or not raw.strip():
        return policy
    namespace = raw.strip()
    allowed = policy.allowed_namespaces
    merged = frozenset({namespace}) if allowed is None else allowed | frozenset({namespace})
    return MemoryAccessPolicy(
        allowed_namespaces=merged,
        read_only=policy.read_only,
        write_denied_namespaces=policy.write_denied_namespaces,
        list_limit=policy.list_limit,
    )

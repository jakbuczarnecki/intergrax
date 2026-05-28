# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Policy types for runtime-controlled task memory access (§42.35)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from intergrax.contracts.memory_write_policy import MemoryWritePolicy

__all__ = ["MemoryAccessPolicy", "MemoryWritePolicy", "memory_access_policy_from_metadata"]


@dataclass(frozen=True)
class MemoryAccessPolicy:
    """
    Tier-3 / task-scoped access rules for ``MemoryView``.

    ``allowed_namespaces=None`` means all namespaces are permitted.
    """

    allowed_namespaces: Optional[FrozenSet[str]] = None
    read_only: bool = False
    write_denied_namespaces: Optional[FrozenSet[str]] = None
    list_limit: int = 100


def memory_access_policy_from_metadata(metadata: Dict[str, Any]) -> MemoryAccessPolicy:
    """Resolve policy from task/request metadata (composition root / Tier-3 config)."""
    raw = metadata.get("memory_access_policy")
    if isinstance(raw, MemoryAccessPolicy):
        return raw
    if not isinstance(raw, dict):
        raw = {}

    allowed_raw = raw.get("allowed_namespaces", metadata.get("memory_allowed_namespaces"))
    allowed: Optional[FrozenSet[str]] = None
    if allowed_raw:
        allowed = frozenset(str(item).strip() for item in allowed_raw if str(item).strip())

    read_only = bool(raw.get("read_only", metadata.get("memory_read_only", False)))
    denied_raw = raw.get("write_denied_namespaces", metadata.get("memory_write_denied_namespaces"))
    write_denied: Optional[FrozenSet[str]] = None
    if denied_raw:
        write_denied = frozenset(str(item).strip() for item in denied_raw if str(item).strip())
    list_limit = int(raw.get("list_limit", 100))
    return MemoryAccessPolicy(
        allowed_namespaces=allowed,
        read_only=read_only,
        write_denied_namespaces=write_denied,
        list_limit=max(1, list_limit),
    )

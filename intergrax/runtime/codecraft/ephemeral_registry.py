# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""EphemeralToolRegistry — task-scoped virtual tools (ECC-5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class EphemeralToolRegistry:
    """Tools visible only inside a craft_id — never in global ToolRegistry."""

    craft_id: str
    _tool_ids: set[str] = field(default_factory=set)
    _lock: RLock = field(default_factory=RLock)

    def register(self, tool_id: str) -> None:
        normalized = tool_id.strip()
        if not normalized:
            return
        with self._lock:
            self._tool_ids.add(normalized)

    def list_tools(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tool_ids))

    def clear(self) -> None:
        with self._lock:
            self._tool_ids.clear()


class EphemeralToolRegistryStore:
    """Index of ephemeral registries keyed by craft_id."""

    def __init__(self) -> None:
        self._registries: dict[str, EphemeralToolRegistry] = {}
        self._lock = RLock()

    def for_craft(self, craft_id: str) -> EphemeralToolRegistry:
        with self._lock:
            registry = self._registries.get(craft_id)
            if registry is None:
                registry = EphemeralToolRegistry(craft_id=craft_id)
                self._registries[craft_id] = registry
            return registry

    def dispose(self, craft_id: str) -> None:
        with self._lock:
            registry = self._registries.pop(craft_id, None)
        if registry is not None:
            registry.clear()


_default_store: EphemeralToolRegistryStore | None = None


def get_ephemeral_registry_store(ctx) -> EphemeralToolRegistryStore:
    raw = ctx.extras.get("codecraft_ephemeral_registry")
    if isinstance(raw, EphemeralToolRegistryStore):
        return raw
    global _default_store
    if _default_store is None:
        _default_store = EphemeralToolRegistryStore()
    return _default_store

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context plugin protocol and catalog registration (Phase CE-1.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.context.registry import (
    ContextPluginEntry,
    ContextPluginRegistry,
    register_context_plugin_entry,
)


@runtime_checkable
class ContextPlugin(Protocol):
    """Class-based registration for custom context plugins (CE-2 entry points)."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def plugin_version(cls) -> str: ...

    @classmethod
    def plugin_description(cls) -> str: ...

    @classmethod
    def register(cls, registry: ContextPluginRegistry) -> None:
        """Register providers and optional ranker/allocator/formatter/validator."""


def register_context_plugin(
    plugin: type[ContextPlugin],
    *,
    override: bool = False,
) -> ContextPluginEntry:
    """Register a :class:`ContextPlugin` implementation in the global catalog."""

    plugin_id = plugin.plugin_id().strip().lower()
    if not plugin_id:
        raise ValueError("plugin_id must be non-empty")

    def _register(registry: ContextPluginRegistry) -> None:
        plugin.register(registry)

    entry = ContextPluginEntry(
        plugin_id=plugin_id,
        version=plugin.plugin_version(),
        description=plugin.plugin_description(),
        register=_register,
    )
    register_context_plugin_entry(entry, override=override)
    return entry

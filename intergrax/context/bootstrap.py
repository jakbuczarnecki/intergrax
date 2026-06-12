# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context plugin catalog bootstrap (Phase CE-2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.context.plugin import ContextPlugin, register_context_plugin
from intergrax.context.registry import (
    clear_context_plugin_catalog,
    get_context_plugin,
    list_context_plugin_ids,
)
from intergrax.core.catalog_conflict import (
    catalog_registration_override,
    entry_point_conflict_policy,
    should_skip_catalog_registration,
)
from intergrax.core.plugins.discovery import EP_CONTEXT, ConflictPolicy, register_plugins

_context_shipped_done = False


def reset_context_catalog_bootstrap_for_tests() -> None:
    """Allow tests to re-run shipped context catalog registration."""
    global _context_shipped_done
    _context_shipped_done = False
    clear_context_plugin_catalog()


@dataclass(frozen=True, slots=True)
class ContextCatalogBootstrapResult:
    context_plugins: int
    catalog_plugin_ids: tuple[str, ...]


def bootstrap_context_catalog(
    *,
    register_shipped: bool = True,
    context_plugins: Sequence[type[ContextPlugin]] = (),
    discover_entry_points: bool = False,
    on_conflict: ConflictPolicy = "error",
) -> ContextCatalogBootstrapResult:
    """
    Register shipped builtin context plugin and optional third-party plugins.

    Idempotent per process for shipped registration.
    """
    global _context_shipped_done
    if register_shipped and not _context_shipped_done:
        from intergrax.context.providers.builtin import BuiltinContextPlugin

        register_context_plugin(BuiltinContextPlugin, override=True)
        _context_shipped_done = True

    ep_policy = entry_point_conflict_policy(on_conflict)

    def _register_context(plugin_type: type[ContextPlugin]) -> bool:
        plugin_id = plugin_type.plugin_id().strip().lower()
        registered = plugin_id in list_context_plugin_ids()
        if should_skip_catalog_registration(slug_registered=registered, on_conflict=on_conflict):
            return False
        override = catalog_registration_override(
            slug=plugin_id,
            slug_registered=registered,
            on_conflict=on_conflict,
            catalog_kind="context",
            plugin_type=plugin_type,
        )
        register_context_plugin(plugin_type, override=override)
        return True

    plugin_count = register_plugins(
        EP_CONTEXT,
        _register_context,
        explicit=context_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
    )
    return ContextCatalogBootstrapResult(
        context_plugins=plugin_count,
        catalog_plugin_ids=tuple(list_context_plugin_ids()),
    )


def materialize_context_plugin_registry(
    plugin_ids: Sequence[str] | None = None,
) -> "ContextPluginRegistry":
    """Build a registry instance from catalog entries (enabled plugin ids)."""
    from intergrax.context.registry import ContextPluginRegistry

    bootstrap_context_catalog()
    registry = ContextPluginRegistry()
    enabled = [item.strip().lower() for item in (plugin_ids or []) if item.strip()]
    if not enabled:
        enabled = ["intergrax.builtin"]
    for plugin_id in enabled:
        get_context_plugin(plugin_id).register_into(registry)
    return registry

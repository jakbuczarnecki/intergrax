# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default tool catalog bundles via :class:`ToolPlugin` (Phase P-Ext)."""

from __future__ import annotations

from typing import AbstractSet, Sequence

from intergrax.tools.registry.catalog import is_tool_bundle_registered
from intergrax.tools.registry.plugin_register import register_tool_plugin

_BOOTSTRAPPED = False


def register_default_tools(
    *,
    bundle_ids: Sequence[str] | None = None,
    override: bool = False,
) -> None:
    """
    Idempotent registration of shipped tool catalog bundles.

    When ``bundle_ids`` is set, only those bundles are registered (lazy catalog).
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override and bundle_ids is None:
        return

    from intergrax.tools.registry.shipped_plugins import shipped_tool_bundle_ids, shipped_tool_plugins

    allowed: AbstractSet[str] | None = None
    if bundle_ids is not None:
        allowed = {bid.strip().lower() for bid in bundle_ids if bid.strip()}
        unknown = allowed - shipped_tool_bundle_ids()
        if unknown:
            raise ValueError(f"Unknown tool bundle_id(s): {', '.join(sorted(unknown))}")

    for plugin_type in shipped_tool_plugins():
        manifest = plugin_type.tool_bundle_manifest()
        if allowed is not None and manifest.bundle_id not in allowed:
            continue
        register_tool_plugin(
            plugin_type,
            override=override or is_tool_bundle_registered(manifest.bundle_id),
        )

    if bundle_ids is None:
        _BOOTSTRAPPED = True


def reset_default_tools_bootstrap() -> None:
    """Test helper — allow ``register_default_tools()`` to run again."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False

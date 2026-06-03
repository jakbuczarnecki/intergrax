# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register tool bundles from manifests or plugin classes."""

from __future__ import annotations

from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.core.plugin import ToolPlugin, tool_bundle_manifest_for_plugin
from intergrax.tools.registry.catalog import ToolBundleEntry, register_tool_bundle
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


def register_from_tool_manifest(
    manifest: ToolBundleManifest,
    register_fn,
    *,
    override: bool = False,
) -> ToolBundleManifest:
    """Register catalog row from manifest + ``register(registry, ctx)`` callable."""

    def _bundle_register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        register_fn(registry, ctx)

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id=manifest.bundle_id,
            tool_ids=manifest.tool_ids,
            register=_bundle_register,
            status=manifest.status,
            description=manifest.description,
        ),
        override=override,
    )
    return manifest


def register_tool_plugin(
    plugin: type[ToolPlugin],
    *,
    override: bool = False,
) -> ToolBundleManifest:
    """Register catalog row from a :class:`ToolPlugin` implementation."""

    manifest = tool_bundle_manifest_for_plugin(plugin)

    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        plugin.register_tools(registry, ctx)

    return register_from_tool_manifest(manifest, _register, override=override)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool plugin protocol — external tool bundles (§7.1.6)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


@runtime_checkable
class ToolPlugin(Protocol):
    """
    Optional class-based registration for custom tool bundles.

    Implement on a class and pass to :func:`~intergrax.tools.registry.plugin_register.register_tool_plugin`
    or register via setuptools entry point group ``intergrax.tools``.
    """

    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        """Catalog identity for this tool bundle."""

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        """Register ``ToolContract`` + handler pairs on ``registry``."""


def tool_bundle_manifest_for_plugin(plugin_type: type[ToolPlugin]) -> ToolBundleManifest:
    manifest = plugin_type.tool_bundle_manifest()
    if not isinstance(manifest, ToolBundleManifest):
        raise TypeError(f"{plugin_type.__qualname__}.tool_bundle_manifest() must return ToolBundleManifest")
    return manifest

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for shipped :class:`~intergrax.tools.core.plugin.ToolPlugin` implementations."""

from __future__ import annotations

from typing import Callable

from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.core.plugin import ToolPlugin
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

ToolRegisterFn = Callable[[ToolRegistry, ToolWiringContext], None]


def define_tool_plugin(
    *,
    bundle_id: str,
    tool_ids: tuple[str, ...],
    register_fn: ToolRegisterFn,
    status: ToolBundleStatus = ToolBundleStatus.STABLE,
    description: str = "",
    class_name: str | None = None,
) -> type[ToolPlugin]:
    """Build a concrete ``ToolPlugin`` class for a shipped tool bundle."""

    name = class_name or f"{bundle_id.replace('-', '_').title()}ToolPlugin"

    class _ToolPlugin:
        @classmethod
        def tool_bundle_manifest(cls) -> ToolBundleManifest:
            return ToolBundleManifest(
                bundle_id=bundle_id,
                tool_ids=tool_ids,
                status=status,
                description=description,
            )

        @classmethod
        def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
            register_fn(registry, ctx)

    _ToolPlugin.__name__ = name
    _ToolPlugin.__qualname__ = name
    return _ToolPlugin

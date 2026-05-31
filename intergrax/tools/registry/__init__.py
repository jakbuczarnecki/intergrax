# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool Library registry — runtime registry, catalog, profile, wiring (Phase O.2)."""

from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import (
    ToolBundleEntry,
    ToolBundleMetadata,
    ToolBundleStatus,
    UnknownToolBundleError,
    catalog_snapshot,
    clear_tool_catalog,
    get_bundle,
    iter_bundles,
    list_bundle_ids,
    list_catalog_tool_ids,
    metadata_for_bundle,
    register_tool_bundle,
    unregister_tool_bundle,
)
from intergrax.tools.registry.factory import build_registry_from_profile, enabled_tool_ids_for_profile
from intergrax.tools.registry.profile import ToolProfile, default_lab_tool_profile
from intergrax.tools.registry.runtime import RegisteredTool, ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

__all__ = [
    "RegisteredTool",
    "ToolBundleEntry",
    "ToolBundleMetadata",
    "ToolBundleStatus",
    "ToolProfile",
    "ToolRegistry",
    "ToolWiringContext",
    "UnknownToolBundleError",
    "build_registry_from_profile",
    "catalog_snapshot",
    "clear_tool_catalog",
    "default_lab_tool_profile",
    "enabled_tool_ids_for_profile",
    "get_bundle",
    "iter_bundles",
    "list_bundle_ids",
    "list_catalog_tool_ids",
    "metadata_for_bundle",
    "register_default_tools",
    "register_tool_bundle",
    "reset_default_tools_bootstrap",
    "unregister_tool_bundle",
]

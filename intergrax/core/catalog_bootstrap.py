# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Tier-0 catalog bootstrap — integrations, tools, skills (Phase P-Ext)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from intergrax.core.plugins.discovery import (
    EP_INTEGRATIONS,
    EP_SKILLS,
    EP_TOOLS,
    ConflictPolicy,
    register_plugins,
)
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.skills.registry.plugin_register import register_skill_plugin
from intergrax.tools.registry.plugin_register import register_tool_plugin

if TYPE_CHECKING:
    from intergrax.integrations.registry.bootstrap import IntegrationPreset

_tier0_shipped_done = False


def reset_tier0_catalog_bootstrap_for_tests() -> None:
    """Allow tests to re-run shipped catalog registration after clearing catalogs."""
    global _tier0_shipped_done
    _tier0_shipped_done = False


@dataclass(frozen=True, slots=True)
class CatalogBootstrapResult:
    """Counts of registered plugin classes (excluding shipped first-party bootstrap)."""

    integration_plugins: int
    tool_plugins: int
    skill_plugins: int
    integration_preset: str


def bootstrap_catalogs(
    *,
    register_shipped: bool = True,
    integration_preset: str = "full",
    tool_bundle_ids: Sequence[str] | None = None,
    skill_bundle_ids: Sequence[str] | None = None,
    discover_entry_points: bool = False,
    integration_plugins: Sequence[type] = (),
    tool_plugins: Sequence[type] = (),
    skill_plugins: Sequence[type] = (),
    on_conflict: ConflictPolicy = "error",
) -> CatalogBootstrapResult:
    """
    Register Tier-0 catalogs for a Tier-3 application host.

    Order: shipped first-party bundles → setuptools entry points → explicit plugin classes.

    Shipped registration is idempotent per process. Use ``integration_preset="core"``,
    or ``tool_bundle_ids`` / ``skill_bundle_ids``, for lazy catalog registration.
    """
    global _tier0_shipped_done
    if register_shipped and not _tier0_shipped_done:
        from intergrax.integrations.registry.bootstrap import register_default_integrations
        from intergrax.skills.registry.bootstrap import register_default_skills
        from intergrax.tools.registry.bootstrap import register_default_tools

        register_default_integrations(preset=integration_preset)  # type: ignore[arg-type]
        register_default_tools(bundle_ids=tool_bundle_ids)
        register_default_skills(bundle_ids=skill_bundle_ids)
        _tier0_shipped_done = True

    def _register_integration(plugin_type: type) -> None:
        register_integration_plugin(plugin_type, override=(on_conflict == "override"))

    def _register_tool(plugin_type: type) -> None:
        register_tool_plugin(plugin_type, override=(on_conflict == "override"))

    def _register_skill(plugin_type: type) -> None:
        register_skill_plugin(plugin_type, override=(on_conflict == "override"))

    integration_count = register_plugins(
        EP_INTEGRATIONS,
        _register_integration,
        explicit=integration_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=on_conflict,
    )
    tool_count = register_plugins(
        EP_TOOLS,
        _register_tool,
        explicit=tool_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=on_conflict,
    )
    skill_count = register_plugins(
        EP_SKILLS,
        _register_skill,
        explicit=skill_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=on_conflict,
    )
    return CatalogBootstrapResult(
        integration_plugins=integration_count,
        tool_plugins=tool_count,
        skill_plugins=skill_count,
        integration_preset=integration_preset,
    )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Tier-0 catalog bootstrap — integrations, tools, skills (Phase P-Ext)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from intergrax.core.catalog_conflict import (
    catalog_registration_override,
    entry_point_conflict_policy,
    should_skip_catalog_registration,
)
from intergrax.core.plugins.discovery import (
    EP_INTEGRATIONS,
    EP_SKILLS,
    EP_TOOLS,
    ConflictPolicy,
    register_plugins,
)
from intergrax.integrations.core.plugin import (
    IntegrationPlugin,
    integration_manifest_for_plugin,
)
from intergrax.integrations.registry.catalog import catalog_snapshot as integration_catalog_snapshot
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.skills.core.plugin import SkillPlugin, skill_bundle_manifest_for_plugin
from intergrax.skills.registry.catalog import catalog_snapshot as skill_catalog_snapshot
from intergrax.skills.registry.plugin_register import register_skill_plugin
from intergrax.tools.core.plugin import ToolPlugin, tool_bundle_manifest_for_plugin
from intergrax.tools.registry.catalog import catalog_snapshot as tool_catalog_snapshot
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
    if register_shipped:
        from intergrax.integrations.registry.bootstrap import register_default_integrations
        from intergrax.skills.registry.bootstrap import register_default_skills
        from intergrax.tools.registry.bootstrap import register_default_tools

        if not _tier0_shipped_done or not integration_catalog_snapshot():
            register_default_integrations(
                preset=integration_preset,  # type: ignore[arg-type]
                override=not integration_catalog_snapshot(),
            )
        if not _tier0_shipped_done or not tool_catalog_snapshot():
            register_default_tools(
                bundle_ids=tool_bundle_ids,
                override=not tool_catalog_snapshot(),
            )
        if not _tier0_shipped_done or not skill_catalog_snapshot():
            register_default_skills(
                bundle_ids=skill_bundle_ids,
                override=not skill_catalog_snapshot(),
            )
        _tier0_shipped_done = True

    ep_policy = entry_point_conflict_policy(on_conflict)

    def _register_integration(plugin_type: type[IntegrationPlugin]) -> bool:
        manifest = integration_manifest_for_plugin(plugin_type)
        slug = manifest.slug.strip().lower()
        slug_registered = slug in integration_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=slug_registered, on_conflict=on_conflict):
            return False
        override = catalog_registration_override(
            slug=slug,
            slug_registered=slug_registered,
            on_conflict=on_conflict,
            catalog_kind="integration",
            plugin_type=plugin_type,
        )
        register_integration_plugin(plugin_type, override=override)
        return True

    def _register_tool(plugin_type: type[ToolPlugin]) -> bool:
        manifest = tool_bundle_manifest_for_plugin(plugin_type)
        bundle_id = manifest.bundle_id.strip().lower()
        bundle_registered = bundle_id in tool_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=bundle_registered, on_conflict=on_conflict):
            return False
        override = catalog_registration_override(
            slug=bundle_id,
            slug_registered=bundle_registered,
            on_conflict=on_conflict,
            catalog_kind="tool",
            plugin_type=plugin_type,
        )
        register_tool_plugin(plugin_type, override=override)
        return True

    def _register_skill(plugin_type: type[SkillPlugin]) -> bool:
        manifest = skill_bundle_manifest_for_plugin(plugin_type)
        bundle_id = manifest.bundle_id.strip().lower()
        bundle_registered = bundle_id in skill_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=bundle_registered, on_conflict=on_conflict):
            return False
        override = catalog_registration_override(
            slug=bundle_id,
            slug_registered=bundle_registered,
            on_conflict=on_conflict,
            catalog_kind="skill",
            plugin_type=plugin_type,
        )
        register_skill_plugin(plugin_type, override=override)
        return True

    integration_count = register_plugins(
        EP_INTEGRATIONS,
        _register_integration,
        explicit=integration_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
    )
    tool_count = register_plugins(
        EP_TOOLS,
        _register_tool,
        explicit=tool_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
    )
    skill_count = register_plugins(
        EP_SKILLS,
        _register_skill,
        explicit=skill_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
    )
    return CatalogBootstrapResult(
        integration_plugins=integration_count,
        tool_plugins=tool_count,
        skill_plugins=skill_count,
        integration_preset=integration_preset,
    )

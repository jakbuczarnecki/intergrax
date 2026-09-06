# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Tier-0 catalog bootstrap — integrations, tools, skills (Phase P-Ext)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.core.catalog_conflict import (
    catalog_registration_override,
    entry_point_conflict_policy,
    should_skip_catalog_registration,
)
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_INTEGRATIONS,
    EP_SKILLS,
    EP_TOOLS,
    ConflictPolicy,
    EntryPointSpec,
    register_plugins,
    register_plugins_with_report,
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
    tool_plugin_load_report: DomainPluginLoadReport
    skill_plugin_load_report: DomainPluginLoadReport


def bootstrap_catalogs(
    *,
    register_shipped: bool = True,
    integration_preset: str = "full",
    tool_bundle_ids: Sequence[str] | None = None,
    skill_bundle_ids: Sequence[str] | None = None,
    discover_entry_points: bool = False,
    discover_tool_entry_points: bool | None = None,
    discover_skill_entry_points: bool | None = None,
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
            skill_snap = skill_catalog_snapshot()
            register_default_skills(
                bundle_ids=skill_bundle_ids,
                override=not bool(skill_snap) or skill_bundle_ids is not None,
            )
        _tier0_shipped_done = True

    ep_policy = entry_point_conflict_policy(on_conflict)
    discover_tools = (
        discover_tool_entry_points
        if discover_tool_entry_points is not None
        else discover_entry_points
    )
    discover_skills = (
        discover_skill_entry_points
        if discover_skill_entry_points is not None
        else discover_entry_points
    )

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

    def _register_tool_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if not isinstance(plugin_type, type):
            message = f"Tool entry point {spec.name!r} does not implement ToolPlugin"
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        try:
            manifest = tool_bundle_manifest_for_plugin(plugin_type)
        except (TypeError, AttributeError) as exc:
            message = (
                f"Tool entry point {spec.name!r} does not implement ToolPlugin: {exc}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        bundle_id = manifest.bundle_id.strip().lower()
        bundle_registered = bundle_id in tool_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=bundle_registered, on_conflict=on_conflict):
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED,
                reason=f"Tool bundle {bundle_id!r} already registered; skipping",
                plugin_id=bundle_id,
                fail_closed=False,
            )
        try:
            override = catalog_registration_override(
                slug=bundle_id,
                slug_registered=bundle_registered,
                on_conflict=on_conflict,
                catalog_kind="tool",
                plugin_type=plugin_type,
            )
        except ValueError as exc:
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                reason=str(exc),
                plugin_id=bundle_id,
                fail_closed=True,
            )
        register_tool_plugin(plugin_type, override=override)
        return True, None

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

    def _register_skill_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if not isinstance(plugin_type, type):
            message = f"Skill entry point {spec.name!r} does not implement SkillPlugin"
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        try:
            manifest = skill_bundle_manifest_for_plugin(plugin_type)
        except (TypeError, AttributeError) as exc:
            message = (
                f"Skill entry point {spec.name!r} does not implement SkillPlugin: {exc}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        bundle_id = manifest.bundle_id.strip().lower()
        bundle_registered = bundle_id in skill_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=bundle_registered, on_conflict=on_conflict):
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED,
                reason=f"Skill bundle {bundle_id!r} already registered; skipping",
                plugin_id=bundle_id,
                fail_closed=False,
            )
        try:
            override = catalog_registration_override(
                slug=bundle_id,
                slug_registered=bundle_registered,
                on_conflict=on_conflict,
                catalog_kind="skill",
                plugin_type=plugin_type,
            )
        except ValueError as exc:
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                reason=str(exc),
                plugin_id=bundle_id,
                fail_closed=True,
            )
        register_skill_plugin(plugin_type, override=override)
        return True, None

    integration_count = register_plugins(
        EP_INTEGRATIONS,
        _register_integration,
        explicit=integration_plugins,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
    )
    tool_explicit_count = 0
    for plugin_type in tool_plugins:
        if _register_tool(plugin_type):
            tool_explicit_count += 1
    tool_report = register_plugins_with_report(
        EP_TOOLS,
        _register_tool_entry_point,
        discover_entry_points=discover_tools,
        on_conflict=ep_policy,
    )
    tool_count = tool_explicit_count + tool_report.registered_count
    skill_explicit_count = 0
    for plugin_type in skill_plugins:
        if _register_skill(plugin_type):
            skill_explicit_count += 1
    skill_report = register_plugins_with_report(
        EP_SKILLS,
        _register_skill_entry_point,
        discover_entry_points=discover_skills,
        on_conflict=ep_policy,
    )
    skill_count = skill_explicit_count + skill_report.registered_count
    return CatalogBootstrapResult(
        integration_plugins=integration_count,
        tool_plugins=tool_count,
        skill_plugins=skill_count,
        integration_preset=integration_preset,
        tool_plugin_load_report=tool_report,
        skill_plugin_load_report=skill_report,
    )

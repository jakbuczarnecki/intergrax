# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Tier-0 catalog bootstrap — integrations, tools, skills (Phase P-Ext)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.core.catalog_conflict import (
    catalog_registration_override,
    entry_point_conflict_policy,
    should_skip_catalog_registration,
)
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
    PluginProvenanceAdmission,
)
from intergrax.core.plugins.discovery import (
    EP_INTEGRATIONS,
    EP_SKILLS,
    EP_TOOLS,
    ConflictPolicy,
    EntryPointLoadResult,
    EntryPointSpec,
    RegisterEntryPointFn,
    iter_entry_point_specs,
    load_entry_point_targets_for_specs,
    register_plugins_with_report,
    resolve_entry_point_plugin_type,
)
from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError
from intergrax.core.plugins.platform_semantics import PlatformPluginConflictKind
from intergrax.core.security.security_configuration import (
    PLUGIN_TRUST_POLICY_MISSING,
    SecurityConfigurationError,
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


def _register_entry_points_with_provenance(
    group: str,
    register_entry_point: RegisterEntryPointFn,
    *,
    discover_entry_points: bool,
    on_conflict: ConflictPolicy,
    provenance: PluginProvenanceAdmission | None,
) -> DomainPluginLoadReport:
    """Entry-point registration with optional publisher provenance (PDC-D6-R016).

    Without ``provenance`` this is exactly ``register_plugins_with_report``. With it, every
    entry point is checked against the agent package trust mechanism *before* its target is
    imported; rejected specs are never loaded and appear in the report as fail-closed
    rejections.
    """
    if provenance is None:
        return register_plugins_with_report(
            group,
            register_entry_point,
            discover_entry_points=discover_entry_points,
            on_conflict=on_conflict,
        )
    if not discover_entry_points:
        return DomainPluginLoadReport.empty(group)

    admitted, provenance_rejections = provenance.partition(iter_entry_point_specs(group))
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(provenance_rejections)
    failed: list[EntryPointLoadResult] = []
    seen_names: set[str] = set()
    for spec in admitted:
        if spec.name in seen_names:
            if on_conflict == "skip":
                continue
            if on_conflict not in ("override", "warn_override"):
                raise PluginConflictError(
                    f"Duplicate entry point {spec.name!r} in group {group!r}",
                    plugin_name=spec.name,
                    group=group,
                    conflict_kind=PlatformPluginConflictKind.ENTRY_POINT_NAME,
                )
        seen_names.add(spec.name)
        (result,) = load_entry_point_targets_for_specs((spec,), on_load_failure="isolate")
        if result.error is not None:
            failed.append(result)
            continue
        try:
            plugin_type = resolve_entry_point_plugin_type(result.target, spec.value)
        except PluginLoadError as exc:
            failed.append(EntryPointLoadResult(spec=spec, error=exc))
            continue
        try:
            registered, rejection = register_entry_point(plugin_type, spec)
        except Exception as exc:
            failed.append(EntryPointLoadResult(spec=spec, error=exc))
            continue
        if registered:
            accepted.append(spec)
        elif rejection is not None:
            rejected.append(rejection)
        else:
            raise ValueError("registration callback must return registered=True or a rejection")
    return DomainPluginLoadReport(
        group=group,
        accepted=tuple(sorted(accepted, key=lambda item: (item.name, item.value))),
        rejected=tuple(sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))),
        failed=tuple(sorted(failed, key=lambda item: (item.spec.name, item.spec.value))),
        registered_count=len(accepted),
    )


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
    integration_plugin_load_report: DomainPluginLoadReport
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
    plugin_provenance: PluginProvenanceAdmission | None = None,
    require_plugin_provenance: bool = False,
) -> CatalogBootstrapResult:
    """
    Register Tier-0 catalogs for a Tier-3 application host.

    Order: shipped first-party bundles → setuptools entry points → explicit plugin classes.

    Shipped registration is idempotent per process. Use ``integration_preset="core"``,
    or ``tool_bundle_ids`` / ``skill_bundle_ids``, for lazy catalog registration.

    ``plugin_provenance`` enforces publisher attestation on entry-point plugins before they
    are imported (PDC-D6-R016, reusing the agent package trust mechanism). Strict hosts set
    ``require_plugin_provenance=True``: entry-point discovery without a provenance policy
    then raises :class:`SecurityConfigurationError` instead of admitting unsigned plugins.
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
    if (
        require_plugin_provenance
        and plugin_provenance is None
        and (discover_entry_points or bool(discover_tool_entry_points) or bool(discover_skill_entry_points))
    ):
        raise SecurityConfigurationError(
            item="plugin_provenance",
            reason_code=PLUGIN_TRUST_POLICY_MISSING,
            message="entry-point discovery on a strict host requires a PluginProvenanceAdmission",
        )
    discover_tools = discover_tool_entry_points if discover_tool_entry_points is not None else discover_entry_points
    discover_skills = discover_skill_entry_points if discover_skill_entry_points is not None else discover_entry_points

    def _register_integration_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if not isinstance(plugin_type, type):
            message = f"Integration entry point {spec.name!r} does not implement IntegrationPlugin"
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        try:
            manifest = integration_manifest_for_plugin(plugin_type)
        except (TypeError, AttributeError) as exc:
            message = f"Integration entry point {spec.name!r} does not implement IntegrationPlugin: {exc}"
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        slug = manifest.slug.strip().lower()
        slug_registered = slug in integration_catalog_snapshot()
        if should_skip_catalog_registration(slug_registered=slug_registered, on_conflict=on_conflict):
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED,
                reason=f"Integration {slug!r} already registered; skipping",
                plugin_id=slug,
                fail_closed=False,
            )
        try:
            override = catalog_registration_override(
                slug=slug,
                slug_registered=slug_registered,
                on_conflict=on_conflict,
                catalog_kind="integration",
                plugin_type=plugin_type,
            )
        except ValueError as exc:
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                reason=str(exc),
                plugin_id=slug,
                fail_closed=True,
            )
        try:
            register_integration_plugin(plugin_type, override=override)
        except ValueError as exc:
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.MANIFEST_INVALID,
                reason=str(exc),
                plugin_id=slug,
                fail_closed=True,
            )
        return True, None

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
            message = f"Tool entry point {spec.name!r} does not implement ToolPlugin: {exc}"
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
            message = f"Skill entry point {spec.name!r} does not implement SkillPlugin: {exc}"
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

    integration_explicit_count = 0
    for plugin_type in integration_plugins:
        if _register_integration(plugin_type):
            integration_explicit_count += 1
    integration_report = _register_entry_points_with_provenance(
        EP_INTEGRATIONS,
        _register_integration_entry_point,
        discover_entry_points=discover_entry_points,
        on_conflict=ep_policy,
        provenance=plugin_provenance,
    )
    integration_count = integration_explicit_count + integration_report.registered_count
    tool_explicit_count = 0
    for plugin_type in tool_plugins:
        if _register_tool(plugin_type):
            tool_explicit_count += 1
    tool_report = _register_entry_points_with_provenance(
        EP_TOOLS,
        _register_tool_entry_point,
        discover_entry_points=discover_tools,
        on_conflict=ep_policy,
        provenance=plugin_provenance,
    )
    tool_count = tool_explicit_count + tool_report.registered_count
    skill_explicit_count = 0
    for plugin_type in skill_plugins:
        if _register_skill(plugin_type):
            skill_explicit_count += 1
    skill_report = _register_entry_points_with_provenance(
        EP_SKILLS,
        _register_skill_entry_point,
        discover_entry_points=discover_skills,
        on_conflict=ep_policy,
        provenance=plugin_provenance,
    )
    skill_count = skill_explicit_count + skill_report.registered_count
    return CatalogBootstrapResult(
        integration_plugins=integration_count,
        tool_plugins=tool_count,
        skill_plugins=skill_count,
        integration_preset=integration_preset,
        integration_plugin_load_report=integration_report,
        tool_plugin_load_report=tool_report,
        skill_plugin_load_report=skill_report,
    )

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
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_CONTEXT,
    ConflictPolicy,
    EntryPointLoadResult,
    EntryPointSpec,
    load_entry_point_targets,
    resolve_entry_point_plugin_type,
)
from intergrax.core.plugins.errors import PluginLoadError

_context_shipped_done = False
_SHIPPED_BUILTIN_PLUGIN_ID = "intergrax.builtin"


def reset_context_catalog_bootstrap_for_tests() -> None:
    """Allow tests to re-run shipped context catalog registration."""
    global _context_shipped_done
    _context_shipped_done = False
    clear_context_plugin_catalog()


@dataclass(frozen=True, slots=True)
class ContextCatalogBootstrapResult:
    context_plugins: int
    catalog_plugin_ids: tuple[str, ...]
    load_report: DomainPluginLoadReport


def _is_context_plugin_type(plugin_type: type) -> bool:
    if not isinstance(plugin_type, type):
        return False
    required = ("plugin_id", "plugin_version", "plugin_description", "register")
    return all(callable(getattr(plugin_type, name, None)) for name in required)


def _register_explicit_context_plugin(
    plugin_type: type[ContextPlugin],
    *,
    on_conflict: ConflictPolicy,
) -> bool:
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


def _register_context_entry_point(
    plugin_type: type,
    spec: EntryPointSpec,
    *,
    on_conflict: ConflictPolicy,
) -> tuple[bool, PluginAdmissionRejection | None]:
    if not _is_context_plugin_type(plugin_type):
        message = (
            f"Context entry point {spec.name!r} does not implement ContextPlugin"
        )
        return False, PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
            reason=message,
            fail_closed=True,
        )

    plugin_id = plugin_type.plugin_id().strip().lower()
    registered = plugin_id in list_context_plugin_ids()
    if should_skip_catalog_registration(slug_registered=registered, on_conflict=on_conflict):
        reason_code = (
            PluginAdmissionReasonCode.SHIPPED_ID_COLLISION
            if plugin_id == _SHIPPED_BUILTIN_PLUGIN_ID
            else PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED
        )
        return False, PluginAdmissionRejection(
            spec=spec,
            reason_code=reason_code,
            reason=(
                f"Context plugin {plugin_id!r} already registered; skipping"
            ),
            plugin_id=plugin_id,
            fail_closed=False,
        )

    override = catalog_registration_override(
        slug=plugin_id,
        slug_registered=registered,
        on_conflict=on_conflict,
        catalog_kind="context",
        plugin_type=plugin_type,
    )
    register_context_plugin(plugin_type, override=override)
    return True, None


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
    plugin_count = 0

    for plugin_type in context_plugins:
        if _register_explicit_context_plugin(plugin_type, on_conflict=on_conflict):
            plugin_count += 1

    if not discover_entry_points:
        return ContextCatalogBootstrapResult(
            context_plugins=plugin_count,
            catalog_plugin_ids=tuple(list_context_plugin_ids()),
            load_report=DomainPluginLoadReport.empty(EP_CONTEXT),
        )

    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = []
    failed: list[EntryPointLoadResult] = []

    for result in load_entry_point_targets(
        EP_CONTEXT,
        on_conflict=ep_policy,
        on_load_failure="fail_fast",
    ):
        if result.error is not None:
            failed.append(result)
            continue
        try:
            plugin_type = resolve_entry_point_plugin_type(
                result.target,
                result.spec.value,
            )
        except PluginLoadError as exc:
            raise PluginLoadError(
                f"Failed to load {EP_CONTEXT}:{result.spec.name} "
                f"({result.spec.value}): {exc}"
            ) from exc

        registered, rejection = _register_context_entry_point(
            plugin_type,
            result.spec,
            on_conflict=on_conflict,
        )
        if rejection is not None:
            rejected.append(rejection)
            continue
        if registered:
            plugin_count += 1
            accepted.append(result.spec)

    load_report = DomainPluginLoadReport(
        group=EP_CONTEXT,
        accepted=tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value))),
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(sorted(failed, key=lambda item: (item.spec.name, item.spec.value))),
        registered_count=len(accepted),
    )
    return ContextCatalogBootstrapResult(
        context_plugins=plugin_count,
        catalog_plugin_ids=tuple(list_context_plugin_ids()),
        load_report=load_report,
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

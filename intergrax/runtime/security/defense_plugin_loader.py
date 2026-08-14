# © Artur Czarnecki. All rights reserved.

"""Load security defense plugins from entry points (Phase SEC-EXT-2 / ENTERPRISE-2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    ConflictPolicy,
    EP_SECURITY_DEFENSES,
    EntryPointLoadResult,
    EntryPointSpec,
    LoadIsolation,
    instantiate_entry_point_target,
    load_entry_point_targets,
)
from intergrax.runtime.security.defense_plugin import SecurityDefensePlugin
from intergrax.runtime.security.defense_registry import (
    get_security_defense_plugin,
    list_shipped_defense_bundle_ids,
    register_security_defense_plugin,
)

logger = logging.getLogger(__name__)

ShippedIdOverride = Literal["error", "warn_override", "allow"]


@dataclass(frozen=True, slots=True)
class SecurityDefenseAdmissionPolicy:
    """Security-owned EP admission. Qualification enforcement is a deferred host seam."""

    ep_name_conflict: ConflictPolicy = "error"
    plugin_id_conflict: ConflictPolicy = "error"
    shipped_id_override: ShippedIdOverride = "error"
    require_production_qualification: bool = True
    on_load_failure: LoadIsolation = "isolate"


# OAD-002: production default is fail-closed ``error``. Explicit opt-in restores
# pre-ENTERPRISE-2 unconditional override + fail-fast load behavior.
LEGACY_UNCONDITIONAL_OVERRIDE_POLICY = SecurityDefenseAdmissionPolicy(
    ep_name_conflict="override",
    plugin_id_conflict="override",
    shipped_id_override="allow",
    require_production_qualification=False,
    on_load_failure="fail_fast",
)


def load_security_defense_plugin_report(
    *,
    discover_entry_points: bool = True,
    admission: SecurityDefenseAdmissionPolicy | None = None,
) -> DomainPluginLoadReport:
    """Load and admit ``intergrax.security_defenses`` EPs; return structured evidence."""
    policy = admission if admission is not None else SecurityDefenseAdmissionPolicy()
    if not discover_entry_points:
        return DomainPluginLoadReport.empty(EP_SECURITY_DEFENSES)

    accepted_by_id: dict[str, EntryPointSpec] = {}
    rejected: list[PluginAdmissionRejection] = []
    failed: list[EntryPointLoadResult] = []
    shipped_ids = frozenset(list_shipped_defense_bundle_ids())

    for result in load_entry_point_targets(
        EP_SECURITY_DEFENSES,
        on_conflict=policy.ep_name_conflict,
        on_load_failure=policy.on_load_failure,
    ):
        if result.error is not None:
            failed.append(result)
            continue
        try:
            plugin = instantiate_entry_point_target(result.target)
        except Exception as exc:
            if policy.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue
        if not isinstance(plugin, SecurityDefensePlugin):
            message = (
                f"Security defense entry point {result.spec.name!r} "
                "must return SecurityDefensePlugin"
            )
            if policy.on_load_failure == "fail_fast":
                raise TypeError(message)
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=message,
                    fail_closed=True,
                )
            )
            continue
        decision = _admit_security_plugin_id(
            plugin_id=plugin.plugin_id,
            spec=result.spec,
            policy=policy,
            shipped_ids=shipped_ids,
            accepted_ids=frozenset(accepted_by_id),
        )
        if decision is not None:
            rejected.append(decision)
            continue
        occupation = _plugin_id_occupation(
            plugin.plugin_id,
            shipped_ids=shipped_ids,
            accepted_ids=frozenset(accepted_by_id),
        )
        register_security_defense_plugin(
            plugin,
            override=occupation != "free",
        )
        accepted_by_id[plugin.plugin_id] = result.spec

    accepted = tuple(
        sorted(accepted_by_id.values(), key=lambda spec: (spec.name, spec.value))
    )
    return DomainPluginLoadReport(
        group=EP_SECURITY_DEFENSES,
        accepted=accepted,
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(
            sorted(failed, key=lambda item: (item.spec.name, item.spec.value))
        ),
        registered_count=len(accepted),
    )


def load_security_defense_plugins(
    *,
    discover_entry_points: bool = True,
    admission: SecurityDefenseAdmissionPolicy | None = None,
) -> int:
    """Compatibility wrapper: registered count from :func:`load_security_defense_plugin_report`."""
    return load_security_defense_plugin_report(
        discover_entry_points=discover_entry_points,
        admission=admission,
    ).registered_count


def _plugin_id_occupation(
    plugin_id: str,
    *,
    shipped_ids: frozenset[str],
    accepted_ids: frozenset[str],
) -> Literal["free", "shipped", "batch", "registered"]:
    if plugin_id in accepted_ids:
        return "batch"
    if plugin_id in shipped_ids:
        return "shipped"
    if get_security_defense_plugin(plugin_id) is not None:
        return "registered"
    return "free"


def _admit_security_plugin_id(
    *,
    plugin_id: str,
    spec: EntryPointSpec,
    policy: SecurityDefenseAdmissionPolicy,
    shipped_ids: frozenset[str],
    accepted_ids: frozenset[str],
) -> PluginAdmissionRejection | None:
    occupation = _plugin_id_occupation(
        plugin_id,
        shipped_ids=shipped_ids,
        accepted_ids=accepted_ids,
    )
    if occupation == "free":
        return None
    if occupation == "shipped":
        return _shipped_id_decision(plugin_id=plugin_id, spec=spec, policy=policy)
    return _plugin_id_conflict_decision(
        plugin_id=plugin_id,
        spec=spec,
        occupation=occupation,
        policy=policy,
    )


def _shipped_id_decision(
    *,
    plugin_id: str,
    spec: EntryPointSpec,
    policy: SecurityDefenseAdmissionPolicy,
) -> PluginAdmissionRejection | None:
    mode = policy.shipped_id_override
    if mode == "allow":
        return None
    if mode == "warn_override":
        logger.warning(
            "Overriding shipped security defense %r from entry point %s",
            plugin_id,
            spec.name,
        )
        return None
    reason = (
        f"Security defense entry point {spec.name!r} collides with shipped "
        f"plugin_id {plugin_id!r}"
    )
    logger.warning(reason)
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.SHIPPED_ID_COLLISION,
        reason=reason,
        plugin_id=plugin_id,
        fail_closed=True,
    )


def _plugin_id_conflict_decision(
    *,
    plugin_id: str,
    spec: EntryPointSpec,
    occupation: Literal["batch", "registered"],
    policy: SecurityDefenseAdmissionPolicy,
) -> PluginAdmissionRejection | None:
    mode = policy.plugin_id_conflict
    if mode == "override":
        return None
    if mode == "warn_override":
        logger.warning(
            "Overriding security defense plugin_id %r from entry point %s",
            plugin_id,
            spec.name,
        )
        return None
    if occupation == "registered":
        reason_code = PluginAdmissionReasonCode.ALREADY_REGISTERED
        reason = (
            f"Security defense entry point {spec.name!r} collides with already "
            f"registered plugin_id {plugin_id!r}"
        )
    else:
        reason_code = PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
        reason = (
            f"Security defense entry point {spec.name!r} collides with another "
            f"entry point plugin_id {plugin_id!r}"
        )
    if mode == "skip":
        logger.warning("Skipping %s", reason)
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.PLUGIN_ID_SKIPPED,
            reason=reason,
            plugin_id=plugin_id,
            fail_closed=False,
        )
    logger.warning(reason)
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=reason_code,
        reason=reason,
        plugin_id=plugin_id,
        fail_closed=True,
    )

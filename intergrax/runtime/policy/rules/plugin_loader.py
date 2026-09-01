# © Artur Czarnecki. All rights reserved.

"""Load policy rule handlers from entry points (Phase DX-5.8 / ENTERPRISE-2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    ConflictPolicy,
    EP_POLICY_RULES,
    EntryPointLoadResult,
    EntryPointSpec,
    LoadIsolation,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_targets,
)
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationResult,
    evaluate_external_package_entry_point_production_admission,
    resolve_host_platform_version,
)
from intergrax.runtime.policy.rules.provenance import PolicyHandlerProvenance, handler_provenance_from_spec
from intergrax.runtime.policy.rules.registry import PolicyRuleHandler, PolicyRuleRegistry


@dataclass(frozen=True, slots=True)
class PolicyRulePluginLoadOutcome:
    report: DomainPluginLoadReport
    handler_provenance: tuple[PolicyHandlerProvenance, ...]


@dataclass(frozen=True, slots=True)
class PolicyRuleLoadPolicy:
    """Policy-owned EP load isolation and handler allowlist governance."""

    ep_name_conflict: ConflictPolicy = "error"
    on_load_failure: LoadIsolation = "isolate"
    allowed_handler_ids: frozenset[str] | None = None
    require_production_admission: bool = False
    package_qualification_lookup: (
        Callable[[EntryPointSpec], PluginQualificationResult | None] | None
    ) = None
    platform_version: str | None = None


def _production_admission_rejections(
    policy: PolicyRuleLoadPolicy,
) -> tuple[frozenset[str], list[PluginAdmissionRejection]]:
    if not policy.require_production_admission:
        return frozenset(), []

    platform_version = policy.platform_version or resolve_host_platform_version()
    lookup = policy.package_qualification_lookup
    rejected_names: set[str] = set()
    rejected: list[PluginAdmissionRejection] = []

    for spec in iter_entry_point_specs(EP_POLICY_RULES):
        qualification = lookup(spec) if lookup is not None else None
        admission = evaluate_external_package_entry_point_production_admission(
            spec,
            qualification,
            platform_version=platform_version,
        )
        if admission.admitted:
            continue
        rejected_names.add(spec.name)
        rejected.append(
            PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED,
                reason=admission.reason,
                fail_closed=True,
            )
        )

    return frozenset(rejected_names), rejected


def load_policy_rule_plugin_report(
    registry: PolicyRuleRegistry,
    *,
    policy: PolicyRuleLoadPolicy | None = None,
) -> PolicyRulePluginLoadOutcome:
    """Load ``intergrax.policy_rules`` EPs with structured evidence."""
    chosen = policy if policy is not None else PolicyRuleLoadPolicy()
    skip_names, production_rejected = _production_admission_rejections(chosen)
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(production_rejected)
    failed: list[EntryPointLoadResult] = []
    handler_provenance: list[PolicyHandlerProvenance] = []

    for result in load_entry_point_targets(
        EP_POLICY_RULES,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
        skip_names=skip_names,
    ):
        if result.error is not None:
            failed.append(result)
            continue
        try:
            instance = instantiate_entry_point_target(result.target)
        except Exception as exc:
            if chosen.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue
        if not isinstance(instance, PolicyRuleHandler):
            message = (
                f"Policy rule entry point {result.spec.name!r} must return PolicyRuleHandler"
            )
            if chosen.on_load_failure == "fail_fast":
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

        handler_id = instance.rule_id
        if (
            chosen.allowed_handler_ids is not None
            and handler_id not in chosen.allowed_handler_ids
        ):
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.NOT_IN_ALLOWLIST,
                    reason=(
                        f"Policy handler {handler_id!r} is not in configured allowlist."
                    ),
                    plugin_id=handler_id,
                    fail_closed=True,
                )
            )
            continue

        registration = registry.register(instance)
        if not registration.accepted:
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=registration.reason_code
                    or PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
                    reason=registration.reason or "handler registration rejected",
                    plugin_id=handler_id,
                    fail_closed=True,
                )
            )
            continue

        accepted.append(result.spec)
        handler_provenance.append(
            handler_provenance_from_spec(handler_id, result.spec)
        )

    accepted_tuple = tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value)))
    report = DomainPluginLoadReport(
        group=EP_POLICY_RULES,
        accepted=accepted_tuple,
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(
            sorted(failed, key=lambda item: (item.spec.name, item.spec.value))
        ),
        registered_count=len(accepted_tuple),
    )
    return PolicyRulePluginLoadOutcome(
        report=report,
        handler_provenance=tuple(
            sorted(handler_provenance, key=lambda item: item.rule_id)
        ),
    )

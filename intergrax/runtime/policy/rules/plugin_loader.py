# © Artur Czarnecki. All rights reserved.

"""Load policy rule handlers from entry points (Phase DX-5.8 / ENTERPRISE-2)."""

from __future__ import annotations

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
    load_entry_point_targets,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleHandler, PolicyRuleRegistry


@dataclass(frozen=True, slots=True)
class PolicyRuleLoadPolicy:
    """Policy-owned EP load isolation. Handler allowlist/provenance is CAND-008."""

    ep_name_conflict: ConflictPolicy = "error"
    on_load_failure: LoadIsolation = "isolate"


def load_policy_rule_plugin_report(
    registry: PolicyRuleRegistry,
    *,
    policy: PolicyRuleLoadPolicy | None = None,
) -> DomainPluginLoadReport:
    """Load ``intergrax.policy_rules`` EPs with structured evidence. Does not wire CAND-006."""
    chosen = policy if policy is not None else PolicyRuleLoadPolicy()
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = []
    failed: list[EntryPointLoadResult] = []

    for result in load_entry_point_targets(
        EP_POLICY_RULES,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
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
        registry.register(instance)
        accepted.append(result.spec)

    accepted_tuple = tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value)))
    return DomainPluginLoadReport(
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


def load_policy_rule_plugins(
    registry: PolicyRuleRegistry,
    *,
    policy: PolicyRuleLoadPolicy | None = None,
) -> int:
    """Compatibility wrapper. Default remains fail-fast when ``policy`` is omitted."""
    chosen = (
        policy
        if policy is not None
        else PolicyRuleLoadPolicy(on_load_failure="fail_fast")
    )
    return load_policy_rule_plugin_report(registry, policy=chosen).registered_count

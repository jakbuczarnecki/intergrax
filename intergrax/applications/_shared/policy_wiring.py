# © Artur Czarnecki. All rights reserved.

"""Tier-3 runtime policy bundle composition (Phase R-Policy.2, H-APP.2.6)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from intergrax.applications._shared.cost_wiring import wire_application_cost
from intergrax.applications._shared.critic_wiring import wire_application_critic
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.execution_mode import (
    ExecutionMode,
    runtime_policies_for_execution_mode,
)
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_POLICY_RULES, EntryPointSpec
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginPackageQualificationBundle,
    PluginQualificationResult,
    resolve_host_platform_version,
)
from intergrax.runtime.policy.policy_bundle import (
    DeclarativePolicyRuntime,
    RuntimePolicyBundle,
)
from intergrax.runtime.policy.rules.loader import load_policy_rules_from_path
from intergrax.runtime.policy.rules.plugin_loader import (
    PolicyRuleLoadPolicy,
    load_policy_rule_plugin_report,
)
from intergrax.runtime.policy.rules.provenance import (
    PolicyBundleProvenance,
    PolicyHandlerProvenance,
    PolicyRulesSourceKind,
    digest_inline_rules,
    digest_policy_rules_file,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule

_STANDARD_POLICY_LOAD_POLICY = PolicyRuleLoadPolicy(on_load_failure="isolate")


def build_runtime_policy_bundle(
    *,
    require_human_on_critical: bool = True,
    domain_fragments: dict[str, object] | None = None,
    execution_mode: ExecutionMode | None = None,
    policy_rules: PolicyRulesProfile | None = None,
    discover_entry_points: bool | None = None,
    package_qualification_lookup: (
        Callable[[EntryPointSpec], PluginQualificationResult | None] | None
    ) = None,
) -> RuntimePolicyBundle:
    """Default harness policy bundle for lab and product hosts."""
    fragments = dict(domain_fragments or {})
    if execution_mode is not None:
        fragments["execution_mode"] = execution_mode.value
        fragments["runtime_policies"] = runtime_policies_for_execution_mode(execution_mode)
    declarative_runtime = _build_declarative_policy_runtime(
        policy_rules,
        discover_entry_points=discover_entry_points,
        execution_mode=execution_mode,
        package_qualification_lookup=package_qualification_lookup,
    )
    return RuntimePolicyBundle(
        require_human_on_critical=require_human_on_critical,
        domain_fragments=fragments,
        declarative_policy_runtime=declarative_runtime,
    )


def wire_policy_bundle(
    env: ApplicationEnvironmentProfile,
    *,
    package_qualifications: PlatformPluginPackageQualificationBundle | None = None,
) -> RuntimePolicyBundle:
    """Merge policy rules, domain fragments, execution mode, cost, and evaluation governance."""
    cost_wiring = wire_application_cost(env)
    evaluation_wiring = wire_application_evaluation(env)
    critic_wiring = wire_application_critic(env)
    qualification_lookup = (
        package_qualifications.lookup_for_entry_point
        if package_qualifications is not None
        else None
    )
    base = build_runtime_policy_bundle(
        domain_fragments={
            **env.domain_policy_fragments,
            **cost_wiring.domain_fragments,
            **evaluation_wiring.domain_fragments,
            **critic_wiring.domain_fragments,
        },
        execution_mode=env.execution_mode,
        policy_rules=env.policy_rules,
        discover_entry_points=discover_plugins_enabled(),
        package_qualification_lookup=qualification_lookup,
    )
    if cost_wiring.budget_policy is None:
        return base
    return RuntimePolicyBundle(
        tool_access=base.tool_access,
        budget=cost_wiring.budget_policy,
        plan_loop=base.plan_loop,
        require_human_on_critical=base.require_human_on_critical,
        domain_fragments=base.domain_fragments,
        declarative_policy_runtime=base.declarative_policy_runtime,
    )


def _build_declarative_policy_runtime(
    policy_rules: PolicyRulesProfile | None,
    *,
    discover_entry_points: bool | None,
    execution_mode: ExecutionMode | None = None,
    package_qualification_lookup: (
        Callable[[EntryPointSpec], PluginQualificationResult | None] | None
    ) = None,
) -> DeclarativePolicyRuntime | None:
    if policy_rules is None:
        return None
    rules = _resolve_policy_rules(policy_rules)
    enforcement_mode = policy_rules.policy_enforcement_mode
    allowlist = _resolve_handler_allowlist(policy_rules)
    registry = PolicyRuleRegistry()
    discover = (
        discover_plugins_enabled()
        if discover_entry_points is None
        else discover_entry_points
    )
    require_production_admission = execution_mode is ExecutionMode.STRICT
    load_policy = PolicyRuleLoadPolicy(
        on_load_failure=_STANDARD_POLICY_LOAD_POLICY.on_load_failure,
        allowed_handler_ids=allowlist,
        require_production_admission=require_production_admission,
        package_qualification_lookup=package_qualification_lookup,
        platform_version=resolve_host_platform_version(),
    )
    if discover:
        plugin_outcome = load_policy_rule_plugin_report(registry, policy=load_policy)
        load_report = plugin_outcome.report
        external_handler_provenance = plugin_outcome.handler_provenance
    else:
        load_report = DomainPluginLoadReport.empty(EP_POLICY_RULES)
        external_handler_provenance = ()
    provenance = _build_provenance(
        policy_rules,
        rules=rules,
        load_report=load_report,
        external_handler_provenance=external_handler_provenance,
    )
    return DeclarativePolicyRuntime(
        registry=registry,
        rules=rules,
        load_report=load_report,
        enforcement_mode=enforcement_mode,
        provenance=provenance,
    )


def _resolve_policy_rules(profile: PolicyRulesProfile) -> tuple[DeclarativePolicyRule, ...]:
    rules: list[DeclarativePolicyRule] = []
    if profile.rules_path is not None:
        rules.extend(load_policy_rules_from_path(Path(profile.rules_path)))
    for item in profile.inline_rules:
        rules.append(DeclarativePolicyRule.model_validate(item))
    return tuple(rules)


def _resolve_handler_allowlist(
    profile: PolicyRulesProfile,
) -> frozenset[str] | None:
    if not profile.allowed_handler_ids:
        return None
    return frozenset(profile.allowed_handler_ids)


def _build_provenance(
    profile: PolicyRulesProfile,
    *,
    rules: tuple[DeclarativePolicyRule, ...],
    load_report: DomainPluginLoadReport,
    external_handler_provenance: tuple[PolicyHandlerProvenance, ...],
) -> PolicyBundleProvenance:
    has_file = profile.rules_path is not None
    has_inline = bool(profile.inline_rules)
    if has_file and has_inline:
        source_kind: PolicyRulesSourceKind = "mixed"
    elif has_file:
        source_kind = "file"
    else:
        source_kind = "inline"

    rules_digest = _rules_digest(profile, rules)
    rejected_handler_ids = tuple(
        sorted(
            item.plugin_id
            for item in load_report.rejected
            if item.plugin_id is not None
        )
    )
    return PolicyBundleProvenance(
        source_kind=source_kind,
        rules_path=str(profile.rules_path) if profile.rules_path is not None else None,
        rules_digest_sha256=rules_digest,
        handler_provenance=external_handler_provenance,
        rejected_handler_ids=rejected_handler_ids,
    )


def _rules_digest(
    profile: PolicyRulesProfile,
    rules: tuple[DeclarativePolicyRule, ...],
) -> str:
    if profile.rules_path is not None and not profile.inline_rules:
        return digest_policy_rules_file(Path(profile.rules_path))
    if profile.rules_path is not None and profile.inline_rules:
        file_digest = digest_policy_rules_file(Path(profile.rules_path))
        inline_digest = digest_inline_rules(rules)
        combined = f"{file_digest}:{inline_digest}".encode("utf-8")
        return hashlib.sha256(combined).hexdigest()
    return digest_inline_rules(rules)

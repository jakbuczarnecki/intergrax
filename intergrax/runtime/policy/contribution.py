# © Artur Czarnecki. All rights reserved.

"""Typed plugin PolicyDefinition contribution and catalog composition (G4B-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.core.distribution import DistributionPackageIdentity
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_POLICY_DEFINITIONS,
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
    resolve_entry_point_distribution_identity,
    resolve_host_platform_version,
)
from intergrax.runtime.policy.builtin_catalog import build_policy_catalog
from intergrax.runtime.policy.catalog import PolicyCatalog
from intergrax.runtime.policy.configuration_contract import (
    ConfigurationContractBinding,
    ConfigurationContractRegistry,
    build_configuration_contract_registry,
    built_in_configuration_contract_ids,
)
from intergrax.runtime.policy.rules.plugin_loader import PolicyRuleLoadPolicy
from intergrax.runtime.policy.rules.provenance import PolicyHandlerProvenance
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry

PolicyDefinitionLoadPolicy = PolicyRuleLoadPolicy


@dataclass(frozen=True, slots=True)
class GovernancePolicyContribution:
    """Immutable typed contribution binding one plugin PolicyDefinition to package identity."""

    definition: PolicyDefinition
    package_identity: DistributionPackageIdentity
    configuration_contract_binding: ConfigurationContractBinding | None = None


@dataclass(frozen=True, slots=True)
class PolicyDefinitionPluginLoadOutcome:
    report: DomainPluginLoadReport
    contributions: tuple[GovernancePolicyContribution, ...]


def _production_admission_rejections(
    policy: PolicyDefinitionLoadPolicy,
) -> tuple[frozenset[str], list[PluginAdmissionRejection]]:
    if not policy.require_production_admission:
        return frozenset(), []

    platform_version = policy.platform_version or resolve_host_platform_version()
    lookup = policy.package_qualification_lookup
    rejected_names: set[str] = set()
    rejected: list[PluginAdmissionRejection] = []

    for spec in iter_entry_point_specs(EP_POLICY_DEFINITIONS):
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


def _resolve_contribution_definitions(target: object) -> tuple[tuple[PolicyDefinition, ConfigurationContractBinding | None], ...]:
    resolved = instantiate_entry_point_target(target)
    if callable(resolved) and not isinstance(resolved, type):
        resolved = resolved()
    if isinstance(resolved, GovernancePolicyContribution):
        return ((resolved.definition, resolved.configuration_contract_binding),)
    if isinstance(resolved, PolicyDefinition):
        return ((resolved, None),)
    if isinstance(resolved, tuple):
        items: list[tuple[PolicyDefinition, ConfigurationContractBinding | None]] = []
        for item in resolved:
            if isinstance(item, GovernancePolicyContribution):
                items.append((item.definition, item.configuration_contract_binding))
            elif isinstance(item, PolicyDefinition):
                items.append((item, None))
            else:
                raise TypeError(
                    "Policy definition entry point must return "
                    "GovernancePolicyContribution or PolicyDefinition values"
                )
        return tuple(items)
    raise TypeError(
        "Policy definition entry point must return GovernancePolicyContribution, "
        "PolicyDefinition, or a tuple thereof"
    )


def _bind_contribution(
    definition: PolicyDefinition,
    spec: EntryPointSpec,
    *,
    configuration_contract_binding: ConfigurationContractBinding | None = None,
) -> GovernancePolicyContribution | PluginAdmissionRejection:
    package_identity = resolve_entry_point_distribution_identity(spec)
    if package_identity is None:
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.UNRESOLVED_PACKAGE_IDENTITY,
            reason="Policy definition contribution package identity could not be resolved.",
            plugin_id=definition.policy_id,
            fail_closed=True,
        )
    if definition.source is not PolicyDefinitionSource.PLUGIN:
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.INVALID_POLICY_CONTRIBUTION_SOURCE,
            reason=(
                f"Plugin policy contribution must use source "
                f"{PolicyDefinitionSource.PLUGIN.value!r}, got {definition.source.value!r}."
            ),
            plugin_id=definition.policy_id,
            fail_closed=True,
        )
    return GovernancePolicyContribution(
        definition=definition,
        package_identity=package_identity,
        configuration_contract_binding=configuration_contract_binding,
    )


def _validate_handler_binding(
    contribution: GovernancePolicyContribution,
    spec: EntryPointSpec,
    registry: PolicyRuleRegistry,
    handler_provenance: dict[str, PolicyHandlerProvenance],
) -> PluginAdmissionRejection | None:
    handler_id = contribution.definition.handler_id
    if registry.resolve(handler_id) is None:
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.POLICY_HANDLER_BINDING_MISSING,
            reason=f"Policy handler {handler_id!r} is not registered.",
            plugin_id=contribution.definition.policy_id,
            fail_closed=True,
        )

    provenance = handler_provenance.get(handler_id)
    if provenance is None or provenance.package_identity is None:
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.POLICY_HANDLER_PROVENANCE_MISMATCH,
            reason=(
                f"Policy handler {handler_id!r} has no admitted package provenance."
            ),
            plugin_id=contribution.definition.policy_id,
            fail_closed=True,
        )
    if provenance.package_identity != contribution.package_identity:
        return PluginAdmissionRejection(
            spec=spec,
            reason_code=PluginAdmissionReasonCode.POLICY_HANDLER_PROVENANCE_MISMATCH,
            reason=(
                "Policy definition contribution package identity does not match "
                f"admitted handler provenance for {handler_id!r}."
            ),
            plugin_id=contribution.definition.policy_id,
            fail_closed=True,
        )
    return None


def _validate_configuration_contract_binding(
    contribution: GovernancePolicyContribution,
    spec: EntryPointSpec,
) -> PluginAdmissionRejection | None:
    contract_id = contribution.definition.configuration_contract_id
    binding = contribution.configuration_contract_binding
    reserved = built_in_configuration_contract_ids()

    if binding is not None:
        if binding.contract_id in reserved:
            return PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.CONFIGURATION_CONTRACT_BUILTIN_RESERVED,
                reason=(
                    f"Plugin configuration contract {binding.contract_id!r} is reserved "
                    "by built-in bindings."
                ),
                plugin_id=contribution.definition.policy_id,
                fail_closed=True,
            )
        if binding.contract_id != contract_id:
            return PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.POLICY_CONFIGURATION_CONTRACT_ID_MISMATCH,
                reason=(
                    "Policy definition configuration_contract_id "
                    f"{contract_id!r} does not match binding contract_id "
                    f"{binding.contract_id!r}."
                ),
                plugin_id=contribution.definition.policy_id,
                fail_closed=True,
            )
        return None

    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.POLICY_CONFIGURATION_CONTRACT_BINDING_MISSING,
        reason=(
            f"No configuration contract binding admitted for {contract_id!r} "
            f"from package {contribution.package_identity.name!r}."
        ),
        plugin_id=contribution.definition.policy_id,
        fail_closed=True,
    )


def load_policy_definition_plugin_report(
    registry: PolicyRuleRegistry,
    handler_provenance: tuple[PolicyHandlerProvenance, ...],
    *,
    policy: PolicyDefinitionLoadPolicy | None = None,
) -> PolicyDefinitionPluginLoadOutcome:
    """Load ``intergrax.policy_definitions`` EPs with provenance-bound validation."""
    chosen = policy if policy is not None else PolicyDefinitionLoadPolicy()
    skip_names, production_rejected = _production_admission_rejections(chosen)
    provenance_by_handler = {item.rule_id: item for item in handler_provenance}
    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = list(production_rejected)
    failed: list[EntryPointLoadResult] = []
    contributions: list[GovernancePolicyContribution] = []

    for result in load_entry_point_targets(
        EP_POLICY_DEFINITIONS,
        on_conflict=chosen.ep_name_conflict,
        on_load_failure=chosen.on_load_failure,
        skip_names=skip_names,
    ):
        if result.error is not None:
            failed.append(result)
            continue
        try:
            definition_items = _resolve_contribution_definitions(result.target)
        except Exception as exc:
            if chosen.on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue

        pending: list[GovernancePolicyContribution] = []
        entry_rejections: list[PluginAdmissionRejection] = []
        entry_rejected = False
        for definition, configuration_contract_binding in definition_items:
            bound = _bind_contribution(
                definition,
                result.spec,
                configuration_contract_binding=configuration_contract_binding,
            )
            if isinstance(bound, PluginAdmissionRejection):
                entry_rejections.append(bound)
                entry_rejected = True
                continue
            handler_rejection = _validate_handler_binding(
                bound,
                result.spec,
                registry,
                provenance_by_handler,
            )
            if handler_rejection is not None:
                entry_rejections.append(handler_rejection)
                entry_rejected = True
                continue
            binding_rejection = _validate_configuration_contract_binding(
                bound,
                result.spec,
            )
            if binding_rejection is not None:
                entry_rejections.append(binding_rejection)
                entry_rejected = True
                continue
            pending.append(bound)

        if entry_rejected:
            rejected.extend(entry_rejections)
        else:
            contributions.extend(pending)
            if definition_items:
                accepted.append(result.spec)

    accepted_tuple = tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value)))
    report = DomainPluginLoadReport(
        group=EP_POLICY_DEFINITIONS,
        accepted=accepted_tuple,
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(
            sorted(failed, key=lambda item: (item.spec.name, item.spec.value))
        ),
        registered_count=len(contributions),
    )
    return PolicyDefinitionPluginLoadOutcome(
        report=report,
        contributions=tuple(
            sorted(
                contributions,
                key=lambda item: (item.definition.policy_id, item.definition.version),
            )
        ),
    )


def build_composed_policy_catalog(
    registry: PolicyRuleRegistry,
    handler_provenance: tuple[PolicyHandlerProvenance, ...],
    *,
    policy: PolicyDefinitionLoadPolicy | None = None,
) -> tuple[PolicyCatalog, PolicyDefinitionPluginLoadOutcome]:
    """Admit plugin policy definitions and compose the final immutable PolicyCatalog."""
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=policy,
    )
    plugin_definitions = tuple(
        contribution.definition for contribution in outcome.contributions
    )
    catalog = build_policy_catalog(plugin_definitions=plugin_definitions)
    return catalog, outcome


def build_composed_configuration_contract_registry(
    contributions: tuple[GovernancePolicyContribution, ...],
) -> ConfigurationContractRegistry:
    """Compose built-in and plugin configuration contract bindings from admitted contributions."""
    plugin_bindings = tuple(
        binding
        for contribution in contributions
        if (binding := contribution.configuration_contract_binding) is not None
    )
    return build_configuration_contract_registry(plugin_bindings=plugin_bindings)

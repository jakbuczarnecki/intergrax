# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre-load admission planning for Decision platform plugins (P0-A-R2/R3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import EntryPointSpec, iter_entry_point_specs
from intergrax.core.plugins.selection_ref import (
    PlatformPluginSelectionRef,
    entry_point_spec_matches_selection_ref,
)
from intergrax.runtime.decision_plugin_manifest_binding import (
    ManifestCapabilityBindingDisposition,
    validate_manifest_capability_binding,
)
from intergrax.runtime.decision_plugin_policy import (
    DecisionPluginLoadPolicy,
    production_admission_rejection_for_spec,
)


@dataclass(frozen=True, slots=True)
class AdmittedDecisionPlugin:
    """One entry point approved for target load after metadata admission."""

    spec: EntryPointSpec
    expected_plugin_id: str | None


@dataclass(frozen=True, slots=True)
class DecisionPluginAdmissionPlan:
    """Metadata-only admission outcome before any entry-point target import."""

    admitted: tuple[AdmittedDecisionPlugin, ...]
    rejected: tuple[PluginAdmissionRejection, ...]

    def expected_plugin_id_for(self, spec: EntryPointSpec) -> str | None:
        for item in self.admitted:
            if item.spec.name == spec.name and item.spec.value == spec.value:
                return item.expected_plugin_id
        return None


def _requested_locator_not_found(
    ref: PlatformPluginSelectionRef,
    *,
    group: str,
) -> PluginAdmissionRejection:
    return PluginAdmissionRejection(
        spec=ref.unresolved_entry_point_spec(),
        reason_code=PluginAdmissionReasonCode.REQUESTED_PLUGIN_LOCATOR_NOT_FOUND,
        reason=(
            f"Requested Decision plugin {ref.plugin_id!r} was not discovered for "
            f"distribution {ref.distribution!r}, entry point group {group!r}, "
            f"name {ref.entry_point_name!r}."
        ),
        plugin_id=ref.plugin_id,
        fail_closed=True,
    )


def _requested_locator_ambiguous(
    ref: PlatformPluginSelectionRef,
    *,
    matches: tuple[EntryPointSpec, ...],
) -> PluginAdmissionRejection:
    names = ", ".join(sorted(spec.name for spec in matches))
    return PluginAdmissionRejection(
        spec=matches[0],
        reason_code=PluginAdmissionReasonCode.REQUESTED_PLUGIN_LOCATOR_AMBIGUOUS,
        reason=(
            f"Requested Decision plugin {ref.plugin_id!r} matches multiple entry points "
            f"for locator {ref.location_key!r}: {names}."
        ),
        plugin_id=ref.plugin_id,
        fail_closed=True,
    )


def _manifest_plugin_id_mismatch(
    spec: EntryPointSpec,
    *,
    requested_plugin_id: str,
    manifest_plugin_id: str,
) -> PluginAdmissionRejection:
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.MANIFEST_PLUGIN_ID_MISMATCH,
        reason=(
            f"Manifest plugin_id {manifest_plugin_id!r} does not match requested "
            f"plugin_id {requested_plugin_id!r} for entry point {spec.name!r} in "
            f"group {spec.group!r}."
        ),
        plugin_id=requested_plugin_id,
        fail_closed=True,
    )


def _resolve_requested_matches(
    group: str,
    requested_plugins: tuple[PlatformPluginSelectionRef, ...],
    discovered: tuple[EntryPointSpec, ...],
) -> tuple[
    dict[EntryPointSpec, PlatformPluginSelectionRef],
    list[PluginAdmissionRejection],
]:
    rejected: list[PluginAdmissionRejection] = []
    spec_to_ref: dict[EntryPointSpec, PlatformPluginSelectionRef] = {}

    for ref in requested_plugins:
        if ref.entry_point_group != group:
            rejected.append(_requested_locator_not_found(ref, group=group))
            continue
        matches = [
            spec
            for spec in discovered
            if entry_point_spec_matches_selection_ref(spec, ref)
        ]
        if not matches:
            rejected.append(_requested_locator_not_found(ref, group=group))
            continue
        if len(matches) > 1:
            rejected.append(_requested_locator_ambiguous(ref, matches=tuple(matches)))
            continue
        spec_to_ref[matches[0]] = ref

    return spec_to_ref, rejected


def _admit_selected_spec(
    spec: EntryPointSpec,
    *,
    ref: PlatformPluginSelectionRef,
    domain: str,
    required_capability_id: str,
    policy: DecisionPluginLoadPolicy,
    plugin_id_owner: dict[str, EntryPointSpec],
) -> tuple[AdmittedDecisionPlugin | None, list[PluginAdmissionRejection], EntryPointSpec | None]:
    rejected: list[PluginAdmissionRejection] = []
    needs_manifest = policy.require_manifest_capability_binding

    binding = validate_manifest_capability_binding(
        spec,
        domain=domain,
        capability_id=required_capability_id,
    )
    if binding.disposition is not ManifestCapabilityBindingDisposition.VALID:
        if binding.rejection is not None:
            rejected.append(binding.rejection)
        return None, rejected, None

    declared_plugin_id = (
        binding.descriptor.plugin_id if binding.descriptor is not None else None
    )
    if needs_manifest and declared_plugin_id is None:
        rejected.append(
            PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
                reason=(
                    f"Entry point {spec.name!r} in group {spec.group!r} "
                    "must declare plugin_id in Platform Plugin manifest "
                    "capabilities for pre-load selection."
                ),
                fail_closed=True,
            ),
        )
        return None, rejected, None

    if declared_plugin_id is not None and declared_plugin_id != ref.plugin_id:
        rejected.append(
            _manifest_plugin_id_mismatch(
                spec,
                requested_plugin_id=ref.plugin_id,
                manifest_plugin_id=declared_plugin_id,
            ),
        )
        return None, rejected, None

    production = production_admission_rejection_for_spec(spec, policy)
    if production is not None:
        rejected.append(production)
        return None, rejected, None

    metadata_plugin_id = ref.plugin_id
    if metadata_plugin_id in plugin_id_owner:
        prior = plugin_id_owner[metadata_plugin_id]
        collision_reason = (
            f"Duplicate canonical plugin_id {metadata_plugin_id!r} declared for "
            f"entry points {prior.name!r} and {spec.name!r} in group {spec.group!r}."
        )
        rejected.append(
            PluginAdmissionRejection(
                spec=prior,
                reason_code=PluginAdmissionReasonCode.METADATA_PLUGIN_ID_COLLISION,
                reason=collision_reason,
                plugin_id=metadata_plugin_id,
                fail_closed=True,
            ),
        )
        rejected.append(
            PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.METADATA_PLUGIN_ID_COLLISION,
                reason=collision_reason,
                plugin_id=metadata_plugin_id,
                fail_closed=True,
            ),
        )
        return None, rejected, prior

    plugin_id_owner[metadata_plugin_id] = spec
    return (
        AdmittedDecisionPlugin(spec=spec, expected_plugin_id=metadata_plugin_id),
        rejected,
        None,
    )


def _admit_unrestricted_spec(
    spec: EntryPointSpec,
    *,
    domain: str,
    required_capability_id: str,
    policy: DecisionPluginLoadPolicy,
    plugin_id_owner: dict[str, EntryPointSpec],
) -> tuple[AdmittedDecisionPlugin | None, list[PluginAdmissionRejection], EntryPointSpec | None]:
    """Admit one discovered entry point when no explicit application selection is configured."""
    rejected: list[PluginAdmissionRejection] = []
    declared_plugin_id: str | None = None

    if policy.require_manifest_capability_binding:
        binding = validate_manifest_capability_binding(
            spec,
            domain=domain,
            capability_id=required_capability_id,
        )
        if binding.disposition is not ManifestCapabilityBindingDisposition.VALID:
            if binding.rejection is not None:
                rejected.append(binding.rejection)
            return None, rejected, None
        if binding.descriptor is not None:
            declared_plugin_id = binding.descriptor.plugin_id

    if policy.require_production_admission:
        production = production_admission_rejection_for_spec(spec, policy)
        if production is not None:
            rejected.append(production)
            return None, rejected, None

    metadata_plugin_id = declared_plugin_id
    if metadata_plugin_id is not None:
        if metadata_plugin_id in plugin_id_owner:
            prior = plugin_id_owner[metadata_plugin_id]
            collision_reason = (
                f"Duplicate canonical plugin_id {metadata_plugin_id!r} declared for "
                f"entry points {prior.name!r} and {spec.name!r} in group {spec.group!r}."
            )
            rejected.append(
                PluginAdmissionRejection(
                    spec=prior,
                    reason_code=PluginAdmissionReasonCode.METADATA_PLUGIN_ID_COLLISION,
                    reason=collision_reason,
                    plugin_id=metadata_plugin_id,
                    fail_closed=True,
                ),
            )
            rejected.append(
                PluginAdmissionRejection(
                    spec=spec,
                    reason_code=PluginAdmissionReasonCode.METADATA_PLUGIN_ID_COLLISION,
                    reason=collision_reason,
                    plugin_id=metadata_plugin_id,
                    fail_closed=True,
                ),
            )
            return None, rejected, prior
        plugin_id_owner[metadata_plugin_id] = spec

    return (
        AdmittedDecisionPlugin(spec=spec, expected_plugin_id=metadata_plugin_id),
        rejected,
        None,
    )


def plan_decision_plugin_admission(
    group: str,
    *,
    domain: str,
    required_capability_id: str,
    policy: DecisionPluginLoadPolicy,
    requested_plugins: tuple[PlatformPluginSelectionRef, ...] | None,
) -> DecisionPluginAdmissionPlan:
    """Discover metadata, select, and admit entry points without importing targets."""
    discovered = iter_entry_point_specs(group)
    rejected: list[PluginAdmissionRejection] = []
    admitted_candidates: list[AdmittedDecisionPlugin] = []
    plugin_id_owner: dict[str, EntryPointSpec] = {}

    if requested_plugins is not None:
        spec_to_ref, locator_rejections = _resolve_requested_matches(
            group,
            requested_plugins,
            discovered,
        )
        rejected.extend(locator_rejections)
        for spec, ref in sorted(
            spec_to_ref.items(),
            key=lambda item: (item[0].name, item[0].value),
        ):
            admitted, admission_rejections, revoke_prior = _admit_selected_spec(
                spec,
                ref=ref,
                domain=domain,
                required_capability_id=required_capability_id,
                policy=policy,
                plugin_id_owner=plugin_id_owner,
            )
            rejected.extend(admission_rejections)
            if revoke_prior is not None:
                admitted_candidates = [
                    item
                    for item in admitted_candidates
                    if item.spec.name != revoke_prior.name
                ]
            if admitted is not None:
                admitted_candidates.append(admitted)
    else:
        for spec in discovered:
            admitted, admission_rejections, revoke_prior = _admit_unrestricted_spec(
                spec,
                domain=domain,
                required_capability_id=required_capability_id,
                policy=policy,
                plugin_id_owner=plugin_id_owner,
            )
            rejected.extend(admission_rejections)
            if revoke_prior is not None:
                admitted_candidates = [
                    item
                    for item in admitted_candidates
                    if item.spec.name != revoke_prior.name
                ]
            if admitted is not None:
                admitted_candidates.append(admitted)

    return DecisionPluginAdmissionPlan(
        admitted=tuple(admitted_candidates),
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value)),
        ),
    )

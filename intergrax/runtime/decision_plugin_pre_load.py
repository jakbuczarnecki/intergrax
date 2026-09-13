# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre-load admission planning for Decision platform plugins (P0-A-R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import EntryPointSpec, iter_entry_point_specs
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


def _not_selected_rejection(
    spec: EntryPointSpec,
    *,
    kind_value: str,
    label: str,
) -> PluginAdmissionRejection:
    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.PLUGIN_NOT_SELECTED,
        reason=(
            f"{label} {kind_value!r} is installed but not selected by application profile."
        ),
        plugin_id=kind_value,
        fail_closed=False,
    )


def _plugin_kind_selected(
    kind_value: str,
    allowed_kinds: frozenset[str] | None,
) -> bool:
    if allowed_kinds is None:
        return True
    return kind_value in allowed_kinds


def plan_decision_plugin_admission(
    group: str,
    *,
    domain: str,
    required_capability_id: str,
    policy: DecisionPluginLoadPolicy,
    kind_allowlist: frozenset[str] | None,
) -> DecisionPluginAdmissionPlan:
    """Discover metadata, select, and admit entry points without importing targets."""
    rejected: list[PluginAdmissionRejection] = []
    admitted_candidates: list[AdmittedDecisionPlugin] = []
    plugin_id_owner: dict[str, EntryPointSpec] = {}

    for spec in iter_entry_point_specs(group):
        declared_plugin_id: str | None = None
        binding = None
        needs_manifest = policy.require_manifest_capability_binding or kind_allowlist is not None

        if needs_manifest:
            binding = validate_manifest_capability_binding(
                spec,
                domain=domain,
                capability_id=required_capability_id,
            )
            if binding.disposition is ManifestCapabilityBindingDisposition.VALID:
                if binding.descriptor is not None and binding.descriptor.plugin_id is not None:
                    declared_plugin_id = binding.descriptor.plugin_id
            elif binding.rejection is not None:
                if kind_allowlist is not None:
                    rejected.append(binding.rejection)
                    continue
                if policy.require_manifest_capability_binding:
                    rejected.append(binding.rejection)
                    continue

        if kind_allowlist is not None:
            if declared_plugin_id is None:
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
                continue
            if not _plugin_kind_selected(declared_plugin_id, kind_allowlist):
                rejected.append(
                    _not_selected_rejection(
                        spec,
                        kind_value=declared_plugin_id,
                        label="DecisionPlugin",
                    ),
                )
                continue
            production = production_admission_rejection_for_spec(spec, policy)
            if production is not None:
                rejected.append(production)
                continue

        if kind_allowlist is None and policy.require_production_admission:
            production = production_admission_rejection_for_spec(spec, policy)
            if production is not None:
                rejected.append(production)
                continue

        metadata_plugin_id = declared_plugin_id
        if metadata_plugin_id is not None:
            if metadata_plugin_id in plugin_id_owner:
                prior = plugin_id_owner[metadata_plugin_id]
                admitted_candidates = [
                    item for item in admitted_candidates if item.spec.name != prior.name
                ]
                collision_reason = (
                    f"Duplicate canonical plugin_id {metadata_plugin_id!r} declared for "
                    f"entry points {prior.name!r} and {spec.name!r} in group {group!r}."
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
                continue
            plugin_id_owner[metadata_plugin_id] = spec

        admitted_candidates.append(
            AdmittedDecisionPlugin(
                spec=spec,
                expected_plugin_id=metadata_plugin_id,
            ),
        )

    return DecisionPluginAdmissionPlan(
        admitted=tuple(admitted_candidates),
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value)),
        ),
    )

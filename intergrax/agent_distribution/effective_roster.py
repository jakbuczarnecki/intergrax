# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic effective roster merge and installed-agent requirement builder (AP-6)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intergrax.agent_distribution._immutable_json import (
    DistributionJsonValue,
    assert_distribution_json_object,
    freeze_distribution_json_object,
)
from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.dependency import (
    InstalledAgentPackageRequirement,
    InstalledAgentRequirementSet,
)
from intergrax.agent_distribution.errors import (
    DependencySpecificationError,
    EffectiveRosterConflict,
)
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.roster import (
    EffectiveRoster,
    EffectiveRosterEntry,
    ManifestDefaultAgentDeclaration,
)
from intergrax.agent_distribution.stores import (
    AgentArtifactMetadataStore,
    AgentInstallationStore,
)


def _deep_merge_config(
    base: Mapping[str, DistributionJsonValue],
    override: Mapping[str, DistributionJsonValue],
) -> Mapping[str, DistributionJsonValue]:
    merged: dict[str, DistributionJsonValue] = assert_distribution_json_object(
        base, field_name="merged_config"
    )
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge_config(base_value, override_value)  # type: ignore[assignment]
        else:
            merged[key] = override_value
    return freeze_distribution_json_object(merged)


def _index_manifest_defaults(
    manifest_defaults: Sequence[ManifestDefaultAgentDeclaration],
) -> dict[str, ManifestDefaultAgentDeclaration]:
    indexed: dict[str, ManifestDefaultAgentDeclaration] = {}
    for declaration in manifest_defaults:
        if declaration.logical_agent_id in indexed:
            raise EffectiveRosterConflict(
                f"duplicate manifest logical_agent_id {declaration.logical_agent_id}"
            )
        indexed[declaration.logical_agent_id] = declaration
    return indexed


def _index_durable_bindings(
    durable_bindings: Sequence[ApplicationAgentBinding],
    *,
    application_id: str,
    application_environment_id: str,
) -> tuple[dict[str, ApplicationAgentBinding], frozenset[str]]:
    active_by_logical: dict[str, ApplicationAgentBinding] = {}
    tombstoned_ids: set[str] = set()
    for binding in durable_bindings:
        if binding.application_id != application_id:
            continue
        if binding.application_environment_id != application_environment_id:
            continue
        if binding.tombstone:
            tombstoned_ids.add(binding.logical_agent_id)
            continue
        if binding.logical_agent_id in active_by_logical:
            raise EffectiveRosterConflict(
                f"duplicate durable binding for logical_agent_id {binding.logical_agent_id}"
            )
        active_by_logical[binding.logical_agent_id] = binding
    return active_by_logical, frozenset(tombstoned_ids)


def _binding_revisions_for_environment(
    durable_bindings: Sequence[ApplicationAgentBinding],
    *,
    application_id: str,
    application_environment_id: str,
) -> tuple[int, ...]:
    revisions = sorted(
        binding.binding_revision
        for binding in durable_bindings
        if binding.application_id == application_id
        and binding.application_environment_id == application_environment_id
    )
    return tuple(revisions)


def _resolve_package_identity(
    *,
    installation_store: AgentInstallationStore,
    application_environment_id: str,
    installation_slot_id: str,
    manifest: ManifestDefaultAgentDeclaration | None,
    binding: ApplicationAgentBinding | None,
    effective_enablement: bool,
) -> tuple[str | None, str, str]:
    active = installation_store.get_active_installation_for_slot(
        application_environment_id,
        installation_slot_id,
    )
    if active is not None:
        _validate_installation_package_line(
            active=active,
            manifest=manifest,
            binding=binding,
        )
        identity = active.package_identity
        return (
            active.installation_id,
            identity.package_digest,
            identity.distribution_package_id,
        )

    if binding is not None and binding.builtin_package_ref is not None:
        return _resolve_builtin_digest(manifest=manifest, binding=binding)

    if manifest is not None and manifest.builtin_package_ref is not None:
        return None, manifest.package_digest, manifest.distribution_package_id

    if effective_enablement:
        raise EffectiveRosterConflict(
            f"enabled roster entry {binding.logical_agent_id if binding else manifest.logical_agent_id if manifest else installation_slot_id} "
            "requires resolvable active installation"
        )

    if manifest is not None:
        return None, manifest.package_digest, manifest.distribution_package_id

    if binding is not None and binding.active_installation_id is not None:
        cached = installation_store.get_installation(binding.active_installation_id)
        if cached is not None:
            identity = cached.package_identity
            return (
                cached.installation_id,
                identity.package_digest,
                identity.distribution_package_id,
            )

    raise EffectiveRosterConflict(
        f"roster entry for slot {installation_slot_id} lacks resolvable package identity"
    )


def _resolve_builtin_digest(
    *,
    manifest: ManifestDefaultAgentDeclaration | None,
    binding: ApplicationAgentBinding,
) -> tuple[None, str, str]:
    if manifest is None:
        raise EffectiveRosterConflict(
            f"builtin binding {binding.application_binding_id} requires manifest defaults for digest"
        )
    return None, manifest.package_digest, manifest.distribution_package_id


def _validate_installation_package_line(
    *,
    active: AgentInstallationRecord,
    manifest: ManifestDefaultAgentDeclaration | None,
    binding: ApplicationAgentBinding | None,
) -> None:
    if active.installation_state is not InstallationState.INSTALLED_ACTIVE:
        raise EffectiveRosterConflict(
            f"slot {active.installation_slot_id} active record is not installed_active"
        )
    if not active.active_for_slot:
        raise EffectiveRosterConflict(
            f"slot {active.installation_slot_id} installation is not active_for_slot"
        )
    active_package_id = active.package_identity.distribution_package_id
    if manifest is not None and manifest.distribution_package_id != active_package_id:
        raise EffectiveRosterConflict(
            f"active installation package line {active_package_id} "
            f"conflicts with manifest {manifest.distribution_package_id}"
        )


def _resolve_effective_default_agent(
    *,
    manifest: ManifestDefaultAgentDeclaration | None,
    binding: ApplicationAgentBinding | None,
) -> bool:
    if binding is not None and binding.default_agent is not None:
        return binding.default_agent
    if manifest is not None:
        return manifest.default_agent
    return False


def _merge_entry(
    *,
    logical_agent_id: str,
    application_environment_id: str,
    manifest: ManifestDefaultAgentDeclaration | None,
    binding: ApplicationAgentBinding | None,
    installation_store: AgentInstallationStore,
) -> EffectiveRosterEntry:
    if binding is None and manifest is None:
        raise EffectiveRosterConflict(
            f"missing merge inputs for logical_agent_id {logical_agent_id}"
        )

    installation_slot_id = (
        binding.installation_slot_id
        if binding is not None
        else manifest.installation_slot_id  # type: ignore[union-attr]
    )
    if installation_slot_id is None:
        raise EffectiveRosterConflict(
            f"logical_agent_id {logical_agent_id} requires installation_slot_id"
        )

    effective_enablement = (
        binding.enablement if binding is not None else manifest.enabled
    )  # type: ignore[union-attr]
    effective_default_agent = _resolve_effective_default_agent(
        manifest=manifest,
        binding=binding,
    )
    if binding is not None and binding.tombstone:
        raise EffectiveRosterConflict(
            "tombstoned binding cannot produce effective roster entry"
        )

    base_config = manifest.config if manifest is not None else {}
    override_config = binding.config if binding is not None else {}
    merged_config = _deep_merge_config(base_config, override_config)

    secret_refs = binding.secret_refs if binding is not None else manifest.secret_refs  # type: ignore[union-attr]
    if binding is not None and binding.policy_overrides is not None:
        policy_overrides = binding.policy_overrides
    else:
        policy_overrides = manifest.policy_overrides if manifest is not None else None

    if binding is not None and binding.factory_reference is not None:
        factory_reference = binding.factory_reference
    else:
        factory_reference = manifest.factory_reference if manifest is not None else None

    active_installation_id, package_digest, distribution_package_id = (
        _resolve_package_identity(
            installation_store=installation_store,
            application_environment_id=application_environment_id,
            installation_slot_id=installation_slot_id,
            manifest=manifest,
            binding=binding,
            effective_enablement=effective_enablement,
        )
    )

    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id=installation_slot_id,
        active_installation_id=active_installation_id,
        package_digest=package_digest,
        distribution_package_id=distribution_package_id,
        effective_enablement=effective_enablement,
        effective_default_agent=effective_default_agent,
        merged_config=merged_config,
        secret_refs=secret_refs,
        policy_overrides=policy_overrides,
        factory_reference=factory_reference,
        application_binding_id=binding.application_binding_id
        if binding is not None
        else None,
        manifest_origin_ref=(
            binding.manifest_origin_ref
            if binding is not None and binding.manifest_origin_ref is not None
            else manifest.manifest_origin_ref
            if manifest is not None
            else None
        ),
    )


class EffectiveRosterBuilder:
    """Deterministic manifest-default + durable-binding merge (§13.2)."""

    def __init__(self, installation_store: AgentInstallationStore) -> None:
        self._installation_store = installation_store

    def build(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        manifest_release_id: str,
        manifest_defaults: Sequence[ManifestDefaultAgentDeclaration],
        durable_bindings: Sequence[ApplicationAgentBinding],
    ) -> EffectiveRoster:
        manifest_by_id = _index_manifest_defaults(manifest_defaults)
        bindings_by_id, tombstoned_ids = _index_durable_bindings(
            durable_bindings,
            application_id=application_id,
            application_environment_id=application_environment_id,
        )

        logical_agent_ids: set[str] = set(bindings_by_id)
        for logical_agent_id in manifest_by_id:
            if logical_agent_id not in tombstoned_ids:
                logical_agent_ids.add(logical_agent_id)

        entries: list[EffectiveRosterEntry] = []
        for logical_agent_id in sorted(logical_agent_ids):
            manifest = manifest_by_id.get(logical_agent_id)
            binding = bindings_by_id.get(logical_agent_id)
            entries.append(
                _merge_entry(
                    logical_agent_id=logical_agent_id,
                    application_environment_id=application_environment_id,
                    manifest=manifest,
                    binding=binding,
                    installation_store=self._installation_store,
                )
            )

        default_agent_ids = [
            entry.logical_agent_id
            for entry in entries
            if entry.effective_default_agent
        ]
        if len(default_agent_ids) > 1:
            raise EffectiveRosterConflict(
                "multiple default_agent=true entries after merge: "
                + ", ".join(sorted(default_agent_ids))
            )

        roster = EffectiveRoster(
            application_id=application_id,
            application_environment_id=application_environment_id,
            manifest_release_id=manifest_release_id,
            binding_revisions=_binding_revisions_for_environment(
                durable_bindings,
                application_id=application_id,
                application_environment_id=application_environment_id,
            ),
            entries=tuple(entries),
        )
        return roster.with_revision_id()


class InstalledAgentRequirementSetBuilder:
    """Convert EffectiveRoster to digest-pinned L2 requirements (§15.2)."""

    def __init__(self, metadata_store: AgentArtifactMetadataStore) -> None:
        self._metadata_store = metadata_store

    def build(self, effective_roster: EffectiveRoster) -> InstalledAgentRequirementSet:
        if effective_roster.effective_roster_revision_id is None:
            raise DependencySpecificationError(
                "effective roster must include effective_roster_revision_id"
            )

        requirements_by_package: dict[str, InstalledAgentPackageRequirement] = {}
        for entry in effective_roster.entries:
            if not entry.effective_enablement:
                continue
            requirement = self._requirement_for_entry(entry)
            existing = requirements_by_package.get(requirement.distribution_package_id)
            if existing is not None:
                if existing.package_digest != requirement.package_digest:
                    raise DependencySpecificationError(
                        f"conflicting digests for distribution_package_id "
                        f"{requirement.distribution_package_id}"
                    )
                continue
            requirements_by_package[requirement.distribution_package_id] = requirement

        ordered = tuple(
            sorted(
                requirements_by_package.values(),
                key=lambda item: (
                    item.distribution_package_id,
                    item.package_digest,
                    item.agent_project_metadata_ref,
                ),
            )
        )
        return InstalledAgentRequirementSet(
            effective_roster_revision_id=effective_roster.effective_roster_revision_id,
            agent_packages=ordered,
        )

    def _requirement_for_entry(
        self,
        entry: EffectiveRosterEntry,
    ) -> InstalledAgentPackageRequirement:
        metadata = self._metadata_store.get_by_digest(entry.package_digest)
        if metadata is None:
            raise DependencySpecificationError(
                f"missing artifact metadata for digest {entry.package_digest}"
            )
        if metadata.tombstoned:
            raise DependencySpecificationError(
                f"artifact metadata for digest {entry.package_digest} is tombstoned"
            )
        if metadata.distribution_package_id != entry.distribution_package_id:
            raise DependencySpecificationError(
                f"metadata package line {metadata.distribution_package_id} "
                f"conflicts with roster entry {entry.distribution_package_id}"
            )
        return InstalledAgentPackageRequirement(
            distribution_package_id=entry.distribution_package_id,
            package_digest=entry.package_digest,
            agent_project_metadata_ref=metadata.agent_project_metadata_ref,
        )

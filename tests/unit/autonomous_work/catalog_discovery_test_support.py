# © Artur Czarnecki. All rights reserved.

"""Shared catalog-backed AW discovery test wiring (Stage 9)."""

from __future__ import annotations

from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CapabilityCatalogDiscoveryDependencies,
    CapabilityCatalogSkillDiscoveryAdapter,
    CapabilityCatalogToolDiscoveryAdapter,
    SkillRegistryManifestLookup,
    identity_key_from_entry_identity,
)
from intergrax.capability_catalog import AvailabilityPreservingGovernanceEvaluator
from intergrax.capability_catalog.adapters.skill import SKILL_BUILTIN_CATALOG_SOURCE_ID
from intergrax.capability_catalog.adapters.tool import TOOL_BUILTIN_CATALOG_SOURCE_ID
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.evidence import CapabilityDiscoveryAvailabilityEvidence
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
)
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.contracts.capability_catalog.scope import (
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
)
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry

_BUILTIN_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id=TOOL_BUILTIN_CATALOG_SOURCE_ID,
    source_kind=CapabilitySourceKind.BUILTIN,
)
_BUILTIN_SKILL_SOURCE = CapabilitySourceIdentity(
    source_id=SKILL_BUILTIN_CATALOG_SOURCE_ID,
    source_kind=CapabilitySourceKind.BUILTIN,
)


def tool_catalog_entry(
    logical_id: str,
    *,
    source: CapabilitySourceIdentity = _BUILTIN_TOOL_SOURCE,
    version_label: str | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=logical_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
        ),
        display_label=logical_id,
    )


def skill_catalog_entry(
    logical_id: str,
    *,
    source: CapabilitySourceIdentity = _BUILTIN_SKILL_SOURCE,
    version_label: str | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=logical_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
        ),
        display_label=logical_id,
    )


def catalog_snapshot_from_registries(
    *,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    extra_entries: tuple[CapabilityCatalogEntry, ...] = (),
) -> CapabilityCatalogSnapshot:
    entries: list[CapabilityCatalogEntry] = list(extra_entries)
    for registered in tool_registry.list():
        entries.append(tool_catalog_entry(registered.contract.tool_id))
    for registered in skill_registry.list():
        manifest = registered.manifest
        entries.append(
            skill_catalog_entry(
                manifest.skill_id,
                version_label=manifest.version,
            ),
        )
    ordered = tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
    return CapabilityCatalogSnapshot(
        source_ids=("test.catalog",),
        entries=ordered,
    )


def host_availability_for_entries(
    *entries: CapabilityCatalogEntry,
) -> CapabilityDiscoveryAvailabilityEvidence:
    return CapabilityDiscoveryAvailabilityEvidence(
        host_available_keys=tuple(
            identity_key_from_entry_identity(entry.identity) for entry in entries
        ),
    )


def catalog_discovery_dependencies(
    *,
    snapshot: CapabilityCatalogSnapshot,
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence,
) -> CapabilityCatalogDiscoveryDependencies:
    return CapabilityCatalogDiscoveryDependencies(
        snapshot=snapshot,
        availability_evidence=availability_evidence,
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.NON_STRICT,
        ),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def catalog_tool_skill_adapters(
    *,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    extra_entries: tuple[CapabilityCatalogEntry, ...] = (),
    host_tool_ids: tuple[str, ...] | None = None,
    host_skill_ids: tuple[str, ...] | None = None,
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
) -> tuple[CapabilityCatalogToolDiscoveryAdapter, CapabilityCatalogSkillDiscoveryAdapter]:
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        extra_entries=extra_entries,
    )
    if availability_evidence is None:
        host_keys: dict[tuple[str, str, str, str], CapabilityCatalogEntry] = {}
        for entry in extra_entries:
            host_keys[entry.identity.sort_key] = entry
        resolved_tool_ids = (
            host_tool_ids
            if host_tool_ids is not None
            else tuple(reg.contract.tool_id for reg in tool_registry.list())
        )
        resolved_skill_ids = (
            host_skill_ids
            if host_skill_ids is not None
            else tuple(reg.manifest.skill_id for reg in skill_registry.list())
        )
        for entry in snapshot.entries:
            logical_id = entry.identity.logical.logical_id
            if entry.identity.kind is CapabilityKind.TOOL and logical_id in resolved_tool_ids:
                host_keys[entry.identity.sort_key] = entry
            if entry.identity.kind is CapabilityKind.SKILL and logical_id in resolved_skill_ids:
                host_keys[entry.identity.sort_key] = entry
        availability_evidence = host_availability_for_entries(*host_keys.values())
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=availability_evidence,
    )
    return (
        CapabilityCatalogToolDiscoveryAdapter(dependencies),
        CapabilityCatalogSkillDiscoveryAdapter(
            dependencies,
            manifest_lookup=SkillRegistryManifestLookup(skill_registry),
        ),
    )

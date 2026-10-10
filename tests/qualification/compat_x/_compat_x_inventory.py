# © Artur Czarnecki. All rights reserved.

"""COMPAT-X inventory SSOT — one record per discovered compatibility surface."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.compat_x._compat_x_discovery import discover_all_compat_surfaces
from tests.qualification.compat_x._compat_x_types import (
    CompatDomain,
    CompatSurfaceRecord,
    DiscoveredCompatSurface,
    EvolutionState,
    ExposureFacet,
    MigrationMechanismClass,
    ShimClass,
)

_PERSISTED_RUNTIME_KEYS = frozenset(
    {
        "runtime_checkpoint",
        "task_checkpoint",
        "pause_record",
        "scheduled_resume",
        "shared_task_context",
        "agent_context_bundle",
        "task_memory",
        "nexus_task_worker",
    }
)

_EVENT_RUNTIME_KEYS = frozenset({"runtime_event"})

_WIRE_RUNTIME_KEYS = frozenset(
    {
        "agent_decision",
        "human_request",
        "governance_resolution",
        "execution_interrupt",
        "policy_decision",
        "handoff",
        "agent_step",
        "partial_result",
        "validation_contract",
        "task_context_assembly",
    }
)


def _domain_for_surface(surface: DiscoveredCompatSurface) -> CompatDomain:
    path = surface.owner_module_path
    if surface.discovery_kind == "defect.persisted_without_version":
        return CompatDomain.RUNTIME
    if surface.discovery_kind == "compat.shim":
        return CompatDomain.COMPAT_SHIM
    if surface.discovery_kind.startswith("migration"):
        return CompatDomain.MIGRATION
    if surface.discovery_kind == "mechanism.policy_owner" and "plugin" in surface.surface_id:
        return CompatDomain.PLUGINS
    if surface.discovery_kind == "mechanism.policy_owner" and "external" in surface.surface_id:
        return CompatDomain.INTEGRATIONS
    if surface.discovery_kind == "mechanism.plugin":
        return CompatDomain.PLUGINS
    if surface.discovery_kind == "mechanism.integrations":
        return CompatDomain.INTEGRATIONS
    if surface.discovery_kind == "registry.contracts":
        return CompatDomain.CONTRACTS
    if surface.discovery_kind == "registry.runtime":
        return CompatDomain.RUNTIME
    if surface.discovery_kind == "event.payload":
        return CompatDomain.EVENTS
    if path.startswith("intergrax/agent_distribution"):
        return CompatDomain.DISTRIBUTION
    if path.startswith("intergrax/tools"):
        return CompatDomain.TOOLS
    if path.startswith("intergrax/skills"):
        return CompatDomain.SKILLS
    if path.startswith("intergrax/marketplace"):
        return CompatDomain.MARKETPLACE
    if path.startswith("intergrax/hosting"):
        return CompatDomain.HOSTING
    if path.startswith("intergrax/proofs") or path.startswith("intergrax/proof_data"):
        return CompatDomain.PROOFS
    if path.startswith("applications/contracts"):
        return CompatDomain.APPLICATIONS_BOUNDARY
    if path.startswith("intergrax/integrations"):
        return CompatDomain.INTEGRATIONS
    if path.startswith("intergrax/contracts"):
        return CompatDomain.CONTRACTS
    if path.startswith("intergrax/runtime"):
        return CompatDomain.RUNTIME
    return CompatDomain.RUNTIME


def _exposure_for_surface(surface: DiscoveredCompatSurface) -> frozenset[ExposureFacet]:
    facets: set[ExposureFacet] = set()
    if surface.discovery_kind == "compat.shim":
        facets.add(ExposureFacet.COMPATIBILITY_ADAPTER)
        return frozenset(facets)
    if surface.discovery_kind == "mechanism.policy_owner" and "plugin" in surface.surface_id:
        facets.add(ExposureFacet.PLUGIN_PROVIDER_CONTRACT)
        return frozenset(facets)
    if surface.discovery_kind == "mechanism.policy_owner" and "external" in surface.surface_id:
        facets.add(ExposureFacet.PLUGIN_PROVIDER_CONTRACT)
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
        return frozenset(facets)
    if surface.discovery_kind == "defect.persisted_without_version":
        facets.add(ExposureFacet.PERSISTED_SCHEMA)
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
        return frozenset(facets)
    if surface.discovery_kind == "mechanism.plugin":
        facets.add(ExposureFacet.PLUGIN_PROVIDER_CONTRACT)
        return frozenset(facets)
    if surface.discovery_kind == "mechanism.integrations":
        facets.add(ExposureFacet.PLUGIN_PROVIDER_CONTRACT)
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
        return frozenset(facets)
    if surface.discovery_kind == "registry.contracts":
        facets.add(ExposureFacet.PUBLIC_STABLE)
        facets.add(ExposureFacet.PERSISTED_SCHEMA)
        return frozenset(facets)
    if surface.discovery_kind == "registry.runtime":
        key = surface.contract_schema_identity
        if key in _PERSISTED_RUNTIME_KEYS:
            facets.add(ExposureFacet.PERSISTED_SCHEMA)
        if key in _EVENT_RUNTIME_KEYS:
            facets.add(ExposureFacet.EVENT_SCHEMA)
        if key in _WIRE_RUNTIME_KEYS or not facets:
            facets.add(ExposureFacet.WIRE_SCHEMA)
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
        return frozenset(facets)
    if surface.discovery_kind == "event.payload":
        facets.add(ExposureFacet.EVENT_SCHEMA)
        facets.add(ExposureFacet.WIRE_SCHEMA)
        return frozenset(facets)
    if surface.discovery_kind.startswith("migration"):
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
        return frozenset(facets)
    path = surface.owner_module_path
    if path.startswith("intergrax/agent_distribution"):
        facets.add(ExposureFacet.DISTRIBUTION_PACKAGE_CONTRACT)
    elif path.startswith("intergrax/tools") or path.startswith("intergrax/skills"):
        facets.add(ExposureFacet.PLUGIN_PROVIDER_CONTRACT)
    elif path.startswith("intergrax/proofs"):
        facets.add(ExposureFacet.PERSISTED_SCHEMA)
        facets.add(ExposureFacet.PUBLIC_STABLE)
    elif path.startswith("intergrax/hosting"):
        facets.add(ExposureFacet.WIRE_SCHEMA)
    else:
        facets.add(ExposureFacet.PLATFORM_INTERNAL)
    return frozenset(facets)


def _evolution_for_surface(surface: DiscoveredCompatSurface) -> frozenset[EvolutionState]:
    if surface.discovery_kind == "defect.persisted_without_version":
        return frozenset(
            {
                EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION,
                EvolutionState.VERSIONED_AND_POLICY_DEFINED,
            }
        )
    states: set[EvolutionState] = {EvolutionState.VERSIONED_AND_POLICY_DEFINED}
    if surface.discovery_kind == "event.payload":
        states.add(EvolutionState.EVENT_EVOLUTION_MISSING)
    if ExposureFacet.PERSISTED_SCHEMA in _exposure_for_surface(surface):
        states.add(EvolutionState.PERSISTED_MIGRATION_MISSING)
    if surface.discovery_kind == "mechanism.policy_owner" and "external" in surface.surface_id:
        states.update(
            {
                EvolutionState.PLUGIN_COMPATIBILITY_DEFINED,
                EvolutionState.DEPRECATION_POLICY_MISSING,
            }
        )
        states.discard(EvolutionState.PLUGIN_COMPATIBILITY_MISSING)
    if surface.discovery_kind == "mechanism.plugin":
        states.add(EvolutionState.PLUGIN_COMPATIBILITY_MISSING)
    if surface.discovery_kind == "mechanism.integrations":
        states.update(
            {
                EvolutionState.PLUGIN_COMPATIBILITY_DEFINED,
                EvolutionState.DEPRECATION_POLICY_MISSING,
            }
        )
        states.discard(EvolutionState.PLUGIN_COMPATIBILITY_MISSING)
    if surface.discovery_kind == "compat.shim":
        states.update(
            {
                EvolutionState.COMPATIBILITY_SHIM_SANCTIONED,
                EvolutionState.DEPRECATION_POLICY_MISSING,
            }
        )
    states.add(EvolutionState.DEPRECATION_POLICY_MISSING)
    return frozenset(states)


def _migration_class_for_surface(surface: DiscoveredCompatSurface) -> MigrationMechanismClass:
    if surface.discovery_kind != "migration.mechanism":
        return MigrationMechanismClass.NOT_APPLICABLE
    try:
        return MigrationMechanismClass(surface.current_version)
    except ValueError:
        return MigrationMechanismClass.UNOWNED_MIGRATION


def _shim_class_for_surface(surface: DiscoveredCompatSurface) -> ShimClass:
    if surface.discovery_kind != "compat.shim":
        return ShimClass.NOT_APPLICABLE
    try:
        return ShimClass(surface.current_version)
    except ValueError:
        return ShimClass.NOT_APPLICABLE


def _policy_owner(surface: DiscoveredCompatSurface) -> str:
    if surface.discovery_kind == "registry.contracts":
        return "intergrax/contracts/migrations/registry.py (CONTRACT_SCHEMA_REGISTRY)"
    if surface.discovery_kind == "registry.runtime":
        return "intergrax/runtime/schema/registry.py (RUNTIME_SCHEMA_REGISTRY)"
    if surface.discovery_kind == "event.payload":
        return "intergrax/runtime/events/payload_registry.py"
    if surface.discovery_kind == "mechanism.policy_owner":
        return surface.owner_module_path
    if surface.discovery_kind == "mechanism.plugin":
        return "intergrax/core/plugins/package_contract.py"
    if surface.discovery_kind == "mechanism.integrations":
        return "intergrax/integrations/contracts/external_contract_compatibility.py"
    return surface.owner_module_path


def _tenant_relevant(surface: DiscoveredCompatSurface) -> bool:
    if surface.discovery_kind == "mechanism.policy_owner" and "external" in surface.surface_id:
        return True
    if surface.discovery_kind == "mechanism.integrations":
        return True
    if "tenant" in surface.owner_module_path.lower():
        return True
    if surface.discovery_kind == "event.payload":
        return True
    return surface.discovery_kind in {
        "registry.contracts",
        "registry.runtime",
    }


def _record_from_surface(surface: DiscoveredCompatSurface) -> CompatSurfaceRecord:
    exposure = _exposure_for_surface(surface)
    persistence = "persisted" if ExposureFacet.PERSISTED_SCHEMA in exposure else "wire_or_transient"
    if ExposureFacet.EVENT_SCHEMA in exposure and ExposureFacet.PERSISTED_SCHEMA not in exposure:
        persistence = "event_wire"
    evidence = tuple(dict.fromkeys((surface.owner_module_path, surface.version_source, *surface.evidence_kinds)))
    return CompatSurfaceRecord(
        surface_id=surface.surface_id,
        semantic_identity=surface.semantic_identity,
        domain=_domain_for_surface(surface),
        owner_module_path=surface.owner_module_path,
        contract_schema_identity=surface.contract_schema_identity,
        exposure_facets=exposure,
        persistence_wire=persistence,
        version_source=surface.version_source,
        current_version=surface.current_version,
        compatibility_policy_owner=_policy_owner(surface),
        migration_owner_path=surface.owner_module_path
        if surface.discovery_kind.startswith("migration")
        else "UNOWNED — COMPAT-X-R2",
        deprecation_owner_path="UNOWNED — COMPAT-X-R5",
        plugin_provider_relevant=ExposureFacet.PLUGIN_PROVIDER_CONTRACT in exposure,
        tenant_relevant=_tenant_relevant(surface),
        evolution_states=_evolution_for_surface(surface),
        migration_class=_migration_class_for_surface(surface),
        shim_class=_shim_class_for_surface(surface),
        evidence_paths=evidence,
    )


@lru_cache(maxsize=1)
def compat_x_inventory() -> tuple[CompatSurfaceRecord, ...]:
    surfaces = sorted(discover_all_compat_surfaces(), key=lambda s: s.surface_id)
    return tuple(_record_from_surface(surface) for surface in surfaces)


COMPAT_X_INVENTORY: tuple[CompatSurfaceRecord, ...] = compat_x_inventory()

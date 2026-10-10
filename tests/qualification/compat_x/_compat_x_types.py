# © Artur Czarnecki. All rights reserved.

"""COMPAT-X typed classification model (P0 baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompatDomain(StrEnum):
    CONTRACTS = "CONTRACTS"
    RUNTIME = "RUNTIME"
    EVENTS = "EVENTS"
    INTEGRATIONS = "INTEGRATIONS"
    PLUGINS = "PLUGINS"
    DISTRIBUTION = "DISTRIBUTION"
    TOOLS = "TOOLS"
    SKILLS = "SKILLS"
    MARKETPLACE = "MARKETPLACE"
    HOSTING = "HOSTING"
    PROOFS = "PROOFS"
    COMPAT_SHIM = "COMPAT_SHIM"
    APPLICATIONS_BOUNDARY = "APPLICATIONS_BOUNDARY"
    MIGRATION = "MIGRATION"


class ExposureFacet(StrEnum):
    PUBLIC_STABLE = "PUBLIC_STABLE"
    PLATFORM_INTERNAL = "PLATFORM_INTERNAL"
    PERSISTED_SCHEMA = "PERSISTED_SCHEMA"
    WIRE_SCHEMA = "WIRE_SCHEMA"
    EVENT_SCHEMA = "EVENT_SCHEMA"
    PLUGIN_PROVIDER_CONTRACT = "PLUGIN_PROVIDER_CONTRACT"
    DISTRIBUTION_PACKAGE_CONTRACT = "DISTRIBUTION_PACKAGE_CONTRACT"
    COMPATIBILITY_ADAPTER = "COMPATIBILITY_ADAPTER"


class EvolutionState(StrEnum):
    VERSIONED_AND_POLICY_DEFINED = "VERSIONED_AND_POLICY_DEFINED"
    VERSIONED_POLICY_MISSING = "VERSIONED_POLICY_MISSING"
    PERSISTED_MIGRATION_DEFINED = "PERSISTED_MIGRATION_DEFINED"
    PERSISTED_MIGRATION_MISSING = "PERSISTED_MIGRATION_MISSING"
    EVENT_EVOLUTION_DEFINED = "EVENT_EVOLUTION_DEFINED"
    EVENT_EVOLUTION_MISSING = "EVENT_EVOLUTION_MISSING"
    PLUGIN_COMPATIBILITY_DEFINED = "PLUGIN_COMPATIBILITY_DEFINED"
    PLUGIN_COMPATIBILITY_MISSING = "PLUGIN_COMPATIBILITY_MISSING"
    DEPRECATION_POLICY_DEFINED = "DEPRECATION_POLICY_DEFINED"
    DEPRECATION_POLICY_MISSING = "DEPRECATION_POLICY_MISSING"
    COMPATIBILITY_SHIM_SANCTIONED = "COMPATIBILITY_SHIM_SANCTIONED"
    COMPATIBILITY_SHIM_PARALLEL_AUTHORITY = "COMPATIBILITY_SHIM_PARALLEL_AUTHORITY"
    UNCLASSIFIED = "UNCLASSIFIED"


class MigrationMechanismClass(StrEnum):
    CANONICAL_MIGRATION_OWNER = "CANONICAL_MIGRATION_OWNER"
    LOCAL_FORMAT_MIGRATION = "LOCAL_FORMAT_MIGRATION"
    LEGACY_COMPAT_READER = "LEGACY_COMPAT_READER"
    UNOWNED_MIGRATION = "UNOWNED_MIGRATION"
    DUPLICATE_MIGRATION_AUTHORITY = "DUPLICATE_MIGRATION_AUTHORITY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ShimClass(StrEnum):
    TRANSLATION_ONLY = "TRANSLATION_ONLY"
    READ_COMPATIBILITY_ONLY = "READ_COMPATIBILITY_ONLY"
    BOUNDED_MIGRATION_ADAPTER = "BOUNDED_MIGRATION_ADAPTER"
    PARALLEL_AUTHORITY = "PARALLEL_AUTHORITY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FrzCmpCandidate(StrEnum):
    PASS_CANDIDATE = "PASS_CANDIDATE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class CompatSurfaceRecord:
    surface_id: str
    domain: CompatDomain
    owner_module_path: str
    contract_schema_identity: str
    exposure_facets: frozenset[ExposureFacet]
    persistence_wire: str
    version_source: str
    current_version: str
    compatibility_policy_owner: str
    migration_owner_path: str
    deprecation_owner_path: str
    plugin_provider_relevant: bool
    tenant_relevant: bool
    evolution_states: frozenset[EvolutionState]
    migration_class: MigrationMechanismClass
    shim_class: ShimClass
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompatOwnerMatrixRow:
    concern: str
    semantic_owner_path: str
    composition_owner_path: str
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompatDiscoveryExclusion:
    surface_id: str
    reason: str
    evidence_path: str

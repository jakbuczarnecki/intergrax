# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability Catalog-backed Tool/Skill discovery projections for AW-7A (Stage 9).

Thin AW adapters over canonical Stage 3–5 governed discovery. AW retains A0–A4
decision authority; catalog owns discovery, ranking, and governance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.errors import (
    CapabilityCatalogDiscoveryError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    CapabilityGovernanceError,
)
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.capability_catalog.governance import (
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
)
from intergrax.capability_catalog.ranking import CapabilityRanker, rank_capability_candidates
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.capability_catalog.work_stage_effective import (
    select_effective_executable_candidates,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityDiscoveryDisposition,
    CapabilityOperationCoverage,
    WorkerAutonomyLevel,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
    WorkerCapabilityNeed,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.identity import CapabilityDiscoveryIdentity
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.query import (
    CapabilityDiscoveryQuery,
    LogicalIdentityFilter,
)
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingContext
from intergrax.contracts.capability_catalog.scope import CapabilityDiscoveryScope
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.catalog_manifests import catalog_manifest_for_skill_id
from intergrax.skills.registry.runtime import SkillRegistry


@runtime_checkable
class SkillManifestLookupPort(Protocol):
    """Read-only Skill manifest operation evidence — not discovery authority."""

    def supported_operations(self, skill_logical_id: str) -> tuple[str, ...] | None:
        """Return manifest-backed operations for ``skill_logical_id``, or None."""
        ...


class SkillRegistryManifestLookup(SkillManifestLookupPort):
    """Resolve skill operations from runtime registry with catalog manifest fallback."""

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self._skill_registry = skill_registry

    def supported_operations(self, skill_logical_id: str) -> tuple[str, ...] | None:
        if self._skill_registry.has(skill_logical_id):
            manifest = self._skill_registry.get(skill_logical_id).manifest
            return _skill_manifest_operations(manifest)
        catalog_manifest = catalog_manifest_for_skill_id(skill_logical_id)
        if catalog_manifest is None:
            return None
        return _skill_manifest_operations(catalog_manifest)


def _skill_manifest_operations(manifest: SkillManifest) -> tuple[str, ...]:
    return tuple(manifest.tool_ids) + (manifest.skill_id,)


@dataclass(frozen=True, slots=True)
class CapabilityCatalogDiscoveryDependencies:
    """Immutable catalog discovery inputs for one AW recovery attempt."""

    snapshot: CapabilityCatalogSnapshot
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence
    governance_context: CapabilityGovernanceContext
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...]
    scope: CapabilityDiscoveryScope
    ranker: CapabilityRanker | None = None


def map_worker_capability_need_to_discovery_query(
    need: WorkerCapabilityNeed,
    *,
    kind: CapabilityKind,
    scope: CapabilityDiscoveryScope,
) -> CapabilityDiscoveryQuery:
    """Map AW need fields to a typed catalog query without registry heuristics."""
    logical_identity: LogicalIdentityFilter | None = None
    if kind is CapabilityKind.TOOL and need.required_operations:
        logical_identity = LogicalIdentityFilter(
            exact_logical_ids=need.required_operations,
        )
    return CapabilityDiscoveryQuery(
        scope=scope,
        kinds=(kind,),
        logical_identity=logical_identity,
    )


def encode_source_qualified_capability_ref(identity: CapabilityDiscoveryIdentity) -> str:
    """Deterministic source-qualified capability reference for AW projection."""
    source = identity.source
    return (
        f"{identity.kind.value}:"
        f"{source.source_kind.value}:"
        f"{source.source_id}:"
        f"{identity.logical.logical_id}"
    )


def encode_catalog_discovery_evidence_ref(identity: CapabilityDiscoveryIdentity) -> ProblemReference:
    """Deterministic AW evidence reference preserving catalog identity."""
    source = identity.source
    return ProblemReference(
        "capability/catalog/"
        f"{identity.kind.value}/"
        f"{source.source_kind.value}/"
        f"{source.source_id}/"
        f"{identity.logical.logical_id}",
    )


def _operation_coverage(
    required: tuple[str, ...],
    offered: tuple[str, ...],
) -> CapabilityOperationCoverage | None:
    if not required or not offered:
        return None
    offered_set = frozenset(offered)
    if all(item in offered_set for item in required):
        return CapabilityOperationCoverage.EXACT
    if any(item in offered_set for item in required):
        return CapabilityOperationCoverage.PARTIAL
    return None


def _tool_operations(identity: CapabilityDiscoveryIdentity) -> tuple[str, ...]:
    """Return canonical Tool operation evidence for Stage 9 exact-ID coverage.

    Stage 9 Tool operation evidence currently proves exact support only when
    required operation IDs correspond to canonical Tool logical IDs.
    Richer Tool operation semantics remain owned by the Tools domain.
    """
    return (identity.logical.logical_id,)


def _tool_supports_required_operations(
    *,
    identity: CapabilityDiscoveryIdentity,
    required_operations: tuple[str, ...],
) -> tuple[tuple[str, ...], CapabilityOperationCoverage] | None:
    offered = _tool_operations(identity)
    coverage = _operation_coverage(required_operations, offered)
    if coverage is not CapabilityOperationCoverage.EXACT:
        return None
    return offered, coverage


def _candidate_sort_key(candidate: WorkerCapabilityCandidate) -> tuple[str, str, str]:
    return (
        candidate.candidate_kind.value,
        candidate.capability_ref,
        candidate.candidate_id,
    )


def _project_tool_candidate(
    governed: GovernedCapabilityCandidate,
    *,
    request: WorkerCapabilityDiscoveryRequest,
    operations: tuple[str, ...],
    coverage: CapabilityOperationCoverage,
) -> WorkerCapabilityCandidate:
    identity = governed.identity
    capability_ref = encode_source_qualified_capability_ref(identity)
    version = governed.provenance.version_label
    return WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.TOOL,
            capability_ref=capability_ref,
            version=version,
        ),
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref=capability_ref,
        source_domain=identity.source.source_id,
        version=version,
        operations=operations,
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(encode_catalog_discovery_evidence_ref(identity),),
        discovered_at=request.need.requested_at,
        operation_coverage=coverage,
    )


def _project_skill_candidate(
    governed: GovernedCapabilityCandidate,
    *,
    request: WorkerCapabilityDiscoveryRequest,
    operations: tuple[str, ...],
    coverage: CapabilityOperationCoverage,
) -> WorkerCapabilityCandidate:
    identity = governed.identity
    capability_ref = encode_source_qualified_capability_ref(identity)
    version = governed.provenance.version_label
    return WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.SKILL,
            capability_ref=capability_ref,
            version=version,
        ),
        candidate_kind=WorkerCapabilityCandidateKind.SKILL,
        capability_ref=capability_ref,
        source_domain=identity.source.source_id,
        version=version,
        operations=operations,
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(encode_catalog_discovery_evidence_ref(identity),),
        discovered_at=request.need.requested_at,
        operation_coverage=coverage,
    )


def _map_governed_layer_disposition(
    *,
    operation_relevant_count: int,
    governed_result: GovernedDiscoveryResult,
    executable: tuple[GovernedCapabilityCandidate, ...],
) -> CapabilityDiscoveryDisposition:
    if operation_relevant_count == 0:
        return CapabilityDiscoveryDisposition.NO_MATCH
    if executable:
        return CapabilityDiscoveryDisposition.MATCH_FOUND
    if governed_result.allowed:
        return CapabilityDiscoveryDisposition.NO_MATCH
    if governed_result.blocked:
        return CapabilityDiscoveryDisposition.POLICY_BLOCKED
    return CapabilityDiscoveryDisposition.NO_MATCH


def _discover_rank_and_govern(
    query: CapabilityDiscoveryQuery,
    dependencies: CapabilityCatalogDiscoveryDependencies,
    candidates: tuple[CapabilityDiscoveryCandidate, ...] | None = None,
) -> GovernedDiscoveryResult:
    discovered = (
        candidates
        if candidates is not None
        else discover_capability_candidates(
            dependencies.snapshot,
            query,
            availability_evidence=dependencies.availability_evidence,
        )
    )
    ranker = dependencies.ranker
    if ranker is None:
        from intergrax.capability_catalog.ranking import StableIdentityRanker

        ranker = StableIdentityRanker()
    ranked = rank_capability_candidates(
        discovered,
        ranker,
        context=CapabilityRankingContext(),
    )
    return govern_capability_candidates(
        ranked,
        evaluators=dependencies.governance_evaluators,
        context=dependencies.governance_context,
    )


def _run_tool_discovery_layer(
    request: WorkerCapabilityDiscoveryRequest,
    dependencies: CapabilityCatalogDiscoveryDependencies,
) -> WorkerCapabilityDiscoveryLayerOutcome:
    query = map_worker_capability_need_to_discovery_query(
        request.need,
        kind=CapabilityKind.TOOL,
        scope=dependencies.scope,
    )
    try:
        discovered = discover_capability_candidates(
            dependencies.snapshot,
            query,
            availability_evidence=dependencies.availability_evidence,
        )
    except CapabilityCatalogIdentityConflict:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.CONFLICT,
        )
    except (
        CapabilityCatalogDiscoveryError,
        CapabilityCatalogSourceFailure,
        CapabilityGovernanceError,
    ):
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )
    operation_relevant: list[
        tuple[CapabilityDiscoveryCandidate, tuple[str, ...], CapabilityOperationCoverage]
    ] = []
    for candidate in discovered:
        resolved = _tool_supports_required_operations(
            identity=candidate.identity,
            required_operations=request.need.required_operations,
        )
        if resolved is not None:
            operations, coverage = resolved
            operation_relevant.append((candidate, operations, coverage))
    operation_relevant_count = len(operation_relevant)
    if operation_relevant_count == 0:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.NO_MATCH,
        )
    filtered_candidates = tuple(item[0] for item in operation_relevant)
    try:
        governed_result = _discover_rank_and_govern(
            query,
            dependencies,
            candidates=filtered_candidates,
        )
    except CapabilityCatalogIdentityConflict:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.CONFLICT,
        )
    except (CapabilityCatalogSourceFailure, CapabilityGovernanceError):
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )
    executable = select_effective_executable_candidates(governed_result.allowed)
    disposition = _map_governed_layer_disposition(
        operation_relevant_count=operation_relevant_count,
        governed_result=governed_result,
        executable=executable,
    )
    if disposition is not CapabilityDiscoveryDisposition.MATCH_FOUND:
        return WorkerCapabilityDiscoveryLayerOutcome(disposition=disposition)
    coverage_by_identity = {
        item[0].identity.sort_key: (item[1], item[2]) for item in operation_relevant
    }
    projected: list[WorkerCapabilityCandidate] = []
    for governed in executable:
        operations, coverage = coverage_by_identity[governed.identity.sort_key]
        projected.append(
            _project_tool_candidate(
                governed,
                request=request,
                operations=operations,
                coverage=coverage,
            ),
        )
    return WorkerCapabilityDiscoveryLayerOutcome(
        disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
        candidates=tuple(sorted(projected, key=_candidate_sort_key)),
    )


def _skill_supports_required_operations(
    *,
    skill_logical_id: str,
    required_operations: tuple[str, ...],
    manifest_lookup: SkillManifestLookupPort,
) -> tuple[tuple[str, ...], CapabilityOperationCoverage] | None:
    offered = manifest_lookup.supported_operations(skill_logical_id)
    if offered is None:
        return None
    coverage = _operation_coverage(required_operations, offered)
    if coverage is None or coverage is CapabilityOperationCoverage.PARTIAL:
        return None
    return offered, coverage


def _run_skill_discovery_layer(
    request: WorkerCapabilityDiscoveryRequest,
    dependencies: CapabilityCatalogDiscoveryDependencies,
    manifest_lookup: SkillManifestLookupPort,
) -> WorkerCapabilityDiscoveryLayerOutcome:
    query = map_worker_capability_need_to_discovery_query(
        request.need,
        kind=CapabilityKind.SKILL,
        scope=dependencies.scope,
    )
    try:
        discovered = discover_capability_candidates(
            dependencies.snapshot,
            query,
            availability_evidence=dependencies.availability_evidence,
        )
    except CapabilityCatalogIdentityConflict:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.CONFLICT,
        )
    except (
        CapabilityCatalogDiscoveryError,
        CapabilityCatalogSourceFailure,
        CapabilityGovernanceError,
    ):
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )
    operation_relevant: list[tuple[CapabilityDiscoveryCandidate, tuple[str, ...], CapabilityOperationCoverage]] = []
    for candidate in discovered:
        resolved = _skill_supports_required_operations(
            skill_logical_id=candidate.identity.logical.logical_id,
            required_operations=request.need.required_operations,
            manifest_lookup=manifest_lookup,
        )
        if resolved is not None:
            operations, coverage = resolved
            operation_relevant.append((candidate, operations, coverage))
    operation_relevant_count = len(operation_relevant)
    if operation_relevant_count == 0:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.NO_MATCH,
        )
    filtered_candidates = tuple(item[0] for item in operation_relevant)
    try:
        governed_result = _discover_rank_and_govern(
            query,
            dependencies,
            candidates=filtered_candidates,
        )
    except CapabilityCatalogIdentityConflict:
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.CONFLICT,
        )
    except (CapabilityCatalogSourceFailure, CapabilityGovernanceError):
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )
    executable = select_effective_executable_candidates(governed_result.allowed)
    disposition = _map_governed_layer_disposition(
        operation_relevant_count=operation_relevant_count,
        governed_result=governed_result,
        executable=executable,
    )
    if disposition is not CapabilityDiscoveryDisposition.MATCH_FOUND:
        return WorkerCapabilityDiscoveryLayerOutcome(disposition=disposition)
    coverage_by_identity = {
        item[0].identity.sort_key: (item[1], item[2]) for item in operation_relevant
    }
    projected: list[WorkerCapabilityCandidate] = []
    for governed in executable:
        operations, coverage = coverage_by_identity[governed.identity.sort_key]
        projected.append(
            _project_skill_candidate(
                governed,
                request=request,
                operations=operations,
                coverage=coverage,
            ),
        )
    return WorkerCapabilityDiscoveryLayerOutcome(
        disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
        candidates=tuple(sorted(projected, key=_candidate_sort_key)),
    )


class CapabilityCatalogToolDiscoveryAdapter:
    """Discover Tool candidates through governed Capability Catalog projection."""

    def __init__(self, dependencies: CapabilityCatalogDiscoveryDependencies) -> None:
        self._dependencies = dependencies

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        return _run_tool_discovery_layer(request, self._dependencies)


class CapabilityCatalogSkillDiscoveryAdapter:
    """Discover Skill candidates through governed Capability Catalog projection."""

    def __init__(
        self,
        dependencies: CapabilityCatalogDiscoveryDependencies,
        *,
        manifest_lookup: SkillManifestLookupPort,
    ) -> None:
        self._dependencies = dependencies
        self._manifest_lookup = manifest_lookup

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        return _run_skill_discovery_layer(
            request,
            self._dependencies,
            self._manifest_lookup,
        )


def identity_key_from_entry_identity(
    identity: CapabilityDiscoveryIdentity,
) -> CapabilityIdentityKey:
    """Public helper for wiring host availability evidence."""
    return CapabilityIdentityKey.from_discovery_identity(identity)

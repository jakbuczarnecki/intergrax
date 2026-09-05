# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Registry-backed capability discovery adapters for AW-7A.

Adapters wrap canonical ToolRegistry, SkillRegistry, and Integration catalog
public APIs. AW core consumes only typed ports/contracts.
"""

from __future__ import annotations

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityDiscoveryDisposition,
    CapabilityOperationCoverage,
    WorkerAutonomyLevel,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.integrations.registry import catalog as integration_catalog
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry


def _coverage_for_operations(
    required: tuple[str, ...],
    offered: tuple[str, ...],
) -> CapabilityOperationCoverage | None:
    if not required or not offered:
        return None
    offered_set = set(offered)
    if all(item in offered_set for item in required):
        return CapabilityOperationCoverage.EXACT
    if any(item in offered_set for item in required):
        return CapabilityOperationCoverage.PARTIAL
    return None


def _candidate_evidence(capability_ref: str) -> tuple[ProblemReference, ...]:
    return (ProblemReference(f"capability/discovery/{capability_ref}"),)


class ToolRegistryCapabilityDiscoveryAdapter:
    """Discover Tool candidates from canonical ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        required = request.need.required_operations
        matches: list[WorkerCapabilityCandidate] = []
        for registered in self._tool_registry.list():
            contract = registered.contract
            offered = (contract.tool_id,)
            coverage = _coverage_for_operations(required, offered)
            if coverage is None:
                for tag in contract.tags:
                    coverage = _coverage_for_operations(required, (tag,))
                    if coverage is not None:
                        offered = (tag,)
                        break
            if coverage is None and contract.category:
                coverage = _coverage_for_operations(required, (contract.category,))
                if coverage is not None:
                    offered = (contract.category,)
            if coverage is None:
                continue
            if coverage is CapabilityOperationCoverage.PARTIAL:
                continue
            capability_ref = f"tool:{contract.tool_id}"
            matches.append(
                WorkerCapabilityCandidate(
                    candidate_id=derive_worker_capability_candidate_id(
                        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
                        capability_ref=capability_ref,
                        version=contract.version,
                    ),
                    candidate_kind=WorkerCapabilityCandidateKind.TOOL,
                    capability_ref=capability_ref,
                    source_domain="tools",
                    version=contract.version,
                    operations=offered,
                    risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                    evidence_refs=_candidate_evidence(capability_ref),
                    discovered_at=request.need.requested_at,
                    operation_coverage=coverage,
                ),
            )
        if not matches:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=tuple(sorted(matches, key=_candidate_sort_key)),
        )


class SkillRegistryCapabilityDiscoveryAdapter:
    """Discover Skill candidates from canonical SkillRegistry."""

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self._skill_registry = skill_registry

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        required = request.need.required_operations
        matches: list[WorkerCapabilityCandidate] = []
        for registered in self._skill_registry.list():
            manifest = registered.manifest
            offered = tuple(manifest.tool_ids) + (manifest.skill_id,)
            coverage = _coverage_for_operations(required, offered)
            if coverage is None:
                for tag in manifest.tags:
                    coverage = _coverage_for_operations(required, (tag,))
                    if coverage is not None:
                        offered = (tag,)
                        break
            if coverage is None or coverage is CapabilityOperationCoverage.PARTIAL:
                continue
            capability_ref = f"skill:{manifest.qualified_id}"
            matches.append(
                WorkerCapabilityCandidate(
                    candidate_id=derive_worker_capability_candidate_id(
                        candidate_kind=WorkerCapabilityCandidateKind.SKILL,
                        capability_ref=capability_ref,
                        version=manifest.version,
                    ),
                    candidate_kind=WorkerCapabilityCandidateKind.SKILL,
                    capability_ref=capability_ref,
                    source_domain="skills",
                    version=manifest.version,
                    operations=offered,
                    risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                    evidence_refs=_candidate_evidence(capability_ref),
                    discovered_at=request.need.requested_at,
                    operation_coverage=coverage,
                ),
            )
        if not matches:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=tuple(sorted(matches, key=_candidate_sort_key)),
        )


class IntegrationCatalogCapabilityDiscoveryAdapter:
    """Discover Integration candidates from canonical integration catalog metadata."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        required = request.need.required_operations
        required_domains = set(request.need.required_data_domains)
        required_protocols = set(request.need.required_protocols)
        matches: list[WorkerCapabilityCandidate] = []
        for entry in integration_catalog.iter_entries():
            offered = (f"integration:{entry.slug}",)
            coverage = _coverage_for_operations(required, offered)
            if coverage is None and required_domains and entry.categories:
                if entry.categories[0].value in required_domains:
                    coverage = CapabilityOperationCoverage.EXACT
                    offered = (entry.categories[0].value,)
            if coverage is None and required_protocols:
                if entry.slug in required_protocols:
                    coverage = CapabilityOperationCoverage.EXACT
                    offered = (entry.slug,)
            if coverage is None or coverage is CapabilityOperationCoverage.PARTIAL:
                continue
            capability_ref = f"integration:{entry.slug}"
            matches.append(
                WorkerCapabilityCandidate(
                    candidate_id=derive_worker_capability_candidate_id(
                        candidate_kind=WorkerCapabilityCandidateKind.INTEGRATION,
                        capability_ref=capability_ref,
                    ),
                    candidate_kind=WorkerCapabilityCandidateKind.INTEGRATION,
                    capability_ref=capability_ref,
                    source_domain="integrations",
                    operations=offered,
                    risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                    evidence_refs=_candidate_evidence(capability_ref),
                    discovered_at=request.need.requested_at,
                    operation_coverage=coverage,
                ),
            )
        if not matches:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=tuple(sorted(matches, key=_candidate_sort_key)),
        )


class MappingApprovedAlternateDiscoveryAdapter:
    """Minimal provider-neutral approved alternate read port."""

    def __init__(
        self,
        alternates: dict[tuple[str, ...], tuple[WorkerCapabilityCandidate, ...]],
    ) -> None:
        self._alternates = alternates

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        key = tuple(sorted(request.need.required_operations))
        candidates = self._alternates.get(key, ())
        if not candidates:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=candidates,
        )


class MappingConfigurationOpportunityDiscoveryAdapter:
    """Discover existing approved capabilities requiring configuration only."""

    def __init__(
        self,
        opportunities: dict[tuple[str, ...], tuple[WorkerCapabilityCandidate, ...]],
    ) -> None:
        self._opportunities = opportunities

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        key = tuple(sorted(request.need.required_operations))
        candidates = self._opportunities.get(key, ())
        if not candidates:
            return WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            )
        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.MATCH_FOUND,
            candidates=candidates,
        )


def _candidate_sort_key(candidate: WorkerCapabilityCandidate) -> tuple[str, str, str]:
    return (
        candidate.candidate_kind.value,
        candidate.capability_ref,
        candidate.candidate_id,
    )


def normalize_candidates(
    candidates: tuple[WorkerCapabilityCandidate, ...],
) -> tuple[WorkerCapabilityCandidate, ...] | CapabilityDiscoveryDisposition:
    """Deduplicate equivalent candidates or detect metadata conflicts."""

    by_id: dict[str, WorkerCapabilityCandidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.candidate_id)
        if existing is None:
            by_id[candidate.candidate_id] = candidate
            continue
        if (
            existing.operations != candidate.operations
            or existing.operation_coverage != candidate.operation_coverage
            or existing.risk_class != candidate.risk_class
        ):
            return CapabilityDiscoveryDisposition.CONFLICT
    return tuple(sorted(by_id.values(), key=_candidate_sort_key))

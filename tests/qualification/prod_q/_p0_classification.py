# © Artur Czarnecki. All rights reserved.

"""Classify discovered PROD-Q production readiness candidates."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
    OwnershipResolutionState,
    ProdQFindingDisposition,
    ProductionReadinessCandidate,
    ProductionReadinessClassification,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
    ProductionReadinessOwner,
)

_INDICATOR_TO_FAMILY: dict[str, ProductionReadinessFamily] = {
    "startup": ProductionReadinessFamily.PQ_F02,
    "lifespan": ProductionReadinessFamily.PQ_F02,
    "shutdown": ProductionReadinessFamily.PQ_F03,
    "health": ProductionReadinessFamily.PQ_F04,
    "readiness": ProductionReadinessFamily.PQ_F05,
    "degraded": ProductionReadinessFamily.PQ_F06,
    "terminal": ProductionReadinessFamily.PQ_F07,
    "recovery": ProductionReadinessFamily.PQ_F08,
    "admission": ProductionReadinessFamily.PQ_F09,
    "unsupported": ProductionReadinessFamily.PQ_F10,
    "mode": ProductionReadinessFamily.PQ_F11,
    "mode_branch": ProductionReadinessFamily.PQ_F11,
    "secrets": ProductionReadinessFamily.PQ_F12,
    "capacity": ProductionReadinessFamily.PQ_F13,
    "backpressure": ProductionReadinessFamily.PQ_F14,
    "throttle": ProductionReadinessFamily.PQ_F15,
    "lifecycle": ProductionReadinessFamily.PQ_F16,
    "idempotency": ProductionReadinessFamily.PQ_F17,
    "materialization": ProductionReadinessFamily.PQ_F18,
    "sandbox": ProductionReadinessFamily.PQ_F19,
    "sandbox_provider": ProductionReadinessFamily.PQ_F09,
    "retention": ProductionReadinessFamily.PQ_F20,
    "backup": ProductionReadinessFamily.PQ_F23,
    "tenant": ProductionReadinessFamily.PQ_F22,
    "contract_import": ProductionReadinessFamily.PQ_F01,
}

_ANCHOR_METADATA: dict[tuple[str, str], tuple[ProductionReadinessOwner, str, str, ProductionReadinessEvidenceState]] = {
    (
        "intergrax/contracts/execution_capacity_admission.py",
        "ExecutionCapacityAdmissionPort",
    ): (
        ProductionReadinessOwner.EXECUTION_CAPACITY,
        "intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
        "intergrax/runtime/execution/capacity.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/runtime/execution/local_execution_capacity_admission.py",
        "LocalExecutionCapacityAdmission",
    ): (
        ProductionReadinessOwner.EXECUTION_CAPACITY,
        "intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
        "intergrax/runtime/execution/local_execution_capacity_admission.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/hosting/engine/engine.py",
        "HostedApplicationEngine",
    ): (
        ProductionReadinessOwner.HOSTING_ENGINE,
        "intergrax.hosting.engine.engine.HostedApplicationEngine",
        "intergrax/hosting/engine/engine.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
    (
        "intergrax/applications/_shared/auditability_health_wiring.py",
        "project_host_auditability_health_facts",
    ): (
        ProductionReadinessOwner.OBSERVABILITY_AUDITABILITY,
        "intergrax.runtime.observability.auditability_health",
        "intergrax/applications/_shared/auditability_health_wiring.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/integrations/providers/relational_store/sqlite/paths.py",
        "resolve_sqlite_store_paths",
    ): (
        ProductionReadinessOwner.RELATIONAL_STORE_SQLITE,
        "intergrax.integrations.providers.relational_store.sqlite.paths",
        "intergrax/integrations/providers/relational_store/sqlite/paths.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
    (
        "intergrax/runtime/execution/lab_reference_agent_runtime.py",
        "build_lab_agent_runtime_config_from_merged",
    ): (
        ProductionReadinessOwner.EXECUTION_RUNTIME,
        "intergrax.runtime.execution.lab_reference_agent_runtime",
        "intergrax/runtime/execution/lab_reference_agent_runtime.py",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
    ),
}

_FRZ_BY_FAMILY: dict[ProductionReadinessFamily, tuple[str, ...]] = {
    ProductionReadinessFamily.PQ_F01: ("FRZ-PRD-01", "FRZ-PRD-07"),
    ProductionReadinessFamily.PQ_F02: ("FRZ-PRD-04",),
    ProductionReadinessFamily.PQ_F03: ("FRZ-PRD-05",),
    ProductionReadinessFamily.PQ_F04: ("FRZ-PRD-09",),
    ProductionReadinessFamily.PQ_F05: ("FRZ-PRD-09",),
    ProductionReadinessFamily.PQ_F06: ("FRZ-PRD-06", "FRZ-PRD-09"),
    ProductionReadinessFamily.PQ_F07: ("FRZ-PRD-10",),
    ProductionReadinessFamily.PQ_F08: ("FRZ-PRD-12",),
    ProductionReadinessFamily.PQ_F09: ("FRZ-PRD-02", "FRZ-SEC-06"),
    ProductionReadinessFamily.PQ_F10: ("FRZ-PRD-03",),
    ProductionReadinessFamily.PQ_F11: ("FRZ-PRD-07", "FRZ-PRD-08"),
    ProductionReadinessFamily.PQ_F12: ("FRZ-SEC-02", "FRZ-SEC-03"),
    ProductionReadinessFamily.PQ_F13: ("FRZ-PRD-11", "FRZ-REL-08", "FRZ-REL-10"),
    ProductionReadinessFamily.PQ_F14: ("FRZ-PRD-11", "FRZ-REL-09"),
    ProductionReadinessFamily.PQ_F15: ("FRZ-REL-11",),
    ProductionReadinessFamily.PQ_F16: ("FRZ-REL-01", "FRZ-REL-02", "FRZ-REL-04"),
    ProductionReadinessFamily.PQ_F17: ("FRZ-REL-03", "FRZ-REL-05"),
    ProductionReadinessFamily.PQ_F18: ("FRZ-PRD-04", "FRZ-PRD-03"),
    ProductionReadinessFamily.PQ_F19: ("FRZ-SEC-05", "FRZ-SEC-07"),
    ProductionReadinessFamily.PQ_F20: ("FRZ-SEC-09",),
    ProductionReadinessFamily.PQ_F21: ("FRZ-SEC-10",),
    ProductionReadinessFamily.PQ_F22: ("FRZ-SEC-01",),
    ProductionReadinessFamily.PQ_F23: ("FRZ-PRD-12", "FRZ-REC-01"),
}


def _default_disposition(
    evidence: ProductionReadinessEvidenceState,
    family: ProductionReadinessFamily,
) -> ProdQFindingDisposition:
    if evidence == ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED:
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER
    if evidence == ProductionReadinessEvidenceState.BLOCKER:
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER
    if family == ProductionReadinessFamily.PQ_F11 and evidence == ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD:
        return ProdQFindingDisposition.TRACKED_FREEZE_DEBT
    if evidence == ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD:
        return ProdQFindingDisposition.TRACKED_FREEZE_DEBT
    return ProdQFindingDisposition.TRACKED_FREEZE_DEBT


def _semantic_responsibility_key(
    family: ProductionReadinessFamily,
    indicator: str,
    candidate: ProductionReadinessCandidate,
) -> str:
    if candidate.discovery_kind == DiscoveryKind.ANCHOR_REGISTRY:
        return f"anchor:{candidate.module_path}:{candidate.symbol}"
    if candidate.discovery_kind == DiscoveryKind.PROVIDER_SANDBOX_HOST:
        return f"provider.sandbox_host:{candidate.module_path}"
    return f"mechanical:{candidate.candidate_id}"


def classify_candidate(candidate: ProductionReadinessCandidate) -> ProductionReadinessClassification:
    mapped_family = _INDICATOR_TO_FAMILY.get(candidate.indicator)
    inventory_status = CandidateInventoryStatus.CLASSIFIED
    if mapped_family is None:
        inventory_status = CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
        family = ProductionReadinessFamily.PQ_UNCLASSIFIED
    else:
        family = mapped_family
    if candidate.discovery_kind == DiscoveryKind.PROVIDER_SANDBOX_HOST:
        family = ProductionReadinessFamily.PQ_F09
        inventory_status = CandidateInventoryStatus.CLASSIFIED

    anchor_key = (candidate.module_path, candidate.symbol)
    ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    if anchor_key in _ANCHOR_METADATA:
        owner, contract, composition, evidence = _ANCHOR_METADATA[anchor_key]
        ownership_resolution = OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
    elif candidate.discovery_kind == DiscoveryKind.PROVIDER_SANDBOX_HOST:
        owner = ProductionReadinessOwner.SANDBOX_HOST_PROVIDER
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED
        ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    elif candidate.discovery_kind == DiscoveryKind.AST_STRICT_LAB_BRANCH:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = candidate.module_path
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.DISCOVERED
    elif candidate.discovery_kind == DiscoveryKind.ANCHOR_REGISTRY:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.DISCOVERED
    else:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.DISCOVERED

    if inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR:
        evidence = ProductionReadinessEvidenceState.BLOCKER
        ownership_resolution = OwnershipResolutionState.AMBIGUOUS

    tenant_relevance = family == ProductionReadinessFamily.PQ_F22 or "tenant" in candidate.indicator

    if "sandbox_host" in candidate.module_path and candidate.discovery_kind != DiscoveryKind.ANCHOR_REGISTRY:
        evidence = ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED

    disposition = _default_disposition(evidence, family)
    if inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR:
        disposition = ProdQFindingDisposition.IN_SCOPE_BLOCKER
    if (
        candidate.module_path == "intergrax/runtime/execution/lab_reference_agent_runtime.py"
        and family == ProductionReadinessFamily.PQ_F11
        and candidate.symbol == "build_lab_agent_runtime_config_from_merged"
    ):
        disposition = ProdQFindingDisposition.TRACKED_FREEZE_DEBT

    semantic_key = _semantic_responsibility_key(family, candidate.indicator, candidate)

    return ProductionReadinessClassification(
        candidate_id=candidate.candidate_id,
        inventory_status=inventory_status,
        family=family,
        module=candidate.module_path,
        symbol=candidate.symbol,
        responsibility=f"{family.value}:{candidate.indicator}",
        semantic_responsibility_key=semantic_key,
        semantic_owner=owner,
        canonical_contract=contract,
        composition_owner=composition,
        ownership_resolution=ownership_resolution,
        production_callers=(candidate.module_path,),
        tenant_relevance=tenant_relevance,
        applicable_frz_ids=_FRZ_BY_FAMILY.get(family, ()),
        current_head_evidence=(
            f"discovery:{candidate.discovery_kind.value}",
            f"indicator:{candidate.indicator}",
        ),
        evidence_state=evidence,
        finding_disposition=disposition,
    )


@lru_cache(maxsize=1)
def classified_inventory() -> tuple[ProductionReadinessClassification, ...]:
    return tuple(classify_candidate(c) for c in raw_discovery_candidates())


def classification_by_candidate_id() -> dict[str, ProductionReadinessClassification]:
    return {record.candidate_id: record for record in classified_inventory()}

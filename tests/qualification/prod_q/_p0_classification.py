# © Artur Czarnecki. All rights reserved.

"""Classify discovered PROD-Q production readiness candidates."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_discovery import raw_discovery_candidates
from tests.qualification.prod_q._p0_findings_register import (
    enrich_sandbox_factory_anchor_classification,
)
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
    MechanismRole,
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
    (
        "intergrax/contracts/execution_capacity/admission_decision.py",
        "assess_root_execution_capacity",
    ): (
        ProductionReadinessOwner.EXECUTION_CAPACITY,
        "intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
        "intergrax/contracts/execution_capacity/admission_decision.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/hosting/engine/engine.py",
        "health_snapshot",
    ): (
        ProductionReadinessOwner.HOSTING_ENGINE,
        "intergrax.hosting.engine.engine.HostedApplicationEngine.health_snapshot",
        "intergrax/hosting/engine/engine.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
    (
        "intergrax/runtime/observability/auditability_health.py",
        "build_auditability_health_snapshot",
    ): (
        ProductionReadinessOwner.OBSERVABILITY_AUDITABILITY,
        "intergrax.runtime.observability.auditability_health",
        "intergrax/runtime/observability/auditability_health.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/applications/_shared/capability_health/providers/dependency_validation.py",
        "DependencyValidationHealthProvider",
    ): (
        ProductionReadinessOwner.CAPABILITY_HEALTH,
        "intergrax.applications._shared.capability_health.providers.dependency_validation",
        "intergrax/applications/_shared/capability_health/providers/dependency_validation.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
    (
        "intergrax/contracts/execution_reliability/shutdown_contract.py",
        "ExecutionRuntimeShutdownPhase",
    ): (
        ProductionReadinessOwner.EXECUTION_RUNTIME,
        "intergrax.contracts.execution_reliability.shutdown_contract",
        "intergrax/contracts/execution_reliability/shutdown_contract.py",
        ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD,
    ),
    (
        "intergrax/fastapi_core/routers/health.py",
        "health",
    ): (
        ProductionReadinessOwner.HOSTING_ENGINE,
        "intergrax.fastapi_core.routers.health",
        "intergrax/fastapi_core/routers/health.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
    (
        "applications/governed_contractor_application/host/factory.py",
        "create_governed_contractor_backend_app",
    ): (
        ProductionReadinessOwner.APPLICATION_HOST_WIRING,
        "applications.governed_contractor_application.host.factory",
        "applications/governed_contractor_application/host/factory.py",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
    ),
}

_CONTRACT_IMPORT_OWNERS: dict[str, tuple[ProductionReadinessOwner, str, str]] = {
    "intergrax.contracts.execution_capacity_admission": (
        ProductionReadinessOwner.EXECUTION_CAPACITY,
        "intergrax.contracts.execution_capacity_admission",
        "intergrax/contracts/execution_capacity_admission.py",
    ),
    "intergrax.contracts.execution_reliability": (
        ProductionReadinessOwner.EXECUTION_RUNTIME,
        "intergrax.contracts.execution_reliability",
        "intergrax/contracts/execution_reliability",
    ),
    "intergrax.contracts.runtime_execution_admission": (
        ProductionReadinessOwner.EXECUTION_RUNTIME,
        "intergrax.contracts.runtime_execution_admission",
        "intergrax/contracts/runtime_execution_admission",
    ),
    "intergrax.contracts.functional_evidence": (
        ProductionReadinessOwner.OBSERVABILITY_AUDITABILITY,
        "intergrax.contracts.functional_evidence",
        "intergrax/contracts/functional_evidence",
    ),
    "intergrax.contracts.idempotency": (
        ProductionReadinessOwner.EXECUTION_RUNTIME,
        "intergrax.contracts.idempotency",
        "intergrax/contracts/idempotency",
    ),
}

_OPERATIONAL_RESPONSIBILITY_INDICATORS: frozenset[str] = frozenset(
    {
        "startup",
        "shutdown",
        "health",
        "readiness",
        "admission",
        "sandbox_provider",
        "sandbox",
        "materialization",
        "capacity",
        "unsupported",
        "mode",
        "degraded",
        "recovery",
    }
)

_ANCHOR_OPERATION_BOUNDARY: dict[tuple[str, str], str] = {
    (
        "intergrax/contracts/execution_capacity_admission.py",
        "ExecutionCapacityAdmissionPort",
    ): "concern:operation:contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
    (
        "intergrax/runtime/execution/local_execution_capacity_admission.py",
        "LocalExecutionCapacityAdmission",
    ): "concern:operation:contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
    (
        "intergrax/contracts/execution_capacity/admission_decision.py",
        "assess_root_execution_capacity",
    ): "concern:operation:contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
    (
        "intergrax/applications/_shared/auditability_health_wiring.py",
        "project_host_auditability_health_facts",
    ): "concern:operation:contract:intergrax.runtime.observability.auditability_health",
    (
        "intergrax/runtime/observability/auditability_health.py",
        "build_auditability_health_snapshot",
    ): "concern:operation:contract:intergrax.runtime.observability.auditability_health",
    (
        "intergrax/hosting/engine/engine.py",
        "HostedApplicationEngine",
    ): "concern:operation:PQ-F04:health:hosted_application_engine",
    (
        "intergrax/hosting/engine/engine.py",
        "health_snapshot",
    ): "concern:operation:PQ-F04:health:hosted_application_engine_snapshot",
    (
        "intergrax/applications/_shared/capability_health/providers/dependency_validation.py",
        "DependencyValidationHealthProvider",
    ): "concern:operation:PQ-F04:health:capability_dependency_validation",
    (
        "intergrax/fastapi_core/routers/health.py",
        "health",
    ): "concern:operation:PQ-F04:health:http_router_surface",
    (
        "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
        "create_e2b_sandbox_host_integration",
    ): "concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor",
    (
        "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
        "create_modal_sandbox_host_integration",
    ): "concern:operation:PQ-F09:sandbox_provider:modal:factory_anchor",
    (
        "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
        "create_daytona_sandbox_host_integration",
    ): "concern:operation:PQ-F09:sandbox_provider:daytona:factory_anchor",
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


def _auxiliary_eligible(candidate: ProductionReadinessCandidate) -> bool:
    if candidate.indicator in _OPERATIONAL_RESPONSIBILITY_INDICATORS:
        return False
    if candidate.discovery_kind in (
        DiscoveryKind.ANCHOR_REGISTRY,
        DiscoveryKind.PROVIDER_SANDBOX_HOST,
        DiscoveryKind.AST_IMPORT_CONTRACT,
    ):
        return False
    return True


def _mechanism_role(candidate: ProductionReadinessCandidate) -> MechanismRole:
    if candidate.indicator not in _INDICATOR_TO_FAMILY:
        return MechanismRole.UNRESOLVED_CLASSIFICATION
    if candidate.discovery_kind in (
        DiscoveryKind.ANCHOR_REGISTRY,
        DiscoveryKind.PROVIDER_SANDBOX_HOST,
        DiscoveryKind.AST_IMPORT_CONTRACT,
        DiscoveryKind.AST_LIFESPAN_HANDLER,
        DiscoveryKind.AST_STRICT_LAB_BRANCH,
    ):
        return MechanismRole.SEMANTIC_MECHANISM
    if candidate.indicator in _OPERATIONAL_RESPONSIBILITY_INDICATORS:
        return MechanismRole.SEMANTIC_MECHANISM
    if _auxiliary_eligible(candidate):
        return MechanismRole.AUXILIARY_DISCOVERY
    return MechanismRole.UNRESOLVED_CLASSIFICATION


def _semantic_concern_key(
    candidate: ProductionReadinessCandidate,
    owner: ProductionReadinessOwner,
    contract: str,
    mechanism_role: MechanismRole,
    family: ProductionReadinessFamily,
) -> str:
    if mechanism_role == MechanismRole.AUXILIARY_DISCOVERY:
        return f"auxiliary:{candidate.module_path}:{candidate.symbol}:{candidate.indicator}"
    if mechanism_role == MechanismRole.UNRESOLVED_CLASSIFICATION:
        return (
            f"concern:unresolved:classification:{family.value}:"
            f"{candidate.module_path}:{candidate.symbol}:{candidate.indicator}"
        )
    anchor_key = (candidate.module_path, candidate.symbol)
    if anchor_key in _ANCHOR_OPERATION_BOUNDARY:
        return _ANCHOR_OPERATION_BOUNDARY[anchor_key]
    if candidate.discovery_kind == DiscoveryKind.PROVIDER_SANDBOX_HOST:
        slug = candidate.module_path.split("/")[-2]
        return f"concern:operation:PQ-F09:sandbox_provider:{slug}"
    if not contract.startswith("CONTRACT_"):
        return f"concern:operation:contract:{contract}"
    if candidate.discovery_kind == DiscoveryKind.ANCHOR_REGISTRY:
        return f"concern:unresolved:anchor:{candidate.module_path}:{candidate.symbol}"
    if candidate.discovery_kind == DiscoveryKind.AST_IMPORT_CONTRACT:
        prefix = candidate.symbol.removeprefix("import:")
        return f"concern:import:{prefix}"
    if candidate.indicator in _OPERATIONAL_RESPONSIBILITY_INDICATORS:
        return f"concern:operation:{family.value}:{candidate.indicator}"
    return (
        f"concern:unresolved:semantic:{family.value}:"
        f"{candidate.module_path}:{candidate.symbol}"
    )


def classify_candidate(candidate: ProductionReadinessCandidate) -> ProductionReadinessClassification:
    mechanism_role = _mechanism_role(candidate)
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
    elif candidate.discovery_kind == DiscoveryKind.AST_IMPORT_CONTRACT:
        prefix = candidate.symbol.removeprefix("import:")
        import_meta = _CONTRACT_IMPORT_OWNERS.get(prefix)
        if import_meta is not None:
            owner, contract, composition = import_meta
            evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
            ownership_resolution = OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
        else:
            owner = ProductionReadinessOwner.DISCOVERED_SURFACE
            contract = "CONTRACT_UNRESOLVED"
            composition = "COMPOSITION_UNRESOLVED"
            evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
            ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    elif candidate.discovery_kind == DiscoveryKind.PROVIDER_SANDBOX_HOST:
        owner = ProductionReadinessOwner.SANDBOX_HOST_PROVIDER
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED
        ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    elif mechanism_role == MechanismRole.AUXILIARY_DISCOVERY:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = candidate.module_path
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    elif mechanism_role == MechanismRole.UNRESOLVED_CLASSIFICATION:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
    elif candidate.discovery_kind == DiscoveryKind.ANCHOR_REGISTRY:
        owner = ProductionReadinessOwner.DISCOVERED_SURFACE
        contract = "CONTRACT_UNRESOLVED"
        composition = "COMPOSITION_UNRESOLVED"
        evidence = ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD
        ownership_resolution = OwnershipResolutionState.NOT_YET_RESOLVED
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

    concern_key = _semantic_concern_key(
        candidate, owner, contract, mechanism_role, family
    )

    head_evidence: tuple[str, ...] = (
        f"discovery:{candidate.discovery_kind.value}",
        f"indicator:{candidate.indicator}",
        f"mechanism_role:{mechanism_role.value}",
    )
    return ProductionReadinessClassification(
        candidate_id=candidate.candidate_id,
        inventory_status=inventory_status,
        family=family,
        module=candidate.module_path,
        symbol=candidate.symbol,
        responsibility=f"{family.value}:{candidate.indicator}",
        mechanism_role=mechanism_role,
        semantic_concern_key=concern_key,
        semantic_owner=owner,
        canonical_contract=contract,
        composition_owner=composition,
        ownership_resolution=ownership_resolution,
        production_callers=(candidate.module_path,),
        tenant_relevance=tenant_relevance,
        applicable_frz_ids=_FRZ_BY_FAMILY.get(family, ()),
        current_head_evidence=head_evidence,
        evidence_state=evidence,
        finding_disposition=disposition,
    )


def _finalize_classified_inventory(
    preliminary: tuple[ProductionReadinessClassification, ...],
    raw: tuple[ProductionReadinessCandidate, ...],
) -> tuple[ProductionReadinessClassification, ...]:
    from tests.qualification.prod_q._p0_ownership import (
        apply_auxiliary_finalization,
        build_auxiliary_justifications,
    )

    justifications = build_auxiliary_justifications(raw, preliminary)
    return tuple(
        enrich_sandbox_factory_anchor_classification(
            apply_auxiliary_finalization(record, justifications.get(record.candidate_id))
        )
        for record in preliminary
    )


@lru_cache(maxsize=1)
def classified_inventory() -> tuple[ProductionReadinessClassification, ...]:
    raw = raw_discovery_candidates()
    preliminary = tuple(classify_candidate(c) for c in raw)
    return _finalize_classified_inventory(preliminary, raw)


def classification_by_candidate_id() -> dict[str, ProductionReadinessClassification]:
    return {record.candidate_id: record for record in classified_inventory()}

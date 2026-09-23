# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R3 specialized memory governance architecture decision SSOT (qualification)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R3_MEMORY_ADR_PATH,
    Gr12Applicability,
    Gr12CoverageStatus,
)


class Gr12MemoryArchitecturePhase(StrEnum):
    ARCHITECTURE_DECISION_CLOSED = "ARCHITECTURE_DECISION_CLOSED"


class Gr12MemoryAuthorityModel(StrEnum):
    """Single permission authority for memory mutations (no parallel CLA-04 today)."""

    SPECIALIZED_DOMAIN_AUTHORITY = "SPECIALIZED_DOMAIN_AUTHORITY"


class Gr12MemoryCla04Applicability(StrEnum):
    """CLA-04 is not authority for current production memory mutation paths."""

    NOT_APPLICABLE_CURRENT_PRODUCTION = "NOT_APPLICABLE_CURRENT_PRODUCTION"
    REQUIRED_FOR_FUTURE_LIVE_OPERATOR_ONLY = "REQUIRED_FOR_FUTURE_LIVE_OPERATOR_ONLY"


class Gr12MemoryMutationContextClass(StrEnum):
    EXECUTION_DATA_PLANE = "EXECUTION_DATA_PLANE"
    BACKGROUND_LIFECYCLE = "BACKGROUND_LIFECYCLE"
    LIVE_OPERATOR_CONTROL_PLANE = "LIVE_OPERATOR_CONTROL_PLANE"
    BOOTSTRAP_MIGRATION = "BOOTSTRAP_MIGRATION"
    TEST_ONLY = "TEST_ONLY"


class Gr12MemoryGr12Applicability(StrEnum):
    """GR-12 control-plane applicability per mutation class (R3 split)."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    APPLICABLE_WHEN_LIVE_OPERATOR_INTRODUCED = "APPLICABLE_WHEN_LIVE_OPERATOR_INTRODUCED"


class Gr12MemoryMutationGovernanceBinding(StrEnum):
    MEMORY_NATIVE_ENFORCED = "MEMORY_NATIVE_ENFORCED"
    MEMORY_NATIVE_DISCLOSURE_ONLY = "MEMORY_NATIVE_DISCLOSURE_ONLY"
    NOT_PRODUCTION = "NOT_PRODUCTION"


@dataclass(frozen=True, slots=True)
class Gr12MemoryMutationSurface:
    mutation: str
    entrypoint: str
    context_class: Gr12MemoryMutationContextClass
    gr12_applicability: Gr12MemoryGr12Applicability
    governance: Gr12MemoryMutationGovernanceBinding
    notes: str


GR12_MEMORY_EVALUATION_REQUEST_CONTRACT: Final[str] = (
    "intergrax.memory.contracts.memory_security_governance.MemoryGovernanceEvaluationRequest"
)

GR12_MEMORY_DECISION_CONTRACT: Final[str] = (
    "intergrax.memory.contracts.memory_security_governance.MemoryGovernanceDecision"
)

GR12_MEMORY_GOVERNANCE_SERVICE: Final[str] = (
    "intergrax.memory.memory_security_governance_service.MemorySecurityGovernanceService"
)

GR12_MEMORY_POLICY_PORT: Final[str] = (
    "intergrax.memory.contracts.memory_security_governance.MemorySecurityStrategySet"
)

GR12_MEMORY_POLICY_EVALUATOR_PORTS: Final[tuple[str, ...]] = (
    "intergrax.memory.contracts.memory_security_governance.MemoryAuthorizationPolicy",
    "intergrax.memory.contracts.memory_security_governance.MemoryTrustEvaluationPolicy",
    "intergrax.memory.contracts.memory_security_governance.MemoryAdmissionPolicy",
    "intergrax.memory.contracts.memory_security_governance.MemoryGovernancePolicy",
    "intergrax.memory.contracts.memory_security_governance.MemoryRetentionPolicy",
)

GR12_MEMORY_MUTATION_ENFORCEMENT_HELPER: Final[str] = (
    "intergrax.memory.memory_specialized_mutation_governance.enforce_specialized_memory_mutation"
)

GR12_MEMORY_PREFERRED_AUTHORITY_FLOW: Final[str] = (
    "RequestIdentity (execution) → domain service builds MemoryGovernanceEvaluationRequest → "
    "MemorySecurityGovernanceService.evaluate → MemoryGovernanceDecision → "
    "domain store mutation (LTM / entity / procedure / long-horizon)"
)

GR12_MEMORY_IDENTITY_MODEL: Final[str] = (
    "MemorySecurityContext.identity: RequestIdentity; scope: MemoryControlScopeRef "
    "(tenant + user via MemoryControlPlaneScope); entity scopes bind tenant_id/user_id; "
    "canonical source authority validates revision for specialized projections"
)

GR12_MEMORY_EVIDENCE_MODEL: Final[str] = (
    "MemoryGovernanceDecision: outcome, reason_code, policy_id, policy_version, operation, "
    "optional trust_class/data_classification/retention_action/constraints/subject_memory_id; "
    "no platform decision_id field; diagnostics via memory observability emitters (not GR-8 fact)"
)

GR12_MEMORY_TOCTOU_STRATEGY: Final[str] = (
    "CanonicalMemoryGovernanceSourceAuthority + validate_canonical_governance_source_snapshot "
    "for projection mutations; user-profile control plane uses entry revision on supersede/delete; "
    "no CLA-04-style post-authorization CAS for memory domain today"
)

GR12_MEMORY_FAIL_CLOSED_STRATEGY: Final[str] = (
    "Missing strategies, evaluator exceptions, invalid merged decision → DENY; "
    "enforce_specialized_memory_mutation raises MemoryGovernanceDenied → zero store write"
)

GR12_MEMORY_NEXT_BOUNDED_TASK: Final[str] = (
    "GR-12-A4-R3-R1 — Memory Specialized Governance Qualification"
)

GR12_MEMORY_QUALIFICATION_PROOF: Final[str] = (
    "tests/qualification/governance/gr12/"
    "test_gr12_a4_r3_memory_specialized_governance_architecture_qualification.py"
)

GR12_MEMORY_MUTATION_SURFACES: tuple[Gr12MemoryMutationSurface, ...] = (
    Gr12MemoryMutationSurface(
        mutation="user_profile.remember / supersede / delete",
        entrypoint="intergrax.memory.default_memory_control_plane",
        context_class=Gr12MemoryMutationContextClass.EXECUTION_DATA_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED,
        notes="MemorySecurityGovernanceService via _enforce_governance; execution RequestIdentity.",
    ),
    Gr12MemoryMutationSurface(
        mutation="long_horizon.compact / promote summaries",
        entrypoint="intergrax.memory.long_horizon_memory_service.LongHorizonMemoryService",
        context_class=Gr12MemoryMutationContextClass.BACKGROUND_LIFECYCLE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED,
        notes="Compaction policy + enforce_specialized_memory_mutation on persist paths.",
    ),
    Gr12MemoryMutationSurface(
        mutation="procedure.remember / supersede / deprecate / delete_projection",
        entrypoint="intergrax.memory.procedural_memory_service.ProceduralMemoryService",
        context_class=Gr12MemoryMutationContextClass.EXECUTION_DATA_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED,
        notes="enforce_specialized_memory_mutation before store upsert/supersede/delete.",
    ),
    Gr12MemoryMutationSurface(
        mutation="entity projection index / relation projection",
        entrypoint="intergrax.memory.entity_memory_indexing.EntityMemoryIndexingService",
        context_class=Gr12MemoryMutationContextClass.EXECUTION_DATA_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED,
        notes="PROJECT/UPDATE/DELETE governance before graph upsert.",
    ),
    Gr12MemoryMutationSurface(
        mutation="procedure projection indexing",
        entrypoint="intergrax.memory.procedural_memory_indexing.ProceduralMemoryIndexingService",
        context_class=Gr12MemoryMutationContextClass.EXECUTION_DATA_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_ENFORCED,
        notes="Governed projection writes from canonical procedure records.",
    ),
    Gr12MemoryMutationSurface(
        mutation="entity / procedure / LTM recall disclosure",
        entrypoint=(
            "intergrax.memory.entity_temporal_memory_service, "
            "intergrax.memory.procedural_memory_service, "
            "intergrax.memory.default_memory_reference_reader"
        ),
        context_class=Gr12MemoryMutationContextClass.EXECUTION_DATA_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.NOT_APPLICABLE,
        governance=Gr12MemoryMutationGovernanceBinding.MEMORY_NATIVE_DISCLOSURE_ONLY,
        notes="RECALL governance filters disclosure; not GR-12 control-plane mutation.",
    ),
    Gr12MemoryMutationSurface(
        mutation="live operator administrative memory API",
        entrypoint="(none in intergrax.applications production)",
        context_class=Gr12MemoryMutationContextClass.LIVE_OPERATOR_CONTROL_PLANE,
        gr12_applicability=Gr12MemoryGr12Applicability.APPLICABLE_WHEN_LIVE_OPERATOR_INTRODUCED,
        governance=Gr12MemoryMutationGovernanceBinding.NOT_PRODUCTION,
        notes="Future path must use single authority (memory validation → CLA-04), not dual ALLOW.",
    ),
)

GR12_MEMORY_REJECTED_ALTERNATIVES: Final[tuple[str, ...]] = (
    "Dual independent MemoryGovernanceDecision AND ControlPlaneMutationDecision for the same mutation.",
    "Mechanical CLA-04 adapter on all execution-time memory writes (fake unification).",
    "Global memory mutation executor in governance/runtime (violates domain-owner mutation).",
    "Default ALLOW or synthetic principal when governance dependency missing.",
    "Provider/storage-specific policy inside repository adapters.",
)

GR12_MEMORY_QUALIFICATION_GATES: Final[tuple[str, ...]] = (
    "memory_mutation_inventory_present",
    "single_authority_per_mutation_no_cla04_in_memory_domain",
    "no_dual_authority_decision_model",
    "public_memory_policy_ports_documented",
    "live_operator_surface_absent_classified",
    "execution_and_background_not_gr12_cp",
    "vector_catalog_gr10_status_unchanged",
    "next_bounded_task_memory_specialized_qualification",
)


@dataclass(frozen=True, slots=True)
class Gr12A4R3MemoryArchitectureDecision:
    architecture_phase: Gr12MemoryArchitecturePhase
    authority_model: Gr12MemoryAuthorityModel
    memory_native_authority: bool
    cla04_applicability: Gr12MemoryCla04Applicability
    live_operator_surface_exists: bool
    execution_write_gr12_applicability: Gr12Applicability
    background_mutation_gr12_applicability: Gr12Applicability
    operator_mutation_gr12_applicability: Gr12Applicability
    policy_port: str
    identity_model: str
    evidence_model: str
    toctou_strategy: str
    fail_closed_strategy: str
    cp_mem_catalog_applicability: Gr12Applicability
    cp_mem_catalog_coverage: Gr12CoverageStatus
    dual_independent_authority: bool
    adr_path: str
    preferred_authority_flow: str
    next_bounded_task: str
    qualification_proof: str
    rejected_alternatives: tuple[str, ...]


GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION: Gr12A4R3MemoryArchitectureDecision = (
    Gr12A4R3MemoryArchitectureDecision(
        architecture_phase=Gr12MemoryArchitecturePhase.ARCHITECTURE_DECISION_CLOSED,
        authority_model=Gr12MemoryAuthorityModel.SPECIALIZED_DOMAIN_AUTHORITY,
        memory_native_authority=True,
        cla04_applicability=Gr12MemoryCla04Applicability.NOT_APPLICABLE_CURRENT_PRODUCTION,
        live_operator_surface_exists=False,
        execution_write_gr12_applicability=Gr12Applicability.NOT_APPLICABLE,
        background_mutation_gr12_applicability=Gr12Applicability.NOT_APPLICABLE,
        operator_mutation_gr12_applicability=Gr12Applicability.NOT_APPLICABLE,
        policy_port=GR12_MEMORY_POLICY_PORT,
        identity_model=GR12_MEMORY_IDENTITY_MODEL,
        evidence_model=GR12_MEMORY_EVIDENCE_MODEL,
        toctou_strategy=GR12_MEMORY_TOCTOU_STRATEGY,
        fail_closed_strategy=GR12_MEMORY_FAIL_CLOSED_STRATEGY,
        cp_mem_catalog_applicability=Gr12Applicability.NOT_APPLICABLE,
        cp_mem_catalog_coverage=Gr12CoverageStatus.NOT_APPLICABLE,
        dual_independent_authority=False,
        adr_path=GR12_A4_R3_MEMORY_ADR_PATH,
        preferred_authority_flow=GR12_MEMORY_PREFERRED_AUTHORITY_FLOW,
        next_bounded_task=GR12_MEMORY_NEXT_BOUNDED_TASK,
        qualification_proof=GR12_MEMORY_QUALIFICATION_PROOF,
        rejected_alternatives=GR12_MEMORY_REJECTED_ALTERNATIVES,
    )
)

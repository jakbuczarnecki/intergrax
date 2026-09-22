# © Artur Czarnecki. All rights reserved.

"""GR-12-A4 residual control-plane path classification SSOT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R2_R1_QUALIFICATION_PROOF,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.gr12_a4_r2_vector_architecture_decision import (
    GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION,
    GR12_VECTOR_CANONICAL_PORT,
    GR12_VECTOR_NEXT_BOUNDED_TASK,
)
from tests.qualification.governance.gr12.gr12_a4_r3_memory_architecture_decision import (
    GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION,
    GR12_MEMORY_EVALUATION_REQUEST_CONTRACT,
)


class Gr12A4PathKind(StrEnum):
    MUTATION_SURFACE = "MUTATION_SURFACE"
    ARCHITECTURE_DECISION_SURFACE = "ARCHITECTURE_DECISION_SURFACE"


class Gr12MemoryCla04Compatibility(StrEnum):
    """§26 specialized memory classification."""

    A_CLA04_COMPATIBLE = "A"
    B_CLA04_AFTER_INTERNAL_ADAPTER = "B"
    C_SEPARATE_POLICY_EVIDENCE_CONTRACT = "C"
    D_NOT_CONTROL_PLANE = "D"


class Gr12CatalogRevisionSemantics(StrEnum):
    PRESENT = "PRESENT"
    ABSENT_REPORTED_GAP = "ABSENT_REPORTED_GAP"


class Gr12OperatorApiExposure(StrEnum):
    OPERATOR_REQUEST_PATH = "OPERATOR_REQUEST_PATH"
    COMPOSE_OR_MAINTENANCE_ONLY = "COMPOSE_OR_MAINTENANCE_ONLY"
    NOT_CURRENTLY_EXPOSED = "NOT_CURRENTLY_EXPOSED"


@dataclass(frozen=True, slots=True)
class Gr12A4ResidualInventoryRow:
    path_id: str
    path_kind: Gr12A4PathKind
    production_entrypoint: str
    mutation_owner: str
    current_authority: str
    consequential: bool
    existing_contract: str
    coverage: Gr12CoverageStatus
    cla04_reuse_blocker: str
    operator_exposure: Gr12OperatorApiExposure
    tenant_scope: str


@dataclass(frozen=True, slots=True)
class Gr12A4CatalogDecision:
    revision_semantics: Gr12CatalogRevisionSemantics
    host_compose_wired: bool
    stale_cas_possible: bool
    preferred_governance_model: str
    architecture_blocker: str


@dataclass(frozen=True, slots=True)
class Gr12A4VectorDecision:
    canonical_port: str
    provider_impl_entrypoint: str
    destructive_ops_on_port: bool
    cla04_mapping_decision_required: bool
    architecture_blocker: str
    architecture_phase: str
    cla04_applicability: Gr12Applicability
    live_operator_surface_exists: bool
    prepare_index_governance: str
    next_bounded_task: str
    qualification_proof: str


@dataclass(frozen=True, slots=True)
class Gr12A4MemoryDecision:
    contract: str
    cla04_compatibility: Gr12MemoryCla04Compatibility
    architecture_blocker: str
    architecture_options: tuple[str, ...]


GR12_A4_RESIDUAL_PATH_IDS: Final[tuple[str, ...]] = (
    "CP-PLUGIN-CATALOG-HOT-RELOAD",
    "CP-VECTOR-INDEX-ADMIN",
    "CP-MEM-SPECIALIZED-MUTATION",
)

GR12_A4_RESIDUAL_INVENTORY: tuple[Gr12A4ResidualInventoryRow, ...] = (
    Gr12A4ResidualInventoryRow(
        path_id="CP-PLUGIN-CATALOG-HOT-RELOAD",
        path_kind=Gr12A4PathKind.MUTATION_SURFACE,
        production_entrypoint=(
            "intergrax.applications._shared.catalog_hot_reload_service."
            "CatalogHotReloadService.reload"
        ),
        mutation_owner="intergrax.integrations.registry (in-process _CATALOG)",
        current_authority="composition-injected ControlPlaneMutationAuthorizationBoundary",
        consequential=True,
        existing_contract="ControlPlaneMutationRequest (CLA-04) + CatalogRevision CAS",
        coverage=Gr12CoverageStatus.QUALIFIED,
        cla04_reuse_blocker="",
        operator_exposure=Gr12OperatorApiExposure.OPERATOR_REQUEST_PATH,
        tenant_scope="host-global in-process registry (platform runtime)",
    ),
    Gr12A4ResidualInventoryRow(
        path_id="CP-VECTOR-INDEX-ADMIN",
        path_kind=Gr12A4PathKind.MUTATION_SURFACE,
        production_entrypoint=(
            "intergrax.applications._shared.vector_index_admin_service."
            "VectorIndexAdminService.prepare"
        ),
        mutation_owner=(
            "VectorIndexAdminService (applications) → VectorIndexAdministration port"
        ),
        current_authority="composition-injected ControlPlaneMutationAuthorizationBoundary",
        consequential=True,
        existing_contract="ControlPlaneMutationRequest (CLA-04) + configuration revision digest",
        coverage=Gr12CoverageStatus.QUALIFIED,
        cla04_reuse_blocker="",
        operator_exposure=Gr12OperatorApiExposure.OPERATOR_REQUEST_PATH,
        tenant_scope="tenant_id on VectorIndexIdentity (required; per-index tenant scope)",
    ),
    Gr12A4ResidualInventoryRow(
        path_id="CP-MEM-SPECIALIZED-MUTATION",
        path_kind=Gr12A4PathKind.ARCHITECTURE_DECISION_SURFACE,
        production_entrypoint="intergrax.memory.memory_specialized_mutation_governance",
        mutation_owner="memory domain (LTM/entity/procedure/summary writes)",
        current_authority="MemorySecurityGovernanceService + MemoryGovernanceEvaluationRequest",
        consequential=False,
        existing_contract=GR12_MEMORY_EVALUATION_REQUEST_CONTRACT,
        coverage=Gr12CoverageStatus.NOT_APPLICABLE,
        cla04_reuse_blocker=(
            "Production paths are execution/background domain writes — GR-12 NOT_APPLICABLE; "
            "future live operator API must use single authority (not dual ALLOW)"
        ),
        operator_exposure=Gr12OperatorApiExposure.NOT_CURRENTLY_EXPOSED,
        tenant_scope="MemoryControlPlaneScope (user/tenant scoped)",
    ),
)

GR12_A4_CATALOG_DECISION: Gr12A4CatalogDecision = Gr12A4CatalogDecision(
    revision_semantics=Gr12CatalogRevisionSemantics.PRESENT,
    host_compose_wired=False,
    stale_cas_possible=True,
    preferred_governance_model=(
        "CatalogHotReloadOperatorRequest → revision snapshot → candidate → CLA-04 → "
        "post-authorization CAS → atomic registry replace → ControlPlaneMutationAuthorizationEvidence"
    ),
    architecture_blocker="",
)

GR12_A4_VECTOR_DECISION: Gr12A4VectorDecision = Gr12A4VectorDecision(
    canonical_port=GR12_VECTOR_CANONICAL_PORT,
    provider_impl_entrypoint=(
        "intergrax.integrations.providers.vector_store.qdrant.index_administration."
        "QdrantVectorIndexAdministration"
    ),
    destructive_ops_on_port=False,
    cla04_mapping_decision_required=False,
    architecture_blocker="",
    architecture_phase=GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION.architecture_phase.value,
    cla04_applicability=GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION.cla04_applicability,
    live_operator_surface_exists=True,
    prepare_index_governance=GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION.prepare_index_decision.value,
    next_bounded_task=GR12_VECTOR_NEXT_BOUNDED_TASK,
    qualification_proof=GR12_A4_R2_R1_QUALIFICATION_PROOF,
)

GR12_A4_MEMORY_DECISION: Gr12A4MemoryDecision = Gr12A4MemoryDecision(
    contract=GR12_MEMORY_EVALUATION_REQUEST_CONTRACT,
    cla04_compatibility=Gr12MemoryCla04Compatibility.D_NOT_CONTROL_PLANE,
    architecture_blocker="",
    architecture_options=GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION.rejected_alternatives,
)

GR12_A4_CLASSIFICATION_PROOF: Final[str] = (
    "tests/qualification/governance/gr12/"
    "test_gr12_a4_residual_control_plane_surface_qualification.py::test_gr12_a4_residual_paths_classified"
)

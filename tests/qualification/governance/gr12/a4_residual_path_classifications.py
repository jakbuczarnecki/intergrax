# © Artur Czarnecki. All rights reserved.

"""GR-12-A4 residual control-plane path classification SSOT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.governance.gr12.catalog import Gr12CoverageStatus


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
        path_kind=Gr12A4PathKind.ARCHITECTURE_DECISION_SURFACE,
        production_entrypoint=(
            "intergrax.integrations.contracts.vector_index_administration.VectorIndexAdministration"
        ),
        mutation_owner="integrations vector index administration port (provider adapters)",
        current_authority="integration credentials / bootstrap callers only",
        consequential=True,
        existing_contract="VectorIndexAdministration (provider-neutral port; no CLA-04 bridge)",
        coverage=Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED,
        cla04_reuse_blocker=(
            "No governed operator admin API; CLA-04 resource mapping for prepare/drop/reindex "
            "undecided; destructive lifecycle not on neutral port"
        ),
        operator_exposure=Gr12OperatorApiExposure.NOT_CURRENTLY_EXPOSED,
        tenant_scope="tenant_id on VectorIndexIdentity (per-index)",
    ),
    Gr12A4ResidualInventoryRow(
        path_id="CP-MEM-SPECIALIZED-MUTATION",
        path_kind=Gr12A4PathKind.ARCHITECTURE_DECISION_SURFACE,
        production_entrypoint="intergrax.memory.memory_specialized_mutation_governance",
        mutation_owner="memory domain (LTM/entity/procedure/summary writes)",
        current_authority="MemorySecurityGovernanceService + MemoryGovernanceEvaluationRequest",
        consequential=True,
        existing_contract="MemoryGovernanceEvaluationRequest (parallel to CLA-04)",
        coverage=Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED,
        cla04_reuse_blocker=(
            "Record revision + canonical source semantics; MemoryGovernanceDecision evidence "
            "≠ ControlPlaneMutationAuthorizationEvidence; not GR-10 MSE"
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
    canonical_port=(
        "intergrax.integrations.contracts.vector_index_administration.VectorIndexAdministration"
    ),
    provider_impl_entrypoint=(
        "intergrax.integrations.providers.vector_store.qdrant.index_administration."
        "QdrantVectorIndexAdministration"
    ),
    destructive_ops_on_port=False,
    cla04_mapping_decision_required=True,
    architecture_blocker=(
        "Provider-neutral port exists but production admin mutations are bootstrap/internal "
        "callers only; CLA-04 mutation_type + revision binding for index admin undecided."
    ),
)

GR12_A4_MEMORY_DECISION: Gr12A4MemoryDecision = Gr12A4MemoryDecision(
    contract="intergrax.memory.contracts.memory_security_governance.MemoryGovernanceEvaluationRequest",
    cla04_compatibility=Gr12MemoryCla04Compatibility.C_SEPARATE_POLICY_EVIDENCE_CONTRACT,
    architecture_blocker=(
        "Specialized memory mutations use memory-native governance; semantic unification via "
        "generic ControlPlaneMutationRequest would be fake unification (§28)."
    ),
    architecture_options=(
        "Option 1: Document memory governance as parallel certified plane (keep CLA-04 for CP mutations only).",
        "Option 2: Internal adapter mapping MemoryGovernanceEvaluationRequest → CLA-04 without new public port.",
        "Option 3: New public MemoryControlPlaneMutation port bridging to GR-8 facts (ADR required).",
    ),
)

GR12_A4_CLASSIFICATION_PROOF: Final[str] = (
    "tests/qualification/governance/gr12/"
    "test_gr12_a4_residual_control_plane_surface_qualification.py::test_gr12_a4_residual_paths_classified"
)

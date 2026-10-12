# © Artur Czarnecki. All rights reserved.

"""Typed models for PROD-Q-P0 closed-world production readiness inventory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProductionReadinessFamily(str, Enum):
    PQ_F01 = "PQ-F01"
    PQ_F02 = "PQ-F02"
    PQ_F03 = "PQ-F03"
    PQ_F04 = "PQ-F04"
    PQ_F05 = "PQ-F05"
    PQ_F06 = "PQ-F06"
    PQ_F07 = "PQ-F07"
    PQ_F08 = "PQ-F08"
    PQ_F09 = "PQ-F09"
    PQ_F10 = "PQ-F10"
    PQ_F11 = "PQ-F11"
    PQ_F12 = "PQ-F12"
    PQ_F13 = "PQ-F13"
    PQ_F14 = "PQ-F14"
    PQ_F15 = "PQ-F15"
    PQ_F16 = "PQ-F16"
    PQ_F17 = "PQ-F17"
    PQ_F18 = "PQ-F18"
    PQ_F19 = "PQ-F19"
    PQ_F20 = "PQ-F20"
    PQ_F21 = "PQ-F21"
    PQ_F22 = "PQ-F22"
    PQ_F23 = "PQ-F23"


ALL_PRODUCTION_READINESS_FAMILIES: frozenset[ProductionReadinessFamily] = frozenset(
    ProductionReadinessFamily
)


class ProductionReadinessEvidenceState(str, Enum):
    PROVEN_CURRENT_HEAD = "PROVEN_CURRENT_HEAD"
    SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED = "SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED"
    PARTIAL_CURRENT_HEAD = "PARTIAL_CURRENT_HEAD"
    BLOCKER = "BLOCKER"
    EXTERNAL_PROVIDER_EVIDENCE_REQUIRED = "EXTERNAL_PROVIDER_EVIDENCE_REQUIRED"
    OUTSIDE_PLATFORM_RESPONSIBILITY_WITH_EVIDENCE = "OUTSIDE_PLATFORM_RESPONSIBILITY_WITH_EVIDENCE"
    N_A_WITH_EVIDENCE = "N_A_WITH_EVIDENCE"


class ProdQFindingDisposition(str, Enum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"


class ProductionReadinessOwner(str, Enum):
    HOSTING_ENGINE = "hosting.engine"
    EXECUTION_RUNTIME = "execution.runtime"
    EXECUTION_CAPACITY = "execution.capacity"
    OBSERVABILITY_AUDITABILITY = "observability.auditability"
    CAPABILITY_HEALTH = "capability.health"
    CONFIGURATION_RESOLUTION = "configuration.resolution"
    INTEGRATION_PROVIDER_SURFACE = "integration.provider_surface"
    RELATIONAL_STORE_SQLITE = "relational_store.sqlite"
    SANDBOX_HOST_PROVIDER = "sandbox_host.provider"
    APPLICATION_HOST_WIRING = "application.host_wiring"
    STATE_RECOVERY = "state.recovery"
    TENANT_ISOLATION_REUSE = "tenant.isolation_reuse"
    DISCOVERED_SURFACE = "discovered.surface"
    DOCUMENTED_ARCHITECTURE_ONLY = "documented.architecture_only"


class DiscoveryKind(str, Enum):
    ANCHOR_REGISTRY = "anchor.registry"
    AST_SYMBOL_INDICATOR = "ast.symbol.indicator"
    AST_LIFESPAN_HANDLER = "ast.lifespan.handler"
    AST_IMPORT_CONTRACT = "ast.import.contract"
    AST_STRICT_LAB_BRANCH = "ast.strict_lab.branch"
    PROVIDER_SANDBOX_HOST = "provider.sandbox_host"


@dataclass(frozen=True, slots=True)
class ProductionReadinessCandidate:
    candidate_id: str
    module_path: str
    symbol: str
    discovery_kind: DiscoveryKind
    line_no: int
    indicator: str


@dataclass(frozen=True, slots=True)
class ProductionReadinessClassification:
    family: ProductionReadinessFamily
    module: str
    symbol: str
    responsibility: str
    semantic_owner: ProductionReadinessOwner
    canonical_contract: str
    composition_owner: str
    production_callers: tuple[str, ...]
    tenant_relevance: bool
    applicable_frz_ids: tuple[str, ...]
    current_head_evidence: tuple[str, ...]
    evidence_state: ProductionReadinessEvidenceState
    finding_disposition: ProdQFindingDisposition


@dataclass(frozen=True, slots=True)
class ScanFailure:
    module_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class ClosedWorldDiscoveryReport:
    module_paths_scanned: tuple[str, ...]
    scan_failures: tuple[ScanFailure, ...]
    raw_candidates: tuple[ProductionReadinessCandidate, ...]
    unclassified_candidate_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class FamilyCoverageRecord:
    family: ProductionReadinessFamily
    title: str
    evidence_state: ProductionReadinessEvidenceState
    finding_disposition: ProdQFindingDisposition
    candidate_count: int
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OwnershipRow:
    responsibility_key: str
    semantic_owner: ProductionReadinessOwner
    canonical_contract: str
    composition_owner: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OwnershipMatrixReport:
    rows: tuple[OwnershipRow, ...]
    duplicate_semantic_owner_keys: tuple[str, ...]
    duplicate_composition_owner_keys: tuple[str, ...]
    unresolved_ownership_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrzCriterionRecord:
    criterion_id: str
    domain: str
    summary: str
    evidence_state: ProductionReadinessEvidenceState
    finding_disposition: ProdQFindingDisposition
    pq_families: tuple[ProductionReadinessFamily, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceItem:
    stage_id: str
    accepted_sha: str
    replay_required: bool


@dataclass(frozen=True, slots=True)
class ProviderQualificationDisposition:
    provider_id: str
    installed: bool
    registered: bool
    configured: bool
    qualified: bool
    production_approved: bool
    evidence_state: ProductionReadinessEvidenceState
    finding_disposition: ProdQFindingDisposition
    notes: str


@dataclass(frozen=True, slots=True)
class R1SqliteEnv01Disposition:
    finding_id: str
    default_db_path: str
    default_db_header_probe: str
    isolated_env_var: str
    evidence_state: ProductionReadinessEvidenceState
    finding_disposition: ProdQFindingDisposition
    recommended_child: str
    reproduction_command: str


@dataclass(frozen=True, slots=True)
class TenantIsolationAuditProdQ:
    tenant_scope_applicable: bool
    verdict: str
    reuse_tenant_x_sha: str
    production_path_contradictions: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProdQP0Matrix:
    production_files_scanned: int
    candidates_discovered: int
    candidates_classified: int
    unclassified_candidates: int
    families_covered: int
    frz_criteria_classified: int
    in_scope_blockers: int


@dataclass(frozen=True, slots=True)
class ProdQP0CertificationReport:
    audited_head: str
    matrix: ProdQP0Matrix
    discovery_report: ClosedWorldDiscoveryReport
    family_coverage: tuple[FamilyCoverageRecord, ...]
    ownership: OwnershipMatrixReport
    frz_records: tuple[FrzCriterionRecord, ...]
    provider_dispositions: tuple[ProviderQualificationDisposition, ...]
    r1_sqlite: R1SqliteEnv01Disposition
    tenant_audit: TenantIsolationAuditProdQ
    recommended_children: tuple[str, ...]

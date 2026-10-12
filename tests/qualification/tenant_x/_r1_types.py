# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 typed certification model (family-based; not per-surface inventory)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.tenant_x._types import (
    PropagationRelation,
    TenantAuthorityOwnerDisposition,
    TenantAuthorityScopeEvidence,
)


class R1ProvenanceStatus(StrEnum):
    PROVEN_CURRENT_HEAD = "PROVEN_CURRENT_HEAD"
    HISTORICAL_REPLAY_CONFIRMED = "HISTORICAL_REPLAY_CONFIRMED"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R1SilentSubstitutionClass(StrEnum):
    LEGAL_NON_TENANT = "LEGAL_NON_TENANT"
    COMPATIBILITY_SEAM = "COMPATIBILITY_SEAM"
    TENANT_REQUIRED_ILLEGAL_SUBSTITUTION = "TENANT_REQUIRED_ILLEGAL_SUBSTITUTION"
    UNKNOWN = "UNKNOWN"


class R1MechanismAuthorityClass(StrEnum):
    """R1 reconciliation — not all P0 mint surfaces are competing tenant authorities."""

    SANCTIONED_SEMANTIC_AUTHORITY = "SANCTIONED_SEMANTIC_AUTHORITY"
    APPLICATION_LOCAL_INGRESS = "APPLICATION_LOCAL_INGRESS"
    CONFORMANCE_OR_QUALIFICATION_ONLY = "CONFORMANCE_OR_QUALIFICATION_ONLY"
    OBSERVABILITY_NON_TENANT = "OBSERVABILITY_NON_TENANT"
    INTEGRATION_DEMO_OR_DEV_SEAM = "INTEGRATION_DEMO_OR_DEV_SEAM"
    EVALUATION_OR_LOAD_HARNESS = "EVALUATION_OR_LOAD_HARNESS"


class R1AuthorityDuplicateParticipation(StrEnum):
    """Whether a P0 mechanism may compete in critical duplicate-authority review."""

    CRITICAL_AUTHORITY = "CRITICAL_AUTHORITY"
    SUBORDINATE_DERIVATION = "SUBORDINATE_DERIVATION"
    APPLICATION_LOCAL = "APPLICATION_LOCAL"
    HARNESS_ONLY = "HARNESS_ONLY"
    OBSERVABILITY_NON_AUTHORITY = "OBSERVABILITY_NON_AUTHORITY"
    DEV_COMPATIBILITY_SEAM = "DEV_COMPATIBILITY_SEAM"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class R1CriticalBoundaryFamily:
    family_id: str
    label: str
    frz_ten_criteria: tuple[str, ...]
    provenance_status: R1ProvenanceStatus
    relation: PropagationRelation | None
    structural_evidence: tuple[str, ...]
    handoff_wave: str | None = None


@dataclass(frozen=True, slots=True)
class R1PropagationEdgeCertification:
    edge_id: str
    source: str
    target: str
    relation: PropagationRelation
    current_head_evidence: tuple[str, ...]
    historical_evidence: tuple[str, ...]
    final_status: R1ProvenanceStatus


@dataclass(frozen=True, slots=True)
class R1AuthorityMechanismResolution:
    mechanism_id: str
    decision_kind: str
    semantic_owner: str
    authority_scope: str
    scope_evidence: TenantAuthorityScopeEvidence
    owner_disposition: TenantAuthorityOwnerDisposition
    authority_class: R1MechanismAuthorityClass
    canonical_contract: str
    notes: str
    owner_evidence: tuple[str, ...] = ()
    scope_evidence_refs: tuple[str, ...] = ()
    production_classification_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class R1StructuralPropagationProof:
    source_symbol: str
    target_symbol: str
    production_file: str
    source_expression: str
    target_expression: str
    relation: PropagationRelation
    fail_closed: str | None = None


@dataclass(frozen=True, slots=True)
class R1SilentSubstitutionRow:
    pattern_id: str
    surface_hint: str
    classification: R1SilentSubstitutionClass
    critical_path: bool
    evidence: tuple[str, ...]


SANCTIONED_ROOT_TENANT_CONTRACT: Final[str] = (
    "intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmissionRequest"
)
